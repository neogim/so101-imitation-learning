"""External labels keep raw dataset metadata immutable."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from offline_labels import update_labels, validate_labels


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class OfflineLabelTests(unittest.TestCase):
    def fixture(self, base, outcomes=True):
        root=base/'raw';(root/'meta/episodes/chunk-000').mkdir(parents=True)
        (root/'meta/info.json').write_text(json.dumps(dict(total_episodes=2)))
        pq.write_table(pa.Table.from_pylist([dict(episode_index=0,length=3),
                                             dict(episode_index=1,length=2)]),
                       root/'meta/episodes/chunk-000/file-000.parquet')
        if outcomes:
            (root/'episode_outcomes.json').write_text(json.dumps([
                dict(episode_index=0,frames=3,end_reason='manual'),
                dict(episode_index=1,frames=2,end_reason='interrupted')]))
        return root

    def test_completed_unknown_can_be_labeled_without_source_write(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=self.fixture(base);before=hashes(root)
            target=base/'labels.json'
            doc=update_labels(root,target,0)
            self.assertEqual(doc['labels'][0]['overall'],'unknown')
            self.assertFalse(doc['labels'][0]['adopt'])
            doc=update_labels(root,target,0,pick='success',place='success',overall='success',adopt=True)
            self.assertTrue(doc['labels'][0]['adopt'])
            self.assertEqual(doc['labels'][1]['end_reason'],'interrupted')
            self.assertEqual(validate_labels(json.loads(target.read_text()),root),doc['labels'])
            self.assertEqual(hashes(root),before)

    def test_interrupted_and_partial_success_cannot_be_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=self.fixture(base);target=base/'labels.json'
            with self.assertRaisesRegex(ValueError,'cannot be adopted'):
                update_labels(root,target,1,pick='success',place='success',overall='success',adopt=True)
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(ValueError,'cannot be adopted'):
                update_labels(root,target,0,pick='success',place='failure',overall='success',adopt=True)
            self.assertFalse(target.exists())

    def test_older_dataset_requires_explicit_end_reason_for_adoption(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=self.fixture(base,outcomes=False);target=base/'labels.json'
            with self.assertRaisesRegex(ValueError,'cannot be adopted'):
                update_labels(root,target,0,pick='success',place='success',overall='success',adopt=True)
            doc=update_labels(root,target,0,end_reason='manual',pick='success',
                              place='success',overall='success',adopt=True)
            self.assertTrue(doc['labels'][0]['adopt'])

    def test_source_mismatch_and_inside_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=self.fixture(base);target=base/'labels.json'
            with self.assertRaisesRegex(ValueError,'outside'):
                update_labels(root,root/'labels.json',0)
            update_labels(root,target,0)
            (root/'meta/info.json').write_text(json.dumps(dict(total_episodes=2,changed=True)))
            with self.assertRaisesRegex(ValueError,'does not match'):
                update_labels(root,target,0)


if __name__=='__main__':unittest.main()
