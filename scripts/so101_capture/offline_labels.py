"""Label completed SO-101 episodes in a separate file; never edit source data."""
import argparse
import hashlib
import json
from pathlib import Path

CHOICES = ('success', 'failure', 'unknown')
NORMAL_ENDINGS = ('manual', 'time_limit')
ENDINGS = NORMAL_ENDINGS + ('interrupted', 'unknown')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def source_identity(source):
    source = source.expanduser().resolve()
    info_path = source / 'meta/info.json'
    info = json.loads(info_path.read_text())
    outcomes_path = source / 'episode_outcomes.json'
    outcomes = json.loads(outcomes_path.read_text()) if outcomes_path.exists() else []
    from pyarrow import parquet as pq
    episodes = []
    for path in sorted((source/'meta/episodes').rglob('*.parquet')):
        episodes.extend(pq.read_table(path).to_pylist())
    counts = {int(row['episode_index']): int(row['length']) for row in episodes}
    if len(counts) != info['total_episodes'] or sorted(counts) != list(range(info['total_episodes'])):
        raise ValueError('Episode metadata does not match info.json')
    if outcomes:
        if sorted(int(r['episode_index']) for r in outcomes) != sorted(counts):
            raise ValueError('Outcome episode IDs do not match metadata')
        for row in outcomes:
            if int(row['frames']) != counts[int(row['episode_index'])]:
                raise ValueError('Outcome frame count does not match metadata')
    ending = {int(r['episode_index']): r['end_reason'] for r in outcomes}
    identity = dict(source=str(source), source_info_sha256=digest(info_path),
                    source_outcomes_sha256=digest(outcomes_path))
    return identity, counts, ending


def validate_labels(document, source):
    identity, counts, endings = source_identity(source)
    if document.get('schema_version') != 1 or any(document.get(k) != v for k, v in identity.items()):
        raise ValueError('Label file does not match this source or its metadata')
    labels = document.get('labels')
    if not isinstance(labels, list) or sorted(r['episode_index'] for r in labels) != sorted(counts):
        raise ValueError('Label episode IDs do not match source')
    for row in labels:
        ep = row['episode_index']
        if row['frames'] != counts[ep] or row['end_reason'] not in ENDINGS:
            raise ValueError(f'Invalid episode metadata in label {ep}')
        if ep in endings and row['end_reason'] != endings[ep]:
            raise ValueError(f'End reason changed for episode {ep}')
        if any(row.get(key) not in CHOICES for key in ('pick','place','overall')):
            raise ValueError(f'Invalid status in label {ep}')
        if type(row.get('adopt')) is not bool or not isinstance(row.get('notes'), str):
            raise ValueError(f'Invalid adopt/notes in label {ep}')
        if row['adopt'] and (row['end_reason'] not in NORMAL_ENDINGS or
                             any(row[key] != 'success' for key in ('pick','place','overall'))):
            raise ValueError(f'Interrupted, failed or unknown episode {ep} cannot be adopted')
    return labels


def update_labels(source, output, episode_index, *, pick=None, place=None, overall=None,
                  adopt=None, notes=None, end_reason=None):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError('Label file must be outside the original dataset')
    if not output.parent.is_dir():
        raise ValueError(f'Create label output directory first: {output.parent}')
    identity, counts, endings = source_identity(source)
    if episode_index not in counts:
        raise ValueError(f'Unknown episode index {episode_index}')
    if output.exists():
        document = json.loads(output.read_text())
        validate_labels(document, source)
    else:
        document = dict(schema_version=1, **identity, labels=[
            dict(episode_index=ep, frames=count, end_reason=endings.get(ep,'unknown'),
                 pick='unknown',place='unknown',overall='unknown',adopt=False,notes='')
            for ep,count in sorted(counts.items())])
    row = next(r for r in document['labels'] if r['episode_index'] == episode_index)
    if end_reason is not None:
        if episode_index in endings and end_reason != endings[episode_index]:
            raise ValueError('End reason cannot contradict original outcome')
        row['end_reason'] = end_reason
    for key,value in (('pick',pick),('place',place),('overall',overall),('notes',notes)):
        if value is not None:
            row[key] = value
    if adopt is not None:
        row['adopt'] = adopt
    elif any(value is not None for value in (pick,place,overall,end_reason)):
        row['adopt'] = False  # Outcome edits require an explicit new adoption decision.
    validate_labels(document, source)
    tmp = output.with_suffix(output.suffix+'.tmp')
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(output)
    return document


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episode', type=int, required=True)
    for field in ('pick','place','overall'):
        p.add_argument('--'+field, choices=CHOICES)
    p.add_argument('--end-reason', choices=ENDINGS,
                   help='Needed for older datasets without episode_outcomes.json')
    p.add_argument('--adopt', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--notes')
    a=p.parse_args()
    doc=update_labels(a.source,a.output,a.episode,pick=a.pick,place=a.place,
                      overall=a.overall,adopt=a.adopt,notes=a.notes,end_reason=a.end_reason)
    print(json.dumps(doc['labels'][a.episode],indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
