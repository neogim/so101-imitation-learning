"""Synthetic regression tests for pairwise ArUco/FK diversity audit."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_aruco_diversity import audit_diversity


class DiversityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.urdf = Path(__file__).resolve().parents[2] / "data/urdf/so101/so101_new_calib.urdf"

    def make_manifest(self, path):
        def sample(frame, q, marker_ids):
            names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
            return {"episode_index": 0, "frame_index": frame,
                    "joints": dict(zip(names, q)),
                    "markers": [{"id": mid} for mid in marker_ids]}
        rows = [
            sample(2, [0, 0, 0, 0, 0], [0, 2]),
            sample(112, [0, 0, 0, 0, 60], [0, 2]),
            sample(300, [20, 5, 15, 10, 60], [0, 2]),
            sample(410, [20, 5, 15, 10, 60], [0, 2]),
        ]
        path.write_text(json.dumps({"samples": rows}), encoding="utf-8")

    def test_j5_only_changes_are_not_distinct_for_id2(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            self.make_manifest(manifest)
            result = audit_diversity(self.urdf, manifest)
        pair = next(pair for pair in result["pairwise_joint_near_repeats"]
                    if pair["a"]["frame"] == 2 and pair["b"]["frame"] == 112)
        self.assertFalse(pair["all_five_near_duplicate"])
        self.assertTrue(pair["wrist_fk_near_duplicate_candidate"])
        wrist_near = {(x["a"]["frame"], x["b"]["frame"])
                      for x in result["per_marker"]["2"]["fk_pose_near_repeat_pairs"]}
        self.assertIn((2, 112), wrist_near)

    def test_exact_repeated_pose_caught_nonconsecutively(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            self.make_manifest(manifest)
            result = audit_diversity(self.urdf, manifest)
        repeats = {(p["a"]["frame"], p["b"]["frame"])
                   for p in result["pairwise_joint_near_repeats"]
                   if p["all_five_near_duplicate"]}
        self.assertIn((300, 410), repeats)
        self.assertEqual(result["per_marker"]["0"]["observations"], 4)
        self.assertEqual(result["per_marker"]["2"]["observations"], 4)

    def test_reject_invalid_threshold_and_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            self.make_manifest(manifest)
            with self.assertRaises(ValueError):
                audit_diversity(self.urdf, manifest, duplicate_deg=0)
            manifest.write_text('{"samples":[]}')
            with self.assertRaises(ValueError):
                audit_diversity(self.urdf, manifest)


if __name__ == "__main__":
    unittest.main()
