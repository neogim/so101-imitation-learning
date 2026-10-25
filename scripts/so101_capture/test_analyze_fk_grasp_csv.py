"""Unit tests for the read-only SO-101 FK summary utility."""
import csv
import tempfile
import unittest
from pathlib import Path

from analyze_fk_grasp_csv import load_rows, summarize


def row(ep, x=0.2):
    return {"ep": str(ep), "open": "1", "close_start": "2", "grasp": "3",
            "x": str(x), "y": "0.2", "z": "0.01",
            "rx": "0", "ry": "0", "rz": "0", "sigma_min": "0.12"}


class FKSummaryTests(unittest.TestCase):
    def save(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "grasp.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row(0)))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_center_and_rms(self):
        result = summarize(load_rows(self.save([row(0, 0.2), row(1, 0.22)])))
        self.assertEqual(result["episodes"], 2)
        self.assertAlmostEqual(result["center_mm"]["x"], 210)
        self.assertAlmostEqual(result["distance_to_mean_mm"]["rms"], 10)

    def test_duplicate_episode_fails(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_rows(self.save([row(0), row(0)]))

    def test_bad_phase_order_fails(self):
        item = row(0)
        item["grasp"] = "0"
        with self.assertRaisesRegex(ValueError, "phase order"):
            load_rows(self.save([item]))

    def test_empty_csv_fails(self):
        with self.assertRaisesRegex(ValueError, "Empty"):
            load_rows(self.save([]))


if __name__ == "__main__":
    unittest.main()
