import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("fit_joint_mapping_offline.py")
SPEC = importlib.util.spec_from_file_location("mapping_fit", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rows(sign, offset, values, noise=None):
    noise = noise or [0.0] * len(values)
    return [
        {"joint": "elbow_flex", "lerobot_deg": x, "urdf_deg": sign * x + offset + n,
         "observation_id": f"pose-{i}", "measurement_source": "external inclinometer"}
        for i, (x, n) in enumerate(zip(values, noise))
    ]


class MappingFitTests(unittest.TestCase):
    def test_positive_sign_and_offset(self):
        result = MODULE.analyze_joint(rows(1, 12.5, [-30, 0, 40]))
        self.assertEqual(result["numerically_preferred_sign"], 1)
        self.assertAlmostEqual(result["hypotheses"][0]["offset_deg"], 12.5)
        self.assertAlmostEqual(result["hypotheses"][0]["leave_one_out_rmse_deg"], 0.0)

    def test_negative_sign_and_offset(self):
        result = MODULE.analyze_joint(rows(-1, -7.0, [-25, 5, 45]))
        self.assertEqual(result["numerically_preferred_sign"], -1)
        self.assertAlmostEqual(result["hypotheses"][1]["offset_deg"], -7.0)

    def test_outlier_is_visible_in_residuals(self):
        result = MODULE.analyze_joint(rows(1, 3.0, [-40, -20, 0, 20, 40], [0, 0, 15, 0, 0]))
        plus = result["hypotheses"][0]
        self.assertEqual(result["numerically_preferred_sign"], 1)
        self.assertGreater(plus["max_absolute_error_deg"], 10)
        self.assertGreater(plus["leave_one_out_rmse_deg"], plus["rmse_deg"])

    def test_degenerate_single_angle_is_insufficient(self):
        result = MODULE.analyze_joint(rows(1, 2.0, [10, 10]))
        self.assertFalse(result["sufficient_angle_diversity"])
        self.assertIsNone(result["numerically_preferred_sign"])
        self.assertEqual(result["mapping_status"], "insufficient_evidence")

    def test_synthetic_sources_are_not_independent(self):
        evidence = rows(1, 0, [-10, 10])
        for row in evidence:
            row["measurement_source"] = "synthetic example"
        result = MODULE.analyze_joint(evidence)
        self.assertFalse(result["sufficient_independent_references"])
        self.assertFalse(result["physically_validated"])

    def test_csv_rejects_unknown_joint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MODULE.REQUIRED)
                writer.writeheader()
                writer.writerow({"joint": "bad", "lerobot_deg": 0, "urdf_deg": 0,
                                 "observation_id": "x", "measurement_source": "fixture"})
            with self.assertRaisesRegex(ValueError, "unknown joint"):
                MODULE.read_evidence(path)


if __name__ == "__main__":
    unittest.main()
