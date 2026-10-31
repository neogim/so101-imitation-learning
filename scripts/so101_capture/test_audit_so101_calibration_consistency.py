import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("audit_so101_calibration_consistency.py")
SPEC = importlib.util.spec_from_file_location("calibration_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CalibrationAuditTests(unittest.TestCase):
    def test_degree_half_span_matches_lerobot_formula(self):
        self.assertTrue(np.isclose(MODULE.degree_half_span(907, 3122), 97.36263736263736))

    def test_tolerance_excludes_float_endpoint_noise_and_one_tick_at_point_one(self):
        values = np.array([97.36264038, 97.45054626])
        high = MODULE.degree_half_span(907, 3122)
        self.assertEqual(MODULE.exceed_count(values, -high, high, 0.0), 2)
        self.assertEqual(MODULE.exceed_count(values, -high, high, 0.1), 0)

    def test_contiguous_runs_preserve_episode_boundaries(self):
        episodes = np.array([1, 1, 1, 2, 2])
        frames = np.array([3, 4, 7, 0, 1])
        mask = np.array([True, True, True, True, False])
        self.assertEqual(MODULE.contiguous_runs(episodes, frames, mask), [
            {"episode": 1, "frame_start": 3, "frame_end": 4},
            {"episode": 1, "frame_start": 7, "frame_end": 7},
            {"episode": 2, "frame_start": 0, "frame_end": 0},
        ])


if __name__ == "__main__":
    unittest.main()
