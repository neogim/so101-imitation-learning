"""Offline-only tests: no hardware, camera, or external dataset required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_aruco_calibration import stationary_candidates


class StationarySelectionTests(unittest.TestCase):
    def test_one_pose_after_motion(self):
        import numpy as np
        moving = np.column_stack([np.linspace(0, 20, 20), np.zeros((20, 5))])
        hold = np.tile([20, 3, -2, 0, 0, 20], (22, 1))
        states = np.vstack([moving, hold])
        chosen = stationary_candidates(states, window=10, tolerance_deg=.5)
        self.assertEqual(len(chosen), 1)
        self.assertGreaterEqual(chosen[0], 20)

    def test_separate_distinct_holds(self):
        import numpy as np
        a = np.tile([0, 0, 0, 0, 0, 0], (15, 1))
        transition = np.column_stack([np.linspace(0, 15, 15), np.zeros((15, 5))])
        b = np.tile([15, 0, 0, 0, 0, 40], (15, 1))
        states = np.vstack([a, transition, b])
        chosen = stationary_candidates(states, window=10)
        self.assertEqual(len(chosen), 2)

    def test_gripper_alone_does_not_count_as_new_pose(self):
        import numpy as np
        states = np.tile([0, 0, 0, 0, 0, 0], (35, 1)).astype(float)
        states[20:, 5] = 80
        self.assertEqual(len(stationary_candidates(states, window=10)), 1)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            stationary_candidates([[float("nan")] * 6] * 20)

    def test_bad_width_rejected(self):
        with self.assertRaises(ValueError):
            stationary_candidates([[0] * 5] * 20)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            stationary_candidates([[0] * 6] * 20, window=1)


if __name__ == "__main__":
    unittest.main()
