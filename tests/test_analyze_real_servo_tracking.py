import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts" / "so101_sim" / "analyze_real_servo_tracking.py"
SPEC = importlib.util.spec_from_file_location("analyze_real_servo_tracking", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class ServoTrackingAnalysisTests(unittest.TestCase):
    def test_regions_join_short_gaps(self):
        mask = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1], dtype=bool)
        self.assertEqual(MOD.contiguous_regions(mask, max_gap=1), [(1, 5), (8, 9)])

    def test_derivative_lag(self):
        action = np.r_[np.zeros(10), np.arange(20), np.full(10, 19.0)]
        state = np.r_[np.zeros(3), action[:-3]]
        self.assertEqual(MOD.derivative_lag_frames(action, state, 6), 3)

    def test_stationary_arm_closure_is_detected(self):
        frames = 50
        action = np.zeros((frames, 6), dtype=float)
        state = np.zeros_like(action)
        action[:, 5] = 80
        action[10:31, 5] = np.linspace(80, 40, 21)
        action[31:, 5] = 40
        state[:, 5] = np.r_[np.full(3, 80.0), action[:-3, 5]]
        _, closures = MOD.analyze_episode(action, state, 20.0, 7, 8)
        self.assertEqual(len(closures), 1)
        self.assertTrue(closures[0]["arm_command_hold"])
        self.assertAlmostEqual(closures[0]["j2_range_motor_deg"], 0.0)


if __name__ == "__main__":
    unittest.main()
