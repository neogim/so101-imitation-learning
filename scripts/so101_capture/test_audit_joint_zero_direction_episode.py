import unittest
import numpy as np

from audit_joint_zero_direction_episode import axis_angle, select_pairs


class PairSelectionTests(unittest.TestCase):
    def test_prefers_isolated_target_motion(self):
        states = np.zeros((40, 5))
        states[10:, 2] = 30
        states[20:, 0] = 80
        states[20:, 1] = 80
        pairs = select_pairs(states, min_gap=2, max_gap=15)
        p = pairs[2]
        self.assertEqual(p["delta_target_deg"], 30)
        self.assertEqual(p["other_joint_normalized_l2"], 0)

    def test_axis_angle_is_right_handed(self):
        r = axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        np.testing.assert_allclose(r @ np.array([1.0, 0.0, 0.0]), [0, 1, 0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
