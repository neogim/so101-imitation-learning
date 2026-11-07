"""Synthetic regression test for initial Eye-to-Hand center-only estimator."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_side_camera_from_id1 import model_centers, solve_center_fit, marker_side_m


class EyeToHandCenterTests(unittest.TestCase):
    def test_mixed_size_id1(self):
        self.assertAlmostEqual(marker_side_m({"marker_sizes_mm_by_id": {"1": 25}}, 1), .025)
        with self.assertRaises(ValueError):
            marker_side_m({"marker_sizes_mm_by_id": {"0": 14}}, 1)

    def test_recovery_of_camera_and_marker_center(self):
        rng = np.random.default_rng(42)
        n = 24
        poses = np.repeat(np.eye(4)[None, :, :], n, axis=0)
        poses[:, :3, :3] = Rotation.from_euler(
            "xyz", rng.uniform(-65, 65, (n, 3)), degrees=True).as_matrix()
        poses[:, :3, 3] = rng.uniform(-.2, .2, (n, 3))
        actual = np.r_[Rotation.from_euler("xyz", [15., -30., 110.],
                                          degrees=True).as_rotvec(),
                       [.25, -.15, .6], [.022, -.012, .03]]
        observed = model_centers(actual, poses)
        result = solve_center_fit(poses, observed)
        self.assertTrue(result["identifiable_9dof"])
        self.assertLess(result["holdout_error_mm"]["rms"], .01)
        np.testing.assert_allclose(result["p_gripper_marker_center_m"], actual[6:], atol=1e-5)
        np.testing.assert_allclose(np.array(result["T_camera_base"])[:3, 3], actual[3:6], atol=1e-5)

    def test_too_few_samples(self):
        with self.assertRaises(ValueError):
            solve_center_fit(np.repeat(np.eye(4)[None, :, :], 7, axis=0), np.zeros((7, 3)))


if __name__ == "__main__":
    unittest.main()
