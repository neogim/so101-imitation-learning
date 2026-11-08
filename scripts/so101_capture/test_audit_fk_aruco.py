"""Unit tests for offline URDF FK audit. No cameras or robot connections."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_fk_aruco import (axis_rotation, forward_kinematics,
                             homogeneous_rotation, parse_urdf,
                             rotation_distance_deg, to_matrix)


class UrdfFkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[2]
        cls.graph, cls.limits = parse_urdf(
            repo / "data/urdf/so101/so101_new_calib.urdf")

    @staticmethod
    def angles():
        return dict(shoulder_pan=5., shoulder_lift=-10.,
                    elbow_flex=25., wrist_flex=30., wrist_roll=15.)

    def test_rpy_is_urdf_rz_ry_rx(self):
        np.testing.assert_allclose(homogeneous_rotation([0, 0, math.pi / 2]),
                                   [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)

    def test_rodrigues_around_z(self):
        r = axis_rotation([0, 0, 3], math.pi / 2)
        np.testing.assert_allclose(r, [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                                   atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1.)

    def test_identity_at_base(self):
        np.testing.assert_allclose(forward_kinematics(
            self.graph, self.angles(), "base_link"), np.eye(4))

    def test_j5_only_changes_gripper_rotation_not_wrist(self):
        q1 = self.angles()
        q2 = dict(q1, wrist_roll=q1["wrist_roll"] + 10.)
        w1 = forward_kinematics(self.graph, q1, "wrist_link")
        w2 = forward_kinematics(self.graph, q2, "wrist_link")
        g1 = forward_kinematics(self.graph, q1, "gripper_link")
        g2 = forward_kinematics(self.graph, q2, "gripper_link")
        np.testing.assert_allclose(w1, w2, atol=1e-11)
        np.testing.assert_allclose(g1[:3, 3], g2[:3, 3], atol=1e-11)
        self.assertAlmostEqual(rotation_distance_deg(g1, g2), 10., places=5)

    def test_wrist_is_affected_by_j4(self):
        q1 = self.angles()
        q2 = dict(q1, wrist_flex=q1["wrist_flex"] + 10.)
        w1 = forward_kinematics(self.graph, q1, "wrist_link")
        w2 = forward_kinematics(self.graph, q2, "wrist_link")
        self.assertAlmostEqual(rotation_distance_deg(w1, w2), 10., places=5)

    def test_j2_lower_limit_from_urdf(self):
        lo, hi = self.limits["shoulder_lift"]
        self.assertAlmostEqual(lo, math.degrees(-1.74533), places=4)
        self.assertLess(-103.21, lo)
        self.assertGreater(hi, 99.)

    def test_invalid_axis_rejected(self):
        with self.assertRaises(ValueError):
            axis_rotation([0, 0, 0], 1.)

    def test_missing_angles_rejected(self):
        with self.assertRaises(ValueError):
            forward_kinematics(self.graph, {"shoulder_pan": 0},
                               "wrist_link")


if __name__ == "__main__":
    unittest.main()
