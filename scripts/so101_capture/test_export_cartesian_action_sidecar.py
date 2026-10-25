"""No robot or placo required for numeric Cartesian target tests."""
import math
import unittest
import numpy as np

from export_cartesian_action_sidecar import rotation_log, target_from_fk


class CartesianActionTest(unittest.TestCase):
    @staticmethod
    def fk(q):
        a = math.radians(q[0])
        T = np.eye(4)
        T[0, 3] = q[1] / 1000
        T[1, 3] = q[2] / 1000
        T[:3, :3] = np.array([
            [math.cos(a), -math.sin(a), 0],
            [math.sin(a), math.cos(a), 0],
            [0, 0, 1],
        ])
        return T

    def test_identity(self):
        s = [0, 1, 2, 3, 4, 5]
        x = target_from_fk(s, s, self.fk)
        np.testing.assert_allclose(x[:6], 0, atol=1e-12)
        self.assertEqual(x[6], 5)

    def test_translation_rotation_gripper(self):
        s = [0, 0, 0, 0, 0, 0]
        a = [90, 20, -10, 0, 0, 30]
        x = target_from_fk(s, a, self.fk)
        np.testing.assert_allclose(x[:3], [0.02, -0.01, 0], atol=1e-12)
        np.testing.assert_allclose(x[3:6], [0, 0, math.pi / 2], atol=1e-10)
        self.assertEqual(x[6], 30)

    def test_rotation_near_pi(self):
        R = np.diag([-1., -1., 1.])
        v = rotation_log(R)
        self.assertAlmostEqual(np.linalg.norm(v), math.pi)

    def test_reject_nan(self):
        with self.assertRaises(ValueError):
            target_from_fk([float("nan")] + [0] * 5, [0] * 6, self.fk)

    def test_reject_bad_shape(self):
        with self.assertRaises(ValueError):
            target_from_fk([0] * 5, [0] * 6, self.fk)


if __name__ == "__main__":
    unittest.main()
