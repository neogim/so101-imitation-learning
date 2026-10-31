import importlib.util
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("evaluate_mapping_hypotheses_offline.py")
SPEC = importlib.util.spec_from_file_location("mapping_ik", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MappingCoordinateTests(unittest.TestCase):
    def test_forward_inverse_round_trip_all_hypotheses(self):
        q = np.array([-20.0, 31.0, -42.0, 53.0, -64.0])
        for signs in MODULE.HYPOTHESES.values():
            urdf = MODULE.motor_to_urdf_degrees(q, signs)
            np.testing.assert_allclose(MODULE.urdf_to_motor_degrees(urdf, signs), q)

    def test_negative_sign_reverses_asymmetric_limits(self):
        lower = np.deg2rad(np.array([-10.0, -20.0]))
        upper = np.deg2rad(np.array([30.0, 40.0]))
        lo, hi = MODULE.urdf_limits_in_motor_degrees(lower, upper, np.array([-1.0, 1.0]))
        np.testing.assert_allclose(lo, [-30.0, -20.0])
        np.testing.assert_allclose(hi, [10.0, 40.0])

    def test_derivatives_use_motor_coordinates_and_timestamps(self):
        q = np.array([[0.0], [1.0], [3.0]])
        velocity, acceleration = MODULE.derivatives(q, np.array([0.0, 0.5, 1.0]))
        self.assertAlmostEqual(velocity[1, 0], 2.0)
        self.assertAlmostEqual(velocity[2, 0], 4.0)
        self.assertAlmostEqual(acceleration[2, 0], 4.0)

    def test_sign_aware_regularization_is_computed_in_motor_coordinates(self):
        signs = np.array([1.0, -1.0])
        q_urdf = np.deg2rad(np.array([10.0, -20.0]))
        terms = MODULE.regularization_terms(
            q_urdf, signs, np.array([8.0, 18.0]), np.array([9.0, 19.0]), .05, .01
        )
        np.testing.assert_allclose(terms, [.1, .1, .01, .01], atol=1e-12)

    def test_peak_reports_frame_and_joint(self):
        values = np.array([[np.nan, np.nan], [1.0, -4.0], [3.0, 2.0]])
        old_names = MODULE.JOINT_NAMES
        MODULE.JOINT_NAMES = ("a", "b")
        try:
            self.assertEqual(MODULE.peak(values, np.array([5, 6, 7]), skip=1),
                             {"value": 4.0, "frame": 6, "joint": "b"})
        finally:
            MODULE.JOINT_NAMES = old_names

    def test_small_deterministic_regularized_position_ik(self):
        fit = MODULE.least_squares(lambda q: np.array([q[0] - 1.0, .1 * q[0]]), np.array([0.0]))
        self.assertTrue(fit.success)
        self.assertAlmostEqual(fit.x[0], 1.0 / 1.01, places=7)


if __name__ == "__main__":
    unittest.main()
