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


if __name__ == "__main__":
    unittest.main()
