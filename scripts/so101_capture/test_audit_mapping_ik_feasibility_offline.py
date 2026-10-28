import importlib.util
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("audit_mapping_ik_feasibility_offline.py")
SPEC = importlib.util.spec_from_file_location("multistart", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MultistartTests(unittest.TestCase):
    def test_seeds_are_deterministic_unique_and_bounded(self):
        q = np.array([0.0, -101.0, 96.0, 0.0, 0.0])
        signs = np.array([1, 1, -1, 1, 1], dtype=float)
        lower = np.deg2rad(np.array([-110, -100, -96.83, -95, -157]))
        upper = np.deg2rad(np.array([110, 100, 96.83, 95, 163]))
        a = MODULE.multistart_seeds(q, signs, lower, upper, None)
        b = MODULE.multistart_seeds(q, signs, lower, upper, None)
        self.assertEqual([name for name, _ in a], [name for name, _ in b])
        self.assertEqual(len(a), len({tuple(np.round(seed, 12)) for _, seed in a}))
        for _, seed in a:
            self.assertTrue(np.all(seed >= lower) and np.all(seed <= upper))

    def test_previous_witness_is_included_and_clipped(self):
        q = np.zeros(5); signs = np.ones(5)
        lower, upper = np.full(5, -1.0), np.full(5, 1.0)
        seeds = dict(MODULE.multistart_seeds(q, signs, lower, upper, np.full(5, 2.0)))
        self.assertIn("previous_witness", seeds)
        self.assertTrue(np.all(seeds["previous_witness"] <= upper))

    def test_residual_threshold_is_not_optimizer_success(self):
        residuals = [0.5, 2.0, 2.1]
        self.assertEqual([value <= 2.0 for value in residuals], [True, True, False])

    def test_peak_reports_absolute_frame_and_joint(self):
        old = MODULE.JOINT_NAMES; MODULE.JOINT_NAMES = ("a", "b")
        try:
            result = MODULE.peak(np.array([[np.nan, np.nan], [1.0, -3.0]]), np.array([570, 571]), 1)
            self.assertEqual(result, {"value": 3.0, "frame": 571, "joint": "b"})
        finally:
            MODULE.JOINT_NAMES = old

    def test_sign_roundtrip_preserves_fk_coordinate(self):
        motor = np.array([-20.0, -100.0, 95.0, 40.0, -90.0])
        for signs in MODULE.HYPOTHESES.values():
            urdf = MODULE.motor_to_urdf_degrees(motor, signs)
            restored = MODULE.urdf_to_motor_degrees(urdf, signs)
            np.testing.assert_allclose(restored, motor)

    def test_stop_rule_requires_both_feasibility_and_temporal_quality(self):
        self.assertEqual(MODULE.expansion_decision(True, True), "ELIGIBLE_FOR_MORE_FRAMES")
        self.assertEqual(MODULE.expansion_decision(True, False), "STOP")
        self.assertEqual(MODULE.expansion_decision(False, True), "STOP")


if __name__ == "__main__":
    unittest.main()
