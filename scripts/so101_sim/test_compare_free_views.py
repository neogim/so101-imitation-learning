"""Unit checks for identical fixed views and hypothesis mapping (offline only)."""
import importlib.util
import math
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("free_view_compare", HERE / "compare_free_views.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FreeViewHypothesisTests(unittest.TestCase):
    def test_signs_apply_to_observation_not_actions(self):
        import real_to_sim as replay
        config = replay.load_config(HERE / "workcell.example.json")
        state = [-2.5, -102.59, 96.57, 62.1, -111.5, 5.3]
        plus = mod.actual_joint_angles(state, mod.mapping(config, "plus"))
        minus = mod.actual_joint_angles(state, mod.mapping(config, "j2j3_minus"))
        self.assertAlmostEqual(plus["shoulder_lift"], -102.59, places=5)
        self.assertAlmostEqual(minus["shoulder_lift"], 102.59, places=5)
        self.assertAlmostEqual(minus["elbow_flex"], -96.57, places=5)
        self.assertAlmostEqual(minus["wrist_flex"], 62.1, places=5)
        self.assertFalse(mod.mapping(config, "plus")["calibrated"])

    def test_free_camera_is_fixed_and_independent_of_hypothesis(self):
        import mujoco
        for name in mod.VIEWS:
            a = mod.viewpoint(name)
            b = mod.viewpoint(name)
            self.assertEqual(a.type, mujoco.mjtCamera.mjCAMERA_FREE)
            self.assertEqual(tuple(a.lookat), tuple(b.lookat))
            self.assertEqual(a.distance, b.distance)
            self.assertEqual(a.azimuth, b.azimuth)
            self.assertEqual(a.elevation, b.elevation)

    def test_invalid_frame_selection_is_rejected_before_touching_data(self):
        with self.assertRaisesRegex(ValueError, "unique non-negative"):
            mod.compare(Path("none"), Path("unused"), 0, [-1], 200, 200,
                        Path("none"), Path("none"), ["oblique"])
        with self.assertRaisesRegex(ValueError, "unique non-negative"):
            mod.compare(Path("none"), Path("unused"), 0, [0, 0], 200, 200,
                        Path("none"), Path("none"), ["oblique"])


if __name__ == "__main__":
    unittest.main()
