"""Phase 1.25 scene-candidate geometry tests (no devices or datasets)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_scene_variations import (
    apply_candidate, capture_reference, generate_offsets,
    quat_product, rotate_translate_xy, z_yaw_quat,
)


class SceneVariationTests(unittest.TestCase):
    def test_seed_repeatability_and_baseline(self):
        a = generate_offsets(20, seed=101)
        self.assertEqual(a, generate_offsets(20, seed=101))
        self.assertNotEqual(a, generate_offsets(20, seed=102))
        self.assertEqual(len(a), 20)
        self.assertEqual(a[0]["muhi_xy_delta_mm"], [0., 0.])
        self.assertEqual(a[0]["muhi_yaw_delta_deg"], 0.)
        self.assertEqual(a[0]["cup_xy_delta_mm"], [0., 0.])
        for row in a:
            self.assertTrue(np.all(np.abs(row["muhi_xy_delta_mm"]) <= 20.))
            self.assertTrue(np.all(np.abs(row["cup_xy_delta_mm"]) <= 20.))
            self.assertLessEqual(abs(row["muhi_yaw_delta_deg"]), 30.)

    def test_explicit_yaw_angles_cover_90_and_minus_90(self):
        from generate_scene_variations import DEFAULT_YAW_ANGLES_DEG
        angles = list(DEFAULT_YAW_ANGLES_DEG)
        rows = generate_offsets(20, seed=101, yaw_angles_deg=angles)
        self.assertEqual([r["muhi_yaw_delta_deg"] for r in rows[:7]], angles)
        self.assertTrue(all(r["muhi_xy_delta_mm"] == [0., 0.]
                            and r["cup_xy_delta_mm"] == [0., 0.]
                            for r in rows[:7]))
        self.assertEqual(rows, generate_offsets(20, seed=101,
                                               yaw_angles_deg=angles))
        self.assertEqual(len(rows), 20)
        with self.assertRaisesRegex(ValueError, "Count"):
            generate_offsets(6, yaw_angles_deg=angles)
        with self.assertRaises(ValueError):
            generate_offsets(20, yaw_angles_deg=[0, 90, 90])
        with self.assertRaises(ValueError):
            generate_offsets(20, yaw_angles_deg=[-90, 90])
        with self.assertRaises(ValueError):
            generate_offsets(20, yaw_angles_deg=[0, float("nan")])

    def test_four_modes_and_ranges(self):
        x = generate_offsets(5, seed=1)
        self.assertEqual(x[1]["cup_xy_delta_mm"], [0., 0.])
        self.assertEqual(x[1]["muhi_yaw_delta_deg"], 0.)
        self.assertEqual(x[2]["muhi_xy_delta_mm"], [0., 0.])
        self.assertEqual(x[2]["cup_xy_delta_mm"], [0., 0.])
        self.assertEqual(x[3]["muhi_xy_delta_mm"], [0., 0.])
        self.assertEqual(x[3]["muhi_yaw_delta_deg"], 0.)
        with self.assertRaises(ValueError):
            generate_offsets(count=0)
        with self.assertRaises(ValueError):
            generate_offsets(yaw_range_deg=float("nan"))
        with self.assertRaises(ValueError):
            generate_offsets(muhi_range_mm=-1)

    def test_rotate_translate_does_not_change_height_or_lengths(self):
        p = np.array([[1., 2., .02], [1., 2.1, .04]])
        q = rotate_translate_xy(p, [1., 2.05], 90, [.1, .2])
        np.testing.assert_allclose(q[0], [1.15, 2.25, .02], atol=1e-10)
        np.testing.assert_allclose(q[1], [1.05, 2.25, .04], atol=1e-10)
        self.assertAlmostEqual(np.linalg.norm(p[0] - p[1]), np.linalg.norm(q[0] - q[1]))
        np.testing.assert_allclose(p, [[1, 2, .02], [1, 2.1, .04]])

    def test_world_quaternion_yaw(self):
        q = quat_product(z_yaw_quat(90), z_yaw_quat(90))
        np.testing.assert_allclose(q, [0., 0., 0., 1.], atol=1e-10)

    def test_only_muhi_and_cup_geoms_move_no_accumulation(self):
        import mujoco
        spec = mujoco.MjSpec()
        w = spec.worldbody
        w.add_geom(name="r2s_muhi_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[.0175, .03, 0], pos=[.1, .1, .02], contype=0, conaffinity=0)
        w.add_geom(name="r2s_muhi_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[.0175, .02, 0], pos=[.1, .15, .02], contype=0, conaffinity=0)
        w.add_geom(name="r2s_cup_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[.039, .001, 0], pos=[.3, .3, .001], contype=0, conaffinity=0)
        for i in range(16):
            w.add_geom(name=f"r2s_cup_wall_{i:02d}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[.01, .002, .04],
                       pos=[.3 + .039 * np.cos(i * np.pi / 8),
                            .3 + .039 * np.sin(i * np.pi / 8), .04],
                       contype=0, conaffinity=0)
        w.add_geom(name="r2s_preview_table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[.4, .25, .02], pos=[0, 0, -.04], contype=0, conaffinity=0)
        model = spec.compile()
        table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_preview_table")
        before = model.geom_pos[table_id].copy()
        ref = capture_reference(model)
        item = {"muhi_xy_delta_mm": [10., 20.],
                "muhi_yaw_delta_deg": 25., "cup_xy_delta_mm": [-20., 0.]}
        apply_candidate(model, ref, item)
        expected = model.geom_pos.copy()
        apply_candidate(model, ref, item)
        np.testing.assert_allclose(model.geom_pos, expected)
        np.testing.assert_allclose(model.geom_pos[table_id], before)
        self.assertAlmostEqual(model.geom_pos[ref["cup_ids"][0], 0], .28, places=7)


if __name__ == "__main__":
    unittest.main()
