"""No-hardware checks for SAM3 alignment on the existing ArUco preview scene."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("align_previous_scene_sam3", HERE / "align_previous_scene_sam3.py")
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)


class MaskAndGeometryTests(unittest.TestCase):
    def test_identical_masks(self):
        real = np.zeros((40, 60), bool)
        real[5:25, 10:35] = True
        result = align.overlap_metrics(real, real)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["area_ratio_sim_over_real"], 1.0)
        self.assertEqual(result["centroid_delta_sim_minus_real_px"], [0., 0.])

    def test_area_and_centroid_mismatch(self):
        real = np.zeros((50, 60), bool)
        sim = np.zeros((50, 60), bool)
        real[5:15, 5:15] = True
        sim[5:25, 10:30] = True
        result = align.overlap_metrics(real, sim)
        self.assertAlmostEqual(result["area_ratio_sim_over_real"], 4.)
        self.assertGreater(result["centroid_delta_sim_minus_real_px"][0], 0)
        self.assertLess(result["iou"], 1.)

    def test_empty_and_size_rejected(self):
        empty = np.zeros((10, 10), bool)
        with self.assertRaisesRegex(ValueError, "empty"):
            align.overlap_metrics(empty, empty)
        with self.assertRaisesRegex(ValueError, "identical"):
            align.overlap_metrics(empty, np.zeros((11, 10), bool))

    def test_camera_relative_front_sign(self):
        # OpenCV camera +Z pointing along +X in base XY; negative front => -X
        camera = np.eye(4)
        camera[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        delta_near = align.offset_xy(camera, 0., .05)
        delta_far = align.offset_xy(camera, 0., -.05)
        self.assertTrue(np.allclose(delta_near, -delta_far))
        self.assertLess(delta_near[0], 0.)
        self.assertGreater(delta_far[0], 0.)

    def test_upper_edge_strip_reprojection(self):
        real = np.zeros((40, 70), bool)
        sim = np.zeros_like(real)
        real[12:, 5:65] = True
        sim[16:, 5:65] = True
        edge = align.visible_upper_edge_residual(real, sim, 20, 50)
        self.assertEqual(edge["sample_columns"], 30)
        self.assertEqual(edge["median_sim_minus_real_px"], 4.)
        self.assertEqual(edge["mean_abs_error_px"], 4.)

    def test_upper_edge_ignores_cutoff_and_rejects_invalid_interval(self):
        real = np.zeros((30, 40), bool)
        real[:15, 5:20] = True
        sim = real.copy()
        edge = align.visible_upper_edge_residual(real, sim, 5, 20)
        self.assertEqual(edge["sample_columns"], 0)
        with self.assertRaisesRegex(ValueError, "Invalid edge"):
            align.visible_upper_edge_residual(real, sim, 20, 5)

    def test_bbox_envelope_errors(self):
        real = np.zeros((20, 30), bool)
        sim = np.zeros_like(real)
        real[3:13, 4:16] = True
        sim[2:14, 6:18] = True
        m = align.silhouette_diagnostics(real, sim)
        self.assertEqual(m["real_bbox_xyxy_px"], [4, 3, 15, 12])
        self.assertEqual(m["bbox_delta_sim_minus_real_px"], [2, -1, 2, 1])
        self.assertEqual(m["bbox_max_abs_error_px"], 2)

    def test_alignment_score_penalizes_same_iou_wrong_apparent_size(self):
        real = np.zeros((50, 70), bool)
        real[10:30, 15:35] = True
        exact = real.copy()
        oversized = np.zeros_like(real)
        oversized[8:35, 10:42] = True
        self.assertGreater(align.alignment_score(real, exact),
                           align.alignment_score(real, oversized))
        self.assertAlmostEqual(align.alignment_score(real, exact), 1.0)

    def test_robot_edge_anchor_keeps_table_border_at_robot_frame(self):
        import mujoco
        xml = """<mujoco><worldbody>
        <geom name="r2s_preview_table" type="box" size=".4 .255 .02" pos=".3 .1 -.02"/>
        <geom name="r2s_table_leg_mm" type="box" size=".02 .02 .1" pos=".1 .1 -.2"/>
        <geom name="r2s_table_leg_mp" type="box" size=".02 .02 .1" pos=".1 .2 -.2"/>
        <geom name="r2s_table_leg_pm" type="box" size=".02 .02 .1" pos=".3 .1 -.2"/>
        <geom name="r2s_table_leg_pp" type="box" size=".02 .02 .1" pos=".3 .2 -.2"/>
        <geom name="r2s_muhi_body" type="sphere" size=".01" pos=".05 .05 .03"/>
        <camera name="side" pos="0 0 1"/>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        ids = align.movie.table_geom_ids(model)
        original_pos = model.geom_pos[ids].copy()
        original_quat = model.geom_quat[ids].copy()
        muhi = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_muhi_body")
        m = model.geom_pos[muhi].copy()
        anchor = [0., 0.]
        self.assertEqual(align.movie.nearest_table_edge(model, anchor), "x-")
        for yaw in (0., 12.5, -20.):
            align.movie.set_table_pose_delta(
                model, original_pos, original_quat, [0., .03, 0.], yaw)
            report = align.movie.anchor_table_edge_to_robot(
                model, "x-", anchor, inset_m=.02)
            self.assertAlmostEqual(report["edge_constraint_residual_m"], 0., places=7)
            self.assertEqual(report["edge"], "x-")
            np.testing.assert_allclose(model.geom_pos[muhi], m)
            np.testing.assert_allclose(model.geom_size[ids[0]], [.4, .255, .02])
        with self.assertRaises(ValueError):
            align.movie.anchor_table_edge_to_robot(model, "wrong")

    def test_table_yaw_rotates_all_parts_only_about_table_center(self):
        import mujoco
        xml = """<mujoco><worldbody>
        <geom name="r2s_preview_table" type="box" size=".4 .255 .02" pos=".2 .1 -.02"/>
        <geom name="r2s_table_leg_mm" type="box" size=".02 .02 .1" pos=".3 .1 -.2"/>
        <geom name="r2s_table_leg_mp" type="box" size=".02 .02 .1" pos=".2 .2 -.2"/>
        <geom name="r2s_table_leg_pm" type="box" size=".02 .02 .1"/>
        <geom name="r2s_table_leg_pp" type="box" size=".02 .02 .1"/>
        <geom name="r2s_muhi_body" type="sphere" size=".01" pos=".3 .3 .03"/>
        <camera name="side" pos="0 0 1"/>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        ids = align.movie.table_geom_ids(model)
        p0 = model.geom_pos[ids].copy()
        q0 = model.geom_quat[ids].copy()
        muhi = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_muhi_body")
        muhi0 = model.geom_pos[muhi].copy()
        cam0 = model.cam_pos.copy()
        align.movie.set_table_pose_delta(model, p0, q0, [0., 0., 0.], 90.)
        np.testing.assert_allclose(model.geom_pos[ids[0]], p0[0], atol=1e-7)
        np.testing.assert_allclose(model.geom_pos[ids[1]], [.2, .2, -.2], atol=1e-7)
        np.testing.assert_allclose(model.geom_pos[ids[2]], [.1, .1, -.2], atol=1e-7)
        np.testing.assert_allclose(model.geom_pos[muhi], muhi0, atol=1e-7)
        np.testing.assert_allclose(model.cam_pos, cam0, atol=1e-7)
        np.testing.assert_allclose(model.geom_size[ids[0]], [.4, .255, .02], atol=1e-7)
        align.movie.set_table_pose_delta(model, p0, q0, [.02, -.03, 0.], 0.)
        np.testing.assert_allclose(model.geom_pos[ids[0]], p0[0] + [.02, -.03, 0.])
        np.testing.assert_allclose(model.geom_quat[ids[0]], q0[0], atol=1e-7)

    def test_negative_yaw_search_bound_rejected(self):
        with self.assertRaisesRegex(ValueError, "Yaw"):
            align.optimize(None, None, None, None, None, None,
                           fit_yaw=True, max_yaw_delta_deg=0.)

    def test_base_axis_translation_does_not_rotate_or_move_other_props(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo unavailable")
        xml = """<mujoco><worldbody>
        <geom name="r2s_preview_table" type="box" pos=".1 .2 -.02"
              quat=".9238795 0 0 .3826834" size=".3 .3 .02"/>
        <geom name="r2s_table_leg_mm" type="box" size=".02 .02 .2"/>
        <geom name="r2s_table_leg_mp" type="box" size=".02 .02 .2"/>
        <geom name="r2s_table_leg_pm" type="box" size=".02 .02 .2"/>
        <geom name="r2s_table_leg_pp" type="box" size=".02 .02 .2"/>
        <geom name="r2s_muhi_body" type="sphere" pos=".2 .3 0" size=".01"/>
        <camera name="fixed" pos="0 0 1"/>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_preview_table")
        muhi = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_muhi_body")
        before_pos = model.geom_pos[table].copy()
        before_quat = model.geom_quat[table].copy()
        before_muhi = model.geom_pos[muhi].copy()
        before_cam = model.cam_pos.copy()
        align.apply_base_offset(model, [10., 0., 0.])
        np.testing.assert_allclose(model.geom_pos[table], before_pos + [.1, 0, 0])
        np.testing.assert_allclose(model.geom_quat[table], before_quat)
        np.testing.assert_allclose(model.geom_pos[muhi], before_muhi)
        np.testing.assert_allclose(model.cam_pos, before_cam)

    def test_invalid_axes_rejected(self):
        self.assertRaisesRegex(ValueError, "axes", align.optimize,
                               None, None, None, None, None, None, axes="camera-front")

    def test_missing_weights_rejected_before_ultralytics(self):
        with self.assertRaisesRegex(ValueError, "weights file is missing"):
            align.load_mask(Path("/does/not/exist.png"), sam3_model=Path("/does/not/exist.pt"))


if __name__ == "__main__":
    unittest.main()
