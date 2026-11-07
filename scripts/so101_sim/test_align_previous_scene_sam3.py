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

    def test_bbox_envelope_errors(self):
        real = np.zeros((20, 30), bool)
        sim = np.zeros_like(real)
        real[3:13, 4:16] = True
        sim[2:14, 6:18] = True
        m = align.silhouette_diagnostics(real, sim)
        self.assertEqual(m["real_bbox_xyxy_px"], [4, 3, 15, 12])
        self.assertEqual(m["bbox_delta_sim_minus_real_px"], [2, -1, 2, 1])
        self.assertEqual(m["bbox_max_abs_error_px"], 2)

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
