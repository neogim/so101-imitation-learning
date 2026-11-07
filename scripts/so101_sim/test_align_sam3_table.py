"""Hardware-free tests for mask alignment and perspective rendering."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("align_sam3_table", HERE / "align_sam3_table.py")
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)


class TestTableAlignment(unittest.TestCase):
    def test_matching_masks(self):
        a = np.zeros((20, 30), dtype=bool)
        a[3:11, 4:15] = True
        result = align.mask_metrics(a, a)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["area_ratio_sim_over_real"], 1.0)
        self.assertEqual(result["centroid_delta_sim_minus_real_px"], [0., 0.])

    def test_mismatch_area_and_centroid(self):
        a = np.zeros((20, 30), dtype=bool)
        b = np.zeros_like(a)
        a[4:8, 5:9] = True
        b[4:12, 6:14] = True
        m = align.mask_metrics(a, b)
        self.assertEqual(m["area_ratio_sim_over_real"], 4)
        self.assertGreater(m["centroid_delta_sim_minus_real_px"][0], 0)
        self.assertLess(m["iou"], 1)

    def test_empty_and_shape_error(self):
        a = np.zeros((20, 30), dtype=bool)
        with self.assertRaisesRegex(ValueError, "Empty"):
            align.mask_metrics(a, a)
        with self.assertRaisesRegex(ValueError, "equally sized"):
            align.mask_metrics(a, np.zeros((30, 20), dtype=bool))

    def test_mujoco_3d_perspective_apparent_size(self):
        """A fixed-size box farther from the SAME camera must occupy fewer pixels."""
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo unavailable")
        xml = """
        <mujoco><worldbody>
          <geom name="target" type="box" size=".1 .1 .03" pos="0 0 -1"/>
          <camera name="side" pos="0 0 0" quat="1 0 0 0" fovy="55"/>
        </worldbody></mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        try:
            with mujoco.Renderer(model, height=240, width=320) as renderer:
                renderer.enable_segmentation_rendering()
                near = align.render_target(renderer, model, data, geom_id, "side").sum()
                model.geom_pos[geom_id, 2] = -1.5
                far = align.render_target(renderer, model, data, geom_id, "side").sum()
        except (mujoco.FatalError, RuntimeError) as exc:
            self.skipTest(f"Offscreen renderer unavailable: {exc}")
        self.assertGreater(near, far)
        self.assertGreater(far, 0)


if __name__ == "__main__":
    unittest.main()
