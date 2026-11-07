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

    def test_missing_weights_rejected_before_ultralytics(self):
        with self.assertRaisesRegex(ValueError, "weights file is missing"):
            align.load_mask(Path("/does/not/exist.png"), sam3_model=Path("/does/not/exist.pt"))


if __name__ == "__main__":
    unittest.main()
