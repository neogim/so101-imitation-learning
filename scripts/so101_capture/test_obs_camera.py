"""Hardware-free checks of the OBS adapter's pixel contract."""
import unittest
from unittest.mock import Mock

import numpy as np
from obs_camera import OBSCamera


class PixelContractTests(unittest.TestCase):
    def test_rgb_and_shape(self):
        camera = OBSCamera()
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr[..., 2] = 255
        rgb = camera._postprocess_image(bgr)
        self.assertEqual(rgb.shape, (480, 640, 3))
        np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])

    def test_reject_changed_resolution(self):
        camera = OBSCamera()
        for shape in [(1080, 1920, 3), (480, 640, 4)]:
            with self.assertRaises(RuntimeError):
                camera._postprocess_image(np.zeros(shape, dtype=np.uint8))

    def test_connect_validates_actual_pixels(self):
        camera = OBSCamera()
        camera.videocapture = Mock()
        camera.videocapture.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        with self.assertRaises(RuntimeError):
            camera._configure_capture_settings()
        camera.videocapture.set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
