"""Camera convention and image-FOV checks; no MuJoCo or hardware needed."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image
from compare_id1_eye_to_hand import mujoco_camera_pose, fovy_from_intrinsics, enhance_sim_brightness


class SideCameraTests(unittest.TestCase):
    def test_opencv_to_mujoco_camera_axes(self):
        T = np.eye(4)
        T[:3, 3] = [.2, -.3, .4]
        pos, quat = mujoco_camera_pose(T)
        np.testing.assert_allclose(pos, [.2, -.3, .4])
        q = np.asarray(quat)
        recovered = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(recovered, np.diag([1, -1, -1]), atol=1e-8)

    def test_arbitrary_rotation(self):
        T = np.eye(4)
        R = Rotation.from_euler("zyx", [50., -17., 12.], degrees=True).as_matrix()
        T[:3, :3] = R
        _, quat = mujoco_camera_pose(T)
        r2 = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(r2, R @ np.diag([1, -1, -1]), atol=1e-8)

    def test_invalid_transform(self):
        T = np.eye(4)
        T[0, 0] = 2
        with self.assertRaises(ValueError):
            mujoco_camera_pose(T)

    def test_display_brightness_only(self):
        original = Image.fromarray(np.array([[[100, 100, 100], [250, 250, 250]]], dtype=np.uint8))
        corrected = enhance_sim_brightness(original)
        pixels = np.asarray(corrected)
        self.assertGreater(int(pixels[0, 0, 0]), 100)
        self.assertEqual(int(pixels[0, 1, 0]), 255)
        np.testing.assert_array_equal(np.asarray(original)[0, 0], [100, 100, 100])
        np.testing.assert_array_equal(np.asarray(enhance_sim_brightness(original, 1, 1)),
                                      np.asarray(original))
        with self.assertRaises(ValueError):
            enhance_sim_brightness(original, brightness=0)

    def test_fovy(self):
        import math
        self.assertAlmostEqual(fovy_from_intrinsics(542.957, 480),
                               math.degrees(2*math.atan(480/(2*542.957))))


if __name__ == "__main__":
    unittest.main()
