"""Camera convention and image-FOV checks; no MuJoCo or hardware needed."""
import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_id1_eye_to_hand import (mujoco_camera_pose, fovy_from_intrinsics,
                                     project_reference_points, add_approximate_workcell, workcell_shift_left_front)


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

    def test_workcell_projection_in_expected_camera_view(self):
        T = np.array([[-.61650, .43981, -.65307, .49516],
                      [.75783, .10642, -.64372, .38170],
                      [-.21361, -.89176, -.39891, .28123],
                      [0, 0, 0, 1]])
        cam = {"fx": 543.17, "fy": 542.96, "cx": 316.39,
               "cy": 229.55, "width": 640, "height": 480}
        projected = project_reference_points(T, cam)
        self.assertTrue(all(p["in_frame"] for p in projected.values()))
        self.assertNotEqual(projected["muhi_center"]["pixel_approx"],
                            projected["cup_center"]["pixel_approx"])

    @unittest.skipUnless(importlib.util.find_spec("mujoco") is not None,
                         "MuJoCo unavailable in this numerical-only environment")
    def test_urdf_visual_only_objects_survive_compile(self):
        import mujoco
        root = Path(__file__).resolve().parents[2]
        spec = mujoco.MjSpec.from_file(str(root / "data/urdf/so101/so101_new_calib.urdf"))
        self.assertTrue(spec.compiler.discardvisual,
                        "URDF import default changed; reassess this regression")
        spec.compiler.discardvisual = False
        add_approximate_workcell(spec)
        model = spec.compile()
        for name in ("r2s_preview_table", "r2s_muhi_body", "r2s_muhi_cap",
                     "r2s_cup_base", "r2s_cup_wall_00"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertGreaterEqual(gid, 0, name)
            self.assertEqual(int(model.geom_contype[gid]), 0)
            self.assertEqual(int(model.geom_conaffinity[gid]), 0)

    def test_shift_moves_workcell_left_and_closer_in_camera_image(self):
        T = np.array([[-.61650, .43981, -.65307, .49516],
                      [.75783, .10642, -.64372, .38170],
                      [-.21361, -.89176, -.39891, .28123],
                      [0, 0, 0, 1]])
        cam = {"fx": 543.17, "fy": 542.96, "cx": 316.39,
               "cy": 229.55, "width": 640, "height": 480}
        shift = workcell_shift_left_front(T, .08, .10)
        self.assertAlmostEqual(shift[2], 0.)
        initial = project_reference_points(T, cam)["table_center"]
        moved = project_reference_points(T, cam, shift)["table_center"]
        self.assertLess(moved["pixel_approx"][0], initial["pixel_approx"][0])
        self.assertLess(moved["depth_m"], initial["depth_m"])
        with self.assertRaises(ValueError):
            workcell_shift_left_front(T, float("nan"), .1)

    def test_fovy(self):
        import math
        self.assertAlmostEqual(fovy_from_intrinsics(542.957, 480),
                               math.degrees(2*math.atan(480/(2*542.957))))


if __name__ == "__main__":
    unittest.main()
