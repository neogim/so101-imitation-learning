"""Offline PnP tests: synthetic data, no hardware."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fit_side_camera_pnp", HERE / "fit_side_camera_pnp.py")
pnp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pnp)


class SidePnPTests(unittest.TestCase):
    def test_known_pose_reconstruction_and_mujoco_convention(self):
        xyz = np.array([[0,0,0], [.2,0,0], [0,.2,0], [0,0,.2],
                        [.15,.1,.15], [-.1,.05,.04], [.08,-.12,.1],
                        [.25,.1,.02], [.2,.2,.15], [.02,-.2,-.08]], dtype=np.float64)
        K = np.array([[600,0,320],[0,590,240],[0,0,1]], dtype=float)
        rvec = np.array([[.2],[-.1],[.05]])
        tvec = np.array([[.0],[.0],[.8]])
        uv = cv2.projectPoints(xyz, rvec, tvec, K, None)[0].reshape(-1, 2)
        result = pnp.solve(xyz, uv, K, np.zeros((5,1)))
        self.assertLess(result["rmse_inlier_px"], 1e-4)
        R = cv2.Rodrigues(rvec)[0]
        np.testing.assert_allclose(result["pos_m"], (-R.T @ tvec).flatten(), atol=1e-4)
        R_mujoco = Rotation.from_quat(np.roll(result["quat_wxyz"], -1)).as_matrix()
        np.testing.assert_allclose(R_mujoco, R.T @ np.diag([1,-1,-1]), atol=1e-4)

    def test_collinear_rejected_and_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_file = root / "landmarks.csv"
            csv_file.write_text("point_id,x_m,y_m,z_m,u_px,v_px\n" +
                                "\n".join(f"{i},{i},0,0,{i},{i}" for i in range(6)))
            with self.assertRaisesRegex(ValueError, "coplanar or degenerate"):
                pnp.read_points(csv_file)
            intr = root / "intrinsics.json"
            intr.write_text(json.dumps({"width": 640, "height": 480,
                                        "fx": 600, "fy": 590, "cx": 320, "cy": 240}))
            w, h, K, d = pnp.read_intrinsics(intr)
            self.assertEqual((w, h), (640, 480))
            self.assertEqual(K.shape, (3,3))
            self.assertEqual(d.shape, (5,1))


if __name__ == "__main__":
    unittest.main()
