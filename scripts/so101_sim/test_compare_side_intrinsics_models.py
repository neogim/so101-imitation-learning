"""Offline regression tests for Side camera intrinsic model comparisons."""
import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np

P=Path(__file__).with_name("compare_side_intrinsics_models.py")
spec=importlib.util.spec_from_file_location("compare_side_intrinsics_models",P)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class SideIntrinsicsModelsTest(unittest.TestCase):
    def test_synthetic_views_constrained_and_full(self):
        xx, yy = np.meshgrid(np.arange(7), np.arange(5))
        obj = np.stack([xx.ravel(),yy.ravel(),np.zeros(35)],axis=1).astype(np.float32)*.02
        K=np.array([[590.,0.,320.],[0.,590.,240.],[0.,0.,1.]])
        objects,detected=[],[]
        for i in range(12):
            r=np.array([.11 + .025*(i%4), -.23 + .08*(i%6), .03*i],dtype=np.float64)
            t=np.array([-.07 + .014*(i%4),-.06+.012*(i%5),.44 + .032*(i%6)])
            uv=cv2.projectPoints(obj,r,t,K,np.zeros(5))[0].astype(np.float32)
            objects.append(obj.copy())
            detected.append({"image":f"synthetic_{i:02d}.png","corners":uv})
        result=mod.compare(objects,detected,(640,480),do_leave_one_out=False)
        self.assertEqual(set(result),set(mod.MODELS))
        for name, row in result.items():
            self.assertLess(row["training_rms_px"],.2, name)
            self.assertEqual(row["intrinsics"]["width"],640)
            self.assertEqual(row["intrinsics"]["height"],480)
            self.assertIsNone(row["loo_rms_mean_px"])
        self.assertLess(abs(result["pinhole"]["intrinsics"]["fx"]-590),10)

    def test_empty_folder_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"Need >=12"):
                mod.collect(Path(tmp),7,5,20)


if __name__=="__main__":
    unittest.main()
