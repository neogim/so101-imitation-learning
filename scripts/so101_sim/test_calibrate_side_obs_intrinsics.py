"""Offline tests for Side OBS chessboard calibration."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cv2

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("obs_cal",HERE/"calibrate_side_obs_intrinsics.py")
cal=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


class IntrinsicsTest(unittest.TestCase):
    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError,"Need at least"):
            cal.calibrate([], 7, 5, 20.0)
        with self.assertRaisesRegex(ValueError,"Invalid chessboard"):
            cal.calibrate([], 2, 5, 20.0)
        with tempfile.TemporaryDirectory() as td:
            for i in range(12):
                cv2.imwrite(str(Path(td)/f"{i:02d}.png"), np.zeros((168,224),dtype=np.uint8))
            with self.assertRaisesRegex(ValueError,"Only 0 usable"):
                cal.calibrate(cal.collect_images(Path(td)),7,5,20.0)


if __name__=="__main__":
    unittest.main()
