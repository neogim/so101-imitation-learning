#!/usr/bin/env python3
"""Hardware-free regression checks for the calibration/URDF audit."""
import importlib.util, pathlib, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]/"scripts"/"so101_capture"/"audit_so101_calibration_readiness.py"
spec=importlib.util.spec_from_file_location("audit_so101_calibration_readiness",ROOT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.c={"range_min":776,"range_max":3170,"drive_mode":0}
    def test_range_endpoints(self):
        self.assertEqual(mod.inverse_normalize(-100,self.c,"range_m100_100"),776)
        self.assertEqual(mod.inverse_normalize(100,self.c,"range_m100_100"),3170)
        self.assertEqual(mod.inverse_normalize(0,self.c,"range_m100_100"),1973)
    def test_drive_mode_inversion(self):
        self.c["drive_mode"]=1
        self.assertEqual(mod.inverse_normalize(100,self.c,"range_m100_100"),776)
        self.assertEqual(mod.inverse_normalize(0,self.c,"range_m100_100"),1973)
    def test_degrees_at_center(self):
        self.assertEqual(mod.inverse_normalize(0,self.c,"degrees"),1973)
        self.assertAlmostEqual(mod.inverse_normalize(90,self.c,"degrees"),1973+4095/4)
    def test_gripper(self):
        self.assertEqual(mod.inverse_normalize(100,self.c,"range_0_100"),3170)
    def test_degrees_not_equal_normalized(self):
        self.assertNotAlmostEqual(mod.inverse_normalize(-100,self.c,"degrees"),
                                  mod.inverse_normalize(-100,self.c,"range_m100_100"))
if __name__=="__main__":unittest.main()
