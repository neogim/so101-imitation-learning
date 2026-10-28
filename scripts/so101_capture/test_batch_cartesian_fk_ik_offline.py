#!/usr/bin/env python3
"""Dependency-light mapping tests for offline Cartesian batch runner."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from batch_cartesian_fk_ik_hypotheses_offline import HYPOTHESES,mapping,to_urdf_rad,to_motor_deg

class MappingTests(unittest.TestCase):
    def test_four_sign_hypotheses_roundtrip(self):
        q=np.array([-4.8,-102.15,96.75,77.05,-97.7])
        for label in HYPOTHESES:
            with self.subTest(label=label):
                signs,offsets=mapping(label,3.5,-2.25)
                self.assertTrue(np.allclose(to_motor_deg(to_urdf_rad(q,signs,offsets),
                                                        signs,offsets),q,atol=1e-10))
    def test_signs_do_not_affect_other_joints(self):
        signs,offsets=mapping("j2-_j3-",2.,-3.)
        self.assertTrue(np.array_equal(signs[[0,3,4]],np.ones(3)))
        self.assertTrue(np.array_equal(offsets[[0,3,4]],np.zeros(3)))
    def test_bad_inputs_rejected(self):
        signs,offsets=mapping("j2+_j3+")
        with self.assertRaises(ValueError):
            to_urdf_rad(np.array([1.,2.]),signs,offsets)
        with self.assertRaises(ValueError):
            mapping("unknown")
        with self.assertRaises(ValueError):
            mapping("j2+_j3+",float("nan"),0.)

if __name__=="__main__":
    unittest.main()
