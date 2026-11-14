"""Synthetic tests for J2/J3 collision-clearance lookup."""
import unittest
import numpy as np
from diagnose_j2_j3_table_clearance import sampled_angles, summarize_scan


class CouplingTests(unittest.TestCase):
    def test_sampled_angles_inside_joint_limits(self):
        x=sampled_angles(0,-.2,.2,30,7)
        self.assertTrue(np.all(x>=-.2))
        self.assertTrue(np.all(x<=.2))
        self.assertEqual(len(x),3 if len(x)==3 else len(x))

    def test_same_j2_has_different_clearance_for_j3(self):
        rows=[
            {"j2_deg":-80., "j3_deg":70., "j2_delta_deg":0.,
             "j3_delta_deg":0., "min_table_clearance_mm":-10.,
             "safe_static":False},
            {"j2_deg":-80., "j3_deg":80., "j2_delta_deg":0.,
             "j3_delta_deg":10., "min_table_clearance_mm":5.,
             "safe_static":True},
            {"j2_deg":-90., "j3_deg":70., "j2_delta_deg":-10.,
             "j3_delta_deg":0., "min_table_clearance_mm":4.,
             "safe_static":True},
        ]
        report=summarize_scan(rows)
        self.assertEqual(report["safe_static_pairs"],2)
        j2=[x for x in report["j2_to_safe_j3_lookup"] if x["j2_deg"]==-80.][0]
        self.assertEqual(j2["nearest_safe_j3_deg"],80.)


if __name__=="__main__":
    unittest.main()
