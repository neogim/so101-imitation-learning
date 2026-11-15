"""Minimal offline smoke tests for collision-guarded DLS/SAC reporting."""
import unittest
import numpy as np

import compare_dls_sac_table_guard as compare


class GuardedComparisonTests(unittest.TestCase):
    def test_row_preserves_reference_diagnostics(self):
        row=compare.record_row(
            "sac","muhi",2,np.zeros(5),
            np.array([.1,.2,.3]),np.array([.1,.2,.31]),
            {"status":"shortened","minimum_distance_mm":3.5,
             "nearest_pair":("tip","r2s_preview_table")},True)
        self.assertEqual(row["guard_status"],"shortened")
        self.assertAlmostEqual(row["tcp_error_mm"],10.)
        self.assertTrue(row["initial_pose_adjusted"])
        self.assertEqual(row["joint_5"],0.)
        self.assertIn("r2s_preview_table",row["nearest_pair"])


if __name__=="__main__":
    unittest.main()
