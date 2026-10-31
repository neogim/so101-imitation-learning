#!/usr/bin/env python3
"""No-hardware Jacobian finite difference / SVD unit tests."""
import unittest
import numpy as np
from audit_dp_jacobian_offline import finite_difference_jacobian, svd_metrics

class JacobianTests(unittest.TestCase):
    def test_linear_fk_exact(self):
        a=np.array([[1.,2.,0.,0.,0.],[0.,3.,4.,0.,0.],[1.,0.,0.,5.,6.]])
        q=np.array([.1,.2,.3,.4,.5])
        np.testing.assert_allclose(finite_difference_jacobian(lambda x:a@x,q),a,atol=1e-9)
    def test_rank_deficient(self):
        j=np.array([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.]])
        x=svd_metrics(j)
        self.assertTrue(x["near_singular_numeric"])
        self.assertIsNone(x["condition_number"])
    def test_regular(self):
        j=np.column_stack([np.eye(3),np.zeros((3,2))])
        x=svd_metrics(j)
        self.assertFalse(x["near_singular_numeric"])
        self.assertAlmostEqual(x["condition_number"],1.)
if __name__=="__main__":unittest.main()
