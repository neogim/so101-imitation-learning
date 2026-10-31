#!/usr/bin/env python3
"""Offline-only focused tests for independent candidate seed construction."""
import unittest

import numpy as np

from generate_independent_ik_candidates_offline import independent_seeds


class TestIndependentSeeds(unittest.TestCase):
    def test_deterministic_and_no_prior(self):
        action = np.array([0., -98., 20., 3., 4.])
        signs = np.ones(5)
        lo = np.deg2rad(np.full(5, -100.))
        hi = np.deg2rad(np.full(5, 100.))
        first = independent_seeds(action, signs, lo, hi)
        second = independent_seeds(action, signs, lo, hi)
        self.assertEqual([name for name, _ in first], [name for name, _ in second])
        for (_, a), (_, b) in zip(first, second):
            np.testing.assert_allclose(a, b, atol=0, rtol=0)
        self.assertTrue(all("previous" not in name for name, _ in first))
        self.assertGreaterEqual(len(first), 5)

    def test_all_seeds_within_urdf_bounds_and_sign_aware(self):
        action = np.array([1., -101., 96., 4., -3.])
        signs = np.array([1., -1., -1., 1., 1.])
        lo = np.deg2rad(np.full(5, -100.))
        hi = np.deg2rad(np.full(5, 100.))
        seeds = independent_seeds(action, signs, lo, hi)
        self.assertTrue(seeds)
        for name, q in seeds:
            self.assertTrue(np.all(q > lo))
            self.assertTrue(np.all(q < hi))
            self.assertTrue(np.isfinite(q).all())


if __name__ == "__main__":
    unittest.main()
