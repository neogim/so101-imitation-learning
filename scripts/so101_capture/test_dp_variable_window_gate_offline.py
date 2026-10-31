#!/usr/bin/env python3
"""Regression: DP quality must depend on the requested frame count, not 31."""
import unittest
from select_ik_trajectory_dp_offline import window_quality_pass


class WindowQualityGateTests(unittest.TestCase):
    def test_120_complete_pass(self):
        self.assertTrue(window_quality_pass(120, 120, [{"quality_pass": True}]))

    def test_31_complete_pass(self):
        self.assertTrue(window_quality_pass(31, 31, [{"quality_pass": True}]))

    def test_incomplete_120_fails(self):
        self.assertFalse(window_quality_pass(119, 120, [{"quality_pass": True}]))

    def test_bad_segment_fails(self):
        self.assertFalse(window_quality_pass(120, 120, [{"quality_pass": True}, {"quality_pass": False}]))

    def test_no_segment_fails(self):
        self.assertFalse(window_quality_pass(120, 120, []))


if __name__ == "__main__":
    unittest.main()
