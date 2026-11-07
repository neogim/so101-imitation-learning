"""Offline tests for region-based ArUco observation selection."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_aruco_calibration import (
    marker_quality, stationary_candidates, stationary_regions,
    parse_marker_sizes_mm, MIXED_MARKER_SIZES_MM,
)


class StationarySelectionTests(unittest.TestCase):
    def test_mixed_marker_size_parse(self):
        value = "0:14,1:25,2:14,3:25,4:25,5:25"
        self.assertEqual(parse_marker_sizes_mm(value), MIXED_MARKER_SIZES_MM)
        self.assertEqual(parse_marker_sizes_mm(None), {i: 14. for i in range(6)})

    def test_mixed_marker_sizes_must_be_complete_and_positive(self):
        for invalid in ("0:14,1:25", "0:14,1:25,2:14,3:25,4:25,5:0",
                        "0:14,1:25,2:14,3:25,4:25,5:nan",
                        "0:14,1:25,2:14,3:25,4:25,5:25,5:25"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_marker_sizes_mm(invalid)

    def test_one_pose_after_motion(self):
        import numpy as np
        moving = np.column_stack([np.linspace(0, 20, 20), np.zeros((20, 5))])
        hold = np.tile([20, 3, -2, 0, 0, 20], (22, 1))
        states = np.vstack([moving, hold])
        chosen = stationary_candidates(states, window=10, tolerance_deg=.5)
        self.assertEqual(len(chosen), 1)
        self.assertGreaterEqual(chosen[0], 20)

    def test_separate_distinct_holds(self):
        import numpy as np
        a = np.tile([0, 0, 0, 0, 0, 0], (15, 1))
        transition = np.column_stack([np.linspace(0, 15, 15), np.zeros((15, 5))])
        b = np.tile([15, 0, 0, 0, 0, 40], (15, 1))
        states = np.vstack([a, transition, b])
        chosen = stationary_candidates(states, window=10)
        self.assertEqual(len(chosen), 2)

    def test_gripper_alone_does_not_count_as_new_pose(self):
        import numpy as np
        states = np.tile([0, 0, 0, 0, 0, 0], (35, 1)).astype(float)
        states[20:, 5] = 80
        self.assertEqual(len(stationary_candidates(states, window=10)), 1)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            stationary_regions([[float("nan")] * 6] * 20)

    def test_bad_width_rejected(self):
        with self.assertRaises(ValueError):
            stationary_regions([[0] * 5] * 20)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            stationary_candidates([[0] * 6] * 20, window=1)

    def test_scan_all_frames_inside_hold_not_only_first_midpoint(self):
        import numpy as np
        states = np.zeros((50, 6))
        regions = stationary_regions(states, window=12)
        self.assertEqual(regions, [list(range(50))])
        self.assertIn(20, regions[0])
        self.assertIn(40, regions[0])

    def test_motion_splits_stationary_regions(self):
        import numpy as np
        a = np.zeros((25, 6))
        moving = np.column_stack([np.arange(1, 21, dtype=float), np.zeros((20, 5))])
        b = np.tile([20, 0, 0, 0, 0, 5], (25, 1))
        regions = stationary_regions(np.vstack([a, moving, b]), window=12)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0], list(range(25)))
        self.assertEqual(regions[1], list(range(44, 70)))

    def test_short_or_moving_sequence_has_no_region(self):
        import numpy as np
        self.assertEqual(stationary_regions(np.zeros((8, 6)), window=12), [])
        moving = np.column_stack([np.arange(30, dtype=float), np.zeros((30, 5))])
        self.assertEqual(stationary_regions(moving, window=12), [])

    def test_marker_quality_prefers_valid_count_then_edge_size(self):
        a = [{"id": 1, "min_edge_px": 12.0}]
        b = [{"id": 0, "min_edge_px": 8.1}, {"id": 4, "min_edge_px": 9.0}]
        self.assertGreater(marker_quality(b, min_edge_px=8),
                           marker_quality(a, min_edge_px=8))
        self.assertIsNone(marker_quality(a, min_edge_px=13))
        self.assertEqual(marker_quality(a, min_edge_px=8), (1, 12., 12.))

    def test_empty_region_validation(self):
        self.assertEqual(stationary_regions([[0] * 6] * 11, window=12), [])


if __name__ == "__main__":
    unittest.main()
