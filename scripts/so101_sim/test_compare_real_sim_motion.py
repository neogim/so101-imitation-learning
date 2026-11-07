"""Synthetic tests for side-by-side replay helpers (no robot or video required)."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_real_sim_motion import sample_evenly, comparison_panel


class MotionComparisonTests(unittest.TestCase):
    def test_sample_five_spread_across_twenty(self):
        rows = [{"frame_index": i} for i in range(20)]
        chosen = sample_evenly(rows, 5)
        self.assertEqual(len(chosen), 5)
        self.assertEqual(chosen[0]["frame_index"], 0)
        self.assertEqual(chosen[-1]["frame_index"], 19)
        self.assertEqual(len(set(item["frame_index"] for item in chosen)), 5)

    def test_sample_limits_and_invalid(self):
        self.assertEqual(len(sample_evenly([1, 2], 5)), 2)
        with self.assertRaises(ValueError):
            sample_evenly([], 5)
        with self.assertRaises(ValueError):
            sample_evenly([1], 0)

    def test_real_sim_panel(self):
        real = Image.new("RGB", (224, 168), "red")
        sim = Image.new("RGB", (640, 480), "blue")
        panel = comparison_panel(real, sim, "test")
        self.assertEqual(panel.size, (1280, 508))
        self.assertEqual(tuple(np.asarray(panel)[28, 0]), (255, 0, 0))
        self.assertEqual(tuple(np.asarray(panel)[28, 640]), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
