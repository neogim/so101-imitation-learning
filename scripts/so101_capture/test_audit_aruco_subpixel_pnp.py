"""Offline regression checks for subpixel ArUco corner refinement."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_aruco_subpixel_pnp import refine_corners, summarize, displacement_pair


class SubpixelArucoTest(unittest.TestCase):
    def test_rounded_corners_refine_on_blurred_square(self):
        image = np.full((480, 640), 255, np.uint8)
        pts = np.array([[295.4, 199.7], [317.2, 199.2],
                        [318.3, 220.7], [294.6, 221.2]], np.float32)
        cv2.fillConvexPoly(image, np.round(pts).astype(np.int32), 0)
        blurred = cv2.GaussianBlur(image, (5, 5), 0.7)
        orig = np.round(pts).astype(np.float32)
        refined, accepted, reason = refine_corners(
            blurred, orig, half_window=3, max_shift_px=2)
        self.assertTrue(accepted, msg=reason)
        self.assertEqual(refined.shape, (4, 2))
        self.assertTrue(np.isfinite(refined).all())

    def test_refuse_image_edge(self):
        image = np.full((480, 640), 255, np.uint8)
        corners = [[1, 2], [15, 2], [15, 18], [1, 18]]
        _, accepted, reason = refine_corners(image, corners)
        self.assertFalse(accepted)
        self.assertEqual(reason, "corner_near_image_boundary")

    def test_empty_summary(self):
        self.assertEqual(summarize([]), {})
        self.assertEqual(displacement_pair([], 2, 631, 680)["status"],
                         "missing_observation")


if __name__ == "__main__":
    unittest.main()
