"""Synthetic-only tests for ArUco IPPE square ambiguity diagnostics."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_aruco_pnp_ambiguity import candidate_poses, marker_object_points


class ArucoPnPAmbiguityTests(unittest.TestCase):
    def test_round_trip_projected_square(self):
        K = np.array([[543., 0., 316.], [0., 543., 230.], [0., 0., 1.]])
        D = np.array([.0765, -.1209, 0., 0., 0.])
        object_points = marker_object_points(.014)
        image = cv2.projectPoints(
            object_points, np.array([.16, -.10, .04]),
            np.array([.02, -.03, .62]), K, D)[0].reshape(4, 2)
        solutions = candidate_poses(image, K, D, .014)
        self.assertGreaterEqual(len(solutions), 1)
        self.assertLess(solutions[0]["reprojection_rms_px"], .05)
        self.assertAlmostEqual(solutions[0]["distance_m"], .62, delta=.05)

    def test_invalid_marker_side(self):
        with self.assertRaises(ValueError):
            marker_object_points(0)


if __name__ == "__main__":
    unittest.main()
