"""Pure geometry tests for three-click Side object alignment."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_episode_objects import targets_from_clicks
from compare_real_sim_motion import pixel_to_base_plane


class ThreeClickAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.T = np.array([[-.61650, .43981, -.65307, .49516],
                           [.75783, .10642, -.64372, .38170],
                           [-.21361, -.89176, -.39891, .28123],
                           [0, 0, 0, 1]])
        self.K = {"fx":543.173,"fy":542.957,"cx":316.395,"cy":229.548}

    def test_three_clicks_create_expected_json_shape_and_yaw(self):
        points = [(250., 370.), (310., 350.), (410., 310.)]
        result = targets_from_clicks(points, self.T, self.K)
        self.assertEqual(result["muhi_center_px"], [280., 360.])
        self.assertEqual(result["cup_center_px"], [410., 310.])
        body = pixel_to_base_plane(self.T, self.K, *points[0], .019)
        cap = pixel_to_base_plane(self.T, self.K, *points[1], .019)
        axis = cap[:2]-body[:2]
        yaw = np.radians(result["muhi_yaw_deg"])
        expected_axis = np.array([-np.sin(yaw), np.cos(yaw)])
        self.assertGreater(np.dot(axis / np.linalg.norm(axis), expected_axis), .999)

    def test_bad_points(self):
        with self.assertRaises(ValueError):
            targets_from_clicks([(0,0),(1,1)], self.T, self.K)
        with self.assertRaises(ValueError):
            targets_from_clicks([(250,370),(250,370),(400,300)], self.T, self.K)
        with self.assertRaises(ValueError):
            targets_from_clicks([(250,370),(300,350),(700,300)], self.T, self.K)


if __name__ == "__main__":
    unittest.main()
