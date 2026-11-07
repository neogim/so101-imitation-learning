"""Offline ChArUco board detection test; no images or hardware required."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_side_charuco_offline import BOARD, detect, make_board


@unittest.skipUnless(hasattr(cv2, "aruco") and hasattr(cv2.aruco, "CharucoBoard"),
                     "OpenCV ArUco/Charuco unavailable")
class CharucoCalibrationTests(unittest.TestCase):
    def test_board_renders_and_all_48_inner_corners_detected(self):
        board = make_board()
        image = board.generateImage((1400, 1800))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        observation = detect(board, image)
        self.assertIsNotNone(observation)
        corners, ids = observation
        self.assertEqual(len(ids), (BOARD["squares_x"] - 1) * (BOARD["squares_y"] - 1))
        self.assertEqual(corners.shape, (48, 1, 2))

    def test_reject_when_all_corners_obscured(self):
        board = make_board()
        blank = np.full((480, 640, 3), 255, np.uint8)
        self.assertIsNone(detect(board, blank))

    def test_reject_invalid_size(self):
        with self.assertRaises(ValueError):
            make_board(dict(BOARD, marker_length_mm=22.))


if __name__ == "__main__":
    unittest.main()
