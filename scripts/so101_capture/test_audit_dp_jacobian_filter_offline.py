#!/usr/bin/env python3
"""Regression: filter by requested direction before checking duplicate frame IDs."""
import unittest
from audit_dp_jacobian_offline import filter_selected_path

class FilterDirectionTests(unittest.TestCase):
    def test_multiple_directions_same_frames(self):
        rows=[{"mapping":"shoulder+_elbow+","method":"dp_second_order",
               "direction":direction,"frame":frame}
              for direction in ("x+","y+","y-","z-") for frame in range(50,101)]
        for direction in ("x+","y+","y-","z-"):
            selected=filter_selected_path(rows,"shoulder+_elbow+","dp_second_order",direction,50,100)
            self.assertEqual(len(selected),51)
            self.assertTrue(all(x["direction"]==direction for x in selected))

    def test_missing_frame_is_error(self):
        rows=[{"mapping":"shoulder+_elbow+","method":"dp_second_order",
               "direction":"x+","frame":50}]
        with self.assertRaisesRegex(ValueError,"missing"):
            filter_selected_path(rows,"shoulder+_elbow+","dp_second_order","x+",50,51)

if __name__=="__main__":
    unittest.main()
