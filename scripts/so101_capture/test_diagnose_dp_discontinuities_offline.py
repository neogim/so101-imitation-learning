#!/usr/bin/env python3
"""Offline discontinuity diagnostic regression checks."""
import unittest
from diagnose_dp_discontinuities_offline import diagnose


def row(frame, angle, mapping="shoulder+_elbow+", direction="x+"):
    return {"mapping":mapping,"method":"dp_second_order","direction":direction,"frame":frame,
      "candidate":{"motor":[0,angle,0,0,0],"seed":"recorded_clipped","active_bound_joints":[]}}


class DiagnosticTests(unittest.TestCase):
    def test_smooth_path(self):
        result=diagnose([row(i,i*0.1) for i in range(5)],"shoulder+_elbow+","dp_second_order")
        self.assertEqual(result["x+"]["violations"],0)

    def test_joint_jump_speed_acceleration(self):
        result=diagnose([row(0,0),row(1,0),row(2,12)],"shoulder+_elbow+","dp_second_order")
        events=result["x+"]["first_events"]
        self.assertEqual({e["kind"] for e in events},{"jump","speed","accel"})
        self.assertTrue(all(e["joint"]=="shoulder_lift" for e in events))

    def test_no_gap_teleport(self):
        result=diagnose([row(0,0),row(2,20)],"shoulder+_elbow+","dp_second_order")
        self.assertEqual(result["x+"]["violations"],0)


if __name__=="__main__":
    unittest.main()
