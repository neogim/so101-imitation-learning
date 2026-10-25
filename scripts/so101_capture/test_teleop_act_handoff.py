import unittest

from act_action_guard import NAMES
from teleop_act_handoff import HandoffFault, HandoffGate


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.writes = []
        self.goal = dict.fromkeys(NAMES, 20.0)
        self.goal[NAMES[-1]] = 50.0
        self.limits = {n: (0, 100) if n == NAMES[-1] else (-90, 90) for n in NAMES}
        self.gate = HandoffGate(self.write, self.limits)

    def write(self, goal):
        self.writes.append(goal.copy())
        return goal

    def snapshot(self, **changes):
        row = dict(goal=self.goal.copy(), present=self.goal.copy(),
                   torque=dict.fromkeys(NAMES, 1), status=dict.fromkeys(NAMES, 0),
                   calibration_ok=True, sample_age_s=0.01,
                   communication_ok=True, queued_goals=0)
        row.update(changes)
        return row

    def handoff(self, **changes):
        self.gate.send_teleop(self.goal)
        self.gate.begin_handoff()
        self.gate.arm_act(**self.snapshot(**changes))

    def test_exclusive_switch_and_first_goal_slew(self):
        self.gate.send_teleop(self.goal)
        self.gate.begin_handoff()
        with self.assertRaises(HandoffFault):
            self.gate.send_teleop(self.goal)
        with self.assertRaises(HandoffFault):
            self.gate.send_act([80] * 6, self.goal, sample_age_s=0.01)
        self.assertEqual(len(self.writes), 1)
        self.gate.arm_act(**self.snapshot())
        with self.assertRaises(HandoffFault):
            self.gate.send_teleop(self.goal)
        sent, reasons = self.gate.send_act([80] * 6, self.goal, sample_age_s=0.01)
        self.assertEqual(len(self.writes), 2)
        self.assertTrue(all(abs(sent[n] - self.goal[n]) <= 1.5 for n in NAMES))
        self.assertTrue(any(r.startswith("speed:") for r in reasons))

    def test_unsafe_handoff_latches_without_write(self):
        for change in (dict(torque={**dict.fromkeys(NAMES, 1), NAMES[0]: 0}),
                       dict(status={**dict.fromkeys(NAMES, 0), NAMES[1]: 1}),
                       dict(communication_ok=False), dict(queued_goals=1),
                       dict(sample_age_s=0.11),
                       dict(present={**self.goal, NAMES[2]: 29})):
            with self.subTest(change=change):
                self.setUp()
                self.gate.send_teleop(self.goal)
                self.gate.begin_handoff()
                with self.assertRaises(HandoffFault):
                    self.gate.arm_act(**self.snapshot(**change))
                self.assertEqual(self.gate.owner, "FAULT")
                self.assertEqual(len(self.writes), 1)
                with self.assertRaises(HandoffFault):
                    self.gate.send_act([20] * 6, self.goal, sample_age_s=0)

    def test_act_communication_fault_no_new_goal(self):
        self.handoff()
        with self.assertRaises(HandoffFault):
            self.gate.send_act([20] * 6, self.goal, sample_age_s=0,
                               communication_ok=False)
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(self.gate.owner, "FAULT")

    def test_act_tracking_fault_no_new_goal(self):
        self.handoff()
        measured = {**self.goal, NAMES[3]: 29}
        with self.assertRaises(HandoffFault):
            self.gate.send_act([20] * 6, measured, sample_age_s=0)
        self.assertEqual(len(self.writes), 1)

    def test_readback_mismatch_no_new_goal(self):
        self.gate.send_teleop(self.goal)
        self.gate.begin_handoff()
        with self.assertRaises(HandoffFault):
            self.gate.arm_act(**self.snapshot(goal={**self.goal, NAMES[0]: 21}))
        self.assertEqual(len(self.writes), 1)

    def test_first_motion_gripper_is_checked_before_writer(self):
        self.gate.send_teleop(self.goal)
        self.gate.begin_handoff()
        self.gate.arm_act(**self.snapshot(), speed_per_s=(3.0,) * 6,
                          max_displacement=1.0)
        with self.assertRaisesRegex(HandoffFault, 'gripper Goal must remain fixed'):
            self.gate.send_act([20.0] * 5 + [0.0], self.goal, sample_age_s=0.01)
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(self.gate.owner, 'FAULT')

    def test_first_motion_writer_change_latches_without_further_write(self):
        self.gate.send_teleop(self.goal)
        self.gate.begin_handoff()
        self.gate.arm_act(**self.snapshot(), speed_per_s=(3.0,) * 6,
                          max_displacement=1.0)
        def changed(goal):
            self.writes.append(goal.copy())
            return {**goal, NAMES[0]: goal[NAMES[0]] + 0.01}
        self.gate.writer = changed
        with self.assertRaisesRegex(HandoffFault, 'writer changed guarded Action'):
            self.gate.send_act([21.0] * 5 + [50.0], self.goal, sample_age_s=0.01)
        self.assertEqual(self.gate.owner, 'FAULT')
        self.assertEqual(len(self.writes), 2)  # one ambiguous write; no retry
        with self.assertRaises(HandoffFault):
            self.gate.send_act([21.0] * 5 + [50.0], self.goal, sample_age_s=0.01)
        self.assertEqual(len(self.writes), 2)


if __name__ == "__main__":
    unittest.main()
