"""Synthetic-only checks; no hardware imports are instantiated by these tests."""

import unittest

from sync_goal_to_present import HardwareGoalIO, MOTORS, SyntheticGoalIO, synchronize, valid_arm_phrase


class GoalSyncTests(unittest.TestCase):
    def test_confirmation_phrase_is_unambiguous_and_required(self):
        self.assertTrue(valid_arm_phrase("SYNC GOAL ONLY"))
        self.assertTrue(valid_arm_phrase(" sync goal only "))
        self.assertFalse(valid_arm_phrase("SO101 GOAL SYNC ONLY"))
        self.assertFalse(valid_arm_phrase(""))

    def run_case(self, io):
        events = []
        result = synchronize(io, events.append)
        return result, events

    def test_success_writes_only_present_raw_values_once(self):
        io = SyntheticGoalIO()
        initial = io.present.copy()
        result, events = self.run_case(io)
        self.assertTrue(result["completed"], result["fault"])
        self.assertEqual(io.writes, [(name, initial[name]) for name in MOTORS])
        self.assertEqual(io.goal, initial)
        self.assertEqual(io.torque, dict.fromkeys(MOTORS, 0))
        self.assertEqual(sum(e["event"] == "all_axes_after_write" for e in events), 6)

    def test_initial_torque_or_status_refuses_all_writes(self):
        for field, value in (("torque", 1), ("status", 1)):
            with self.subTest(field=field):
                io = SyntheticGoalIO()
                getattr(io, field)[MOTORS[0]] = value
                result, _ = self.run_case(io)
                self.assertFalse(result["completed"])
                self.assertEqual(io.writes, [])

    def test_outside_calibration_refuses_all_writes(self):
        io = SyntheticGoalIO()
        io.present[MOTORS[2]] = io.bounds[MOTORS[2]][0] - 1
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual(io.writes, [])

    def test_ambiguous_write_timeout_no_retry_or_later_write(self):
        io = SyntheticGoalIO(fail_write_on=MOTORS[2])
        result, events = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual([name for name, _ in io.writes], list(MOTORS[:3]))
        self.assertEqual(result["writes"], list(MOTORS[:2]))
        self.assertEqual(sum(e["event"] == "before_write" for e in events), 3)

    def test_torque_change_after_write_stops_immediately(self):
        io = SyntheticGoalIO(torque_change_on=MOTORS[1])
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual([name for name, _ in io.writes], list(MOTORS[:2]))

    def test_real_incident_delayed_torque_change_stops_after_first_write(self):
        class DelayedTorqueOnGoal(SyntheticGoalIO):
            def __init__(self):
                super().__init__()
                self.pending = False

            def write_goal(self, motor, raw_value):
                super().write_goal(motor, raw_value)
                self.pending = True

            def read_one(self, motor):
                snapshot = super().read_one(motor)
                if self.pending:
                    # Matches the real trace: immediate read is 0, then 1.
                    self.torque[motor] = 1
                    self.pending = False
                return snapshot

        io = DelayedTorqueOnGoal()
        result, events = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual([name for name, _ in io.writes], [MOTORS[0]])
        self.assertEqual(next(e for e in events if e["event"] == "after_write")["readback"]["Torque_Enable"], 0)
        self.assertEqual(next(e for e in events if e["event"] == "all_axes_after_write")["snapshot"]["Torque_Enable"][MOTORS[0]], 1)

    def test_hardware_io_cannot_open(self):
        with self.assertRaisesRegex(RuntimeError, "suspended"):
            HardwareGoalIO().open()

    def test_position_change_after_write_stops_immediately(self):
        io = SyntheticGoalIO(position_change_on=MOTORS[1])
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual([name for name, _ in io.writes], list(MOTORS[:2]))

    def test_read_timeout_refuses_goal_write(self):
        io = SyntheticGoalIO(fail_read_on=MOTORS[0])
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual(io.writes, [])

    def test_other_axis_torque_change_detected_before_next_write(self):
        class OtherAxisFault(SyntheticGoalIO):
            def write_goal(self, motor, raw_value):
                super().write_goal(motor, raw_value)
                if motor == MOTORS[0]:
                    self.torque[MOTORS[5]] = 1

        io = OtherAxisFault()
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual([name for name, _ in io.writes], [MOTORS[0]])

    def test_goal_readback_mismatch_stops(self):
        class DroppedWrite(SyntheticGoalIO):
            def write_goal(self, motor, raw_value):
                self.writes.append((motor, raw_value))

        io = DroppedWrite()
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual([name for name, _ in io.writes], [MOTORS[0]])


if __name__ == "__main__":
    unittest.main()
