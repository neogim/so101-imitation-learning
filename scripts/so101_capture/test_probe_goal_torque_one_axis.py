"""Offline safety tests for the single-Goal, possible-torque-on probe."""

import unittest

from probe_goal_torque_one_axis import SyntheticProbeIO, run_probe
from sync_goal_to_present import MOTORS


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, delay):
        self.now += max(0, delay)


class ProbeTests(unittest.TestCase):
    def run_case(self, io, *, sleep=None):
        timer = FakeClock()
        events = []
        result = run_probe(io, io.motor, events.append, clock=timer.clock,
                           sleep=sleep or timer.sleep)
        return result, events

    def test_goal_write_may_enable_torque_without_motion(self):
        io = SyntheticProbeIO(MOTORS[0], torque_after_write=True)
        result, events = self.run_case(io)
        self.assertTrue(result["completed"], result["fault"])
        self.assertEqual(io.writes, [(MOTORS[0], 1481)])
        self.assertEqual(result["torque_after"][MOTORS[0]], 1)
        self.assertTrue(all(value == 0 for value in result["position_change_raw"].values()))
        self.assertEqual(sum(e["event"] == "readback" for e in events), 4)

    def test_existing_torque_or_status_refuses_before_write(self):
        for field in ("torque", "status"):
            with self.subTest(field=field):
                io = SyntheticProbeIO(MOTORS[0])
                getattr(io, field)[MOTORS[3]] = 1
                result, _ = self.run_case(io)
                self.assertFalse(result["completed"])
                self.assertFalse(result["write_attempted"])
                self.assertEqual(io.writes, [])

    def test_out_of_range_position_refuses_before_write(self):
        io = SyntheticProbeIO(MOTORS[0])
        io.present[MOTORS[2]] = 0
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual(io.writes, [])

    def test_ambiguous_write_timeout_never_retries(self):
        io = SyntheticProbeIO(MOTORS[0], write_fails=True)
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertTrue(result["write_attempted"])
        self.assertEqual(io.writes, [(MOTORS[0], 1481)])

    def test_movement_after_write_stops_with_no_second_write(self):
        io = SyntheticProbeIO(MOTORS[0], move_after_write=20)
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual(len(io.writes), 1)

    def test_nonzero_status_after_write_stops(self):
        io = SyntheticProbeIO(MOTORS[0], status_after_write=1)
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual(len(io.writes), 1)

    def test_other_axis_torque_change_stops(self):
        class OtherAxisChange(SyntheticProbeIO):
            def write_goal_once(self, motor, raw):
                super().write_goal_once(motor, raw)
                self.torque[MOTORS[3]] = 1

        io = OtherAxisChange(MOTORS[0])
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual(len(io.writes), 1)

    def test_communication_failure_after_write_no_retry(self):
        class ReadFailure(SyntheticProbeIO):
            def read_all(self):
                if self.writes:
                    raise ConnectionError("Injected post-write read timeout")
                return super().read_all()

        io = ReadFailure(MOTORS[0])
        result, _ = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual(len(io.writes), 1)


if __name__ == "__main__":
    unittest.main()
