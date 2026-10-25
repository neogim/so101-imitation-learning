"""No-hardware tests for the independent one-broadcast Goal trial."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import sys

from act_startup import MOTORS
from probe_goal_broadcast import HardwareIO, SyntheticIO, main, probe


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class BroadcastGoalTests(unittest.TestCase):
    @staticmethod
    def run_case(io):
        events = []
        clock = FakeClock()
        result = probe(io, events.append, clock=clock, sleep=clock.sleep)
        return result, events

    def test_normal_exactly_one_broadcast_and_four_readbacks(self):
        io = SyntheticIO()
        initial = io.present.copy()
        result, events = self.run_case(io)
        self.assertTrue(result["completed"])
        self.assertEqual(io.writes, 1)
        self.assertEqual(result["write_attempts"], 1)
        self.assertEqual(io.goal, initial)
        self.assertTrue(io.closed)
        readbacks = [e for e in events if e["event"] == "readback"]
        self.assertEqual([e["delay_target_ms"] for e in readbacks], [0, 20, 100, 500])
        for event in readbacks:
            for register in ("Goal_Position", "Present_Position", "Torque_Enable", "Status"):
                self.assertEqual(set(event["raw"][register]), set(MOTORS))

    def test_unsafe_preflight_never_writes(self):
        for fault in ("calibration_mismatch", "torque", "status", "range", "missing_joint"):
            with self.subTest(fault=fault):
                io = SyntheticIO(fault if fault == "calibration_mismatch" else None)
                if fault == "torque":
                    io.torque[MOTORS[0]] = 1
                elif fault == "status":
                    io.status[MOTORS[1]] = 2
                elif fault == "range":
                    io.present[MOTORS[2]] = io.bounds[MOTORS[2]][0] - 1
                elif fault == "missing_joint":
                    del io.goal[MOTORS[3]]
                result, _ = self.run_case(io)
                self.assertFalse(result["completed"])
                self.assertEqual(io.writes, 0)

    def test_goal_or_torque_or_motion_anomaly_stops_after_one_write(self):
        for fault in ("torque_on", "partial_torque", "late_torque", "goal_mismatch", "position_change"):
            with self.subTest(fault=fault):
                io = SyntheticIO(fault)
                result, events = self.run_case(io)
                self.assertFalse(result["completed"])
                self.assertEqual(io.writes, 1)
                self.assertEqual(result["write_attempts"], 1)
                self.assertTrue(io.closed)
                self.assertIn("fault", [e["event"] for e in events])
                self.assertTrue(result["no_configure_or_alignment"])

    def test_ambiguous_write_timeout_is_never_retried(self):
        io = SyntheticIO("write_timeout")
        result, events = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual(io.writes, 1)
        self.assertEqual(result["write_attempts"], 1)
        self.assertIn("ambiguous", result["fault"])
        self.assertNotIn("write_returned", [e["event"] for e in events])

    def test_readback_timeout_stops_without_retry(self):
        io = SyntheticIO("read_timeout")
        result, events = self.run_case(io)
        self.assertFalse(result["completed"])
        self.assertEqual(io.writes, 1)
        self.assertEqual(result["write_attempts"], 1)
        self.assertEqual([e["event"] for e in events].count("fault_readback"), 1)
        self.assertEqual(events[-1]["current_state"], "unknown")

    def test_stale_fresh_read_refuses_broadcast(self):
        # A slow log step must invalidate the otherwise fresh Present sample.
        io = SyntheticIO()
        clock = FakeClock()
        events = []

        def slow_emit(row):
            events.append(row)
            if row["event"] == "write_intent":
                clock.now += 0.11

        result = probe(io, slow_emit, clock=clock, sleep=clock.sleep)
        self.assertFalse(result["completed"])
        self.assertEqual(io.writes, 0)
        self.assertIn("100 ms", result["fault"])

    def test_existing_lerobot_sync_write_path_uses_raw_and_no_retry(self):
        class Bus:
            def __init__(self):
                self.calls = []

            def sync_write(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        io = HardwareIO()
        bus = Bus()
        io.robot = type("Robot", (), {"bus": bus})()
        target = dict(zip(MOTORS, (1481, 1633, 2894, 2087, 833, 2161)))
        io.write_goals_once(target)
        self.assertEqual(bus.calls, [(('Goal_Position', target),
                                      {'normalize': False, 'num_retry': 0})])

    def test_jsonl_retains_failed_six_axis_readback(self):
        io = SyntheticIO("partial_torque")
        clock = FakeClock()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with path.open("x") as stream:
                def emit(row):
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()

                result = probe(io, emit, clock=clock, sleep=clock.sleep)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertFalse(result["completed"])
        bad = next(row for row in rows if row["event"] == "readback")
        self.assertEqual(bad["raw"]["Torque_Enable"][MOTORS[2]], 1)
        self.assertEqual(bad["raw"]["Torque_Enable"][MOTORS[0]], 0)
        self.assertEqual(rows[-1]["event"], "fault_readback")

    def test_hardware_cli_refuses_without_separate_approval_flags(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            with patch.object(sys, "argv", ["probe_goal_broadcast.py", "--mode", "hardware",
                                            "--output", str(output)]):
                with self.assertRaises(SystemExit):
                    main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
