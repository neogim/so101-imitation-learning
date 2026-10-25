"""One-shot six-axis Goal <- fresh Present probe; synthetic by default.

This is an independent Stage-1 trial, never an ACT startup. Hardware mode is
for a *future, separately approved* operator-supervised trial. Importing this
module and running synthetic mode never opens a serial port. Hardware mode does
not configure motors, enable/disable torque, align, or send ACT Actions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from act_startup import MOTORS, REGISTERS


MAX_POSITION_CHANGE_RAW = 5
MAX_FRESH_AGE_S = 0.10
READBACK_DELAYS_S = (0.0, 0.02, 0.10, 0.50)
APPROVAL_PHRASE = "ONE BROADCAST GOAL WRITE"


class ProbeFault(RuntimeError):
    pass


def _check_snapshot(raw, limits, *, phase, baseline=None, target=None):
    if set(limits) != set(MOTORS):
        raise ProbeFault(f"{phase}: incomplete calibration limits")
    for register in REGISTERS:
        if register not in raw or set(raw[register]) != set(MOTORS):
            raise ProbeFault(f"{phase}: incomplete {register}")
    for motor in MOTORS:
        low, high = limits[motor]
        present = raw["Present_Position"][motor]
        goal = raw["Goal_Position"][motor]
        torque = raw["Torque_Enable"][motor]
        status = raw["Status"][motor]
        if not all(isinstance(v, int) and not isinstance(v, bool)
                   for v in (low, high, present, goal, torque, status)):
            raise ProbeFault(f"{phase}: {motor} invalid raw register value")
        if low > high or not low <= present <= high:
            raise ProbeFault(f"{phase}: {motor} Present {present} outside calibrated [{low},{high}]")
        if torque != 0 or status != 0:
            raise ProbeFault(f"{phase}: {motor} Torque={torque} Status={status}; stop")
        if baseline is not None:
            if abs(present - baseline["Present_Position"][motor]) > MAX_POSITION_CHANGE_RAW:
                raise ProbeFault(f"{phase}: {motor} moved unexpectedly")
            if target is None and goal != baseline["Goal_Position"][motor]:
                raise ProbeFault(f"{phase}: {motor} Goal changed before write")
        if target is not None and goal != target[motor]:
            raise ProbeFault(f"{phase}: {motor} Goal readback {goal} != sent {target[motor]}")


def probe(io, emit, *, clock=time.perf_counter, sleep=time.sleep):
    """Exactly one sync_write attempt after two fresh six-axis preflight reads."""
    started = clock()
    write_attempts = 0
    target = None
    last_snapshot = None
    outcome = None

    def event(name, **fields):
        emit(dict(event=name, utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_ms=(clock() - started) * 1000, **fields))

    try:
        metadata = io.open()
        limits = io.limits()
        if not io.calibration_matches():
            raise ProbeFault("Existing Follower calibration mismatch")
        before = io.read_raw()
        last_snapshot = before
        event("preflight", raw=before, limits=limits, metadata=metadata)
        _check_snapshot(before, limits, phase="preflight")

        sampled_at = clock()  # conservative age: first register may be read now
        fresh = io.read_raw()
        last_snapshot = fresh
        event("fresh_before_write", raw=fresh,
              read_ms=(clock() - sampled_at) * 1000)
        _check_snapshot(fresh, limits, phase="fresh_before_write", baseline=before)
        target = {motor: fresh["Present_Position"][motor] for motor in MOTORS}
        if any(not limits[motor][0] <= target[motor] <= limits[motor][1] for motor in MOTORS):
            raise ProbeFault("Goal target outside calibration")
        event("write_intent", target_raw=target, note="One broadcast, no status ACK and no retry")
        if clock() - sampled_at > MAX_FRESH_AGE_S:
            raise ProbeFault("Fresh Present sample exceeded 100 ms before Goal write")

        write_attempts = 1  # increment BEFORE call: timeout has ambiguous delivery
        write_started = clock()
        io.write_goals_once(target)
        write_returned = clock()
        event("write_returned", target_raw=target,
              write_ms=(write_returned - write_started) * 1000)

        for delay in READBACK_DELAYS_S:
            remaining = write_returned + delay - clock()
            if remaining > 0:
                sleep(remaining)
            raw = io.read_raw()
            last_snapshot = raw
            event("readback", delay_target_ms=delay * 1000,
                  after_write_ms=(clock() - write_returned) * 1000, raw=raw)
            _check_snapshot(raw, limits, phase=f"readback_{delay * 1000:g}ms",
                            baseline=fresh, target=target)

        outcome = dict(completed=True, write_attempts=write_attempts,
                       target_raw=target, last_snapshot=last_snapshot, fault=None,
                       no_configure_or_alignment=True)
        return outcome
    except BaseException as exc:
        event("fault", fault=f"{type(exc).__name__}: {exc}",
              write_attempts=write_attempts,
              last_observed_raw=last_snapshot,
              note="No further Goal/torque write; last observation is not guaranteed current")
        # One read-only diagnostic attempt; never resumes or retries the write.
        try:
            diagnostic = io.read_raw()
            event("fault_readback", raw=diagnostic, validated=False)
        except BaseException as read_exc:
            event("fault_readback", raw=None, current_state="unknown",
                  error=f"{type(read_exc).__name__}: {read_exc}")
        outcome = dict(completed=False, write_attempts=write_attempts,
                       target_raw=target, last_snapshot=last_snapshot,
                       fault=f"{type(exc).__name__}: {exc}", no_configure_or_alignment=True)
        return outcome
    finally:
        try:
            io.close_without_writes()
        except BaseException as exc:
            event("close_fault", fault=f"{type(exc).__name__}: {exc}")
            if outcome is not None:
                outcome["completed"] = False
                outcome["fault"] = outcome["fault"] or f"Close failed: {type(exc).__name__}: {exc}"


class SyntheticIO:
    def __init__(self, fault=None):
        self.fault = fault
        self.present = dict(zip(MOTORS, (1481, 1633, 2894, 2087, 833, 2161)))
        self.goal = dict.fromkeys(MOTORS, 0)
        self.torque = dict.fromkeys(MOTORS, 0)
        self.status = dict.fromkeys(MOTORS, 0)
        self.bounds = dict(zip(MOTORS, ((733, 3468), (776, 3170), (907, 3122),
                                       (863, 3250), (0, 4095), (2014, 2994))))
        self.reads = 0
        self.writes = 0
        self.closed = False

    def open(self):
        return {"device": "synthetic", "no_serial_port": True}

    def calibration_matches(self):
        return self.fault != "calibration_mismatch"

    def limits(self):
        return self.bounds.copy()

    def read_raw(self):
        self.reads += 1
        if self.fault == "read_timeout" and self.reads >= 3:
            raise ConnectionError("synthetic Present_Position timeout")
        if self.fault == "late_torque" and self.reads == 4:
            self.torque[MOTORS[0]] = 1
        return {"Goal_Position": self.goal.copy(), "Present_Position": self.present.copy(),
                "Torque_Enable": self.torque.copy(), "Status": self.status.copy()}

    def write_goals_once(self, target):
        self.writes += 1
        if self.fault == "write_timeout":
            raise ConnectionError("synthetic ambiguous broadcast write")
        self.goal = target.copy()
        if self.fault == "torque_on":
            self.torque[MOTORS[0]] = 1
        elif self.fault == "partial_torque":
            self.torque[MOTORS[2]] = 1
        elif self.fault == "goal_mismatch":
            self.goal[MOTORS[3]] += 1
        elif self.fault == "position_change":
            self.present[MOTORS[1]] += 20

    def close_without_writes(self):
        self.closed = True


class HardwareIO:
    """No Robot.connect(), configure(), torque write or calibration write."""

    def __init__(self):
        self.robot = None

    def open(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from record_episode import FOLLOWER, calibration_paths, deny_calibration

        if not Path(FOLLOWER).exists() or not calibration_paths()[0].is_file():
            raise FileNotFoundError("Follower port or existing calibration missing")
        self.robot = SO101Follower(SO101FollowerConfig(
            port=FOLLOWER, id="my_follower_arm", use_degrees=True,
            disable_torque_on_disconnect=False))
        self.robot.calibrate = deny_calibration
        self.robot._save_calibration = deny_calibration
        self.robot.bus.write_calibration = deny_calibration
        self.robot.bus.connect()  # bus handshake reads only
        return {"port": FOLLOWER, "robot_id": "my_follower_arm", "read_only_connect": True}

    def calibration_matches(self):
        return self.robot.is_calibrated

    def limits(self):
        return {motor: (self.robot.bus.calibration[motor].range_min,
                        self.robot.bus.calibration[motor].range_max) for motor in MOTORS}

    def read_raw(self):
        return {register: self.robot.bus.sync_read(register, normalize=False, num_retry=0)
                for register in REGISTERS}

    def write_goals_once(self, target):
        # Same LeRobot broadcast write path as record_episode.start_hardware().
        # Raw Present values require normalize=False to preserve exact register values.
        self.robot.bus.sync_write("Goal_Position", target, normalize=False, num_retry=0)

    def close_without_writes(self):
        if self.robot is not None and self.robot.bus.is_connected:
            self.robot.bus.disconnect(disable_torque=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "hardware"), default="synthetic")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hardware-goal-broadcast-approved", action="store_true")
    parser.add_argument("--ack-arm-supported", action="store_true")
    parser.add_argument("--ack-clear-workspace", action="store_true")
    parser.add_argument("--ack-power-stop-ready", action="store_true")
    args = parser.parse_args()
    if args.mode == "hardware":
        if not all((args.hardware_goal_broadcast_approved, args.ack_arm_supported,
                    args.ack_clear_workspace, args.ack_power_stop_ready)):
            parser.error("Hardware trial requires separate approval and all physical safety acknowledgments")
        typed = input(f"Type {APPROVAL_PHRASE} only after supporting the arm: ")
        if typed.strip() != APPROVAL_PHRASE:
            raise RuntimeError("Hardware Goal broadcast not armed")
    args.output.mkdir(parents=True, exist_ok=False)
    io = HardwareIO() if args.mode == "hardware" else SyntheticIO()
    with (args.output / "events.jsonl").open("x") as stream:
        def emit(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        result = probe(io, emit)
        emit({"event": "result", **result})
    (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["completed"]:
        raise RuntimeError(result["fault"])


if __name__ == "__main__":
    main()
