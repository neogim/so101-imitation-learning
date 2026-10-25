"""Prepared, independent SO-101 micro-motion test; synthetic is the default.

Hardware mode is NOT part of ACT autonomous control. It is never run by tests.
It requires pre-enabled torque, a supported arm, a clear workspace and separate
operator authorization. It never calibrates, configures, enables/disables torque,
or retries a Goal_Position write. No goal is sent during cleanup or a fault.
"""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from act_action_guard import ActionGuard, GuardConfig, NAMES
from communication_guard import GuardedIO


BODY = NAMES[:5]


class SyntheticIO:
    def __init__(self, fail_read_at=None, fail_write_at=None, delay_write_ms=0, torque=True):
        self.position = dict(zip(NAMES, [-60., -80., 80., 45., -90., 20.]))
        self.last_goal = self.position.copy()
        self.fail_read_at, self.fail_write_at = fail_read_at, fail_write_at
        self.delay_write_ms, self.torque = delay_write_ms, torque
        self.reads = self.writes = 0
        self.joint = "wrist_flex.pos"

    def open(self):
        return {}

    def state(self):
        self.reads += 1
        if self.reads == self.fail_read_at:
            raise TimeoutError("Injected Present_Position timeout")
        return self.position.copy()

    def torque_ok(self):
        return self.torque

    def goal(self):
        return self.last_goal.copy()

    def limits(self):
        return ((-180., 180.),) * 5 + ((0., 100.),)

    def send(self, values):
        self.writes += 1
        time.sleep(self.delay_write_ms / 1000)
        if self.writes == self.fail_write_at:
            raise TimeoutError("Injected Goal_Position write timeout")
        self.last_goal[self.joint] = float(values[self.joint])
        self.position[self.joint] = self.last_goal[self.joint]  # ideal synthetic plant only
        return values.copy()

    def close(self):
        pass


class HardwareIO:
    def __init__(self, joint):
        self.joint = joint
        self.robot = None

    def open(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from record_episode import FOLLOWER, calibration_paths, deny_calibration
        if not Path(FOLLOWER).exists() or not calibration_paths()[0].is_file():
            raise FileNotFoundError("Follower port or calibration missing")
        self.robot = SO101Follower(SO101FollowerConfig(
            port=FOLLOWER, id="my_follower_arm", use_degrees=True,
            disable_torque_on_disconnect=False,
        ))
        self.robot.calibrate = deny_calibration
        self.robot._save_calibration = deny_calibration
        self.robot.bus.write_calibration = deny_calibration
        # Bus.connect() only; Robot.connect() would call configure() and write registers.
        self.robot.bus.connect()
        if not self.robot.is_calibrated:
            raise RuntimeError("Calibration mismatch; no Action allowed")
        return {"port": FOLLOWER, "id": "my_follower_arm", "use_degrees": True,
                "torque_changes": False, "write_joint": self.joint}

    def state(self):
        values = self.robot.bus.sync_read("Present_Position", num_retry=0)
        return {name: float(values[name.removesuffix(".pos")]) for name in NAMES}

    def torque_ok(self):
        values = self.robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=0)
        return len(values) == 6 and all(value == 1 for value in values.values())

    def goal(self):
        values = self.robot.bus.sync_read("Goal_Position", num_retry=0)
        return {name: float(values[name.removesuffix(".pos")]) for name in NAMES}

    def limits(self):
        from capture_session import calibration_limits
        value = calibration_limits(self.robot)
        value["gripper.pos"] = (max(0., value["gripper.pos"][0]),
                                min(100., value["gripper.pos"][1]))
        return tuple(tuple(value[n]) for n in NAMES)

    def communication_settings(self, register, role):
        port = self.robot.bus.port_handler
        return {"register": register, "role": role,
                "baudrate": port.getBaudRate(),
                "pyserial_timeout_s": port.ser.timeout,
                "packet_timeout_snapshot_ms": port.packet_timeout,
                "effective_num_read_retries": 0,
                "goal_write_retries": 0,
                "sync_write_has_status_ack": False}

    def send(self, values):
        # Broadcast sync-write has no servo acknowledgment. Never retry on failure.
        name = self.joint.removesuffix(".pos")
        self.robot.bus.sync_write("Goal_Position", {name: values[self.joint]}, num_retry=0)
        return values.copy()

    def close(self):
        if self.robot is not None and self.robot.bus.is_connected:
            self.robot.bus.disconnect(disable_torque=False)


def run_micro(io, *, joint, delta_deg, cycles=40, clock=time.perf_counter,
              sleep=time.sleep, keypoll=lambda: ""):
    """Run after external authorization; return diagnostics, never send on fault cleanup."""
    if joint not in BODY or not math.isfinite(delta_deg) or not 0 < abs(delta_deg) <= 1:
        raise ValueError("Choose one rotary joint and a signed delta of at most 1 degree")
    if not 2 <= cycles <= 100:
        raise ValueError("Invalid cycle count")
    guarded = GuardedIO(io, clock=clock)
    rows = []
    try:
        initial = guarded.state()
        if not guarded.torque_ok():
            raise RuntimeError("All six axes must already have torque enabled; no write")
        goal = io.goal()  # read only; a failure aborts before any Action
        if set(initial) != set(NAMES) or set(goal) != set(NAMES):
            raise ValueError("Unexpected joint names")
        if max(abs(goal[n] - initial[n]) for n in NAMES) > 1:
            raise RuntimeError("Existing Goal_Position differs from measured position >1; no write")
        limits = io.limits()
        selected = NAMES.index(joint)
        if not limits[selected][0] <= initial[joint] + delta_deg <= limits[selected][1]:
            raise RuntimeError("Requested small motion exceeds calibrated joint range")
        guard = ActionGuard(GuardConfig(limits=limits, speed_per_s=(30,) * 6,
                                          gripper_closing_lead=None, tracking_error_limit=8))
        guard.reset_for_new_episode([initial[n] for n in NAMES])
        # Monitor half a second with no writes. Position drift prevents arming.
        for _ in range(10):
            sample = guarded.state()
            if not guarded.torque_ok() or max(abs(sample[n]-initial[n]) for n in NAMES) > .5:
                raise RuntimeError("Torque missing or pose drift before first Action")
            sleep(.05)
        start = clock()
        previous_tick = None
        for cycle in range(cycles):
            sleep(max(0, start + cycle*.05 - clock()))
            tick = clock()
            if previous_tick is not None and tick-previous_tick > .075:
                raise RuntimeError("Control interval >75ms; no further Action")
            previous_tick = tick
            if keypoll() in ("q", "\x03"):
                raise RuntimeError("Operator stop requested; no new Action")
            guarded.context = {"mode": "FIRST_MOTION", "cycle": cycle,
                               "cycle_start_monotonic": tick}
            t0 = clock()
            state = guarded.state()
            t1 = clock()
            if not guarded.torque_ok():
                raise RuntimeError("Torque lost; no further Action")
            t2 = clock()
            # Hold the small displacement for first half, then return toward baseline.
            requested = np.asarray([initial[n] for n in NAMES], dtype=float)
            if cycle < cycles // 2:
                requested[selected] += delta_deg
            limited, reasons = guard.propose(requested, [state[n] for n in NAMES],
                                             sample_age_s=clock()-t1)
            t3 = clock()
            if limited is None:
                raise RuntimeError("Guard latched: " + ";".join(reasons))
            actual_goal = dict(zip(NAMES, limited.tolist()))
            guarded.send(actual_goal)
            t4 = clock()
            work_ms = (t4-tick)*1000
            rows.append({"cycle": cycle, "state_read_ms": (t1-t0)*1000,
                         "torque_read_ms": (t2-t1)*1000, "guard_ms": (t3-t2)*1000,
                         "goal_write_ms": (t4-t3)*1000, "full_cycle_ms": work_ms,
                         "requested": requested.tolist(), "limited": limited.tolist(),
                         "measured": [state[n] for n in NAMES], "reasons": reasons})
            guarded.cycles.append({"cycle": cycle, "work_s": work_ms/1000, "ok": True})
            if work_ms > 50:
                raise RuntimeError("Cycle work >50ms after write; no further Action")
            if cycle == cycles // 2 - 1 and abs(state[joint] - initial[joint]) < .15:
                raise RuntimeError("Selected joint has not measurably followed; no further Action")
        return {"completed": True, "fault": None, "cycles": rows,
                "communication": guarded.report()}
    except BaseException as exc:
        return {"completed": False, "fault": f"{type(exc).__name__}: {exc}",
                "cycles": rows, "communication": guarded.report()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("synthetic", "hardware"), default="synthetic")
    p.add_argument("--joint", choices=BODY, required=True)
    p.add_argument("--delta-deg", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hardware-write-approved", action="store_true")
    p.add_argument("--ack-arm-supported", action="store_true")
    p.add_argument("--ack-clear-workspace", action="store_true")
    p.add_argument("--ack-stop-ready", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.mode == "hardware":
        if not all((a.hardware_write_approved, a.ack_arm_supported,
                    a.ack_clear_workspace, a.ack_stop_ready)):
            raise RuntimeError("Hardware writes require explicit approval and all safety acknowledgments")
        if not sys.stdin.isatty():
            raise RuntimeError("Hardware writes require an interactive terminal")
        phrase = input("Type SO101 SMALL MOTION to arm this independent test: ")
        if phrase != "SO101 SMALL MOTION":
            raise RuntimeError("Operator did not arm the test")
        io = HardwareIO(a.joint)
    else:
        io = SyntheticIO()
        io.joint = a.joint
    a.output.mkdir(parents=True)
    metadata = {}
    try:
        metadata = io.open()
        if a.mode == "hardware":
            from capture_session import Keyboard
            with Keyboard() as keyboard:
                result = run_micro(io, joint=a.joint, delta_deg=a.delta_deg, keypoll=keyboard.poll)
        else:
            result = run_micro(io, joint=a.joint, delta_deg=a.delta_deg)
    finally:
        io.close()  # Never torque off or send a new goal at exit.
    result["mode"] = a.mode
    result["metadata"] = metadata
    result["warning"] = ("No automatic return on abnormal stop. Servo may continue toward/hold last goal. "
                         "Physical stopping and holding are not guaranteed. Support the arm.")
    (a.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cycles", "communication")}, indent=2))
    if not result["completed"]:
        raise RuntimeError(result["fault"])


if __name__ == "__main__":
    main()
