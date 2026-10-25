"""One Goal write / one axis probe; a Goal write MAY ENABLE TORQUE.

Synthetic is the default. Hardware requires a separately approved, supported,
interactive trial. No torque register writes, no calibration/configure, no ACT,
no retry or cleanup Goal. The servo can remain powered after this exits.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from first_motion_test import HardwareIO
from sync_goal_to_present import MOTORS, SyntheticGoalIO


REGISTER_NAMES = ("Torque_Enable", "Status", "Operating_Mode",
                  "Angular_Resolution", "Present_Position", "Goal_Position")
MAX_MOVEMENT_RAW = 5
OBSERVE_AFTER_WRITE_S = (0.0, 0.02, 0.10, 0.50)


class HardwareProbeIO:
    def __init__(self, motor):
        self.hardware = HardwareIO(motor + ".pos")
        self.bus = None

    def open(self):
        metadata = self.hardware.open()  # bus.connect only; no Robot.configure
        self.bus = self.hardware.robot.bus
        return {"port": metadata["port"], "motor": self.hardware.joint,
                "calibration_matches": True}

    def limits(self):
        return {name: (self.bus.calibration[name].range_min,
                       self.bus.calibration[name].range_max) for name in MOTORS}

    def read_all(self):
        return {reg: self.bus.sync_read(reg, normalize=False, num_retry=0)
                for reg in REGISTER_NAMES}

    def write_goal_once(self, motor, raw):
        # Only Goal_Position. A status reply is expected; no retries.
        self.bus.write("Goal_Position", motor, raw, normalize=False, num_retry=0)

    def close(self):
        self.hardware.close()  # disable_torque=False; no further register writes


class SyntheticProbeIO(SyntheticGoalIO):
    def __init__(self, motor, *, torque_after_write=True, move_after_write=0,
                 write_fails=False, status_after_write=0):
        super().__init__()
        self.motor = motor
        self.torque_after_write = torque_after_write
        self.move_after_write = move_after_write
        self.write_fails = write_fails
        self.status_after_write = status_after_write

    def write_goal_once(self, motor, raw):
        if self.writes:
            raise RuntimeError("Second Goal write forbidden")
        self.writes.append((motor, raw))
        if self.write_fails:
            raise ConnectionError("Injected ambiguous Goal write timeout")
        self.goal[motor] = raw
        if self.torque_after_write:
            self.torque[motor] = 1
        self.present[motor] += self.move_after_write
        self.status[motor] = self.status_after_write


def check_snapshot(snap, limits, baseline=None):
    for register in REGISTER_NAMES:
        if set(snap[register]) != set(MOTORS):
            raise RuntimeError(f"Incomplete {register} read")
    for motor in MOTORS:
        if snap["Status"][motor] != 0:
            raise RuntimeError(f"{motor}: nonzero Status")
        if snap["Operating_Mode"][motor] != 0 or snap["Angular_Resolution"][motor] != 1:
            raise RuntimeError(f"{motor}: unexpected operating mode or resolution")
        present = snap["Present_Position"][motor]
        low, high = limits[motor]
        if not low <= present <= high:
            raise RuntimeError(f"{motor}: Present outside calibrated range")
        if baseline is not None and abs(present-baseline["Present_Position"][motor]) > MAX_MOVEMENT_RAW:
            raise RuntimeError(f"{motor}: movement >{MAX_MOVEMENT_RAW} raw ticks")


def run_probe(io, motor, emit, *, clock=time.perf_counter, sleep=time.sleep):
    """Perform at most one Goal write; all later operations are reads."""
    opened = False
    write_attempted = False
    baseline = None
    started = clock()
    try:
        metadata = io.open()
        opened = True
        limits = io.limits()
        if set(limits) != set(MOTORS):
            raise RuntimeError("Incomplete calibration")
        baseline = io.read_all()
        check_snapshot(baseline, limits)
        if any(baseline["Torque_Enable"][name] != 0 for name in MOTORS):
            raise RuntimeError("Some torque is already enabled; no Goal write")
        emit({"event": "preflight", "t_ms": (clock()-started)*1000,
              "snapshot": baseline, "limits": limits, "metadata": metadata})
        sleep(0.20)
        fresh_start = clock()
        fresh = io.read_all()
        fresh_ms = (clock()-fresh_start)*1000
        check_snapshot(fresh, limits, baseline)
        if fresh_ms > 100 or any(fresh["Torque_Enable"][name] != 0 for name in MOTORS):
            raise RuntimeError("Stale state or torque enabled before write")
        if fresh["Goal_Position"] != baseline["Goal_Position"]:
            raise RuntimeError("Goal changed during no-write observation")
        target = int(fresh["Present_Position"][motor])
        emit({"event": "immediately_before_write", "t_ms": (clock()-started)*1000,
              "snapshot": fresh, "read_ms": fresh_ms, "motor": motor, "target_raw": target})
        write_attempted = True  # Ambiguous timeout must never lead to a retry.
        t0 = clock()
        io.write_goal_once(motor, target)
        emit({"event": "write_returned", "t_ms": (clock()-started)*1000,
              "motor": motor, "target_raw": target, "write_ms": (clock()-t0)*1000})
        after = []
        write_end = clock()
        for offset in OBSERVE_AFTER_WRITE_S:
            sleep(max(0, write_end + offset - clock()))
            t0 = clock()
            snap = io.read_all()
            read_ms = (clock()-t0)*1000
            emit({"event": "readback", "t_ms": (clock()-started)*1000,
                  "offset_s": offset, "read_ms": read_ms, "snapshot": snap})
            check_snapshot(snap, limits, baseline)
            if snap["Goal_Position"][motor] != target:
                raise RuntimeError(f"{motor}: Goal readback mismatch")
            if abs(snap["Goal_Position"][motor]-snap["Present_Position"][motor]) > MAX_MOVEMENT_RAW:
                raise RuntimeError(f"{motor}: Goal/Present mismatch")
            if any(snap["Torque_Enable"][name] != 0 for name in MOTORS if name != motor):
                raise RuntimeError("Another axis unexpectedly enabled torque")
            after.append(snap)
        return {"completed": True, "write_attempted": write_attempted,
                "motor": motor, "target_raw": target,
                "torque_after": after[-1]["Torque_Enable"],
                "position_change_raw": {name: after[-1]["Present_Position"][name]-baseline["Present_Position"][name]
                                        for name in MOTORS},
                "fault": None,
                "warning": "Torque may remain ON. Support arm before physically cutting servo power."}
    except BaseException as exc:
        return {"completed": False, "write_attempted": write_attempted,
                "motor": motor, "fault": f"{type(exc).__name__}: {exc}",
                "warning": "No retry or cleanup write. Torque may remain ON; support arm and use physical power stop."}
    finally:
        if opened or isinstance(io, HardwareProbeIO):
            io.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "hardware"), default="synthetic")
    parser.add_argument("--motor", choices=MOTORS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-torque-risk-approved", action="store_true")
    parser.add_argument("--ack-arm-supported", action="store_true")
    parser.add_argument("--ack-clear-workspace", action="store_true")
    parser.add_argument("--ack-power-stop-ready", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "hardware":
        if not all((args.hardware_torque_risk_approved, args.ack_arm_supported,
                    args.ack_clear_workspace, args.ack_power_stop_ready)):
            raise RuntimeError("One-axis hardware Goal write requires separate torque-risk approval")
        if not sys.stdin.isatty():
            raise RuntimeError("Hardware trial requires an interactive terminal")
        if input("Type ONE AXIS TORQUE PROBE after supporting arm: ").strip().upper() != "ONE AXIS TORQUE PROBE":
            raise RuntimeError("Operator did not arm the trial")
        io = HardwareProbeIO(args.motor)
    else:
        io = SyntheticProbeIO(args.motor)
    args.output.mkdir(parents=True)
    with (args.output / "events.jsonl").open("w") as log:
        def emit(event):
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()
            os.fsync(log.fileno())

        result = run_probe(io, args.motor, emit)
        emit({"event": "result", **result})
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["completed"]:
        raise RuntimeError(result["fault"])


if __name__ == "__main__":
    main()
