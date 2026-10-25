"""Synthetic replay of raw Goal_Position <- Present_Position.

Hardware mode is suspended after the 2026-10-20 trial: writing Goal_Position
caused shoulder_pan Torque_Enable to become 1. This script must not be used for
further hardware writes until a different safety procedure is reviewed.
"""

import argparse
import json
import os
from pathlib import Path
import time

from act_action_guard import NAMES
from first_motion_test import HardwareIO


MOTORS = tuple(name.removesuffix(".pos") for name in NAMES)
MAX_POSITION_CHANGE_RAW = 5
MAX_GOAL_ERROR_RAW = 5
MAX_SAMPLE_AGE_S = 0.10
ARM_PHRASE = "SYNC GOAL ONLY"


def valid_arm_phrase(value):
    return value.strip().upper() == ARM_PHRASE


class HardwareGoalIO:
    def __init__(self):
        self.hardware = HardwareIO("wrist_roll.pos")
        self.bus = None

    def open(self):
        raise RuntimeError("Hardware Goal sync suspended after observed torque activation")

    def limits(self):
        return {motor: (self.bus.calibration[motor].range_min,
                        self.bus.calibration[motor].range_max) for motor in MOTORS}

    def read_all(self):
        return {register: self.bus.sync_read(register, normalize=False, num_retry=0)
                for register in ("Torque_Enable", "Status", "Operating_Mode",
                                 "Angular_Resolution", "Present_Position", "Goal_Position")}

    def read_one(self, motor):
        return {register: self.bus.read(register, motor, normalize=False, num_retry=0)
                for register in ("Torque_Enable", "Status", "Present_Position", "Goal_Position")}

    def write_goal(self, motor, raw_value):
        # Single-servo write expects a status packet; never retry an ambiguous write.
        self.bus.write("Goal_Position", motor, raw_value, normalize=False, num_retry=0)

    def close(self):
        self.hardware.close()  # disable_torque=False; never changes holding state


class SyntheticGoalIO:
    def __init__(self, *, fail_write_on=None, torque_change_on=None, position_change_on=None,
                 fail_read_on=None):
        self.present = dict(zip(MOTORS, (1481, 1633, 2894, 2087, 833, 2161)))
        self.goal = dict.fromkeys(MOTORS, 0)
        self.torque = dict.fromkeys(MOTORS, 0)
        self.status = dict.fromkeys(MOTORS, 0)
        self.bounds = dict(zip(MOTORS, ((733, 3468), (776, 3170), (907, 3122),
                                        (863, 3250), (0, 4095), (2014, 2994))))
        self.fail_write_on = fail_write_on
        self.torque_change_on = torque_change_on
        self.position_change_on = position_change_on
        self.fail_read_on = fail_read_on
        self.writes = []

    def open(self):
        return {"synthetic": True}

    def limits(self):
        return self.bounds.copy()

    def read_all(self):
        return {"Torque_Enable": self.torque.copy(), "Status": self.status.copy(),
                "Operating_Mode": dict.fromkeys(MOTORS, 0),
                "Angular_Resolution": dict.fromkeys(MOTORS, 1),
                "Present_Position": self.present.copy(), "Goal_Position": self.goal.copy()}

    def read_one(self, motor):
        if motor == self.fail_read_on:
            raise ConnectionError("Injected read timeout")
        return {"Torque_Enable": self.torque[motor], "Status": self.status[motor],
                "Present_Position": self.present[motor], "Goal_Position": self.goal[motor]}

    def write_goal(self, motor, raw_value):
        self.writes.append((motor, raw_value))
        if motor == self.fail_write_on:
            raise ConnectionError("Injected ambiguous write timeout")
        self.goal[motor] = raw_value
        if motor == self.torque_change_on:
            self.torque[motor] = 1
        if motor == self.position_change_on:
            self.present[motor] += 20

    def close(self):
        pass


def _validate_all(snapshot, bounds, *, baseline=None):
    for register in ("Torque_Enable", "Status", "Operating_Mode", "Angular_Resolution",
                     "Present_Position", "Goal_Position"):
        if set(snapshot[register]) != set(MOTORS):
            raise RuntimeError(f"Incomplete {register} read")
    for motor in MOTORS:
        if snapshot["Torque_Enable"][motor] != 0 or snapshot["Status"][motor] != 0:
            raise RuntimeError(f"{motor}: torque or status not zero")
        if snapshot["Operating_Mode"][motor] != 0 or snapshot["Angular_Resolution"][motor] != 1:
            raise RuntimeError(f"{motor}: unexpected position mode or angular resolution")
        present = snapshot["Present_Position"][motor]
        low, high = bounds[motor]
        if not low <= present <= high:
            raise RuntimeError(f"{motor}: present raw {present} outside calibrated [{low}, {high}]")
        if baseline is not None:
            if abs(present - baseline["Present_Position"][motor]) > MAX_POSITION_CHANGE_RAW:
                raise RuntimeError(f"{motor}: unexpected position change")
            if abs(snapshot["Goal_Position"][motor] - present) > MAX_GOAL_ERROR_RAW:
                raise RuntimeError(f"{motor}: Goal/Present readback mismatch")


def synchronize(io, emit, *, clock=time.perf_counter):
    """Fail closed on first anomaly; never retry or restore an old Goal."""
    opened = False
    writes = []
    start = clock()
    try:
        metadata = io.open()
        opened = True
        bounds = io.limits()
        if set(bounds) != set(MOTORS):
            raise RuntimeError("Incomplete calibration limits")
        t0 = clock()
        before = io.read_all()
        read_ms = (clock() - t0) * 1000
        _validate_all(before, bounds)
        emit({"event": "preflight", "elapsed_ms": (clock()-start)*1000,
              "read_ms": read_ms, "snapshot": before, "limits": bounds,
              "metadata": metadata})
        written_targets = {}
        for motor in MOTORS:
            t0 = clock()
            before_each = io.read_all()
            _validate_all(before_each, bounds)
            emit({"event": "all_axes_before_write", "motor": motor,
                  "snapshot": before_each, "read_ms": (clock()-t0)*1000,
                  "elapsed_ms": (clock()-start)*1000})
            for name in MOTORS:
                if abs(before_each["Present_Position"][name]-before["Present_Position"][name]) > MAX_POSITION_CHANGE_RAW:
                    raise RuntimeError(f"{name}: unexpected movement before Goal write")
            for name, expected in written_targets.items():
                if before_each["Goal_Position"][name] != expected:
                    raise RuntimeError(f"{name}: Goal changed before next write")
            for name in MOTORS:
                if name not in written_targets and before_each["Goal_Position"][name] != before["Goal_Position"][name]:
                    raise RuntimeError(f"{name}: unprocessed Goal changed outside this script")
            t0 = clock()
            fresh = io.read_one(motor)
            fresh_ms = (clock()-t0)*1000
            if clock()-t0 > MAX_SAMPLE_AGE_S:
                raise RuntimeError(f"{motor}: stale pre-write sample")
            low, high = bounds[motor]
            if fresh["Torque_Enable"] != 0 or fresh["Status"] != 0:
                raise RuntimeError(f"{motor}: torque/status changed before write")
            if not low <= fresh["Present_Position"] <= high:
                raise RuntimeError(f"{motor}: present outside calibration before write")
            if (abs(fresh["Present_Position"]-before["Present_Position"][motor])
                    > MAX_POSITION_CHANGE_RAW):
                raise RuntimeError(f"{motor}: unexpected movement before write")
            if fresh["Goal_Position"] != before["Goal_Position"][motor]:
                raise RuntimeError(f"{motor}: Goal changed outside this script")
            target = int(fresh["Present_Position"])
            emit({"event": "before_write", "motor": motor, "fresh": fresh,
                  "fresh_read_ms": fresh_ms, "target_raw": target,
                  "elapsed_ms": (clock()-start)*1000})
            t0 = clock()
            io.write_goal(motor, target)
            write_ms = (clock()-t0)*1000
            writes.append(motor)
            written_targets[motor] = target
            emit({"event": "write_returned", "motor": motor, "target_raw": target,
                  "write_ms": write_ms, "elapsed_ms": (clock()-start)*1000})
            t0 = clock()
            after_one = io.read_one(motor)
            verify_ms = (clock()-t0)*1000
            emit({"event": "after_write", "motor": motor, "readback": after_one,
                  "read_ms": verify_ms, "elapsed_ms": (clock()-start)*1000})
            if after_one["Torque_Enable"] != 0 or after_one["Status"] != 0:
                raise RuntimeError(f"{motor}: torque/status changed after Goal write")
            if after_one["Goal_Position"] != target:
                raise RuntimeError(f"{motor}: Goal readback differs from written raw value")
            if abs(after_one["Present_Position"]-target) > MAX_GOAL_ERROR_RAW:
                raise RuntimeError(f"{motor}: Goal/Present changed beyond {MAX_GOAL_ERROR_RAW} raw ticks")
            # A different servo changing state is also a stop condition.
            t0 = clock()
            after_all = io.read_all()
            emit({"event": "all_axes_after_write", "motor": motor, "snapshot": after_all,
                  "read_ms": (clock()-t0)*1000, "elapsed_ms": (clock()-start)*1000})
            _validate_all(after_all, bounds)
            for name in MOTORS:
                if abs(after_all["Present_Position"][name]-before["Present_Position"][name]) > MAX_POSITION_CHANGE_RAW:
                    raise RuntimeError(f"{name}: unexpected movement during Goal sync")
            for name, expected in written_targets.items():
                if after_all["Goal_Position"][name] != expected:
                    raise RuntimeError(f"{name}: Goal changed after verified write")
        t0 = clock()
        final = io.read_all()
        final_ms = (clock()-t0)*1000
        emit({"event": "final_readback", "snapshot": final,
              "read_ms": final_ms, "elapsed_ms": (clock()-start)*1000})
        _validate_all(final, bounds, baseline=before)
        for name, expected in written_targets.items():
            if final["Goal_Position"][name] != expected:
                raise RuntimeError(f"{name}: Goal changed before final readback")
        return {"completed": True, "writes": writes, "fault": None,
                "pre_goal_raw": before["Goal_Position"],
                "pre_present_raw": before["Present_Position"],
                "post_goal_raw": final["Goal_Position"],
                "post_present_raw": final["Present_Position"],
                "position_change_raw": {name: final["Present_Position"][name]-before["Present_Position"][name]
                                        for name in MOTORS},
                "elapsed_ms": (clock()-start)*1000}
    except BaseException as exc:
        return {"completed": False, "writes": writes,
                "fault": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": (clock()-start)*1000}
    finally:
        if opened or isinstance(io, HardwareGoalIO):
            try:
                io.close()
            except BaseException as exc:
                emit({"event": "disconnect_fault", "fault": f"{type(exc).__name__}: {exc}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "hardware"), default="synthetic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-goal-sync-approved", action="store_true")
    parser.add_argument("--ack-arm-supported", action="store_true")
    parser.add_argument("--ack-clear-workspace", action="store_true")
    parser.add_argument("--ack-power-stop-ready", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "hardware":
        # Observed on 2026-10-20: a Goal_Position write to shoulder_pan changed
        # Torque_Enable from 0 to 1 shortly afterward. The original safety
        # assumption is false for this device. Keep hardware mode disabled
        # until a separately reviewed procedure is implemented.
        raise RuntimeError("Hardware Goal sync suspended: Goal write enabled shoulder_pan torque in the first trial")
    io = SyntheticGoalIO()
    args.output.mkdir(parents=True)
    with (args.output / "events.jsonl").open("w") as log:
        def emit(event):
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()
            os.fsync(log.fileno())

        result = synchronize(io, emit)
        emit({"event": "result", **result})
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["completed"]:
        raise RuntimeError(result["fault"])


if __name__ == "__main__":
    main()
