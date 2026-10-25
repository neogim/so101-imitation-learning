"""No-hardware 20 Hz ACT cycle with injected serial write delay/failure."""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from act_action_guard import ActionGuard, GuardConfig, NAMES
from observe_act_read_only import SyntheticReader, load_policy, read_only_cycle


class DelayedSyntheticReader(SyntheticReader):
    def read(self):
        t0 = time.perf_counter()
        time.sleep(.002)  # representative position-read cost, not measured hardware latency
        state = self.state.copy()
        t1 = time.perf_counter()
        time.sleep(.001)  # representative torque-read cost
        t2 = time.perf_counter()
        image = self.camera.read_latest()
        t3 = time.perf_counter()
        return state, image, True, (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, 0.0


class DelayedSyntheticWriter:
    def __init__(self, delay_ms, fail_at=None):
        self.delay_ms, self.fail_at = delay_ms, fail_at
        self.attempts = 0
        self.successful = 0
        self.latched = False

    def send(self, goal, reader):
        if self.latched:
            raise RuntimeError("Simulated communication fault latched: no new write")
        self.attempts += 1
        time.sleep(self.delay_ms / 1000)
        if self.attempts == self.fail_at:
            self.latched = True
            raise TimeoutError("Injected Goal_Position write timeout; no retry")
        # Ideal follower only for timing and guard-path testing.
        reader.state = np.asarray(goal, dtype=np.float32)
        self.successful += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "mps"), default="mps")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--write-delay-ms", type=float, default=2)
    p.add_argument("--fail-write-at", type=int)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if not 1 <= a.frames <= 1000 or not math.isfinite(a.write_delay_ms) or a.write_delay_ms < 0:
        raise ValueError("Invalid simulation parameters")
    policy, pre, post = load_policy(a.device)
    policy.config.n_action_steps = 1
    policy.reset()
    reader = DelayedSyntheticReader([-60, -80, 80, 45, -90, 20])
    limits = ((-180, 180),) * 5 + ((0, 100),)
    guard = ActionGuard(GuardConfig(limits=limits, speed_per_s=(30,) * 6,
                                    gripper_closing_lead=None))
    writer = DelayedSyntheticWriter(a.write_delay_ms, a.fail_write_at)
    a.output.mkdir(parents=True)
    rows, fault = [], None
    warmup_ms = []
    with torch.inference_mode():
        for _ in range(3):
            warmup_start = time.perf_counter()
            read_only_cycle(reader, policy, pre, post, guard, a.device, -1)
            warmup_ms.append((time.perf_counter()-warmup_start)*1000)
            policy.reset()
            guard.reset_for_new_episode(reader.state)
    start = time.perf_counter()
    with torch.inference_mode():
        for index in range(a.frames):
            time.sleep(max(0, start + index * .05 - time.perf_counter()))
            tick = time.perf_counter()
            try:
                state, _, _, _, predicted, limited, reasons, refreshed, timing = read_only_cycle(
                    reader, policy, pre, post, guard, a.device, index)
                if limited is None:
                    fault = "guard_latched:" + ";".join(reasons)
                    break
                write_start = time.perf_counter()
                writer.send(limited, reader)
                write_ms = (time.perf_counter()-write_start)*1000
                work_ms = (time.perf_counter()-tick)*1000
                rows.append(dict(frame_index=index, model_ms=timing["model_ms"],
                                 read_ms=timing["state_read_ms"]+timing["torque_read_ms"],
                                 camera_ms=timing["camera_read_ms"],
                                 resize_prepare_ms=timing["resize_ms"]+timing["prepare_ms"],
                                 guard_ms=timing["guard_ms"], write_ms=write_ms,
                                 full_cycle_ms=work_ms, prediction_gripper=float(predicted[5]),
                                 limited_gripper=float(limited[5]),
                                 reasons=";".join(reasons), chunk_refreshed=int(refreshed)))
                if work_ms > 50:
                    fault = f"cycle_overrun_after_write:{work_ms:.3f}ms; no further writes"
                    writer.latched = True
                    break
            except BaseException as exc:
                writer.latched = True
                guard.stop(f"write_or_read_fault:{type(exc).__name__}")
                fault = f"{type(exc).__name__}: {exc}; no retry or further write"
                break
    with (a.output / "cycles.csv").open("w", newline="") as f:
        fields = ["frame_index", "model_ms", "read_ms", "camera_ms", "resize_prepare_ms",
                  "guard_ms", "write_ms", "full_cycle_ms", "prediction_gripper", "limited_gripper",
                  "reasons", "chunk_refreshed"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    result = {"simulation_only": True, "frames_requested": a.frames,
              "frames_with_successful_fake_write": writer.successful,
              "fake_write_attempts": writer.attempts,
              "injected_write_delay_ms": a.write_delay_ms, "fail_write_at": a.fail_write_at,
              "fault": fault, "no_auto_retry": True, "n_action_steps": 1,
              "no_write_warmup_ms": warmup_ms,
              "max_successful_cycle_ms": max((r["full_cycle_ms"] for r in rows), default=None),
              "max_fake_write_ms": max((r["write_ms"] for r in rows), default=None),
              "warning": "Synthetic read/write timing is not a measurement of real Goal_Position transmission or motor response."}
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
