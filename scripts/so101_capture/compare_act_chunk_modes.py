"""Compare already recorded ACT candidate streams; never connects to hardware."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from act_action_guard import ActionGuard, GuardConfig, NAMES, UNITS


def describe(values):
    return {name: {"max": float(np.max(values[:, i])),
                   "p95": float(np.percentile(values[:, i], 95)), "unit": UNITS[i]}
            for i, name in enumerate(NAMES)} if len(values) else {}


def analyze(path):
    summary = json.loads((path / "summary.json").read_text())
    hardware = summary.get("source") == "hardware"
    rows_path = path / ("observations.csv" if hardware else "actions.csv")
    with rows_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in {rows_path}")
    n = int(summary.get("n_action_steps", 50))  # Earlier fixed-50 probe predates this metadata.
    if n not in (1, 5, 10, 50):
        raise ValueError(f"Unexpected action interval {n}")
    if hardware:
        limits = summary["limits"]
        state_prefix = "state"
        time_key = "total_work_ms"
        limit_prefix = "hypothetical_limited"
    else:
        limits_doc = json.loads(Path(summary["limits"]).read_text())
        limits = limits_doc["joint_limits"]
        state_prefix = "measured"
        time_key = "cycle_work_ms"
        limit_prefix = "limited"
    config = GuardConfig(limits=tuple(tuple(limits[k]) for k in NAMES),
                         speed_per_s=(30,) * 6, period_s=.05)
    ideal_guard = ActionGuard(config)
    raw_boundary, ideal_boundary, raw_all = [], [], []
    reason_count = Counter()
    original_reason_count = Counter()
    inference_ms, work_ms, elapsed_ms = [], [], []
    prev_ep, prev_raw, prev_ideal, ideal_state = None, None, None, None
    original_latched = 0
    for row in rows:
        ep = 0 if hardware else int(row["episode_index"])
        idx = int(row["frame_index"])
        raw = np.asarray([float(row[f"predicted.{key}"]) for key in NAMES])
        measured = np.asarray([float(row[f"{state_prefix}.{key}"]) for key in NAMES])
        if ep != prev_ep:
            ideal_guard.reset_for_new_episode(measured)
            ideal_state = measured.copy()
            prev_raw = prev_ideal = None
            prev_ep = ep
        limited, reasons = ideal_guard.propose(raw, ideal_state)
        if limited is None:
            raise RuntimeError(f"Ideal-follower guard unexpectedly latched: {ep}/{idx}: {reasons}")
        ideal_state = limited.copy()
        reason_count.update(reasons)
        original_latched += int(row["hypothetical_guard_latched" if hardware else "guard_latched"])
        original_reason_count.update(filter(None, row["reasons"].split(";")))
        refreshed = bool(int(row["chunk_refreshed"]))
        if refreshed != (idx % n == 0):
            raise ValueError(f"Chunk refresh mismatch {path}: episode={ep} frame={idx}")
        if prev_raw is not None:
            delta = np.abs(raw - prev_raw)
            raw_all.append(delta)
            if refreshed:
                raw_boundary.append(delta)
                ideal_boundary.append(np.abs(limited - prev_ideal))
        prev_raw, prev_ideal = raw, limited
        inference_ms.append(float(row["model_ms"] if hardware else row["inference_ms"]))
        work_ms.append(float(row[time_key]))
        if hardware:
            elapsed_ms.append(float(row["cycle_elapsed_ms"]))
    raw_boundary = np.asarray(raw_boundary).reshape(-1, 6)
    ideal_boundary = np.asarray(ideal_boundary).reshape(-1, 6)
    raw_all = np.asarray(raw_all).reshape(-1, 6)
    refresh_ms = [inference_ms[i] for i, row in enumerate(rows) if int(row["chunk_refreshed"])]
    output = {
        "source": "observation_only" if hardware else "offline_recorded_demo",
        "path": str(path), "n_action_steps": n, "frames": len(rows),
        "boundary_count": len(raw_boundary),
        "raw_boundary_abs_delta": describe(raw_boundary),
        "ideal_follower_guarded_boundary_abs_delta": describe(ideal_boundary),
        "raw_all_frame_abs_delta": describe(raw_all),
        "inference_refresh_ms": {"mean": float(np.mean(refresh_ms)),
                                 "p95": float(np.percentile(refresh_ms, 95)),
                                 "max": float(np.max(refresh_ms))},
        "inference_all_cycles_ms": {"mean": float(np.mean(inference_ms)),
                                    "p95": float(np.percentile(inference_ms, 95))},
        "cycle_work_ms": {"mean": float(np.mean(work_ms)), "p95": float(np.percentile(work_ms, 95)),
                          "max": float(np.max(work_ms)),
                          "over_50ms": int(np.count_nonzero(np.asarray(work_ms) > 50))},
        "ideal_follower_guard_reason_counts": dict(reason_count),
        "recorded_state_counterfactual_latched_frames": original_latched,
        "recorded_state_counterfactual_reason_counts": dict(original_reason_count),
        "note": "Ideal-follower guard replay is only a constraint-count simulation. Recorded demo or stationary observations do not reveal live closed-loop tracking. No Action sent.",
    }
    if hardware:
        output["full_cycle_ms"] = {"mean": float(np.mean(elapsed_ms)),
                                   "p95": float(np.percentile(elapsed_ms, 95)),
                                   "max": float(np.max(elapsed_ms)),
                                   "over_50ms": int(np.count_nonzero(np.asarray(elapsed_ms) > 50))}
        output["torque_disabled_frames"] = summary["torque_disabled_frames"]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = [analyze(path) for path in args.runs]
    if len({(r["source"], r["n_action_steps"]) for r in results}) != len(results):
        raise ValueError("Duplicate mode/source")
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
