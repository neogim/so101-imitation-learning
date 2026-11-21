#!/usr/bin/env python3
"""Offline tracking analysis for a LeRobot SO-101 position dataset.

The recorded ``action`` is treated as the limited Goal_Position returned by
``send_action``.  It is not labelled as measured servo position, torque, or a
hardware acknowledgement.  J1--J5 are normalized motor degrees and J6 is a
normalized 0--100 gripper position in the inspected capture pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")


def contiguous_regions(mask: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    regions = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > max_gap + 1:
            regions.append((start, previous + 1))
            start = index
        previous = index
    regions.append((start, previous + 1))
    return regions


def derivative_lag_frames(action: np.ndarray, state: np.ndarray, max_lag: int) -> int | None:
    da, ds = np.diff(action), np.diff(state)
    if np.std(da) < 1e-6 or np.std(ds) < 1e-6:
        return None
    best = None
    for lag in range(max_lag + 1):
        left = da[: len(da) - lag or None]
        right = ds[lag:]
        if len(left) < 10:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(correlation) and (best is None or correlation > best[0]):
            best = (correlation, lag)
    return None if best is None else best[1]


def analyze_episode(action: np.ndarray, state: np.ndarray, fps: float,
                    episode: int, max_lag: int) -> tuple[dict, list[dict]]:
    error = action - state
    joint = {}
    for index, name in enumerate(JOINTS):
        unit = "normalized_0_100" if name == "J6" else "normalized_motor_deg"
        joint[name] = {
            "unit": unit,
            "mae": float(np.mean(np.abs(error[:, index]))),
            "p95_abs_error": float(np.percentile(np.abs(error[:, index]), 95)),
            "max_abs_error": float(np.max(np.abs(error[:, index]))),
            "derivative_lag_frames": derivative_lag_frames(
                action[:, index], state[:, index], max_lag),
        }

    # Lower J6 values close the inspected gripper convention.  Find sustained
    # command decreases, then retain only material closures.
    delta = np.diff(action[:, 5])
    candidates = contiguous_regions(delta < -0.05, max_gap=2)
    closures = []
    for begin_delta, end_delta in candidates:
        begin = begin_delta
        end = min(len(action) - 1, end_delta + 1)
        command_change = float(action[end, 5] - action[begin, 5])
        if command_change > -5 or (end - begin) < 2:
            continue
        response_threshold = state[begin, 5] + 0.1 * command_change
        hits = np.flatnonzero(state[begin:end + 1, 5] <= response_threshold)
        delay_frames = None if not len(hits) else int(hits[0])
        post = min(len(state) - 1, end + int(round(0.5 * fps)))
        row = {
            "episode": episode,
            "start_frame": begin,
            "end_frame": end,
            "start_s": begin / fps,
            "end_s": end / fps,
            "duration_s": (end - begin) / fps,
            "j6_command_change_norm": command_change,
            "j6_state_change_norm": float(state[end, 5] - state[begin, 5]),
            "j6_command_speed_norm_s": float(-command_change / max((end - begin) / fps, 1 / fps)),
            "j6_state_peak_speed_norm_s": float(np.max(np.maximum(0, -np.diff(state[begin:end + 1, 5]) * fps))),
            "j6_10pct_delay_s": None if delay_frames is None else delay_frames / fps,
            "j6_peak_abs_tracking_error_norm": float(np.max(np.abs(error[begin:end + 1, 5]))),
        }
        for axis, name in zip((1, 2, 3), ("J2", "J3", "J4")):
            segment = state[begin:post + 1, axis]
            command_segment = action[begin:post + 1, axis]
            row[f"{name.lower()}_end_delta_motor_deg"] = float(state[post, axis] - state[begin, axis])
            row[f"{name.lower()}_range_motor_deg"] = float(np.ptp(segment))
            row[f"{name.lower()}_command_range_motor_deg"] = float(np.ptp(command_segment))
        row["arm_command_hold"] = bool(max(
            row["j2_command_range_motor_deg"], row["j3_command_range_motor_deg"],
            row["j4_command_range_motor_deg"]) <= 1.5)
        closures.append(row)
    return {"episode": episode, "frames": len(action), "joint": joint}, closures


def aggregate(episodes: list[dict], closures: list[dict], fps: float) -> dict:
    result = {
        "fps": fps,
        "episodes": len(episodes),
        "frames": int(sum(row["frames"] for row in episodes)),
        "action_semantics": "limited Goal_Position returned by send_action; not measurement or acknowledgement",
        "joint_units": {name: ("normalized_0_100" if name == "J6" else "normalized_motor_deg") for name in JOINTS},
        "closure_direction": "decreasing J6 value",
        "closure_events": len(closures),
        "joint_tracking": {},
    }
    for name in JOINTS:
        rows = [episode["joint"][name] for episode in episodes]
        lags = [row["derivative_lag_frames"] for row in rows if row["derivative_lag_frames"] is not None]
        result["joint_tracking"][name] = {
            "episode_median_mae": float(np.median([row["mae"] for row in rows])),
            "episode_median_p95_abs_error": float(np.median([row["p95_abs_error"] for row in rows])),
            "episode_max_abs_error": float(max(row["max_abs_error"] for row in rows)),
            "median_derivative_lag_s": None if not lags else float(np.median(lags) / fps),
        }
    if closures:
        keys = [
            "j6_command_speed_norm_s", "j6_state_peak_speed_norm_s", "j6_10pct_delay_s",
            "j6_peak_abs_tracking_error_norm", "j2_end_delta_motor_deg", "j3_end_delta_motor_deg",
            "j4_end_delta_motor_deg", "j2_range_motor_deg", "j3_range_motor_deg", "j4_range_motor_deg",
        ]
        result["closure_summary"] = {}
        for key in keys:
            values = [row[key] for row in closures if row[key] is not None]
            result["closure_summary"][key] = {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max_abs": float(np.max(np.abs(values))),
            }
        held = [row for row in closures if row["arm_command_hold"]]
        result["arm_hold_closure_events"] = len(held)
        result["arm_hold_closure_summary"] = {}
        for key in keys:
            values = [row[key] for row in held if row[key] is not None]
            if values:
                result["arm_hold_closure_summary"][key] = {
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "max_abs": float(np.max(np.abs(values))),
                }
    return result


def plot_results(summary: dict, closures: list[dict], out: Path) -> None:
    names = list(JOINTS)
    mae = [summary["joint_tracking"][name]["episode_median_mae"] for name in names]
    p95 = [summary["joint_tracking"][name]["episode_median_p95_abs_error"] for name in names]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, mae, 0.36, label="Median episode MAE")
    axes[0].bar(x + 0.18, p95, 0.36, label="Median episode p95 |error|")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Tracking error (motor deg; J6 normalized units)")
    axes[0].set_xlabel("Joint")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    if closures:
        episodes = [row["episode"] for row in closures]
        axes[1].scatter(episodes, [row["j2_end_delta_motor_deg"] for row in closures], label="J2")
        axes[1].scatter(episodes, [row["j3_end_delta_motor_deg"] for row in closures], label="J3")
        axes[1].scatter(episodes, [row["j4_end_delta_motor_deg"] for row in closures], label="J4")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Episode index")
    axes[1].set_ylabel("State change through close + 0.5 s (motor deg)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("SO-101 recorded goal/state tracking and arm motion during J6 closure")
    fig.savefig(out / "real_servo_tracking.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-lag-frames", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.max_lag_frames < 0:
        parser.error("max-lag-frames must be nonnegative")
    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    table = pq.read_table(args.dataset / "data" / "chunk-000" / "file-000.parquet")
    frame = table.to_pandas()
    actions = np.stack(frame["action"].to_numpy())
    states = np.stack(frame["observation.state"].to_numpy())
    episode_index = frame["episode_index"].to_numpy()
    episodes, closures = [], []
    for episode in np.unique(episode_index):
        selection = episode_index == episode
        result, found = analyze_episode(actions[selection], states[selection], fps,
                                        int(episode), args.max_lag_frames)
        episodes.append(result)
        closures.extend(found)
    summary = aggregate(episodes, closures, fps)
    args.out.mkdir(parents=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "episodes.json").write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    if closures:
        with (args.out / "j6_closure_events.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(closures[0]))
            writer.writeheader()
            writer.writerows(closures)
    plot_results(summary, closures, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
