#!/usr/bin/env python3
"""Post-hoc greedy/first-order/second-order selection over cached IK candidates."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_mapping_hypotheses_offline import derivatives, load_frames
from export_cartesian_action_sidecar import JOINT_NAMES

METHODS = ("greedy", "dp_first_order", "dp_second_order")
SCALES = {"deviation": 15.0, "velocity": 100.0, "acceleration": 2000.0, "orientation": 15.0}
WEIGHTS = {"data": 1.0, "velocity": 1.0, "acceleration": 1.0, "orientation": 0.05, "bound": 0.02}


def candidate_key(item: dict) -> tuple:
    return (float(item["position_residual_mm"]), str(item["seed"]), tuple(item["motor"]))


def deduplicate(rows: list[dict], tolerance_deg: float = 1e-6) -> list[dict]:
    kept: dict[tuple[int, ...], dict] = {}
    for row in rows:
        motor = np.asarray(row["candidate_motor_deg"], dtype=float)
        if motor.shape != (5,) or not np.isfinite(motor).all():
            raise ValueError("Invalid candidate motor vector")
        key = tuple(np.rint(motor / tolerance_deg).astype(np.int64))
        item = {**row, "motor": motor.tolist()}
        if key not in kept or candidate_key(item) < candidate_key(kept[key]):
            kept[key] = item
    return sorted(kept.values(), key=candidate_key)


def contiguous_segments(layers: list[list[dict]]) -> list[tuple[int, int]]:
    segments, start = [], None
    for index, layer in enumerate(layers + [[]]):
        if layer and start is None:
            start = index
        elif not layer and start is not None:
            segments.append((start, index))
            start = None
    return segments


def node_cost(candidate: dict, recorded: np.ndarray) -> float:
    deviation = (np.asarray(candidate["motor"]) - recorded) / SCALES["deviation"]
    orientation = float(candidate["orientation_change_deg"]) / SCALES["orientation"]
    return (WEIGHTS["data"] * float(np.mean(deviation ** 2))
            + WEIGHTS["orientation"] * orientation ** 2
            + WEIGHTS["bound"] * len(candidate["active_bound_joints"]))


def velocity_cost(previous: dict, current: dict, dt: float) -> float:
    velocity = (np.asarray(current["motor"]) - np.asarray(previous["motor"])) / dt
    return WEIGHTS["velocity"] * float(np.mean((velocity / SCALES["velocity"]) ** 2))


def acceleration_cost(a: dict, b: dict, c: dict, dt_ab: float, dt_bc: float) -> float:
    va = (np.asarray(b["motor"]) - np.asarray(a["motor"])) / dt_ab
    vb = (np.asarray(c["motor"]) - np.asarray(b["motor"])) / dt_bc
    acceleration = (vb - va) / ((dt_ab + dt_bc) / 2.0)
    return WEIGHTS["acceleration"] * float(np.mean((acceleration / SCALES["acceleration"]) ** 2))


def select_greedy(layers: list[list[dict]], actions: np.ndarray, timestamps: np.ndarray) -> tuple[list[int], float]:
    selected, total = [], 0.0
    for i, layer in enumerate(layers):
        choices = []
        for j, candidate in enumerate(layer):
            cost = node_cost(candidate, actions[i])
            if selected:
                cost += velocity_cost(layers[i - 1][selected[-1]], candidate, timestamps[i] - timestamps[i - 1])
            choices.append((cost, candidate_key(candidate), j))
        incremental, _, choice = min(choices)
        selected.append(choice); total += incremental
    return selected, total


def select_first_order(layers: list[list[dict]], actions: np.ndarray, timestamps: np.ndarray) -> tuple[list[int], float]:
    costs = np.array([node_cost(c, actions[0]) for c in layers[0]])
    back = []
    for i in range(1, len(layers)):
        next_costs = np.empty(len(layers[i])); pointers = np.empty(len(layers[i]), dtype=int)
        dt = timestamps[i] - timestamps[i - 1]
        for j, current in enumerate(layers[i]):
            options = [(costs[k] + velocity_cost(previous, current, dt), candidate_key(previous), k)
                       for k, previous in enumerate(layers[i - 1])]
            best, _, pointer = min(options)
            next_costs[j] = best + node_cost(current, actions[i]); pointers[j] = pointer
        costs = next_costs; back.append(pointers)
    end = min(range(len(layers[-1])), key=lambda j: (costs[j], candidate_key(layers[-1][j])))
    path = [end]
    for pointers in reversed(back):
        path.append(int(pointers[path[-1]]))
    return list(reversed(path)), float(costs[end])


def select_second_order(layers: list[list[dict]], actions: np.ndarray, timestamps: np.ndarray) -> tuple[list[int], float]:
    if len(layers) < 3:
        return select_first_order(layers, actions, timestamps)
    dt = timestamps[1] - timestamps[0]
    pair_cost = np.empty((len(layers[0]), len(layers[1])))
    for a, first in enumerate(layers[0]):
        for b, second in enumerate(layers[1]):
            pair_cost[a, b] = node_cost(first, actions[0]) + node_cost(second, actions[1]) + velocity_cost(first, second, dt)
    backs = []
    for i in range(2, len(layers)):
        new = np.empty((len(layers[i - 1]), len(layers[i]))); pointer = np.empty_like(new, dtype=int)
        dt_ab, dt_bc = timestamps[i - 1] - timestamps[i - 2], timestamps[i] - timestamps[i - 1]
        for b, previous in enumerate(layers[i - 1]):
            for c, current in enumerate(layers[i]):
                options = [(pair_cost[a, b] + acceleration_cost(first, previous, current, dt_ab, dt_bc),
                            candidate_key(first), a) for a, first in enumerate(layers[i - 2])]
                best, _, a = min(options)
                new[b, c] = best + node_cost(current, actions[i]) + velocity_cost(previous, current, dt_bc)
                pointer[b, c] = a
        pair_cost = new; backs.append(pointer)
    b, c = min(np.ndindex(pair_cost.shape), key=lambda x: (pair_cost[x], candidate_key(layers[-2][x[0]]), candidate_key(layers[-1][x[1]])))
    path = [b, c]
    for pointer in reversed(backs):
        a = int(pointer[path[0], path[1]])
        path.insert(0, a)
    return path, float(pair_cost[b, c])


def metric_peak(values: np.ndarray, frames: np.ndarray, skip: int) -> dict:
    work = np.abs(values[skip:]); flat = int(np.nanargmax(work))
    row, joint = np.unravel_index(flat, work.shape); row += skip
    return {"value": float(work[row - skip, joint]), "frame": int(frames[row]), "joint": JOINT_NAMES[joint]}


def path_metrics(selected: list[dict], actions: np.ndarray, timestamps: np.ndarray, frames: np.ndarray,
                 cal_low: np.ndarray, cal_high: np.ndarray) -> dict:
    motor = np.array([item["motor"] for item in selected])
    jump = np.vstack([np.full((1, 5), np.nan), np.diff(motor, axis=0)])
    velocity, acceleration = derivatives(motor, timestamps)
    residuals = np.array([item["position_residual_mm"] for item in selected])
    deviations = motor - actions
    orientations = np.array([item["orientation_change_deg"] for item in selected])
    metrics = {
        "frames": len(frames), "residual_max_mm": float(residuals.max()),
        "deviation_peak": metric_peak(deviations, frames, 0),
        "jump_peak": metric_peak(jump, frames, 1) if len(frames) > 1 else None,
        "velocity_peak": metric_peak(velocity, frames, 1) if len(frames) > 1 else None,
        "acceleration_peak": metric_peak(acceleration, frames, 2) if len(frames) > 2 else None,
        "orientation_max_deg": float(orientations.max()),
        "outside_calibration_frames": int(np.any((motor < cal_low) | (motor > cal_high), axis=1).sum()),
    }
    metrics["quality_pass"] = bool(
        metrics["residual_max_mm"] <= 2.0 and metrics["deviation_peak"]["value"] <= 15.0
        and (metrics["jump_peak"] is None or metrics["jump_peak"]["value"] <= 5.0)
        and (metrics["velocity_peak"] is None or metrics["velocity_peak"]["value"] <= 100.0)
        and (metrics["acceleration_peak"] is None or metrics["acceleration_peak"]["value"] <= 2000.0)
        and metrics["orientation_max_deg"] <= 15.0
    )
    return metrics


def window_quality_pass(coverage: int, expected_frames: int, segment_metrics: list[dict]) -> bool:
    """Window-aware gate; never assume the historic 31-frame interval."""
    return bool(coverage == expected_frames and segment_metrics
                and all(metric["quality_pass"] for metric in segment_metrics))


def run(args: argparse.Namespace) -> dict:
    raw = json.loads(args.candidates.read_text())
    required = {"mapping", "direction", "frame", "candidate_motor_deg", "position_residual_mm",
                "feasible_witness", "orientation_change_deg", "active_bound_joints", "seed"}
    if any(not required.issubset(row) for row in raw):
        raise ValueError("Candidate schema mismatch")
    rows = load_frames(args.dataset, args.episode, args.frame_start, args.frame_end)
    frames = np.array([row["frame_index"] for row in rows]); timestamps = np.array([row["timestamp"] for row in rows])
    actions = np.array([row["action"][:5] for row in rows], dtype=float)
    dt = np.diff(timestamps)
    if not np.isfinite(timestamps).all() or np.any(dt <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    timestamp_diagnostics = {
        "dt_min_s": float(dt.min()), "dt_median_s": float(np.median(dt)),
        "dt_max_s": float(dt.max()),
        "non_50ms_steps": int(np.sum(~np.isclose(dt, 0.05, atol=2e-6))),
    }
    calibration = json.loads(args.calibration.read_text()); cal_low = np.empty(5); cal_high = np.empty(5)
    for i, name in enumerate(JOINT_NAMES):
        span = (calibration[name]["range_max"] - calibration[name]["range_min"]) * 180 / 4095
        cal_low[i], cal_high[i] = -span, span
    grouped = defaultdict(list)
    for row in raw:
        if not np.isfinite(float(row["position_residual_mm"])) or not np.isfinite(float(row["orientation_change_deg"])):
            raise ValueError("Nonfinite candidate")
        if row["feasible_witness"] and float(row["position_residual_mm"]) <= 2.0:
            grouped[(row["mapping"], row["direction"], int(row["frame"]))].append(row)
    results, traces = {}, []
    for mapping in sorted({row["mapping"] for row in raw}):
        results[mapping] = {}
        for direction in sorted({row["direction"] for row in raw}):
            layers = [deduplicate(grouped[(mapping, direction, int(frame))]) for frame in frames]
            segments = contiguous_segments(layers)
            method_results = {}
            for method in METHODS:
                selected_all, costs = [None] * len(frames), []
                for start, end in segments:
                    segment_layers, segment_actions, segment_times = layers[start:end], actions[start:end], timestamps[start:end]
                    if method == "greedy": path, cost = select_greedy(segment_layers, segment_actions, segment_times)
                    elif method == "dp_first_order": path, cost = select_first_order(segment_layers, segment_actions, segment_times)
                    else: path, cost = select_second_order(segment_layers, segment_actions, segment_times)
                    for local, choice in enumerate(path): selected_all[start + local] = segment_layers[local][choice]
                    costs.append(cost)
                segment_metrics = []
                for start, end in segments:
                    selected = selected_all[start:end]
                    segment_metrics.append(path_metrics(selected, actions[start:end], timestamps[start:end], frames[start:end], cal_low, cal_high))
                coverage = sum(item is not None for item in selected_all)
                aggregate = {
                    "coverage": coverage, "full_coverage": coverage == len(frames),
                    "segments": [[int(frames[a]), int(frames[b - 1])] for a, b in segments],
                    "total_cost": float(sum(costs)), "segment_metrics": segment_metrics,
                    "quality_pass": window_quality_pass(coverage, len(frames), segment_metrics),
                }
                for focus in (571, 572):
                    matches = np.flatnonzero(frames == focus)
                    index = int(matches[0]) if len(matches) else None
                    item = selected_all[index] if index is not None else None
                    aggregate[f"frame_{focus}"] = None if item is None else {
                        "seed": item["seed"], "motor_deg": item["motor"],
                        "residual_mm": item["position_residual_mm"], "active_bound_joints": item["active_bound_joints"],
                    }
                if aggregate["frame_571"] and aggregate["frame_572"]:
                    delta = np.asarray(aggregate["frame_572"]["motor_deg"]) - np.asarray(aggregate["frame_571"]["motor_deg"])
                    j = int(np.argmax(np.abs(delta)))
                    aggregate["jump_571_to_572"] = {"value": float(abs(delta[j])), "joint": JOINT_NAMES[j]}
                else: aggregate["jump_571_to_572"] = None
                method_results[method] = aggregate
                traces.extend({"mapping": mapping, "direction": direction, "method": method,
                               "frame": int(frames[i]), "candidate": selected_all[i]} for i in range(len(frames)))
            results[mapping][direction] = {
                "candidate_counts": [len(layer) for layer in layers],
                "missing_frames": [int(frames[i]) for i, layer in enumerate(layers) if not layer],
                "methods": method_results,
            }
    summary = {
        "classification": "CONDITIONAL_NUMERICAL_SENSITIVITY",
        "candidate_source": str(args.candidates), "episode": args.episode, "frames": [int(frames[0]), int(frames[-1])], "fps": 20,
        "timestamp_diagnostics": timestamp_diagnostics, "dedup_tolerance_deg": 1e-6, "cost_scales": SCALES, "cost_weights": WEIGHTS,
        "results": results,
        "warnings": [
            "DP is optimal only within the supplied finite candidate set; verify candidate seed provenance separately.",
            "No missing layer is filled and no transition crosses a feasibility gap.",
            "Thresholds are research diagnostics, not physical safety limits.",
            "Physical mapping, orientation constraint, collision, torque, ACT generation and hardware replay remain unverified/out of scope.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "selected_paths.json").write_text(json.dumps(traces, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=565)
    parser.add_argument("--frame-end", type=int, default=595)
    args = parser.parse_args()
    if args.frame_start < 0 or args.frame_end < args.frame_start:
        parser.error("Invalid frame interval")
    if not args.calibration.is_file():
        parser.error("--calibration must refer to an existing JSON file; set FOLLOWER_CALIBRATION_JSON")
    if args.output.exists(): parser.error("output exists")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
