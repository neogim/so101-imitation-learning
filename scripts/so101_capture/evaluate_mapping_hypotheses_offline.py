#!/usr/bin/env python3
"""Matched unregularized/regularized offline IK for four sign hypotheses."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import placo
import pyarrow.parquet as pq
from scipy.optimize import least_squares

from export_cartesian_action_sidecar import JOINT_NAMES
from generate_offset_ik_offline import joint_limits

HYPOTHESES = {
    "shoulder+_elbow+": np.array([1, 1, 1, 1, 1], dtype=float),
    "shoulder-_elbow+": np.array([1, -1, 1, 1, 1], dtype=float),
    "shoulder+_elbow-": np.array([1, 1, -1, 1, 1], dtype=float),
    "shoulder-_elbow-": np.array([1, -1, -1, 1, 1], dtype=float),
}
DIRECTIONS = {
    "x+": np.array([1, 0, 0]), "x-": np.array([-1, 0, 0]),
    "y+": np.array([0, 1, 0]), "y-": np.array([0, -1, 0]),
    "z+": np.array([0, 0, 1]), "z-": np.array([0, 0, -1]),
}
METHODS = ("unregularized", "regularized")


def motor_to_urdf_degrees(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * signs


def urdf_to_motor_degrees(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * signs


def urdf_limits_in_motor_degrees(lower: np.ndarray, upper: np.ndarray, signs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.rad2deg(lower) * signs, np.rad2deg(upper) * signs
    return np.minimum(a, b), np.maximum(a, b)


def derivatives(values: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocity, acceleration = np.full_like(values, np.nan), np.full_like(values, np.nan)
    dt = np.diff(timestamps)
    velocity[1:] = np.diff(values, axis=0) / dt[:, None]
    acceleration[2:] = np.diff(velocity[1:], axis=0) / (((dt[:-1] + dt[1:]) / 2)[:, None])
    return velocity, acceleration


def regularization_terms(
    q_urdf_rad: np.ndarray,
    signs: np.ndarray,
    recorded_motor_deg: np.ndarray,
    previous_candidate_motor_deg: np.ndarray | None,
    recorded_weight: float,
    previous_weight: float,
) -> np.ndarray:
    candidate_motor = urdf_to_motor_degrees(np.rad2deg(q_urdf_rad), signs)
    terms = [recorded_weight * (candidate_motor - recorded_motor_deg)]
    if previous_candidate_motor_deg is not None:
        terms.append(previous_weight * (candidate_motor - previous_candidate_motor_deg))
    return np.concatenate(terms)


def rotation_geodesic_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def load_frames(dataset: Path, episode: int, frame_start: int, frame_end: int) -> list[dict]:
    rows = []
    for path in sorted((dataset / "data").rglob("*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["episode_index", "frame_index", "timestamp", "action"]):
            rows.extend(row for row in batch.to_pylist()
                        if int(row["episode_index"]) == episode and frame_start <= int(row["frame_index"]) <= frame_end)
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not np.array_equal([row["frame_index"] for row in rows], np.arange(frame_start, frame_end + 1)):
        raise ValueError("Requested frame interval is not complete and consecutive")
    return rows


def peak(values: np.ndarray, frames: np.ndarray, *, skip: int = 0) -> dict:
    work = np.abs(values[skip:])
    flat = int(np.nanargmax(work))
    row, joint = np.unravel_index(flat, work.shape)
    row += skip
    return {"value": float(work[row - skip, joint]), "frame": int(frames[row]), "joint": JOINT_NAMES[joint]}


def distribution(values: np.ndarray) -> dict:
    finite = np.asarray(values)[np.isfinite(values)]
    return {"p50": float(np.quantile(finite, .5)), "p95": float(np.quantile(finite, .95)), "max": float(finite.max())}


def evaluate(args: argparse.Namespace) -> dict:
    split = json.loads((args.split_dataset / "episode_split.json").read_text())
    train, validation = set(split["train_episode_indices"]), set(split["validation_episode_indices"])
    if train & validation or args.episode not in train:
        raise ValueError("Only a train-split episode from a disjoint split is allowed")
    if args.output.exists() or args.dataset.resolve() in args.output.resolve().parents:
        raise ValueError("Output must be new and outside the dataset")
    rows = load_frames(args.dataset, args.episode, args.frame_start, args.frame_end)
    frames = np.array([row["frame_index"] for row in rows], dtype=int)
    timestamps = np.array([row["timestamp"] for row in rows], dtype=float)
    motor_action = np.array([row["action"][:5] for row in rows], dtype=float)
    if not np.isfinite(motor_action).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Invalid Action or timestamps")
    lower, upper = joint_limits(args.urdf)
    calibration = json.loads(args.calibration.read_text())
    cal_low = np.empty(5); cal_high = np.empty(5)
    for i, name in enumerate(JOINT_NAMES):
        entry = calibration[name]
        half = (int(entry["range_max"]) - int(entry["range_min"])) * 180.0 / 4095.0
        cal_low[i], cal_high[i] = -half, half
    robot = placo.RobotWrapper(str(args.urdf.resolve()))

    def fk(q_rad: np.ndarray) -> np.ndarray:
        for name, value in zip(JOINT_NAMES, q_rad):
            robot.set_joint(name, float(value))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"), dtype=float).copy()

    original_velocity, original_acceleration = derivatives(motor_action, timestamps)
    original_jump = np.vstack([np.full((1, 5), np.nan), np.diff(motor_action, axis=0)])
    original_peaks = {
        "jump_deg": peak(original_jump, frames, skip=1),
        "velocity_deg_s": peak(original_velocity, frames, skip=1),
        "acceleration_deg_s2": peak(original_acceleration, frames, skip=2),
    }
    gate_thresholds = {
        "jump_deg": max(5.0, 2.0 * original_peaks["jump_deg"]["value"]),
        "velocity_deg_s": max(100.0, 2.0 * original_peaks["velocity_deg_s"]["value"]),
        "acceleration_deg_s2": max(2000.0, 2.0 * original_peaks["acceleration_deg_s2"]["value"]),
        "tool_orientation_change_deg": 15.0,
    }
    results, detailed = {}, []
    for mapping, signs in HYPOTHESES.items():
        recorded_urdf_rad = np.deg2rad(motor_to_urdf_degrees(motor_action, signs))
        motor_lower, motor_upper = urdf_limits_in_motor_degrees(lower, upper, signs)
        recorded_outside = np.any((motor_action < motor_lower) | (motor_action > motor_upper), axis=1)
        original_transforms = np.array([fk(q) for q in recorded_urdf_rad])
        mapping_result = {
            "signs": dict(zip(JOINT_NAMES, signs.astype(int).tolist())),
            "recorded_outside_hypothetical_urdf_bounds_frames": int(recorded_outside.sum()),
            "methods": {method: {} for method in METHODS},
        }
        for direction, axis in DIRECTIONS.items():
            targets = original_transforms[:, :3, 3] + axis * args.offset_mm / 1000.0
            for method in METHODS:
                candidates, residuals, recomputed, successes, seed_clipped, orientations = [], [], [], [], [], []
                previous_motor = None
                for i, recorded_rad in enumerate(recorded_urdf_rad):
                    seed = np.clip(recorded_rad, lower + 1e-9, upper - 1e-9)
                    seed_clipped.append(bool(np.any(np.abs(seed - recorded_rad) > 1e-10)))

                    def residual(q: np.ndarray) -> np.ndarray:
                        position = (fk(q)[:3, 3] - targets[i]) * 1000.0
                        if method == "unregularized":
                            return position
                        priors = regularization_terms(q, signs, motor_action[i], previous_motor,
                                                      args.recorded_weight, args.previous_weight)
                        return np.concatenate([position, priors])

                    fit = least_squares(residual, seed, bounds=(lower, upper), max_nfev=args.max_nfev)
                    transform = fk(fit.x)
                    candidate_motor = urdf_to_motor_degrees(np.rad2deg(fit.x), signs)
                    position_error = float(np.linalg.norm((transform[:3, 3] - targets[i]) * 1000.0))
                    candidates.append(candidate_motor); residuals.append(position_error)
                    recomputed.append(position_error); successes.append(bool(fit.success))
                    orientations.append(rotation_geodesic_degrees(original_transforms[i, :3, :3], transform[:3, :3]))
                    previous_motor = candidate_motor.copy()
                candidates = np.asarray(candidates); residuals = np.asarray(residuals)
                recomputed = np.asarray(recomputed); orientations = np.asarray(orientations)
                deviation = candidates - motor_action
                jump = np.vstack([np.full((1, 5), np.nan), np.diff(candidates, axis=0)])
                velocity, acceleration = derivatives(candidates, timestamps)
                screen = np.asarray(successes) & (residuals <= args.max_position_error_mm) & (np.max(np.abs(deviation), axis=1) <= args.max_deviation_deg)
                near_bound = np.any((np.abs(candidates - motor_lower) <= 1e-3) | (np.abs(candidates - motor_upper) <= 1e-3), axis=1)
                outside_cal = np.any((candidates < cal_low) | (candidates > cal_high), axis=1)
                metrics = {
                    "optimizer_successes": int(sum(successes)), "screen_passes": int(screen.sum()),
                    "position_residual_mm": distribution(residuals),
                    "position_residual_peak_frame": int(frames[int(np.argmax(residuals))]),
                    "recomputed_residual_max_abs_difference_mm": float(np.max(np.abs(recomputed - residuals))),
                    "motor_deviation_deg": distribution(np.max(np.abs(deviation), axis=1)),
                    "motor_deviation_peak": peak(deviation, frames),
                    "jump_peak": peak(jump, frames, skip=1),
                    "velocity_peak": peak(velocity, frames, skip=1),
                    "acceleration_peak": peak(acceleration, frames, skip=2),
                    "tool_orientation_change_deg": distribution(orientations),
                    "tool_orientation_peak_frame": int(frames[int(np.argmax(orientations))]),
                    "seed_clipped_frames": int(sum(seed_clipped)),
                    "near_urdf_bound_frames": int(near_bound.sum()),
                    "outside_calibrated_interval_frames": int(outside_cal.sum()),
                }
                at_572 = np.flatnonzero(frames == 572)
                metrics["frame_572"] = None if not len(at_572) else {
                    "screen_pass": bool(screen[at_572[0]]),
                    "position_residual_mm": float(residuals[at_572[0]]),
                    "max_motor_deviation_deg": float(np.max(np.abs(deviation[at_572[0]]))),
                    "shoulder_candidate_motor_deg": float(candidates[at_572[0], 1]),
                    "tool_orientation_change_deg": float(orientations[at_572[0]]),
                    "near_urdf_bound": bool(near_bound[at_572[0]]),
                }
                mapping_result["methods"][method][direction] = metrics
                detailed.extend({
                    "mapping": mapping, "method": method, "direction": direction, "frame": int(frames[i]),
                    "optimizer_success": successes[i], "screen_pass": bool(screen[i]),
                    "position_residual_mm": float(residuals[i]),
                    "max_motor_deviation_deg": float(np.max(np.abs(deviation[i]))),
                    "tool_orientation_change_deg": float(orientations[i]),
                } for i in range(len(frames)))
        results[mapping] = mapping_result

    gate_by_mapping = {}
    for mapping, result in results.items():
        directions = result["methods"]["regularized"].values()
        expected_clips = result["recorded_outside_hypothetical_urdf_bounds_frames"]
        checks = {
            "all_optimizer_successful": all(x["optimizer_successes"] == len(frames) for x in directions),
            "all_rows_pass_screen": all(x["screen_passes"] == len(frames) for x in directions),
            "jump_within_threshold": all(x["jump_peak"]["value"] <= gate_thresholds["jump_deg"] for x in directions),
            "velocity_within_threshold": all(x["velocity_peak"]["value"] <= gate_thresholds["velocity_deg_s"] for x in directions),
            "acceleration_within_threshold": all(x["acceleration_peak"]["value"] <= gate_thresholds["acceleration_deg_s2"] for x in directions),
            "orientation_within_threshold": all(x["tool_orientation_change_deg"]["max"] <= gate_thresholds["tool_orientation_change_deg"] for x in directions),
            "seed_clipping_explained": all(x["seed_clipped_frames"] == expected_clips for x in directions),
            "no_additional_bound_activity": all(x["near_urdf_bound_frames"] <= expected_clips for x in directions),
        }
        gate_by_mapping[mapping] = {"pass": all(checks.values()), "checks": checks}
    overall_go = all(item["pass"] for item in gate_by_mapping.values())
    summary = {
        "classification": "CONDITIONAL_NUMERICAL_SENSITIVITY",
        "episode": args.episode, "train_split_confirmed": True,
        "frame_start": int(frames[0]), "frame_end": int(frames[-1]), "frames": len(frames),
        "median_dt_s": float(np.median(np.diff(timestamps))),
        "settings": {"offset_mm": args.offset_mm, "max_nfev": args.max_nfev,
                     "recorded_weight": args.recorded_weight, "previous_weight": args.previous_weight,
                     "common_seed": "per-frame recorded Action clipped to URDF bounds"},
        "calibrated_motor_degree_intervals": dict(zip(JOINT_NAMES, np.column_stack([cal_low, cal_high]).tolist())),
        "original_action_peaks": original_peaks,
        "gate_thresholds": gate_thresholds,
        "hypotheses": results,
        "gate": {"by_mapping": gate_by_mapping, "overall_go": overall_go,
                 "decision": "GO_FULL_648" if overall_go else "STOP_AT_31"},
        "warnings": [
            "Hypotheses are numerical stress tests and cannot identify physical sign or zero.",
            "TCP position/orientation are in a hypothetical URDF world frame.",
            "Orientation is unconstrained; collision, torque, contact, gripper and physical safety are not evaluated.",
            "Rates are finite differences of counterfactual motor-coordinate waypoints, not telemetry.",
            "No ACT training data or hardware commands were generated.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "per_frame.json").write_text(json.dumps(detailed, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "split-dataset", "urdf", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--offset-mm", type=float, default=3.0)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--max-position-error-mm", type=float, default=2.0)
    parser.add_argument("--max-deviation-deg", type=float, default=15.0)
    parser.add_argument("--recorded-weight", type=float, default=.05)
    parser.add_argument("--previous-weight", type=float, default=.01)
    args = parser.parse_args()
    if (args.frame_start < 0 or args.frame_end < args.frame_start or args.offset_mm <= 0
            or args.max_nfev < 1 or args.recorded_weight < 0 or args.previous_weight < 0):
        parser.error("invalid numeric arguments")
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
