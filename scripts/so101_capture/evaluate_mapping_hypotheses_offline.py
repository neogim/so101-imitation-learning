#!/usr/bin/env python3
"""Conditional offline ±XYZ IK sensitivity for four shoulder/elbow sign maps.

Reads recorded Action only.  Never imports robot hardware modules or writes the
dataset.  Results are hypothetical URDF-world numerics, not physical validation.
"""
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


def motor_to_urdf_degrees(motor_degrees: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return np.asarray(motor_degrees, dtype=float) * signs


def urdf_to_motor_degrees(urdf_degrees: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return np.asarray(urdf_degrees, dtype=float) * signs


def urdf_limits_in_motor_degrees(lower: np.ndarray, upper: np.ndarray, signs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.rad2deg(lower) * signs
    b = np.rad2deg(upper) * signs
    return np.minimum(a, b), np.maximum(a, b)


def derivatives(values: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.full_like(values, np.nan)
    acceleration = np.full_like(values, np.nan)
    dt = np.diff(timestamps)
    velocity[1:] = np.diff(values, axis=0) / dt[:, None]
    acceleration[2:] = np.diff(velocity[1:], axis=0) / (((dt[:-1] + dt[1:]) / 2)[:, None])
    return velocity, acceleration


def load_frames(dataset: Path, episode: int, frame_start: int, frame_end: int) -> list[dict]:
    rows = []
    for path in sorted((dataset / "data").rglob("*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["episode_index", "frame_index", "timestamp", "action"]):
            rows.extend(
                row for row in batch.to_pylist()
                if int(row["episode_index"]) == episode and frame_start <= int(row["frame_index"]) <= frame_end
            )
    rows.sort(key=lambda row: int(row["frame_index"]))
    expected = np.arange(frame_start, frame_end + 1)
    actual = np.array([row["frame_index"] for row in rows], dtype=int)
    if not np.array_equal(actual, expected):
        raise ValueError("Requested frame interval is not complete and consecutive")
    return rows


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
    calibrated_intervals = {}
    for name in JOINT_NAMES:
        entry = calibration[name]
        half_span = (int(entry["range_max"]) - int(entry["range_min"])) * 180.0 / 4095.0
        calibrated_intervals[name] = [-half_span, half_span]
    robot = placo.RobotWrapper(str(args.urdf.resolve()))

    def fk(q_rad: np.ndarray) -> np.ndarray:
        for name, value in zip(JOINT_NAMES, q_rad):
            robot.set_joint(name, float(value))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"), dtype=float)[:3, 3].copy()

    results = {}
    detailed = []
    for mapping, signs in HYPOTHESES.items():
        urdf_recorded_deg = motor_to_urdf_degrees(motor_action, signs)
        urdf_recorded_rad = np.deg2rad(urdf_recorded_deg)
        motor_lower, motor_upper = urdf_limits_in_motor_degrees(lower, upper, signs)
        recorded_outside = np.any((motor_action < motor_lower) | (motor_action > motor_upper), axis=1)
        origins = np.array([fk(q) for q in urdf_recorded_rad])
        mapping_result = {
            "signs": dict(zip(JOINT_NAMES, signs.astype(int).tolist())),
            "motor_coordinate_urdf_bounds_deg": {
                name: [float(motor_lower[i]), float(motor_upper[i])] for i, name in enumerate(JOINT_NAMES)
            },
            "recorded_action_outside_hypothetical_urdf_bounds_frames": int(recorded_outside.sum()),
            "directions": {},
        }
        for direction, axis in DIRECTIONS.items():
            output_motor = []
            residuals = []
            successes = []
            clipped_seed = []
            previous = None
            for index, recorded_rad in enumerate(urdf_recorded_rad):
                target = origins[index] + axis * (args.offset_mm / 1000.0)
                seed_source = recorded_rad if previous is None else previous
                seed = np.clip(seed_source, lower + 1e-9, upper - 1e-9)
                clipped_seed.append(bool(np.any(np.abs(seed - seed_source) > 1e-10)))
                fit = least_squares(
                    lambda q: (fk(q) - target) * 1000.0,
                    seed,
                    bounds=(lower, upper),
                    max_nfev=args.max_nfev,
                )
                candidate_motor = urdf_to_motor_degrees(np.rad2deg(fit.x), signs)
                output_motor.append(candidate_motor)
                residuals.append(float(np.linalg.norm((fk(fit.x) - target) * 1000.0)))
                successes.append(bool(fit.success))
                previous = fit.x.copy()
            output_motor = np.asarray(output_motor)
            residuals = np.asarray(residuals)
            deviations = np.max(np.abs(output_motor - motor_action), axis=1)
            velocity, acceleration = derivatives(output_motor, timestamps)
            candidate_outside = np.any((output_motor < motor_lower) | (output_motor > motor_upper), axis=1)
            screen = np.asarray(successes) & (residuals <= args.max_position_error_mm) & (deviations <= args.max_deviation_deg)
            at_572 = int(np.flatnonzero(frames == 572)[0]) if 572 in frames else None
            stats = {
                "frames": len(frames),
                "optimizer_successes": int(sum(successes)),
                "numerical_screen_passes": int(screen.sum()),
                "max_hypothetical_world_position_residual_mm": float(residuals.max()),
                "max_motor_coordinate_deviation_deg": float(deviations.max()),
                "max_motor_coordinate_velocity_deg_s": float(np.nanmax(np.abs(velocity[1:]))),
                "max_motor_coordinate_acceleration_deg_s2": float(np.nanmax(np.abs(acceleration[2:]))),
                "seed_clipped_frames": int(sum(clipped_seed)),
                "candidate_outside_hypothetical_urdf_bounds_frames": int(candidate_outside.sum()),
                "frame_572": None if at_572 is None else {
                    "recorded_shoulder_motor_deg": float(motor_action[at_572, 1]),
                    "recorded_shoulder_urdf_deg": float(urdf_recorded_deg[at_572, 1]),
                    "candidate_shoulder_motor_deg": float(output_motor[at_572, 1]),
                    "position_residual_mm": float(residuals[at_572]),
                    "seed_was_clipped": clipped_seed[at_572],
                },
            }
            mapping_result["directions"][direction] = stats
            detailed.extend({
                "mapping": mapping, "direction": direction, "frame": int(frames[i]),
                "optimizer_success": successes[i], "position_residual_mm": float(residuals[i]),
                "max_motor_deviation_deg": float(deviations[i]), "numeric_screen_pass": bool(screen[i]),
            } for i in range(len(frames)))
        results[mapping] = mapping_result
    baseline = results["shoulder+_elbow+"]
    for mapping, result in results.items():
        result["difference_from_identity"] = {
            direction: {
                "numerical_screen_passes": result["directions"][direction]["numerical_screen_passes"] - baseline["directions"][direction]["numerical_screen_passes"],
                "max_motor_deviation_deg": result["directions"][direction]["max_motor_coordinate_deviation_deg"] - baseline["directions"][direction]["max_motor_coordinate_deviation_deg"],
            } for direction in DIRECTIONS
        }
    summary = {
        "classification": "CONDITIONAL_NUMERICAL_SENSITIVITY",
        "episode": args.episode, "train_split_confirmed": True,
        "frame_start": int(frames[0]), "frame_end": int(frames[-1]), "frames": len(frames),
        "offset_mm": args.offset_mm, "max_nfev": args.max_nfev,
        "lerobot_calibrated_motor_degree_intervals": calibrated_intervals,
        "hypotheses": results,
        "warnings": [
            "Hypotheses are stress tests, not mapping estimates or physical validation.",
            "Residuals are in a hypothetical URDF world frame; TCP and camera alignment are unverified.",
            "Position-only IK leaves tool orientation unconstrained.",
            "Collision, torque, contact, gripper limits, physical calibration and hardware replay are not evaluated.",
            "No ACT examples or synchronized synthetic images were generated.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "per_frame.json").write_text(json.dumps(detailed, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-dataset", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--offset-mm", type=float, default=3.0)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--max-position-error-mm", type=float, default=2.0)
    parser.add_argument("--max-deviation-deg", type=float, default=15.0)
    args = parser.parse_args()
    if args.frame_start < 0 or args.frame_end < args.frame_start or args.offset_mm <= 0 or args.max_nfev < 1:
        parser.error("invalid numeric arguments")
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
