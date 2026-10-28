#!/usr/bin/env python3
"""Finite deterministic multistart audit of hypothetical SO-101 ±3 mm IK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import placo
from scipy.optimize import least_squares

from evaluate_mapping_hypotheses_offline import (
    DIRECTIONS, HYPOTHESES, derivatives, load_frames, motor_to_urdf_degrees,
    rotation_geodesic_degrees, urdf_limits_in_motor_degrees, urdf_to_motor_degrees,
)
from export_cartesian_action_sidecar import JOINT_NAMES
from generate_offset_ik_offline import joint_limits


def multistart_seeds(
    recorded_motor_deg: np.ndarray,
    signs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    previous_urdf_rad: np.ndarray | None,
) -> list[tuple[str, np.ndarray]]:
    """Predeclared seeds: clipped record, prior witness, and ±2/±5° J2/J3."""
    candidates = [("recorded_clipped", recorded_motor_deg.copy())]
    if previous_urdf_rad is not None:
        candidates.append(("previous_witness", urdf_to_motor_degrees(np.rad2deg(previous_urdf_rad), signs)))
    for joint, index in (("shoulder", 1), ("elbow", 2)):
        for delta in (-5.0, -2.0, 2.0, 5.0):
            value = recorded_motor_deg.copy(); value[index] += delta
            candidates.append((f"{joint}_{delta:+g}deg", value))
    output, seen = [], set()
    for name, motor in candidates:
        q = np.clip(np.deg2rad(motor_to_urdf_degrees(motor, signs)), lower + 1e-9, upper - 1e-9)
        key = tuple(np.round(q, 12))
        if key not in seen:
            seen.add(key); output.append((name, q))
    return output


def peak(values: np.ndarray, frames: np.ndarray, skip: int) -> dict:
    work = np.abs(values[skip:]); flat = int(np.nanargmax(work))
    row, joint = np.unravel_index(flat, work.shape); row += skip
    return {"value": float(work[row - skip, joint]), "frame": int(frames[row]), "joint": JOINT_NAMES[joint]}


def expansion_decision(all_witnesses: bool, temporal_ok: bool) -> str:
    return "ELIGIBLE_FOR_MORE_FRAMES" if all_witnesses and temporal_ok else "STOP"


def run(args: argparse.Namespace) -> dict:
    split = json.loads((args.split_dataset / "episode_split.json").read_text())
    train, validation = set(split["train_episode_indices"]), set(split["validation_episode_indices"])
    if train & validation or args.episode not in train:
        raise ValueError("Episode must be train-only in a disjoint split")
    if args.output.exists() or args.dataset.resolve() in args.output.resolve().parents:
        raise ValueError("Output must be new and outside dataset")
    rows = load_frames(args.dataset, args.episode, args.frame_start, args.frame_end)
    frames = np.array([row["frame_index"] for row in rows], dtype=int)
    timestamps = np.array([row["timestamp"] for row in rows], dtype=float)
    actions = np.array([row["action"][:5] for row in rows], dtype=float)
    lower, upper = joint_limits(args.urdf)
    calibration = json.loads(args.calibration.read_text())
    cal_low, cal_high = np.empty(5), np.empty(5)
    for i, name in enumerate(JOINT_NAMES):
        span = (calibration[name]["range_max"] - calibration[name]["range_min"]) * 180.0 / 4095.0
        cal_low[i], cal_high[i] = -span, span
    robot = placo.RobotWrapper(str(args.urdf.resolve()))

    def fk(q: np.ndarray) -> np.ndarray:
        for name, value in zip(JOINT_NAMES, q):
            robot.set_joint(name, float(value))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"), dtype=float).copy()

    details, hypotheses = [], {}
    for mapping, signs in HYPOTHESES.items():
        recorded_urdf = np.deg2rad(motor_to_urdf_degrees(actions, signs))
        recorded_tf = np.array([fk(q) for q in recorded_urdf])
        motor_lower, motor_upper = urdf_limits_in_motor_degrees(lower, upper, signs)
        recorded_outside = np.any((actions < motor_lower) | (actions > motor_upper), axis=1)
        mapping_summary = {"recorded_outside_urdf_frames": int(recorded_outside.sum()), "directions": {}}
        for direction, axis in DIRECTIONS.items():
            selected_motor, selected_q, selected_residual, selected_orientation = [], [], [], []
            feasible_frames, seed_sensitive_frames, optimizer_successes = 0, 0, 0
            previous = None
            per_frame_seed_counts = []
            for i, frame in enumerate(frames):
                target = recorded_tf[i, :3, 3] + axis * args.offset_mm / 1000.0
                seed_results = []
                for seed_name, seed in multistart_seeds(actions[i], signs, lower, upper, previous):
                    fit = least_squares(lambda q: (fk(q)[:3, 3] - target) * 1000.0,
                                        seed, bounds=(lower, upper), max_nfev=args.max_nfev)
                    transform = fk(fit.x)
                    motor = urdf_to_motor_degrees(np.rad2deg(fit.x), signs)
                    residual = float(np.linalg.norm((transform[:3, 3] - target) * 1000.0))
                    roundtrip_q = np.deg2rad(motor_to_urdf_degrees(motor, signs))
                    roundtrip_residual = float(np.linalg.norm((fk(roundtrip_q)[:3, 3] - target) * 1000.0))
                    active = [JOINT_NAMES[j] for j in range(5)
                              if abs(motor[j] - motor_lower[j]) <= 1e-3 or abs(motor[j] - motor_upper[j]) <= 1e-3]
                    item = {
                        "mapping": mapping, "direction": direction, "frame": int(frame), "seed": seed_name,
                        "optimizer_success": bool(fit.success), "position_residual_mm": residual,
                        "roundtrip_position_residual_mm": roundtrip_residual,
                        "roundtrip_residual_difference_mm": abs(roundtrip_residual - residual),
                        "feasible_witness": residual <= args.feasible_residual_mm,
                        "active_bound_joints": active,
                        "motor_deviation_deg": float(np.max(np.abs(motor - actions[i]))),
                        "orientation_change_deg": rotation_geodesic_degrees(recorded_tf[i, :3, :3], transform[:3, :3]),
                        "candidate_motor_deg": motor.tolist(),
                    }
                    seed_results.append((item, fit.x.copy(), motor.copy()))
                    details.append(item); optimizer_successes += int(fit.success)
                feasible = [entry for entry in seed_results if entry[0]["feasible_witness"]]
                feasible_frames += bool(feasible)
                statuses = {entry[0]["feasible_witness"] for entry in seed_results}
                spread = max(entry[0]["motor_deviation_deg"] for entry in seed_results) - min(entry[0]["motor_deviation_deg"] for entry in seed_results)
                seed_sensitive_frames += len(statuses) > 1 or spread > 5.0
                if feasible:
                    if previous is None:
                        chosen = min(feasible, key=lambda entry: (entry[0]["motor_deviation_deg"], entry[0]["position_residual_mm"], entry[0]["seed"]))
                    else:
                        chosen = min(feasible, key=lambda entry: (float(np.max(np.abs(entry[1] - previous))), entry[0]["motor_deviation_deg"], entry[0]["position_residual_mm"], entry[0]["seed"]))
                else:
                    chosen = min(seed_results, key=lambda entry: (entry[0]["position_residual_mm"], entry[0]["motor_deviation_deg"], entry[0]["seed"]))
                previous = chosen[1].copy()
                selected_q.append(chosen[1]); selected_motor.append(chosen[2])
                selected_residual.append(chosen[0]["position_residual_mm"])
                selected_orientation.append(chosen[0]["orientation_change_deg"])
                per_frame_seed_counts.append(len(seed_results))
            selected_motor = np.asarray(selected_motor)
            jump = np.vstack([np.full((1, 5), np.nan), np.diff(selected_motor, axis=0)])
            velocity, acceleration = derivatives(selected_motor, timestamps)
            deviations = selected_motor - actions
            direction_summary = {
                "frames": len(frames), "seeds_per_frame_minmax": [min(per_frame_seed_counts), max(per_frame_seed_counts)],
                "seed_attempts": int(sum(per_frame_seed_counts)),
                "optimizer_successes_across_seeds": optimizer_successes,
                "feasible_witness_frames": int(feasible_frames),
                "not_found_frames": int(len(frames) - feasible_frames),
                "seed_sensitive_frames": int(seed_sensitive_frames),
                "best_selected_residual_max_mm": float(max(selected_residual)),
                "selected_deviation_peak": peak(deviations, frames, 0),
                "selected_jump_peak": peak(jump, frames, 1),
                "selected_velocity_peak": peak(velocity, frames, 1),
                "selected_acceleration_peak": peak(acceleration, frames, 2),
                "selected_orientation_max_deg": float(max(selected_orientation)),
                "selected_outside_calibration_frames": int(np.any((selected_motor < cal_low) | (selected_motor > cal_high), axis=1).sum()),
                "max_roundtrip_residual_difference_mm": float(max(
                    item["roundtrip_residual_difference_mm"] for item in details
                    if item["mapping"] == mapping and item["direction"] == direction
                )),
            }
            where_572 = np.flatnonzero(frames == 572)
            if len(where_572):
                k = int(where_572[0])
                seeds_572 = [item for item in details if item["mapping"] == mapping and item["direction"] == direction and item["frame"] == 572]
                direction_summary["frame_572"] = {
                    "recorded_shoulder_motor_deg": float(actions[k, 1]),
                    "recorded_shoulder_urdf_deg": float(motor_to_urdf_degrees(actions[k], signs)[1]),
                    "recorded_reference_outside_urdf": bool(recorded_outside[k]),
                    "recorded_shoulder_urdf_after_seed_clip_deg": float(np.rad2deg(np.clip(recorded_urdf[k], lower + 1e-9, upper - 1e-9))[1]),
                    "seeds_tested": len(seeds_572),
                    "feasible_seed_count": sum(item["feasible_witness"] for item in seeds_572),
                    "best_residual_mm": min(item["position_residual_mm"] for item in seeds_572),
                    "best_inbound_shoulder_motor_deg": float(selected_motor[k, 1]),
                    "selected_active_bound_joints": min(seeds_572, key=lambda item: item["position_residual_mm"])["active_bound_joints"],
                }
            mapping_summary["directions"][direction] = direction_summary
        hypotheses[mapping] = mapping_summary
    all_witnesses = all(value["feasible_witness_frames"] == len(frames)
                        for mapping in hypotheses.values() for value in mapping["directions"].values())
    temporal_ok = all(value["selected_jump_peak"]["value"] <= 5.0
                      and value["selected_velocity_peak"]["value"] <= 100.0
                      and value["selected_acceleration_peak"]["value"] <= 2000.0
                      for mapping in hypotheses.values() for value in mapping["directions"].values())
    summary = {
        "classification": "CONDITIONAL_NUMERICAL_SENSITIVITY",
        "episode": args.episode, "train_split_confirmed": True,
        "frame_start": int(frames[0]), "frame_end": int(frames[-1]), "frames": len(frames),
        "median_dt_s": float(np.median(np.diff(timestamps))),
        "settings": {"offset_mm": args.offset_mm, "feasible_residual_mm": args.feasible_residual_mm,
                     "max_nfev": args.max_nfev, "deterministic": True,
                     "seed_family": "recorded clipped; previous witness; J2/J3 ±2° and ±5°"},
        "hypotheses": hypotheses,
        "expansion_gate": {"all_targets_have_witness": all_witnesses, "temporal_thresholds_pass": temporal_ok,
                           "decision": expansion_decision(all_witnesses, temporal_ok)},
        "warnings": [
            "A residual witness proves only finite numerical feasibility in a hypothetical URDF model.",
            "NOT FOUND is inconclusive and is not proof of global unreachability.",
            "Optimizer success is reported separately from residual-based feasibility.",
            "Orientation is unconstrained; collision, torque, contact and physical safety are not evaluated.",
            "No physical mapping, ACT data, URDF edit or hardware operation is authorized.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "per_seed.json").write_text(json.dumps(details, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "split-dataset", "urdf", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--offset-mm", type=float, default=3.0)
    parser.add_argument("--feasible-residual-mm", type=float, default=2.0)
    parser.add_argument("--max-nfev", type=int, default=100)
    args = parser.parse_args()
    if args.frame_start < 0 or args.frame_end < args.frame_start or args.offset_mm <= 0 or args.max_nfev < 1:
        parser.error("invalid arguments")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
