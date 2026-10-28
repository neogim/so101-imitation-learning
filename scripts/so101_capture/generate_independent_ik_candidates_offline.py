#!/usr/bin/env python3
"""Independent-seed, offline-only SO-101 IK candidate generator.

Never connects to hardware. Writes candidate diagnostics to a NEW output directory.
Use the existing select_ik_trajectory_dp_offline.py for greedy/DP comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import placo
from scipy.optimize import least_squares

from evaluate_mapping_hypotheses_offline import (
    DIRECTIONS, HYPOTHESES, load_frames, motor_to_urdf_degrees,
    rotation_geodesic_degrees, urdf_limits_in_motor_degrees, urdf_to_motor_degrees,
)
from export_cartesian_action_sidecar import JOINT_NAMES
from generate_offset_ik_offline import joint_limits


def independent_seeds(recorded_motor: np.ndarray, signs: np.ndarray,
                      lower: np.ndarray, upper: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """No previous-frame seed, no seeded RNG, deterministic, same family at every frame."""
    proposals = [("recorded_clipped", recorded_motor.copy())]
    for joint, j in (("shoulder", 1), ("elbow", 2)):
        for delta in (-5.0, -2.0, 2.0, 5.0):
            changed = recorded_motor.copy()
            changed[j] += delta
            proposals.append((f"{joint}_{delta:+g}deg", changed))
    seeds, seen = [], set()
    for label, motor in proposals:
        q = np.clip(np.deg2rad(motor_to_urdf_degrees(motor, signs)),
                    lower + 1e-9, upper - 1e-9)
        key = tuple(np.round(q, 12))
        if key not in seen:
            seeds.append((label, q))
            seen.add(key)
    return seeds


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise ValueError("Output already exists")
    if args.dataset.resolve() in args.output.resolve().parents:
        raise ValueError("Output must be outside source dataset")
    split = json.loads((args.split_dataset / "episode_split.json").read_text())
    train = set(split["train_episode_indices"])
    valid = set(split["validation_episode_indices"])
    if train & valid or args.episode not in train:
        raise ValueError("Episode must belong only to fixed train split")
    rows = load_frames(args.dataset, args.episode, args.frame_start, args.frame_end)
    frames = np.array([int(x["frame_index"]) for x in rows])
    timestamps = np.array([float(x["timestamp"]) for x in rows])
    actions = np.array([x["action"][:5] for x in rows], dtype=float)
    if not np.isfinite(actions).all() or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Invalid Action / timestamps")
    if not np.allclose(np.diff(timestamps), 0.05, atol=2e-6):
        raise ValueError("Expected recorded 20 Hz timestamps")
    lower, upper = joint_limits(args.urdf)
    robot = placo.RobotWrapper(str(args.urdf.resolve()))

    def fk(q: np.ndarray) -> np.ndarray:
        for name, angle in zip(JOINT_NAMES, q):
            robot.set_joint(name, float(angle))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"), dtype=float).copy()

    details, summary = [], {}
    for mapping, signs in HYPOTHESES.items():
        reference = np.array([fk(np.deg2rad(motor_to_urdf_degrees(motor, signs))) for motor in actions])
        motor_low, motor_high = urdf_limits_in_motor_degrees(lower, upper, signs)
        summary[mapping] = {}
        for direction, axis in DIRECTIONS.items():
            targets_with_witness = 0
            attempts = 0
            for i, frame in enumerate(frames):
                target = reference[i, :3, 3] + axis * args.offset_mm / 1000.0
                witnessed = False
                for label, seed in independent_seeds(actions[i], signs, lower, upper):
                    fit = least_squares(
                        lambda q: (fk(q)[:3, 3] - target) * 1000.0,
                        seed, bounds=(lower, upper), max_nfev=args.max_nfev)
                    transform = fk(fit.x)
                    motor = urdf_to_motor_degrees(np.rad2deg(fit.x), signs)
                    residual = float(np.linalg.norm((transform[:3, 3] - target) * 1000.0))
                    witness = residual <= args.feasible_residual_mm
                    witnessed |= witness
                    active = [name for j, name in enumerate(JOINT_NAMES)
                              if abs(motor[j] - motor_low[j]) <= 1e-3
                              or abs(motor[j] - motor_high[j]) <= 1e-3]
                    details.append({
                        "mapping": mapping, "direction": direction, "frame": int(frame),
                        "seed": label, "optimizer_success": bool(fit.success),
                        "position_residual_mm": residual, "feasible_witness": bool(witness),
                        "active_bound_joints": active,
                        "motor_deviation_deg": float(np.max(np.abs(motor - actions[i]))),
                        "orientation_change_deg": rotation_geodesic_degrees(
                            reference[i, :3, :3], transform[:3, :3]),
                        "candidate_motor_deg": motor.tolist(),
                    })
                    attempts += 1
                targets_with_witness += int(witnessed)
            summary[mapping][direction] = {
                "witness_targets": targets_with_witness,
                "not_found_targets": len(frames) - targets_with_witness,
                "seed_attempts": attempts,
            }
    result = {
        "classification": "CONDITIONAL_NUMERICAL_SENSITIVITY",
        "seed_family": "independent recorded-clipped; shoulder/elbow +/-2,+/-5 degrees; NO previous-witness seed",
        "episode": args.episode, "frames": [int(frames[0]), int(frames[-1])],
        "hypotheses": summary,
        "warnings": [
            "No physical sign/offset mapping validated.",
            "Finite multistart search cannot prove a target unreachable.",
            "All position residuals are hypothetical URDF-world numerics.",
            "No ACT training, hardware playback, collision or torque checks.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "per_seed.json").write_text(json.dumps(details, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "split-dataset", "urdf", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame-start", type=int, default=565)
    p.add_argument("--frame-end", type=int, default=595)
    p.add_argument("--offset-mm", type=float, default=3.0)
    p.add_argument("--feasible-residual-mm", type=float, default=2.0)
    p.add_argument("--max-nfev", type=int, default=100)
    a = p.parse_args()
    if a.frame_start < 0 or a.frame_end < a.frame_start or a.offset_mm <= 0 or a.max_nfev < 1:
        p.error("Invalid arguments")
    print(json.dumps(run(a), indent=2))


if __name__ == "__main__":
    main()
