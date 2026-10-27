#!/usr/bin/env python3
"""Generate offline IK waypoint candidates from recorded TRAIN trajectories.

Produces counterfactual joint waypoints, NOT synchronized observations or ACT
training episodes. No hardware, no motor calls, no collision validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import placo
from scipy.optimize import least_squares

from evaluate_ik_replay_offline import rows_for
from export_cartesian_action_sidecar import JOINT_NAMES


def joint_limits(urdf: Path):
    root = ET.parse(urdf).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    limits = []
    for name in JOINT_NAMES:
        lim = joints[name].find("limit")
        if lim is None:
            raise ValueError(f"Missing URDF joint limit: {name}")
        limits.append((float(lim.get("lower")), float(lim.get("upper"))))
    a = np.asarray(limits)
    if not np.isfinite(a).all() or np.any(a[:, 1] <= a[:, 0]):
        raise ValueError("Invalid joint limits")
    return a[:, 0], a[:, 1]


def run(args):
    source = args.dataset.resolve()
    destination = args.output.resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("Output must be new and outside the original dataset")
    if args.stride < 1 or args.max_frames < 2 or args.max_nfev < 1:
        raise ValueError("Invalid arguments")
    if not all(np.isfinite([args.offset_mm, args.max_position_error_mm, args.max_joint_step_deg])):
        raise ValueError("Invalid numeric thresholds")
    if args.offset_mm <= 0 or args.max_position_error_mm <= 0 or args.max_joint_step_deg <= 0:
        raise ValueError("Thresholds must be positive")
    info = json.loads((source / "meta/info.json").read_text())
    if info["features"]["action"]["shape"] != [6]:
        raise ValueError("Require original 6D joint-action data")
    split = json.loads((args.split_dataset / "episode_split.json").read_text())
    train = set(map(int, split["train_episode_indices"]))
    val = set(map(int, split["validation_episode_indices"]))
    if train & val or args.episode not in train:
        raise ValueError("Episode must belong to training split only")
    rows = rows_for(source, args.episode, args.stride, args.max_frames)
    lower, upper = joint_limits(args.urdf)
    robot = placo.RobotWrapper(str(args.urdf.resolve()))
    if not set(JOINT_NAMES).issubset(set(robot.joint_names())):
        raise ValueError("URDF joint names incompatible")

    def fk(q):
        for name, angle in zip(JOINT_NAMES, q):
            robot.set_joint(name, float(angle))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"), dtype=float).copy()

    # Only perturb target XYZ. Existing rotation is retained for documentation
    # but the solver does not constrain orientation (5-DOF arm).
    directions = np.eye(3)
    cases = [(f"{name}{sign}", directions[i] * sign)
             for i, name in enumerate("xyz") for sign in (1, -1)]
    results = []
    for label, direction in cases:
        prev = None
        for row in rows:
            recorded = np.deg2rad(np.asarray(row["action"][:5], dtype=float))
            if not np.isfinite(recorded).all():
                raise ValueError("Nonfinite recorded action")
            target = fk(recorded)[:3, 3] + direction * (args.offset_mm / 1000.0)
            seed = np.clip(recorded if prev is None else prev, lower + 1e-9, upper - 1e-9)
            fit = least_squares(lambda q: (fk(q)[:3, 3] - target) * 1000.0,
                                seed, bounds=(lower, upper), max_nfev=args.max_nfev)
            angle = fit.x
            err = float(np.linalg.norm((fk(angle)[:3, 3] - target) * 1000))
            step = None if prev is None else float(np.max(np.abs(np.rad2deg(angle - prev))))
            accepted = bool(fit.success and err <= args.max_position_error_mm
                            and (step is None or step <= args.max_joint_step_deg))
            results.append({
                "variant": label, "episode": args.episode,
                "frame": int(row["frame_index"]), "optimizer_success": bool(fit.success),
                "position_error_mm": err, "sampled_joint_step_deg": step,
                "passes_numerical_screen": accepted,
                "target_xyz_m": target.tolist(),
                "ik_arm_deg": np.rad2deg(angle).tolist(),
                "original_arm_deg": np.rad2deg(recorded).tolist(),
                "original_gripper": float(row["action"][5]),
            })
            # Preserve continuity even for rejected candidates; do not imply
            # accepted continuous trajectory if any waypoint fails.
            prev = angle.copy()
    destination.mkdir(parents=True)
    (destination / "ik_waypoints.json").write_text(json.dumps(results, indent=2) + "\n")
    with (destination / "ik_waypoints.csv").open("w", newline="") as fh:
        fields = ["variant", "episode", "frame", "optimizer_success",
                  "position_error_mm", "sampled_joint_step_deg", "passes_numerical_screen"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: entry[k] for k in fields} for entry in results])
    summary = {
        "source_episode": args.episode, "sampled_frames": len(rows),
        "variants": [c[0] for c in cases], "offset_mm": args.offset_mm,
        "samples_total": len(results),
        "samples_passing_numerical_screen": sum(r["passes_numerical_screen"] for r in results),
        "all_waypoints_pass_by_variant": {
            name: all(r["passes_numerical_screen"] for r in results if r["variant"] == name)
            for name, _ in cases
        },
        "important_limitations": [
            "OFFLINE ONLY: no robot commands; candidates are not approved for hardware replay",
            "Not an ACT training dataset: generated counterfactual waypoints have no corresponding camera observations",
            "URDF joint limits may not match calibrated physical joint limits",
            "No collision, contact, dynamic or complete swept-path checking",
            "Sampling stride means step check is NOT a per-control-cycle velocity limit",
            "Position-only IK; 5-DOF wrist orientation is unconstrained",
            "TCP/tool coordinate and camera extrinsics are not physically verified",
        ],
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split-dataset", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--offset-mm", type=float, default=3.0)
    parser.add_argument("--max-position-error-mm", type=float, default=2.0)
    parser.add_argument("--max-joint-step-deg", type=float, default=25.0)
    parser.add_argument("--max-nfev", type=int, default=150)
    run(parser.parse_args())
