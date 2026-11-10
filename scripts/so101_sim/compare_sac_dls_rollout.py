#!/usr/bin/env python3
"""Compare deterministic SAC and Jacobian-DLS SO-101 kinematic rollouts.

This is a forward-kinematics-only comparison. Joint positions are assigned
directly; it does not simulate dynamics, contact, torque, or a physical robot.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learn_reach_kinematic import KinematicReachEnv, jacobian_reach_action


MAX_STEPS = 60
ACTION_DELTA_RAD = 0.04
SUCCESS_RADIUS_M = 0.025
POLICY_NAMES = ("sac", "jacobian_dls")
CSV_FIELDS = [
    "policy", "goal_seed", "step",
    *(f"joint_{i}" for i in range(1, 6)),
    *(f"action_{i}" for i in range(1, 6)),
    "tcp_x", "tcp_y", "tcp_z",
    "target_x", "target_y", "target_z",
    "tcp_distance_mm", "success",
]


def _record(env, *, policy, goal_seed, step, action):
    joints = np.asarray(env.data.qpos[env.qids], dtype=float).copy()
    tcp = env._tcp()
    target = np.asarray(env.target_tcp, dtype=float).copy()
    distance_mm = float(np.linalg.norm(target - tcp) * 1000.0)
    success = distance_mm <= env.success_radius_m * 1000.0
    values = {
        "policy": policy, "goal_seed": goal_seed, "step": step,
        "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
        "target_x": target[0], "target_y": target[1], "target_z": target[2],
        "tcp_distance_mm": distance_mm, "success": success,
    }
    values.update({f"joint_{i + 1}": value for i, value in enumerate(joints)})
    values.update({f"action_{i + 1}": value
                   for i, value in enumerate(np.asarray(action, dtype=float))})
    return values


def _run_policy(env, model, *, policy, goal_seed):
    obs, _ = env.reset(seed=goal_seed)
    rows = [_record(env, policy=policy, goal_seed=goal_seed, step=0,
                    action=np.zeros(5))]
    for step in range(1, MAX_STEPS + 1):
        if policy == "sac":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = jacobian_reach_action(env)
        obs, _, terminated, truncated, _ = env.step(action)
        rows.append(_record(env, policy=policy, goal_seed=goal_seed,
                            step=step, action=action))
        if terminated or truncated:
            break
    return rows


def collect_rollouts(model_path, goal_seed, goal_joint_delta_rad=0.35):
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required to load the SAC model") from exc

    model = SAC.load(str(model_path), device="cpu")
    rollouts = {}
    initial_states = {}
    for policy in POLICY_NAMES:
        env = KinematicReachEnv(
            max_steps=MAX_STEPS, action_delta_rad=ACTION_DELTA_RAD,
            success_radius_m=SUCCESS_RADIUS_M,
            goal_joint_delta_rad=goal_joint_delta_rad)
        try:
            rows = _run_policy(env, model, policy=policy, goal_seed=goal_seed)
        finally:
            env.close()
        rollouts[policy] = rows
        initial_states[policy] = np.array(
            [rows[0][f"joint_{i}"] for i in range(1, 6)]
            + [rows[0][axis] for axis in ("tcp_x", "tcp_y", "tcp_z")]
            + [rows[0][axis] for axis in ("target_x", "target_y", "target_z")])
    if not np.array_equal(initial_states["sac"], initial_states["jacobian_dls"]):
        raise RuntimeError("SAC and DLS initial conditions do not match exactly")
    return rollouts


def _path_length_m(rows):
    points = np.array([[row["tcp_x"], row["tcp_y"], row["tcp_z"]]
                       for row in rows])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def make_summary(rollouts, goal_seed, goal_joint_delta_rad=0.35):
    summary = {
        "goal_seed": goal_seed,
        "goal_joint_delta_rad": goal_joint_delta_rad,
        "initial_tcp_distance_mm": rollouts["sac"][0]["tcp_distance_mm"],
    }
    for policy in POLICY_NAMES:
        rows = rollouts[policy]
        summary[policy] = {
            "success": bool(rows[-1]["success"]),
            "final_tcp_distance_mm": rows[-1]["tcp_distance_mm"],
            "steps": int(rows[-1]["step"]),
            "tcp_path_length_m": _path_length_m(rows),
        }
    return summary


def _plot_distance(rollouts, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in POLICY_NAMES:
        rows = rollouts[policy]
        ax.plot([r["step"] for r in rows], [r["tcp_distance_mm"] for r in rows],
                marker="o", label=policy)
    ax.axhline(25, color="black", linestyle="--", linewidth=1,
               label="success threshold (25 mm)")
    ax.set(xlabel="Step", ylabel="TCP distance (mm)")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "distance_curve.png", dpi=160)
    plt.close(fig)


def _plot_joints(rollouts, out):
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    for joint, ax in enumerate(axes, start=1):
        for policy in POLICY_NAMES:
            rows = rollouts[policy]
            ax.plot([r["step"] for r in rows],
                    [r[f"joint_{joint}"] for r in rows], label=policy)
        ax.set_ylabel(f"Joint {joint}\n(rad)")
        ax.grid(alpha=.3)
    axes[0].legend()
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(out / "joint_curves.png", dpi=160)
    plt.close(fig)


def _plot_path(rollouts, out, axes_names, filename):
    x_name, y_name = axes_names
    fig, ax = plt.subplots(figsize=(6, 6))
    for policy in POLICY_NAMES:
        rows = rollouts[policy]
        ax.plot([r[x_name] for r in rows], [r[y_name] for r in rows],
                marker="o", label=policy)
    first = rollouts["sac"][0]
    ax.scatter(first[x_name], first[y_name], s=80, marker="s", color="black",
               label="common start", zorder=5)
    target_names = (x_name.replace("tcp_", "target_"),
                    y_name.replace("tcp_", "target_"))
    ax.scatter(first[target_names[0]], first[target_names[1]], s=120, marker="*",
               color="gold", edgecolor="black", label="common target", zorder=5)
    ax.set(xlabel=f"{x_name[-1].upper()} (m)", ylabel=f"{y_name[-1].upper()} (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / filename, dpi=160)
    plt.close(fig)


def save_results(rollouts, summary, out):
    out.mkdir(parents=True)
    with (out / "rollout.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for policy in POLICY_NAMES:
            writer.writerows(rollouts[policy])
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_distance(rollouts, out)
    _plot_joints(rollouts, out)
    _plot_path(rollouts, out, ("tcp_x", "tcp_y"), "tcp_path_xy.png")
    _plot_path(rollouts, out, ("tcp_x", "tcp_z"), "tcp_path_xz.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--goal-seed", type=int, required=True)
    parser.add_argument("--goal-joint-delta-rad", type=float, default=0.35,
                        help="goal sampling range in joint space (default: 0.35)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"SAC model ZIP is required but was not found: {args.model}")
    if args.out.exists():
        parser.error(f"--out must not already exist: {args.out}")
    if not 0 < args.goal_joint_delta_rad <= 1.0:
        parser.error("--goal-joint-delta-rad must be in (0, 1.0]")
    rollouts = collect_rollouts(
        args.model, args.goal_seed, args.goal_joint_delta_rad)
    summary = make_summary(
        rollouts, args.goal_seed, args.goal_joint_delta_rad)
    save_results(rollouts, summary, args.out)
    print(f"Saved comparison to {args.out}")
    for policy in POLICY_NAMES:
        result = summary[policy]
        print(f"{policy}: success={result['success']}, steps={result['steps']}, "
              f"final={result['final_tcp_distance_mm']:.3f} mm, "
              f"path={result['tcp_path_length_m'] * 1000:.3f} mm")


if __name__ == "__main__":
    main()
