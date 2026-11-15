#!/usr/bin/env python3
"""Trace real MuJoCo PD substeps for an oracle FK goal, with CSV/plots.

Offline diagnostic, not SAC, not robot hardware. Uses exactly the same
SO101ReachEnv.step controller as RL via optional trace_hook instrumentation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

import real_to_sim as r2s
from so101_reach_env import SO101ReachEnv


def run_trace(*, seed=101, episode=0, out: Path,
              near_goal_joint_rad=None, max_steps=150, frame_skip=10,
              kp=12., kd=1.1, max_joint_torque=3.):
    """Record one episode at every physics substep; caller chooses new directory."""
    if episode < 0 or max_steps <= 0 or frame_skip <= 0 or out.exists():
        raise ValueError("episode>=0, steps>0, frame_skip>0, and new --out required")
    env = SO101ReachEnv(seed=seed, max_steps=max_steps, frame_skip=frame_skip,
                        goal_joint_delta_rad=near_goal_joint_rad,
                        kp=kp, kd=kd, max_joint_torque=max_joint_torque)
    trace = []
    try:
        _, initial = env.reset(seed=seed + episode)
        target = env.goal_joint_angles.copy()
        initial_joint = env.data.qpos[env.qids].copy()
        env.targets = target.copy()  # Direct oracle target: bypass RL actions.
        for _ in range(max_steps):
            _, _, terminated, truncated, info = env.step(
                np.zeros(5), trace_hook=trace.append)
            if terminated or truncated:
                break
        if not trace:
            raise RuntimeError("No physics substeps recorded")
        contact_counter = Counter(
            tuple(sorted(pair)) for row in trace for pair in row["contact_pairs"])
        saturated = np.array([row["torque_saturated"] for row in trace], dtype=bool)
        positions = np.array([row["joint_position_rad"] for row in trace])
        err = np.abs(positions - target)
        record = {
            "status": "SIM_PD_TRACE_ONLY_NOT_RL_OR_ACTUAL_ROBOT",
            "seed": int(seed), "episode": int(episode),
            "goal_distribution": "near" if near_goal_joint_rad is not None else "full",
            "near_goal_joint_rad": near_goal_joint_rad,
            "controller": {
                "kp": kp, "kd": kd, "max_joint_torque_nm": max_joint_torque,
                "frame_skip": frame_skip, "mj_timestep_s": float(env.model.opt.timestep),
                "max_rl_steps": max_steps,
            },
            "joint_names": list(r2s.JOINTS),
            "initial_joint_rad": initial_joint.tolist(),
            "goal_joint_rad": target.tolist(),
            "goal_tcp_proxy_base_m": initial["goal_base_m"],
            "steps": int(info["episode_steps"]),
            "physics_substeps": len(trace),
            "success": bool(info["is_success"]),
            "final_tcp_distance_m": float(info["distance_m"]),
            "final_joint_error_l2_rad": float(np.linalg.norm(positions[-1] - target)),
            "max_joint_error_rad": np.max(err, axis=0).tolist(),
            "saturated_substep_fraction_per_joint": saturated.mean(axis=0).tolist(),
            "any_saturation_fraction": float(saturated.any(axis=1).mean()),
            "substeps_with_contacts_fraction": float(np.mean([
                len(row["contact_pairs"]) > 0 for row in trace])),
            "contact_pairs_by_substep_count": [
                {"pair": list(pair), "substeps": count}
                for pair, count in contact_counter.most_common()
            ],
            "notes": [
                "Contact geometry pairs are detected, not proof of harmful collision.",
                "Torque saturation is measured on raw PD+bias force before clipping.",
                "FK goal is kinematically reachable; collision-free path not guaranteed.",
                "TCP is uncalibrated gripper_link origin; no grasp, no hardware.",
            ],
        }
        out.mkdir(parents=True, exist_ok=False)
        with (out / "pd_trace.csv").open("x", newline="", encoding="utf-8") as f:
            cols = (["time_s", "tcp_distance_m", "contact_count", "contact_pairs"] +
                    [f"{j}_{s}" for j in r2s.JOINTS
                     for s in ("target_rad", "actual_rad", "velocity_rad_s",
                               "torque_command_nm", "torque_applied_nm", "saturated")])
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for item in trace:
                row = {
                    "time_s": item["time_s"],
                    "tcp_distance_m": item["tcp_distance_m"],
                    "contact_count": len(item["contact_pairs"]),
                    "contact_pairs": ";".join(
                        f"{a}|{b}" for a, b in item["contact_pairs"]),
                }
                for index, j in enumerate(r2s.JOINTS):
                    for key, field in [
                        ("joint_target_rad", "target_rad"),
                        ("joint_position_rad", "actual_rad"),
                        ("joint_velocity_rad_s", "velocity_rad_s"),
                        ("torque_command_nm", "torque_command_nm"),
                        ("torque_applied_nm", "torque_applied_nm"),
                        ("torque_saturated", "saturated"),
                    ]:
                        value = item[key][index]
                        row[f"{j}_{field}"] = (
                            int(value) if field == "saturated" else float(value))
                writer.writerow(row)
        (out / "summary.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_plots(trace, out, max_joint_torque)
        return record
    finally:
        env.close()


def write_plots(trace, out: Path, torque_limit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.array([r["time_s"] for r in trace])
    target = np.array([r["joint_target_rad"] for r in trace])
    actual = np.array([r["joint_position_rad"] for r in trace])
    speed = np.array([r["joint_velocity_rad_s"] for r in trace])
    raw = np.array([r["torque_command_nm"] for r in trace])
    applied = np.array([r["torque_applied_nm"] for r in trace])
    distances = np.array([r["tcp_distance_m"] for r in trace])
    contacts = np.array([len(r["contact_pairs"]) for r in trace])

    def plot_joints(filename, title, measured, ylabel, desired=None, upper=None):
        fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(time, measured[:, i], label="actual" if desired is not None else "value")
            if desired is not None:
                ax.plot(time, desired[:, i], "--", label="target", alpha=.8)
            if upper is not None:
                ax.axhline(upper, color="gray", linestyle=":", label="torque limit")
                ax.axhline(-upper, color="gray", linestyle=":")
            ax.set_ylabel(r2s.JOINTS[i] + "\n" + ylabel)
            ax.grid(True, alpha=.3)
            if i == 0:
                ax.legend(loc="best")
        axes[-1].set_xlabel("MuJoCo time [s]")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=130)
        plt.close(fig)

    plot_joints("joint_angles.png", "Joint target vs actual",
                actual, "rad", desired=target)
    # For each joint, the raw (possibly saturated) command vs actual applied torque.
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time, raw[:, i], label="raw PD+bias", alpha=.7)
        ax.plot(time, applied[:, i], label="applied (clipped)")
        ax.axhline(torque_limit, color="gray", linestyle=":", label="limit")
        ax.axhline(-torque_limit, color="gray", linestyle=":")
        ax.set_ylabel(r2s.JOINTS[i] + "\nNm")
        ax.grid(True, alpha=.3)
        if i == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("MuJoCo time [s]")
    fig.suptitle("PD generalized torque demand vs applied")
    fig.tight_layout()
    fig.savefig(out / "joint_torques.png", dpi=130)
    plt.close(fig)

    plot_joints("joint_velocities.png", "Joint velocities", speed, "rad/s")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, distances)
    axes[0].axhline(.025, color="gray", linestyle="--", label="25 mm reach threshold")
    axes[0].set_ylabel("TCP proxy distance [m]")
    axes[0].legend()
    axes[1].step(time, contacts, where="post")
    axes[1].set_ylabel("Active contacts")
    axes[1].set_xlabel("MuJoCo time [s]")
    for ax in axes:
        ax.grid(True, alpha=.3)
    fig.suptitle("Reach distance and contacts")
    fig.tight_layout()
    fig.savefig(out / "tcp_distance_contacts.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--frame-skip", type=int, default=10)
    p.add_argument("--near-goal-joint-rad", type=float, default=None)
    p.add_argument("--kp", type=float, default=12.)
    p.add_argument("--kd", type=float, default=1.1)
    p.add_argument("--max-joint-torque", type=float, default=3.)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    summary = run_trace(
        seed=args.seed, episode=args.episode, out=args.out,
        near_goal_joint_rad=args.near_goal_joint_rad,
        max_steps=args.max_steps, frame_skip=args.frame_skip,
        kp=args.kp, kd=args.kd, max_joint_torque=args.max_joint_torque)
    print(f"Saved diagnostic: {args.out}")
    print(f"reach={summary['success']} "
          f"distance={summary['final_tcp_distance_m']:.4f}m "
          f"joint_error={summary['final_joint_error_l2_rad']:.4f}rad")
    print("Per-joint saturation fractions:",
          dict(zip(summary["joint_names"],
                   [round(x, 3) for x in summary["saturated_substep_fraction_per_joint"]])))
    print("Substeps with contacts:", round(summary["substeps_with_contacts_fraction"], 3))


if __name__ == "__main__":
    main()
