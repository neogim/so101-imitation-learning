#!/usr/bin/env python3
"""Export DLS expert transitions for 9D-observation / 5D-action BC.

MuJoCo FK only; no dynamics, contact, hardware, SAC retraining or oracle q_goal.
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


def generate(out: Path, *, episodes: int, seed_start: int, radius_mm: float,
             deltas: tuple[float, ...], max_steps: int = 60,
             min_initial_distance_mm: float = 0.0,
             max_initial_distance_mm: float | None = None,
             max_goal_attempts: int = 500) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    if episodes <= 0 or radius_mm <= 0 or max_steps <= 0:
        raise ValueError("episodes, radius_mm and max_steps must be positive")
    if not deltas or any(not 0 < d <= 1 for d in deltas):
        raise ValueError("goal deltas must lie in (0, 1] rad")
    if min_initial_distance_mm < 0 or not np.isfinite(min_initial_distance_mm):
        raise ValueError("min initial distance must be finite and non-negative")
    if (max_initial_distance_mm is not None and
            (not np.isfinite(max_initial_distance_mm) or
             max_initial_distance_mm <= min_initial_distance_mm)):
        raise ValueError("max initial distance must exceed min initial distance")
    if max_goal_attempts <= 0:
        raise ValueError("max_goal_attempts must be positive")
    out.mkdir(parents=True)
    observations, actions, next_obs, rewards, terminated, truncated = [], [], [], [], [], []
    episode_ids, goal_seeds, goal_ranges, step_ids = [], [], [], []
    episode_summary, trajectories = [], {}

    for delta in deltas:
        env = KinematicReachEnv(max_steps=max_steps, action_delta_rad=0.04,
                                goal_joint_delta_rad=delta, reward_scale=10.0,
                                success_radius_m=radius_mm / 1000)
        try:
            for ep in range(episodes):
                # Keep train/validation and the 300-goal evaluation seeds disjoint.
                # Deterministic rejection sampling on INITIAL Cartesian
                # distance; disjoint seed ranges across episodes, even when
                # a difficult distance band needs multiple attempts.
                for attempt in range(max_goal_attempts):
                    goal_seed = seed_start + ep + attempt * episodes
                    obs, _ = env.reset(seed=goal_seed)
                    start_dist = float(obs[-1]) * 1000
                    if (start_dist >= min_initial_distance_mm and
                            (max_initial_distance_mm is None or
                             start_dist <= max_initial_distance_mm)):
                        break
                else:
                    raise RuntimeError(
                        f"Could not sample a goal in [{min_initial_distance_mm}, "
                        f"{max_initial_distance_mm or 'inf'}] mm "
                        f"for delta={delta}, episode={ep} after "
                        f"{max_goal_attempts} attempts. Lower the minimum "
                        "distance or increase the goal range.")
                episode_id = len(episode_summary)
                trace = []
                for step in range(max_steps):
                    action = np.asarray(jacobian_reach_action(env), dtype=np.float32)
                    # Action is NORMALIZED [-1,1], not raw radians.
                    new_obs, reward, done, timeout, info = env.step(action)
                    observations.append(obs.copy())
                    actions.append(action.copy())
                    next_obs.append(new_obs.copy())
                    rewards.append(float(reward))
                    terminated.append(bool(done))
                    truncated.append(bool(timeout))
                    episode_ids.append(episode_id)
                    goal_seeds.append(goal_seed)
                    goal_ranges.append(delta)
                    step_ids.append(step)
                    trace.append(float(info["distance_m"]) * 1000)
                    obs = new_obs
                    if done or timeout:
                        break
                if not trace:
                    raise RuntimeError("Empty DLS rollout")
                episode_summary.append({
                    "episode_id": episode_id, "goal_seed": goal_seed,
                    "sampling_attempts": attempt + 1,
                    "goal_joint_delta_rad": delta, "success": bool(done),
                    "initial_tcp_distance_mm": start_dist,
                    "final_tcp_distance_mm": trace[-1],
                    "steps": len(trace),
                })
                if ep == 0:
                    trajectories[f"{delta:.2f}"] = trace
        finally:
            env.close()

    obs = np.asarray(observations, np.float32)
    act = np.asarray(actions, np.float32)
    next_obs = np.asarray(next_obs, np.float32)
    ids = np.asarray(episode_ids, np.int32)
    rewards_np = np.asarray(rewards, np.float32)
    term = np.asarray(terminated, bool)
    trunc = np.asarray(truncated, bool)
    if obs.shape != (len(ids), 9) or act.shape != (len(ids), 5):
        raise AssertionError("Unexpected observation/action dimensions")
    if next_obs.shape != obs.shape or not np.isfinite(obs).all() or not np.isfinite(act).all():
        raise AssertionError("Invalid teacher samples")
    if np.any(np.abs(act) > 1.00001):
        raise AssertionError("Actions must be normalized to [-1,1]")
    if len(set((s["goal_joint_delta_rad"], s["goal_seed"]) for s in episode_summary)) != len(episode_summary):
        raise AssertionError("Duplicate goal")

    np.savez_compressed(out / "dls_transitions.npz",
                        observations=obs, actions=act,
                        next_observations=next_obs, rewards=rewards_np,
                        terminated=term, truncated=trunc, episode_ids=ids,
                        evaluation_seeds=np.asarray(goal_seeds, np.int32),
                        goal_joint_delta_rad=np.asarray(goal_ranges, np.float32),
                        step=np.asarray(step_ids, np.int32))
    with (out / "episodes.csv").open("x", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(episode_summary[0]))
        writer.writeheader()
        writer.writerows(episode_summary)
    by_range = {}
    for delta in deltas:
        rows = [r for r in episode_summary if r["goal_joint_delta_rad"] == delta]
        by_range[f"{delta:.2f}"] = {
            "episodes": len(rows), "success": sum(r["success"] for r in rows),
            "mean_initial_mm": float(np.mean([r["initial_tcp_distance_mm"] for r in rows])),
            "mean_final_mm": float(np.mean([r["final_tcp_distance_mm"] for r in rows])),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
        }
    summary = {
        "status": "KINEMATIC_ONLY_DLS_TEACHER_NO_HARDWARE",
        "teacher": "jacobian_reach_action", "damping": 0.02, "gain": 0.75,
        "observation": "[q1..q5, goal_xyz - tcp_xyz, distance_m] (9D)",
        "action": "normalized bounded joint increment in [-1,1]^5; environment multiplies by 0.04rad",
        "reward": "-10*distance_m", "success_radius_mm": radius_mm,
        "max_steps": max_steps, "episodes_per_range": episodes,
        "goal_seed_start": seed_start,
        "min_initial_distance_mm": min_initial_distance_mm,
        "max_initial_distance_mm": max_initial_distance_mm,
        "max_goal_attempts": max_goal_attempts,
        "total_episodes": len(episode_summary), "transitions": len(ids),
        "by_range": by_range,
        "train_validation_split": "SPLIT BY EPISODE ID, not transitions",
        "warning": "No SAC/BC evaluation yet; teacher dataset is not real-world manipulation.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (out / "report.md").open("x", encoding="utf-8") as fp:
        fp.write("# DLS expert dataset for BC\n\n")
        fp.write(f"- {len(episode_summary)} episodes / {len(ids)} transitions\n")
        fp.write(f"- Success radius: {radius_mm} mm, reward scale: 10\n")
        fp.write("- No physical robot, dynamics, collision or SAC training\n\n")
        fp.write("| Goal range | Success | Mean initial error | Mean final error | Mean steps |\n")
        fp.write("| --- | ---: | ---: | ---: | ---: |\n")
        for key, row in by_range.items():
            fp.write(f"| ±{key}rad | {row['success']}/{row['episodes']} | "
                     f"{row['mean_initial_mm']:.2f}mm | {row['mean_final_mm']:.2f}mm | "
                     f"{row['mean_steps']:.2f} |\n")
    keys = list(by_range)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(keys, [100 * by_range[k]["success"] / by_range[k]["episodes"] for k in keys])
    ax.set(xlabel="Goal joint range ±rad", ylabel="DLS success rate (%)", ylim=(0, 105))
    fig.tight_layout(); fig.savefig(out / "success_rate.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for delta in deltas:
        rows = [r["final_tcp_distance_mm"] for r in episode_summary if r["goal_joint_delta_rad"] == delta]
        ax.hist(rows, bins=20, alpha=0.4, label=f"±{delta:.2f}")
    ax.set(xlabel="Final TCP error (mm)", ylabel="Episodes"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "final_error_hist.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for delta in deltas:
        rows = [r["steps"] for r in episode_summary if r["goal_joint_delta_rad"] == delta]
        ax.hist(rows, bins=range(1, max_steps + 3), alpha=0.4, label=f"±{delta:.2f}")
    ax.set(xlabel="Episode steps", ylabel="Episodes"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "episode_steps_hist.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, values in trajectories.items():
        ax.plot(range(1, len(values) + 1), values, label=f"±{key} rad")
    ax.axhline(radius_mm, linestyle="--", color="gray", label="Success radius")
    ax.set(xlabel="DLS step", ylabel="TCP error (mm)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "representative_error_curves.png", dpi=160); plt.close(fig)
    print(f"Saved {len(episode_summary)} episodes, {len(ids)} transitions to {out}")
    for key, item in by_range.items():
        print(f"±{key}rad: {item['success']}/{item['episodes']} success")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes-per-range", type=int, default=100)
    p.add_argument("--seed-start", type=int, default=2000001)
    p.add_argument("--success-mm", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--goal-deltas", type=float, nargs="+", default=[.35, .50, .70])
    p.add_argument("--min-initial-distance-mm", type=float, default=0.0,
                   help="only accept goals at least this far from start TCP")
    p.add_argument("--max-initial-distance-mm", type=float, default=None,
                   help="optional upper bound for start-to-goal TCP distance")
    p.add_argument("--max-goal-attempts", type=int, default=500,
                   help="deterministic bounded rejection-sampling attempts per episode")
    args = p.parse_args()
    generate(args.out, episodes=args.episodes_per_range, seed_start=args.seed_start,
             radius_mm=args.success_mm, deltas=tuple(args.goal_deltas),
             max_steps=args.max_steps,
             min_initial_distance_mm=args.min_initial_distance_mm,
             max_initial_distance_mm=args.max_initial_distance_mm,
             max_goal_attempts=args.max_goal_attempts)


if __name__ == "__main__":
    main()
