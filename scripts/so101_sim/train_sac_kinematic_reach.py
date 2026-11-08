#!/usr/bin/env python3
"""One small, reproducible SAC learning experiment on KINEMATIC SO-101 Reach.

Compare (1) untrained SAC, (2) trained SAC and (3) Jacobian-DLS using the
SAME evaluation goals. The agent sees only the existing 9D observations.
No hidden target_joint is passed to SAC. No physics, PD, contact model,
hardware control, or demonstration data is involved.

pip install "stable-baselines3>=2.4,<3"
python scripts/so101_sim/train_sac_kinematic_reach.py --out outputs/sac-quick
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from learn_reach_kinematic import KinematicReachEnv, jacobian_reach_action


def evaluate(policy, *, name, episodes=10, seed=1001, max_steps=60):
    """Evaluate deterministic policy using identical held-out goal seeds."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = KinematicReachEnv(max_steps=max_steps)
    rows = []
    try:
        for episode in range(episodes):
            goal_seed = seed + episode
            obs, _ = env.reset(seed=goal_seed)
            initial_mm = float(obs[-1]) * 1000
            for step in range(max_steps):
                if name == "jacobian_dls":
                    action = jacobian_reach_action(env)
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            rows.append({
                "policy": name, "episode": episode, "goal_seed": goal_seed,
                "initial_tcp_mm": round(initial_mm, 3),
                "final_tcp_mm": round(float(info["distance_m"]) * 1000, 3),
                "steps": step + 1,
                "success": bool(info["is_success"]),
            })
    finally:
        env.close()
    return rows


def summarize(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["policy"], []).append(r)
    return {
        name: {
            "success_count": sum(r["success"] for r in part),
            "episodes": len(part),
            "success_rate": sum(r["success"] for r in part) / len(part),
            "mean_final_tcp_mm": round(float(np.mean(
                [r["final_tcp_mm"] for r in part])), 3),
            "mean_initial_tcp_mm": round(float(np.mean(
                [r["initial_tcp_mm"] for r in part])), 3),
        } for name, part in grouped.items()
    }


def run(*, timesteps=3000, eval_episodes=10, train_seed=101,
        eval_seed=1001, max_steps=60):
    if timesteps < 1 or eval_episodes < 1 or max_steps < 1:
        raise ValueError("timesteps/eval_episodes/max_steps must be positive")
    if eval_seed <= train_seed + timesteps:
        # This is a heuristic against accidentally using nearby goal seeds,
        # not a rigorous guarantee of held-out goal distribution.
        raise ValueError("Set eval_seed > train_seed + timesteps")
    try:
        import torch
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError(
            "Install optional SAC dependency: "
            "python -m pip install 'stable-baselines3>=2.4,<3'") from exc
    torch.set_num_threads(1)  # small MLPs on Mac/CPU; avoid thread overhead
    train_env = KinematicReachEnv(seed=train_seed, max_steps=max_steps)
    try:
        start = time.perf_counter()
        model = SAC(
            "MlpPolicy", train_env, seed=train_seed, device="cpu",
            learning_rate=3e-4, buffer_size=max(10000, timesteps),
            learning_starts=min(100, max(1, timesteps // 4)),
            batch_size=64, train_freq=1, gradient_steps=1,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=0,
        )
        rows = evaluate(model, name="sac_before", episodes=eval_episodes,
                        seed=eval_seed, max_steps=max_steps)
        model.learn(total_timesteps=timesteps)
        train_seconds = time.perf_counter() - start
        rows += evaluate(model, name="sac_after", episodes=eval_episodes,
                         seed=eval_seed, max_steps=max_steps)
        rows += evaluate(None, name="jacobian_dls", episodes=eval_episodes,
                         seed=eval_seed, max_steps=max_steps)
        return model, {
            "status": "KINEMATIC_REACH_ONLY_NO_PHYSICS_NO_HARDWARE",
            "algorithm": "Stable-Baselines3 SAC MlpPolicy",
            "train_timesteps": timesteps,
            "train_seed": train_seed,
            "evaluation_goal_seeds": [eval_seed+i for i in range(eval_episodes)],
            "max_steps_per_episode": max_steps,
            "success_threshold_mm": 25,
            "training_and_model_setup_seconds": round(train_seconds, 3),
            "policy_device": "cpu",
            "network": [64, 64],
            "rows": rows, "summary": summarize(rows),
            "notes": [
                "SAC sees 9D observation, not the hidden correct target joints.",
                "Evaluation uses deterministic actions and fixed identical goal seeds.",
                "Evaluation goals are sampled from the same task distribution, not a distinct task.",
                "Jacobian DLS is analytic control, not a learned policy.",
                "Training duration/seed is one small pilot; SAC may fail to learn.",
                "No physical dynamics or collision checking; NOT safe for hardware transfer.",
            ],
        }
    finally:
        train_env.close()


def save(model, report, out):
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    model.save(str(out / "sac_kinematic_reach"))
    with (out / "evaluation.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timesteps", type=int, default=3000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--train-seed", type=int, default=101)
    p.add_argument("--eval-seed", type=int, default=10001)
    args = p.parse_args()
    if args.out.exists():
        p.error("--out must be a new directory")
    model, report = run(
        timesteps=args.timesteps, eval_episodes=args.eval_episodes,
        train_seed=args.train_seed, eval_seed=args.eval_seed)
    save(model, report, args.out)
    print("Saved", args.out / "summary.json")
    print("Training + model setup (seconds):",
          report["training_and_model_setup_seconds"])
    for name, metric in report["summary"].items():
        print(f"{name}: {metric['success_count']}/{metric['episodes']} "
              f"({metric['success_rate']:.0%}) success, "
              f"mean final TCP={metric['mean_final_tcp_mm']:.1f}mm")
    print("One small pilot only; no contact physics, no real robot control.")


if __name__ == "__main__":
    main()
