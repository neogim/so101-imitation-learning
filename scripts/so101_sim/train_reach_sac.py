#!/usr/bin/env python3
"""Small offline SAC Reach training experiment; never opens robot devices.

Requires optional: pip install "stable-baselines3>=2.4,<3" tensorboard
This script is a training entry point, not evidence of successful learning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from so101_reach_env import SO101ReachEnv


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback
    import real_to_sim as r2s

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    args = parser.parse_args()
    if args.steps <= 0 or args.out.exists():
        raise ValueError("steps must be positive and output directory must be new")
    train = Monitor(SO101ReachEnv(urdf=args.urdf, seed=args.seed))
    evaluation = Monitor(SO101ReachEnv(urdf=args.urdf, seed=args.seed + 1000))
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        model = SAC("MlpPolicy", train, seed=args.seed, verbose=1,
                    tensorboard_log=str(args.out / "tb"),
                    learning_starts=1000, batch_size=128, buffer_size=50000)
        eval_callback = EvalCallback(
            evaluation, best_model_save_path=str(args.out / "best"),
            log_path=str(args.out / "evaluation"), eval_freq=5000,
            n_eval_episodes=10, deterministic=True, verbose=1)
        model.learn(total_timesteps=args.steps, callback=eval_callback)
        model.save(str(args.out / "sac_reach_final"))
        # Separate final deterministic evaluation. Training is not proof of success.
        records = []
        for episode in range(10):
            obs, _ = evaluation.reset(seed=args.seed + 2000 + episode)
            for step in range(evaluation.unwrapped.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = evaluation.step(action)
                if terminated or truncated:
                    break
            records.append({"episode": episode, "success": bool(info["is_success"]),
                            "distance_m": float(info["distance_m"]), "steps": step + 1})
        summary = {
            "status": "SIMULATED_REACH_ONLY_NOT_PICK",
            "algorithm": "SAC",
            "train_steps": args.steps,
            "seed": args.seed,
            "evaluation_count": len(records),
            "evaluation_reach_rate": sum(v["success"] for v in records) / len(records),
            "evaluation": records,
            "limitations": ["uncalibrated gripper_link TCP proxy",
                            "sampled FK targets, not Muhi grasp poses",
                            "unvalidated servo/contact dynamics",
                            "no real SO-101 commands"],
        }
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print("Saved:", args.out / "summary.json")
    finally:
        train.close()
        evaluation.close()


if __name__ == "__main__":
    main()
