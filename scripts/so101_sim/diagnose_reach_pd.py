#!/usr/bin/env python3
"""Check whether existing MuJoCo PD controller can reach known FK targets.

Diagnostic oracle sets the controller's joint-angle target directly.
It is NOT an RL policy, IK solution, or hardware verification.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from so101_reach_env import SO101ReachEnv


def run_diagnostic(seed=101, episodes=10, near_goal_joint_rad=None,
                   max_steps=150, frame_skip=10):
    env = SO101ReachEnv(seed=seed, max_steps=max_steps, frame_skip=frame_skip,
                        goal_joint_delta_rad=near_goal_joint_rad)
    rows = []
    try:
        for i in range(episodes):
            _, reset_info = env.reset(seed=seed + i)
            desired_joint = env.goal_joint_angles.copy()
            env.targets = desired_joint  # Oracle: bypass actions to isolate PD physics.
            for step in range(max_steps):
                _, _, done, truncated, info = env.step(np.zeros(5))
                if done or truncated:
                    break
            joint_err = float(np.linalg.norm(
                env.data.qpos[env.qids] - desired_joint))
            rows.append({
                "episode": i,
                "goal_base_m": reset_info["goal_base_m"],
                "success": bool(info["is_success"]),
                "final_tcp_distance_m": float(info["distance_m"]),
                "final_joint_error_rad_l2": joint_err,
                "steps": step + 1,
            })
    finally:
        env.close()
    return {
        "status": "SIM_PD_ORACLE_ONLY_NOT_IK_OR_RL_OR_HARDWARE",
        "seed": seed,
        "episodes": episodes,
        "goal_joint_delta_rad": near_goal_joint_rad,
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "results": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--near-goal-joint-rad", type=float, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.episodes < 1 or args.out.exists():
        p.error("episodes must be positive and --out must not exist")
    result = run_diagnostic(args.seed, args.episodes, args.near_goal_joint_rad)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"PD oracle success: {result['success_rate']:.1%}; saved: {args.out}")
    for r in result["results"]:
        print(f"episode={r['episode']} reach={r['success']} "
              f"distance={r['final_tcp_distance_m']:.4f}m "
              f"joint_error={r['final_joint_error_rad_l2']:.4f}rad")


if __name__ == "__main__":
    main()
