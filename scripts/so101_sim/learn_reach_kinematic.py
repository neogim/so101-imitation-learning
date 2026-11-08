#!/usr/bin/env python3
"""Simple SO-101 Reach learning baseline: kinematics only, NOT physical control.

The 5D action changes joint angles by at most action_delta_rad per step.
MuJoCo is used ONLY for forward kinematics (mj_forward), with no mj_step,
PD torques, self-contact forces, or servo dynamics. This intentionally small
exercise is the reference before teaching RL, not a Sim2Real controller.
Original URDF and real robot are untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import real_to_sim as r2s
from so101_reach_env import joint_addresses, reach_reward


class KinematicReachEnv(gym.Env):
    """Observation = [five q, three target-TCP delta, distance] (9D)."""

    def __init__(self, *, seed=101, action_delta_rad=0.04, max_steps=60,
                 goal_joint_delta_rad=0.35, success_radius_m=0.025):
        super().__init__()
        if not 0 < action_delta_rad <= 0.2 or max_steps < 1:
            raise ValueError("Invalid action step or horizon")
        if not 0 < goal_joint_delta_rad <= 1.0 or success_radius_m <= 0:
            raise ValueError("Invalid goal range or success threshold")
        spec = mujoco.MjSpec.from_file(str(r2s.DEFAULT_URDF.resolve()))
        spec.compiler.discardvisual = True
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qids, _, self.qmin, self.qmax = joint_addresses(self.model)
        self.gripper_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
        if self.gripper_id < 0:
            raise ValueError("Missing gripper_link")
        self.action_delta_rad = float(action_delta_rad)
        self.max_steps = int(max_steps)
        self.goal_joint_delta_rad = float(goal_joint_delta_rad)
        self.success_radius_m = float(success_radius_m)
        self.action_space = spaces.Box(-1, 1, (5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.target_joint = np.zeros(5)
        self.target_tcp = np.zeros(3)
        self.initial_joint = np.zeros(5)
        self.steps = 0

    def _tcp(self):
        return np.asarray(self.data.xpos[self.gripper_id], dtype=float).copy()

    def _obs(self):
        delta = self.target_tcp - self._tcp()
        return np.concatenate([self.data.qpos[self.qids], delta,
                               [np.linalg.norm(delta)]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        start = np.clip(np.zeros(5), self.qmin + .05, self.qmax - .05)
        self.initial_joint = start.copy()
        self.data.qpos[self.qids] = start
        mujoco.mj_forward(self.model, self.data)
        start_tcp = self._tcp()
        low = np.maximum(self.qmin + .05, start - self.goal_joint_delta_rad)
        high = np.minimum(self.qmax - .05, start + self.goal_joint_delta_rad)
        # Sample goals that require real movement; fixed bounded retries.
        for _ in range(64):
            q = self.rng.uniform(low, high)
            self.data.qpos[self.qids] = q
            mujoco.mj_forward(self.model, self.data)
            goal_tcp = self._tcp()
            if np.linalg.norm(goal_tcp - start_tcp) >= .05:
                break
        else:
            raise RuntimeError("Could not sample nontrivial FK goal")
        self.target_joint = q.copy()  # Oracle-only known solution; not RL observation.
        self.target_tcp = goal_tcp.copy()
        self.data.qpos[self.qids] = start
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._obs(), {"goal_tcp_m": self.target_tcp.tolist(),
                             "status": "KINEMATIC_REFERENCE_NOT_PHYSICAL_SIM"}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Expected finite 5D action")
        self.data.qpos[self.qids] = np.clip(
            self.data.qpos[self.qids] +
            np.clip(action, -1, 1) * self.action_delta_rad,
            self.qmin, self.qmax)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        distance = float(np.linalg.norm(self.target_tcp - self._tcp()))
        reward, success = reach_reward(distance, self.success_radius_m)
        return self._obs(), reward, success, self.steps >= self.max_steps and not success, {
            "is_success": success, "distance_m": distance,
            "status": "KINEMATIC_REFERENCE_NOT_PHYSICAL_SIM"
        }


def run_experiment(*, seed=101, episodes=3, max_steps=60, action_delta_rad=.04,
                   goal_joint_delta_rad=.35):
    env = KinematicReachEnv(seed=seed, max_steps=max_steps,
                            action_delta_rad=action_delta_rad,
                            goal_joint_delta_rad=goal_joint_delta_rad)
    rows = []
    try:
        for ep in range(episodes):
            for policy in ("zero_action", "known_joint_oracle"):
                env.reset(seed=seed + ep)
                initial_distance = float(np.linalg.norm(env.target_tcp - env._tcp()))
                for step in range(max_steps):
                    if policy == "known_joint_oracle":
                        # This uses the hidden FK-generated answer: a validation
                        # baseline only, NOT a learned observation-driven agent.
                        action = np.clip((env.target_joint -
                                         env.data.qpos[env.qids]) /
                                         env.action_delta_rad, -1, 1)
                    else:
                        action = np.zeros(5)
                    _, _, done, truncated, info = env.step(action)
                    if done or truncated:
                        break
                rows.append({
                    "episode": ep, "policy": policy,
                    "initial_tcp_mm": round(initial_distance*1000, 3),
                    "final_tcp_mm": round(info["distance_m"]*1000, 3),
                    "steps": step + 1, "success": bool(info["is_success"]),
                })
    finally:
        env.close()
    return {
        "status": "KINEMATIC_FK_REACH_ONLY_NOT_PD_NOT_RL_NOT_SIM2REAL",
        "seed": seed,
        "success_threshold_mm": 25,
        "results": rows,
        "limitations": [
            "Joint positions are directly assigned; dynamics and contact physics are NOT simulated.",
            "Known-joint oracle has access to hidden target joint values and is NOT an RL agent.",
            "This baseline demonstrates coordinate/goal/action mechanics, not collision safety.",
            "Do NOT use its actions for the physical robot.",
        ]
    }


def save(report, out):
    if out.exists():
        raise ValueError(f"Output path already exists: {out}")
    out.mkdir(parents=True)
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (out / "reach_baselines.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["results"][0]))
        writer.writeheader()
        writer.writerows(report["results"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()
    if args.episodes <= 0:
        p.error("--episodes must be positive")
    report = run_experiment(seed=args.seed, episodes=args.episodes)
    save(report, args.out)
    print("Saved", args.out / "reach_baselines.csv")
    for policy in ("zero_action", "known_joint_oracle"):
        rows = [r for r in report["results"] if r["policy"] == policy]
        print(f"{policy}: success {sum(r['success'] for r in rows)}/{len(rows)}, "
              f"mean_final_tcp_mm={np.mean([r['final_tcp_mm'] for r in rows]):.2f}")
    print("Kinematic demonstration only: no PD physics, no self-collision, no SAC.")


if __name__ == "__main__":
    main()
