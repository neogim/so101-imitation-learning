#!/usr/bin/env python3
"""Gymnasium-compatible, low-dimensional SO-101 MuJoCo Reach (offline only).

Five joint targets use a bounded PD joint-force controller + MuJoCo mj_step.
The training goal is *reach*, NOT grabbing the bottle or insertion into a cup.
Neither the gripper-link origin nor servo dynamics are mechanically calibrated.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import real_to_sim as r2s


def joint_addresses(model):
    qpos_ids, dof_ids, lows, highs = [], [], [], []
    for name in r2s.JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Missing joint: {name}")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected hinge joint: {name}")
        qpos_ids.append(int(model.jnt_qposadr[jid]))
        dof_ids.append(int(model.jnt_dofadr[jid]))
        if model.jnt_limited[jid]:
            lows.append(float(model.jnt_range[jid, 0]))
            highs.append(float(model.jnt_range[jid, 1]))
        else:
            # Fallback is an explicit conservative example, not hardware limits.
            lows.append(-np.pi / 2)
            highs.append(np.pi / 2)
    return (np.array(qpos_ids), np.array(dof_ids), np.array(lows),
            np.array(highs))


def reach_reward(distance_m, success_radius_m=.025):
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError("Invalid goal distance")
    return -float(distance_m), bool(distance_m <= success_radius_m)


class SO101ReachEnv(gym.Env):
    """5D action: per-step target-angle increments. 14D state observation.

    obs = 5 joint angles, 5 velocities, 3 target-minus-TCP, 1 distance.
    Goal distribution samples *reachable* TCP positions via FK of random q.
    This avoids arbitrary unreachable goals but is NOT collision validation.
    """

    metadata = {"render_modes": []}

    def __init__(self, urdf=r2s.DEFAULT_URDF, seed=None, max_steps=150,
                 action_delta_rad=.045, success_radius_m=.025,
                 frame_skip=10, kp=12., kd=1.1, max_joint_torque=3.,
                 goal_joint_delta_rad=None):
        super().__init__()
        if max_steps <= 0 or frame_skip <= 0 or action_delta_rad <= 0:
            raise ValueError("Invalid horizon or controller resolution")
        if kp <= 0 or kd < 0 or max_joint_torque <= 0 or success_radius_m <= 0:
            raise ValueError("Invalid reach or controller parameters")
        if goal_joint_delta_rad is not None and not (0 < goal_joint_delta_rad <= np.pi):
            raise ValueError("goal_joint_delta_rad must be in (0, pi]")
        self.goal_joint_delta_rad = goal_joint_delta_rad
        spec = mujoco.MjSpec.from_file(str(Path(urdf).resolve(strict=True)))
        spec.compiler.discardvisual = True
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qids, self.dids, self.qmin, self.qmax = joint_addresses(self.model)
        self.gripper_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
        if self.gripper_id < 0:
            raise ValueError("gripper_link TCP proxy body missing")
        self.action_space = spaces.Box(-1., 1., shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,),
                                            dtype=np.float32)
        self.max_steps = max_steps
        self.action_delta_rad = float(action_delta_rad)
        self.success_radius_m = float(success_radius_m)
        self.frame_skip = int(frame_skip)
        self.kp, self.kd, self.max_joint_torque = float(kp), float(kd), float(max_joint_torque)
        self.goal = np.zeros(3)
        self.goal_joint_angles = np.zeros(5)
        self.targets = np.zeros(5)
        self.t = 0
        self.np_random = np.random.default_rng(seed)
        self.reset(seed=seed)

    def _tcp(self):
        return self.data.xpos[self.gripper_id].copy()

    def _observation(self):
        q = self.data.qpos[self.qids].copy()
        v = self.data.qvel[self.dids].copy()
        delta = self.goal - self._tcp()
        return np.concatenate([q, v, delta, [np.linalg.norm(delta)]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mid = np.clip(np.zeros(5), self.qmin + .05, self.qmax - .05)
        self.data.qpos[self.qids] = mid
        mujoco.mj_forward(self.model, self.data)
        start_tcp = self._tcp()
        # Choose a reachable joint configuration, then turn its TCP into goal.
        # This is only kinematic reachability; no obstacle/physics guarantee.
        margin = np.minimum(.15, (self.qmax - self.qmin) * .1)
        low, high = self.qmin + margin, self.qmax - margin
        if self.goal_joint_delta_rad is not None:
            low = np.maximum(low, mid - self.goal_joint_delta_rad)
            high = np.minimum(high, mid + self.goal_joint_delta_rad)
            if np.any(low >= high):
                raise ValueError("No valid near-goal joint sampling interval")
        sampled = self.np_random.uniform(low, high)
        self.data.qpos[self.qids] = sampled
        mujoco.mj_forward(self.model, self.data)
        sampled_goal = self._tcp()
        # Reject trivially identical goal if possible; deterministic bounded retry.
        for _ in range(8):
            if np.linalg.norm(sampled_goal - start_tcp) > .06:
                break
            sampled = self.np_random.uniform(low, high)
            self.data.qpos[self.qids] = sampled
            mujoco.mj_forward(self.model, self.data)
            sampled_goal = self._tcp()
        self.goal = sampled_goal.copy()
        self.goal_joint_angles = sampled.copy()
        self.data.qpos[self.qids] = mid
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.targets = mid.copy()
        self.t = 0
        return self._observation(), {"goal_base_m": self.goal.tolist(),
                                     "goal_mode": ("near" if self.goal_joint_delta_rad is not None else "full"),
                                     "status": "REACH_ONLY_NOT_PICK"}

    def step(self, action, *, trace_hook=None):
        action = np.asarray(action, dtype=float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("Expected finite 5D joint delta action")
        action = np.clip(action, -1., 1.)
        self.targets = np.clip(self.targets + self.action_delta_rad * action,
                               self.qmin, self.qmax)
        for _ in range(self.frame_skip):
            mujoco.mj_forward(self.model, self.data)
            qerr = self.targets - self.data.qpos[self.qids]
            torques = self.kp * qerr - self.kd * self.data.qvel[self.dids]
            # qfrc_bias includes gravity and Coriolis terms (PD+feedforward).
            torques += self.data.qfrc_bias[self.dids]
            self.data.qfrc_applied[:] = 0
            applied = np.clip(torques, -self.max_joint_torque, self.max_joint_torque)
            self.data.qfrc_applied[self.dids] = applied
            mujoco.mj_step(self.model, self.data)
            if trace_hook is not None:
                # Diagnostic-only; no extra work on the regular SAC training path.
                contacts = []
                for ci in range(self.data.ncon):
                    contact = self.data.contact[ci]
                    g1 = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
                    g2 = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
                    contacts.append((g1 or f"geom{contact.geom1}",
                                     g2 or f"geom{contact.geom2}"))
                trace_hook({
                    "time_s": float(self.data.time),
                    "joint_target_rad": self.targets.copy(),
                    "joint_position_rad": self.data.qpos[self.qids].copy(),
                    "joint_velocity_rad_s": self.data.qvel[self.dids].copy(),
                    "torque_command_nm": torques.copy(),
                    "torque_applied_nm": applied.copy(),
                    "torque_saturated": (np.abs(torques) > self.max_joint_torque).copy(),
                    "tcp_distance_m": float(np.linalg.norm(self.goal - self._tcp())),
                    "contact_pairs": contacts,
                })
        self.t += 1
        delta = float(np.linalg.norm(self.goal - self._tcp()))
        reward, success = reach_reward(delta, self.success_radius_m)
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        terminated = success or not finite
        truncated = self.t >= self.max_steps and not terminated
        obs = self._observation()
        return obs, reward, terminated, truncated, {
            "is_success": success and finite,
            "distance_m": delta,
            "episode_steps": self.t,
            "status": "REACH_ONLY_NOT_PICK",
        }

    def close(self):
        pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    args = p.parse_args()
    env = SO101ReachEnv(urdf=args.urdf, seed=args.seed)
    try:
        for i in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + i)
            reward = 0.
            for step in range(env.max_steps):
                # Smoke baseline: random agent, NOT trained reinforcement learning.
                obs, reward, done, timeout, info = env.step(env.action_space.sample())
                if done or timeout:
                    break
            print(f"episode={i} random_agent_steps={step+1} "
                  f"reach={info['is_success']} distance={info['distance_m']:.4f}m")
    finally:
        env.close()
