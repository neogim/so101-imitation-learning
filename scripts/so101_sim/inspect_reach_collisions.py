#!/usr/bin/env python3
"""Offline SO-101 URDF/MuJoCo collision-geometry audit and contact ablation.

Maps *compiled* geom IDs to mesh/body names. Compares identical FK-goal
PD-oracle rollouts with all contacts enabled vs a diagnostic contact-disabled
model. Never changes URDF, repository assets, or physical robot parameters.

Disabled contacts are NOT a safe/valid production controller configuration.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

import real_to_sim as r2s
from so101_reach_env import SO101ReachEnv


def object_name(model, kind, idx):
    name = mujoco.mj_id2name(model, kind, int(idx))
    return name or f"unnamed_{int(idx)}"


def geom_inventory(model):
    rows = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        mesh_id = int(model.geom_dataid[gid])
        geom_type = int(model.geom_type[gid])
        mesh = (object_name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH) and mesh_id >= 0
                else None)
        body = object_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        parent = int(model.body_parentid[bid])
        parent_name = (object_name(model, mujoco.mjtObj.mjOBJ_BODY, parent)
                       if bid != 0 else None)
        rows.append({
            "geom_id": gid,
            "geom_name": object_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
            "body_id": bid,
            "body_name": body,
            "parent_body_name": parent_name,
            "type": mujoco.mjtGeom(geom_type).name,
            "mesh_name": mesh,
            "contype": int(model.geom_contype[gid]),
            "conaffinity": int(model.geom_conaffinity[gid]),
        })
    return rows


def contact_inventory(model, data):
    out = Counter()
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = sorted((int(c.geom1), int(c.geom2)))
        out[(a, b)] += 1
    return [
        {"geom_id_pair": [a, b], "contact_points": n}
        for (a, b), n in sorted(out.items())
    ]


# Mesh IDs verified on the compiled, pinned SO-101 URDF.
BASE_SHOULDER_PAIRS = ((0, 6), (1, 6), (2, 6))


def install_selective_geom_filter(model, excluded_pairs):
    """Keep every originally eligible geom pair except selected pairs.

    Distinct one-hot contype bits and per-geom conaffinity bitsets let MuJoCo
    reject individual *geom* pairs, unlike body-level contact exclusions.
    Only valid for this small model with 1/1 original masks: validate first.
    Call before mj_forward / mj_step; never apply on the normal training model.
    """
    if model.ngeom > 30:
        raise ValueError("One-hot geom contact filter supports <=30 geoms")
    if not np.all(model.geom_contype == 1) or not np.all(model.geom_conaffinity == 1):
        raise ValueError("Expected original contype=conaffinity=1 for all geoms")
    pairs = set()
    for raw_a, raw_b in excluded_pairs:
        a, b = sorted((int(raw_a), int(raw_b)))
        if a == b or not (0 <= a < model.ngeom and 0 <= b < model.ngeom):
            raise ValueError(f"Invalid geom pair: {(raw_a, raw_b)}")
        pairs.add((a, b))
    full_mask = (1 << model.ngeom) - 1
    masks = [full_mask & ~(1 << g) for g in range(model.ngeom)]
    for a, b in pairs:
        masks[a] &= ~(1 << b)
        masks[b] &= ~(1 << a)
    # MuJoCo accepts a candidate when either direction shares contype+affinity;
    # remove both directions for each excluded pair.
    model.geom_contype[:] = np.asarray([1 << g for g in range(model.ngeom)], dtype=int)
    model.geom_conaffinity[:] = np.asarray(masks, dtype=int)
    return sorted(pairs)


def rollout(model, seed, episode, max_steps, frame_skip, contact_disabled,
            near_goal_joint_rad=None, excluded_geom_pairs=None):
    env = SO101ReachEnv(seed=seed, max_steps=max_steps, frame_skip=frame_skip,
                        goal_joint_delta_rad=near_goal_joint_rad)
    try:
        if excluded_geom_pairs is not None:
            install_selective_geom_filter(env.model, excluded_geom_pairs)
        if contact_disabled:
            env.model.opt.disableflags = (
                int(env.model.opt.disableflags)
                | int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
        # Save the same goal with the same RNG seed for both conditions.
        _, info0 = env.reset(seed=seed + episode)
        desired = env.goal_joint_angles.copy()
        initial_pose = env.data.qpos[env.qids].copy()
        initial_contact_count = int(env.data.ncon)
        initial_contacts = contact_inventory(env.model, env.data)
        env.targets = desired.copy()
        seen_pairs = Counter()
        n_contact_frames = 0
        max_contact_points = 0
        max_abs_velocity = np.zeros(5)
        max_abs_raw_torque = np.zeros(5)
        torque_saturation_count = np.zeros(5, dtype=int)
        physics_steps = 0
        min_tcp_distance = np.inf

        def hook(item):
            nonlocal n_contact_frames, max_contact_points, physics_steps, min_tcp_distance
            physics_steps += 1
            cs = item["contact_pairs"]
            if cs:
                n_contact_frames += 1
            max_contact_points = max(max_contact_points, len(cs))
            seen_pairs.update(tuple(sorted(pair)) for pair in cs)
            max_abs_velocity[:] = np.maximum(
                max_abs_velocity, np.abs(item["joint_velocity_rad_s"]))
            max_abs_raw_torque[:] = np.maximum(
                max_abs_raw_torque, np.abs(item["torque_command_nm"]))
            torque_saturation_count[:] += np.asarray(item["torque_saturated"], dtype=int)
            min_tcp_distance = min(min_tcp_distance, float(item["tcp_distance_m"]))

        for step in range(max_steps):
            _, _, done, truncated, info = env.step(
                np.zeros(5), trace_hook=hook)
            if done or truncated:
                break

        return {
            "contact_disabled": contact_disabled,
            "excluded_geom_pairs": [list(v) for v in (excluded_geom_pairs or ())],
            "goal_joint_rad": desired.tolist(),
            "initial_joint_rad": initial_pose.tolist(),
            "goal_tcp_base_m": info0["goal_base_m"],
            "initial_contact_points": initial_contact_count,
            "initial_contact_geom_pairs": initial_contacts,
            "success": bool(info["is_success"]),
            "steps": step + 1,
            "physics_substeps": physics_steps,
            "min_tcp_distance_m": float(min_tcp_distance),
            "final_tcp_distance_m": float(info["distance_m"]),
            "final_joint_error_l2_rad": float(
                np.linalg.norm(env.data.qpos[env.qids] - desired)),
            "max_abs_joint_velocity_rad_s": max_abs_velocity.tolist(),
            "max_abs_raw_torque_nm": max_abs_raw_torque.tolist(),
            "torque_saturation_fraction": (
                torque_saturation_count / physics_steps).tolist(),
            "contact_substep_fraction": n_contact_frames / physics_steps,
            "max_contact_points": max_contact_points,
            "contact_pair_record_counts": [
                {"geom_names": list(pair), "contact_records": n}
                for pair, n in seen_pairs.most_common()
            ],
        }
    finally:
        env.close()


def run_audit(*, seed=101, episode=0, max_steps=150, frame_skip=10,
              near_goal_joint_rad=None):
    if episode < 0 or max_steps <= 0 or frame_skip <= 0:
        raise ValueError("Invalid episode or physics horizon")
    env = SO101ReachEnv(seed=seed, frame_skip=frame_skip)
    try:
        inventory = geom_inventory(env.model)
        model_info = {
            "mujoco_version": mujoco.__version__,
            "urdf": str(r2s.DEFAULT_URDF),
            "ngeom": int(env.model.ngeom),
            "nbody": int(env.model.nbody),
            "njnt": int(env.model.njnt),
            "mesh_geoms": sum(v["mesh_name"] is not None for v in inventory),
        }
    finally:
        env.close()
    normal = rollout(None, seed, episode, max_steps, frame_skip, False,
                     near_goal_joint_rad=near_goal_joint_rad)
    ablation = rollout(None, seed, episode, max_steps, frame_skip, True,
                       near_goal_joint_rad=near_goal_joint_rad)
    np.testing.assert_allclose(normal["goal_joint_rad"], ablation["goal_joint_rad"])
    np.testing.assert_allclose(normal["goal_tcp_base_m"], ablation["goal_tcp_base_m"])
    return {
        "status": "OFFLINE_DIAGNOSTIC_CONTACT_ABLATION_NOT_ROBOT_CONTROL",
        "seed": seed, "episode": episode, "max_steps": max_steps,
        "frame_skip": frame_skip, "near_goal_joint_rad": near_goal_joint_rad,
        "model": model_info,
        "geoms": inventory,
        "baseline_contact_enabled": normal,
        "contact_disabled_diagnostic_only": ablation,
        "warnings": [
            "Geometry mesh names are compiled model names, not a safety classification.",
            "Contact-disabled dynamics are unphysical; use only to isolate a cause.",
            "A contact count is a count of contact points, not proof of harmful self-collision.",
            "Targets are FK-reachable in joint space; paths may still be obstructed.",
            "Never apply contact-disabled settings or arbitrary torques to physical SO-101.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--frame-skip", type=int, default=10)
    p.add_argument("--near-goal-joint-rad", type=float, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("--out must be a new path")
    report = run_audit(
        seed=args.seed, episode=args.episode,
        max_steps=args.max_steps, frame_skip=args.frame_skip,
        near_goal_joint_rad=args.near_goal_joint_rad)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("Saved:", args.out)
    print("Compiled collision geometry inventory:")
    for g in report["geoms"]:
        print(f"geom{g['geom_id']:02d} body={g['body_name']:<28} "
              f"mesh={str(g['mesh_name']):<42} "
              f"type={g['type']} mask={g['contype']}/{g['conaffinity']}")
    for name in ("baseline_contact_enabled", "contact_disabled_diagnostic_only"):
        r = report[name]
        print(f"{name}: success={r['success']} "
              f"final_distance={r['final_tcp_distance_m']:.4f}m "
              f"joint_error={r['final_joint_error_l2_rad']:.4f}rad "
              f"contact_fraction={r['contact_substep_fraction']:.3f}")
    print("Diagnostics only; do not use no-contact physics for actual RL training.")


if __name__ == "__main__":
    main()
