#!/usr/bin/env python3
"""Isolated SO-101 J6 collision/actuation audit; no Muhi, table or hardware.

Two deliberately separated tests:
  1. Kinematic J6 sweep: report each collidable mesh pair at each angle.
  2. Free J6 PD: check whether J6 follows an open-to-close reference with
     actual MuJoCo dynamics, with J1-J5 fixed at their initial joint pose.
The test does NOT simulate a grasp or alter URDF collision geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import real_to_sim as r2s
import physical_grasp_j6_bench as bench

TCP_IN_GRIPPER = np.array([-.0079, -.000218121, -.0981274], dtype=float)


def geometry_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"


def geometry_info(model, gid):
    gid = int(gid)
    body = int(model.geom_bodyid[gid])
    return {
        "geom_id": gid,
        "geom": geometry_name(model, gid),
        "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
        "geom_type": int(model.geom_type[gid]),
        "geom_group": int(model.geom_group[gid]),
        "contype": int(model.geom_contype[gid]),
        "conaffinity": int(model.geom_conaffinity[gid]),
        "geom_size_m": [float(x) for x in model.geom_size[gid]],
    }


def jaw_pair_rows(model, data, fixed, moving, angle_deg, tcp_world):
    """Measure all jaw mesh pairs; retain the actual geom names and witnesses."""
    rows = []
    for f in fixed:
        for m in moving:
            witness = np.zeros(6)
            gap = float(mujoco.mj_geomDistance(
                model, data, int(f), int(m), .3, witness))
            rows.append({
                "j6_deg": float(angle_deg),
                "fixed_geom": geometry_name(model, f),
                "fixed_geom_id": int(f),
                "moving_geom": geometry_name(model, m),
                "moving_geom_id": int(m),
                "signed_distance_mm": gap * 1000.,
                "fixed_witness_tcp_distance_mm":
                    float(np.linalg.norm(witness[:3] - tcp_world) * 1000.),
                "moving_witness_tcp_distance_mm":
                    float(np.linalg.norm(witness[3:] - tcp_world) * 1000.),
                "fixed_witness_xyz_m": [float(x) for x in witness[:3]],
                "moving_witness_xyz_m": [float(x) for x in witness[3:]],
            })
    return rows


def current_jaw_contact_rows(model, data, fixed, moving):
    fs, ms = set(fixed), set(moving)
    rows = []
    for i, contact in enumerate(data.contact[:data.ncon]):
        a, b = int(contact.geom1), int(contact.geom2)
        if not ((a in fs and b in ms) or (b in fs and a in ms)):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        rows.append({
            "fixed_geom": geometry_name(model, a if a in fs else b),
            "moving_geom": geometry_name(model, b if a in fs else a),
            "penetration_mm": min(0., float(contact.dist) * 1000.),
            "normal_force_n": float(force[0]),
            "constraint_address": int(contact.efc_address),
        })
    return rows


def run(args):
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")
    spec = mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    spec.compiler.discardvisual = False
    model = spec.compile()
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    qids, dids = [], []
    for name in r2s.JOINTS:
        q, d, limits = bench.joint_addr(model, name)
        qids.append(q)
        dids.append(d)
        data.qpos[q] = 0.
    j5_index = r2s.JOINTS.index("wrist_roll")
    j5_id = qids[j5_index]
    data.qpos[j5_id] = math.radians(args.j5_deg)
    jawq, jawd, jawlim = bench.joint_addr(model, "gripper")
    open_angle, close_angle = float(jawlim[1]), float(jawlim[0])
    data.qpos[jawq] = open_angle
    fixed = bench.owned_collision_geoms(model, bench.FINGER_BODY)
    moving = bench.owned_collision_geoms(model, bench.MOVING_BODY)
    if not fixed or not moving:
        raise RuntimeError("URDF has no collidable geoms for a jaw")
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bench.FINGER_BODY)
    if grip < 0:
        raise RuntimeError("No gripper_link body")
    mujoco.mj_forward(model, data)
    initial_state = data.qpos.copy()
    q_hold = np.array([initial_state[i] for i in qids])
    d_hold = np.array(dids, dtype=int)

    # Store both mesh-distance witnesses and MuJoCo collision contacts;
    # a negative mesh distance alone does NOT prove that solver contact
    # between those geoms is active.
    kinematic, pairs, contacts = [], [], []
    for angle in np.linspace(open_angle, close_angle, args.sweep_count):
        data.qpos[:] = initial_state
        data.qpos[jawq] = angle
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        R = np.asarray(data.xmat[grip]).reshape(3, 3)
        tip = np.asarray(data.xpos[grip]) + R @ TCP_IN_GRIPPER
        jawpairs = jaw_pair_rows(model, data, fixed, moving, math.degrees(angle), tip)
        active = current_jaw_contact_rows(model, data, fixed, moving)
        pairs.extend(jawpairs)
        contacts.extend([{"j6_deg": math.degrees(angle), **c} for c in active])
        closest = min(jawpairs, key=lambda row: row["signed_distance_mm"])
        kinematic.append({
            "j6_deg": float(math.degrees(angle)),
            "min_jaw_mesh_gap_mm": float(closest["signed_distance_mm"]),
            "min_pair_fixed_geom": closest["fixed_geom"],
            "min_pair_moving_geom": closest["moving_geom"],
            "min_pair_fixed_witness_tcp_distance_mm":
                closest["fixed_witness_tcp_distance_mm"],
            "min_pair_moving_witness_tcp_distance_mm":
                closest["moving_witness_tcp_distance_mm"],
            "jaw_solver_contact_count": len(active),
            "jaw_contacts_deepest_mm":
                min((c["penetration_mm"] for c in active), default=0.),
        })

    # Separate the three hypotheses with ONE run / one Mac command:
    # (1) prior low-torque PD, (2) same torque with gravity/bias feedforward,
    # (3) compensated torque with a larger diagnostic saturation bound.
    # NO Muhi or table, same joint pose / same URDF per comparison.
    model.dof_armature[jawd] = max(float(model.dof_armature[jawd]), args.armature)
    model.dof_damping[jawd] = max(float(model.dof_damping[jawd]), args.damping)
    cases = (
        ("baseline_pd_015", False, args.torque_nm),
        ("bias_comp_pd_015", True, args.torque_nm),
        ("bias_comp_pd_060", True, args.comparison_torque_nm),
    )
    all_dynamic = []
    case_results = []
    for case_name, bias_compensated, torque_limit in cases:
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_state
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        case_rows, peak_j6_overrun_deg = [], 0.
        start_bias = float(data.qfrc_bias[jawd])
        for step in range(int(math.ceil(args.duration_s / args.dt))):
            t = float(data.time)
            progress = min(1., t / args.close_s)
            smooth = progress * progress * (3.-2.*progress)
            target = open_angle + smooth * (close_angle - open_angle)
            data.qpos[qids] = q_hold
            data.qvel[d_hold] = 0.
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = 0.
            bias = float(data.qfrc_bias[jawd])
            feedback = (args.kp * (target - data.qpos[jawq])
                        - args.kd * data.qvel[jawd])
            # MuJoCo's qfrc_bias is the generalized force of gravity and
            # Coriolis terms, NOT a motor command. It must be included as
            # feedforward to compensate opposing gravity in this diagnosis.
            desired_torque = feedback + (bias if bias_compensated else 0.)
            applied = float(np.clip(desired_torque, -torque_limit, torque_limit))
            data.qfrc_applied[jawd] = applied
            mujoco.mj_step(model, data)
            solver_contacts = current_jaw_contact_rows(model, data, fixed, moving)
            data.qpos[qids] = q_hold
            data.qvel[d_hold] = 0.
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError(
                    f"Nonfinite J6 simulation: {case_name} step {step}")
            angle_deg = float(math.degrees(data.qpos[jawq]))
            peak_j6_overrun_deg = max(
                peak_j6_overrun_deg,
                max(0., angle_deg-math.degrees(open_angle),
                    math.degrees(close_angle)-angle_deg))
            case_rows.append({
                "case": case_name, "step": step, "time_s": float(data.time),
                "j6_target_deg": math.degrees(target),
                "j6_actual_deg": angle_deg,
                "j6_velocity_rad_s": float(data.qvel[jawd]),
                "j6_bias_nm": bias,
                "j6_pd_feedback_nm": float(feedback),
                "j6_requested_torque_nm": float(desired_torque),
                "j6_commanded_torque_nm": applied,
                "torque_limit_nm": float(torque_limit),
                "bias_compensated": bool(bias_compensated),
                "jaw_solver_contact_count": len(solver_contacts),
                "jaw_contact_max_normal_force_n": max(
                    (c["normal_force_n"] for c in solver_contacts),
                    default=0.),
            })
        all_dynamic.extend(case_rows)
        last = case_rows[-1]
        case_results.append({
            "case": case_name,
            "bias_compensated": bool(bias_compensated),
            "torque_limit_nm": float(torque_limit),
            "j6_initial_bias_nm": start_bias,
            "final_target_deg": last["j6_target_deg"],
            "final_actual_deg": last["j6_actual_deg"],
            "max_abs_tracking_error_deg": max(abs(
                row["j6_actual_deg"]-row["j6_target_deg"])
                for row in case_rows),
            "max_joint_limit_overrun_deg": peak_j6_overrun_deg,
            "torque_saturation_fraction": (
                sum(abs(row["j6_commanded_torque_nm"]) >=
                    .99*torque_limit for row in case_rows)/len(case_rows)),
            "steps_with_jaw_contact": sum(
                row["jaw_solver_contact_count"]>0 for row in case_rows),
        })
    dynamic = all_dynamic
    args.out.mkdir(parents=True, exist_ok=False)

    def write_csv(name, records):
        if not records:
            return
        with (args.out / name).open("x", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv("jaw_geometries.csv", [geometry_info(model, g)
              for g in list(fixed) + list(moving)])
    write_csv("kinematic_sweep.csv", kinematic)
    write_csv("all_jaw_geom_pairs.csv", pairs)
    write_csv("jaw_solver_contacts.csv", contacts)
    write_csv("j6_dynamic_no_object.csv", dynamic)
    write_csv("j6_controller_comparison.csv", case_results)
    baseline = kinematic[0]
    self_overlap_at_open = baseline["min_jaw_mesh_gap_mm"]
    summary = {
        "status": "J6_DIAGNOSTIC_ONLY",
        "simulation_only": True,
        "robot_source": str(args.urdf),
        "muhi_present": False,
        "table_present": False,
        "wrist_rig_present": False,
        "original_urdf_modified": False,
        "joint_initial_j5_deg": args.j5_deg,
        "j6_limits_deg": [math.degrees(close_angle), math.degrees(open_angle)],
        "fixed_collision_geom_count": len(fixed),
        "moving_collision_geom_count": len(moving),
        "baseline_open_mesh_pair_min_gap_mm": self_overlap_at_open,
        "baseline_open_mesh_pair": [
            baseline["min_pair_fixed_geom"], baseline["min_pair_moving_geom"]],
        "baseline_open_pair_fixed_witness_tcp_distance_mm":
            baseline["min_pair_fixed_witness_tcp_distance_mm"],
        "baseline_open_pair_moving_witness_tcp_distance_mm":
            baseline["min_pair_moving_witness_tcp_distance_mm"],
        "baseline_open_active_solver_contact_count":
            baseline["jaw_solver_contact_count"],
        "sweep_steps": len(kinematic),
        "sweep_steps_with_active_jaw_contacts": sum(
            r["jaw_solver_contact_count"] > 0 for r in kinematic),
        "controller_comparison": case_results,
        "j6_dof_properties": {
            "armature_kg_m2": float(model.dof_armature[jawd]),
            "damping_nm_s_rad": float(model.dof_damping[jawd]),
            "frictionloss_nm": float(model.dof_frictionloss[jawd]),
            "stiffness_nm_rad": float(model.jnt_stiffness[
                mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"gripper")]),
        },
        "dynamic_j6_initial_deg": case_results[0]["final_actual_deg"],
        "dynamic_j6_last_deg": case_results[0]["final_actual_deg"],
        "dynamic_last_target_deg": case_results[0]["final_target_deg"],
        "dynamic_max_abs_tracking_error_deg":
            case_results[0]["max_abs_tracking_error_deg"],
        "dynamic_max_joint_limit_overrun_deg":
            case_results[0]["max_joint_limit_overrun_deg"],
        "dynamic_steps_with_jaw_contact":
            case_results[0]["steps_with_jaw_contact"],
        "interpretation": (
            "A negative pair distance may arise from overlapping root/collision "
            "hulls, not actual fingertip collision. Inspect witnesses, geom "
            "names and solver contacts before changing any collision meshes."
        ),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--j5-deg", type=float, default=-90.)
    p.add_argument("--sweep-count", type=int, default=45)
    p.add_argument("--duration-s", type=float, default=3.)
    p.add_argument("--close-s", type=float, default=2.2)
    p.add_argument("--dt", type=float, default=.002)
    p.add_argument("--armature", type=float, default=.002)
    p.add_argument("--damping", type=float, default=.06)
    p.add_argument("--kp", type=float, default=.15)
    p.add_argument("--kd", type=float, default=.08)
    p.add_argument("--torque-nm", type=float, default=.015)
    p.add_argument("--comparison-torque-nm", type=float, default=.06)
    args = p.parse_args()
    if not (10 <= args.sweep_count <= 200 and .0001 <= args.dt <= .01
            and args.duration_s > args.close_s > 0
            and args.kp > 0 and args.kd >= 0 and args.torque_nm > 0
            and args.armature >= 0 and args.damping >= 0
            and args.comparison_torque_nm >= args.torque_nm):
        p.error("invalid simulation or J6 sweep parameters")
    run(args)


if __name__ == "__main__":
    main()
