#!/usr/bin/env python3
"""SO-101 J6: isolate base/shoulder collision, then bounded PD torque sensitivity.

SIMULATION-ONLY DIAGNOSTIC. Never changes source URDF, physical-grasp bench,
hardware settings or saved tasks. Pair filtering uses independent collision
bitmasks, leaving other 1/1 contacts enabled; inspect mask preconditions.
The torque values are exploratory and are NOT calibrated STS3215 limits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import physical_grasp_j6_bench as bench
import real_to_sim as r2s


def label(model, obj_type, idx):
    return mujoco.mj_id2name(model, obj_type, int(idx)) or f"id_{idx}"


def contact_inventory(model, data):
    pairs = {}
    for c in data.contact[:data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        a = label(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1])
        b = label(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2])
        key = (min(g1, g2), max(g1, g2))
        item = pairs.setdefault(key, {
            "geom1_id": key[0], "geom2_id": key[1],
            "body1": a, "body2": b, "contact_points": 0,
            "deepest_penetration_mm": 0.0,
        })
        item["contact_points"] += 1
        item["deepest_penetration_mm"] = min(
            item["deepest_penetration_mm"], float(c.dist)*1000.)
    return list(pairs.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    p.add_argument("--duration-s", type=float, default=3.)
    p.add_argument("--close-s", type=float, default=2.2)
    p.add_argument("--dt", type=float, default=.002)
    p.add_argument("--torques", type=float, nargs="+", default=[.015, .06, .12])
    args = p.parse_args()
    if args.out.exists():
        p.error(f"output already exists: {args.out}")
    if not (0 < args.dt <= .01 and args.duration_s > args.close_s > 0 and
            len(args.torques) <= 6 and all(0 < t <= .3 for t in args.torques)):
        p.error("invalid duration/timestep/diagnostic torque arguments")

    spec = mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    model = spec.compile()
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    arm_q, arm_d = [], []
    for name in r2s.JOINTS:
        q, d, _ = bench.joint_addr(model, name)
        arm_q.append(q)
        arm_d.append(d)
    jaw_q, jaw_d, limits = bench.joint_addr(model, "gripper")
    model.dof_armature[jaw_d] = max(float(model.dof_armature[jaw_d]), .002)
    model.dof_damping[jaw_d] = max(float(model.dof_damping[jaw_d]), .06)
    initial = np.array(model.qpos0, copy=True)
    initial[arm_q] = 0.
    initial[arm_q[r2s.JOINTS.index("wrist_roll")]] = math.radians(-90.)
    initial[jaw_q] = float(limits[1])
    original_flags = int(model.opt.disableflags)
    original_masks = [
        (int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
        for g in range(model.ngeom)
    ]
    mujoco.mj_resetData(model, data)
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    initial_pairs = contact_inventory(model, data)

    # Filter ONLY cross-contacts of static world/body0 and shoulder_link/body.
    # Default 1/1 geoms still collide with BOTH categories:
    # world (contype=2, conaffinity=1), shoulder (contype=4, conaffinity=1).
    # They do not collide with each other: 2&1 == 0, 4&1 == 0.
    shoulder = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")
    if shoulder < 0:
        raise RuntimeError("shoulder_link missing")
    base_ids = [g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) == 0 and
                (model.geom_contype[g] or model.geom_conaffinity[g])]
    shoulder_ids = [g for g in range(model.ngeom)
                    if int(model.geom_bodyid[g]) == shoulder and
                    (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not base_ids or not shoulder_ids:
        raise RuntimeError("world/shoulder collision geoms missing")
    affected = set(base_ids + shoulder_ids)
    if any(original_masks[g] != (1, 1) for g in affected):
        raise RuntimeError(
            "World/shoulder masks not uniformly 1/1; refusing category change")
    # Other geoms are required to retain default 1/1 for this *diagnostic*
    # to ensure that filtering does not silently suppress other pairs.
    others = [g for g in range(model.ngeom) if g not in affected
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if any(original_masks[g] != (1, 1) for g in others):
        raise RuntimeError("Other geom masks not uniformly 1/1; refuse pair-filter diagnostic")
    if not any(
        {int(model.geom_bodyid[p["geom1_id"]]),
         int(model.geom_bodyid[p["geom2_id"]])} == {0, shoulder}
        for p in initial_pairs):
        raise RuntimeError("No world/shoulder contact in initial pose")

    def restore_masks():
        for g, (ct, ca) in enumerate(original_masks):
            model.geom_contype[g] = ct
            model.geom_conaffinity[g] = ca
        model.opt.disableflags = original_flags

    def configure(mode):
        restore_masks()
        if mode == "base_shoulder_pair_filtered_DIAGNOSTIC":
            for g in base_ids:
                model.geom_contype[g] = 2
                model.geom_conaffinity[g] = 1
            for g in shoulder_ids:
                model.geom_contype[g] = 4
                model.geom_conaffinity[g] = 1
        elif mode == "all_contact_off_REFERENCE":
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    cases = [("normal_contact", args.torques[0])]
    cases += [("base_shoulder_pair_filtered_DIAGNOSTIC", t)
              for t in args.torques]
    cases += [("all_contact_off_REFERENCE", args.torques[0])]

    rows, summaries, setup_contacts = [], [], []
    try:
        for idx, (mode, torque_cap) in enumerate(cases):
            configure(mode)
            mujoco.mj_resetData(model, data)
            data.qpos[:] = initial
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            initial_contact_count = int(data.ncon)
            if mode == "base_shoulder_pair_filtered_DIAGNOSTIC":
                remaining = contact_inventory(model, data)
                if any(
                    {int(model.geom_bodyid[p["geom1_id"]]),
                     int(model.geom_bodyid[p["geom2_id"]])} == {0, shoulder}
                    for p in remaining):
                    raise RuntimeError("world/shoulder contacts persist after pair filter")
            setup_contacts.append(dict(
                case_index=idx, mode=mode, torque_cap_nm=torque_cap,
                initial_ncon=initial_contact_count))
            case_rows = []
            nsteps = math.ceil(args.duration_s / args.dt)
            for step in range(nsteps):
                t = float(data.time)
                u = min(1., t/args.close_s)
                target = float(limits[1]) + u*u*(3-2*u)*(
                    float(limits[0])-float(limits[1]))
                data.qpos[arm_q] = initial[arm_q]
                data.qvel[arm_d] = 0.
                mujoco.mj_forward(model, data)
                err = target - float(data.qpos[jaw_q])
                pd = .15*err - .08*float(data.qvel[jaw_d])
                cmd = float(np.clip(pd, -torque_cap, torque_cap))
                data.qfrc_applied[:] = 0.
                data.qfrc_applied[jaw_d] = cmd
                mujoco.mj_step(model, data)
                constraint = float(data.qfrc_constraint[jaw_d])
                passive = float(data.qfrc_passive[jaw_d])
                ncon = int(data.ncon)
                data.qpos[arm_q] = initial[arm_q]
                data.qvel[arm_d] = 0.
                mujoco.mj_forward(model, data)
                if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                    raise FloatingPointError(f"{mode}/{torque_cap}: nonfinite at {step}")
                case_rows.append(dict(
                    case_index=idx, mode=mode, torque_cap_nm=torque_cap,
                    time_s=float(data.time),
                    j6_target_deg=math.degrees(target),
                    j6_actual_deg=math.degrees(float(data.qpos[jaw_q])),
                    j6_velocity_rad_s=float(data.qvel[jaw_d]),
                    pd_requested_nm=float(pd), applied_nm=cmd,
                    j6_constraint_nm=constraint, j6_passive_nm=passive,
                    ncon=ncon))
            rows += case_rows
            last = case_rows[-1]
            summaries.append(dict(
                case_index=idx, mode=mode, torque_cap_nm=torque_cap,
                final_target_deg=last["j6_target_deg"],
                final_actual_deg=last["j6_actual_deg"],
                final_error_deg=last["j6_actual_deg"]-last["j6_target_deg"],
                saturation_fraction=sum(
                    abs(r["applied_nm"]) >= .99*torque_cap
                    for r in case_rows)/len(case_rows),
                max_abs_constraint_nm=max(
                    abs(r["j6_constraint_nm"]) for r in case_rows),
                steps_with_contact=sum(r["ncon"]>0 for r in case_rows),
                within_5deg=abs(last["j6_actual_deg"]-last["j6_target_deg"]) <= 5))
    finally:
        restore_masks()

    args.out.mkdir(parents=True)
    def writecsv(filename, data_rows):
        with (args.out/filename).open("w", newline="", encoding="utf8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(data_rows[0]))
            writer.writeheader()
            writer.writerows(data_rows)
    writecsv("j6_contact_torque_trace.csv", rows)
    writecsv("j6_contact_torque_summary.csv", summaries)
    writecsv("initial_contact_counts.csv", setup_contacts)
    report = {
        "status": "J6_COLLISION_AND_CONTROL_SENSITIVITY_ONLY",
        "original_urdf_modified": False,
        "production_contact_masks_modified": False,
        "model_initial_contact_pairs": initial_pairs,
        "world_geom_ids": base_ids,
        "shoulder_geom_ids": shoulder_ids,
        "pair_filter_notes": (
            "Uses 2/1 vs 4/1 collision categories on world vs shoulder geoms; "
            "all other 1/1 geoms retain mutual contact. "
            "This is a diagnosis, not an approved production collision filter."),
        "controller_notes": (
            "Torque test caps are exploratory, not calibrated STS3215 values; "
            "J1-J5 are overwritten kinematically at each simulation step."),
        "cases": summaries}
    (args.out/"summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for s in summaries:
            cr = [r for r in rows if r["case_index"]==s["case_index"]]
            ax.plot([r["time_s"] for r in cr],
                    [r["j6_actual_deg"] for r in cr],
                    label=f'{s["mode"]} {s["torque_cap_nm"]:.3f} Nm')
        base = [r for r in rows if r["case_index"]==0]
        ax.plot([r["time_s"] for r in base],
                [r["j6_target_deg"] for r in base],
                "--", color="black", label="target")
        ax.set(xlabel="Time (s)", ylabel="J6 angle (deg)",
               title="J6 base/shoulder contact filter and PD torque sensitivity")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.out/"j6_contact_torque_comparison.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        xs = list(range(len(summaries)))
        ax.bar(xs, [s["final_actual_deg"] for s in summaries])
        ax.axhline(math.degrees(float(limits[0])), linestyle="--",
                   color="black", label="target")
        ax.set_xticks(xs, [f'{s["mode"][:15]}\n{s["torque_cap_nm"]:.3f}Nm'
                           for s in summaries], fontsize=7)
        ax.set(ylabel="J6 final angle (deg)",
               title="Final J6 angle by collision and torque condition")
        ax.grid(alpha=.2, axis="y")
        fig.tight_layout()
        fig.savefig(args.out/"j6_final_angle_bars.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
