#!/usr/bin/env python3
"""Identify base/shoulder collision in isolated SO-101 J6 actuation.

Diagnostic only. Never changes URDF or production grasp collision settings.
Three reset-identical runs: all contacts, only the shoulder geoms touching
world masked (diagnostic), and all contacts disabled (diagnostic).
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


def name(model, kind, idx):
    return mujoco.mj_id2name(model, kind, int(idx)) or f"unnamed_{idx}"


def geom_info(model, gid):
    bid = int(model.geom_bodyid[gid])
    mesh_id = int(model.geom_dataid[gid]) if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH) else -1
    return {
        "id": gid,
        "name": name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
        "body": name(model, mujoco.mjtObj.mjOBJ_BODY, bid),
        "body_id": bid,
        "geom_type": int(model.geom_type[gid]),
        "mesh_id": mesh_id,
        "mesh_name": name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) if mesh_id >= 0 else None,
        "contype": int(model.geom_contype[gid]),
        "conaffinity": int(model.geom_conaffinity[gid]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"output exists: {args.out}")
    spec = mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    model = spec.compile()
    model.opt.timestep = .002
    data = mujoco.MjData(model)

    armq, armd = [], []
    for joint in r2s.JOINTS:
        q, d, _ = bench.joint_addr(model, joint)
        armq.append(q)
        armd.append(d)
    jawq, jawd, limits = bench.joint_addr(model, "gripper")
    model.dof_armature[jawd] = max(float(model.dof_armature[jawd]), .002)
    model.dof_damping[jawd] = max(float(model.dof_damping[jawd]), .06)
    initial = np.array(model.qpos0, copy=True)
    initial[armq] = 0.
    initial[armq[r2s.JOINTS.index("wrist_roll")]] = math.radians(-90.)
    initial[jawq] = float(limits[1])

    mujoco.mj_resetData(model, data)
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    initial_contacts = []
    shoulder_world_geoms = set()
    for c in data.contact[:data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        initial_contacts.append({
            "geom1_id": g1, "geom2_id": g2,
            "body1": name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
            "body2": name(model, mujoco.mjtObj.mjOBJ_BODY, b2),
            "penetration_mm": min(0., float(c.dist) * 1000.),
        })
        if b1 == 0 and name(model, mujoco.mjtObj.mjOBJ_BODY, b2) == "shoulder_link":
            shoulder_world_geoms.add(g2)
        elif b2 == 0 and name(model, mujoco.mjtObj.mjOBJ_BODY, b1) == "shoulder_link":
            shoulder_world_geoms.add(g1)
    if not shoulder_world_geoms:
        raise RuntimeError("No world-to-shoulder contact at starting pose; diagnostic assumptions changed")

    all_contact_geoms = sorted(set(
        [r["geom1_id"] for r in initial_contacts] +
        [r["geom2_id"] for r in initial_contacts]))
    inventory = [geom_info(model, i) for i in all_contact_geoms]
    original_flags = int(model.opt.disableflags)
    original_masks = {g: (int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
                      for g in shoulder_world_geoms}

    cases = ("unmodified_contact_on", "mask_world_touching_shoulder_DIAGNOSTIC",
             "all_contact_off_DIAGNOSTIC")
    traces, summaries = [], []
    try:
        for case in cases:
            model.opt.disableflags = original_flags
            for g, (ct, ca) in original_masks.items():
                model.geom_contype[g] = ct
                model.geom_conaffinity[g] = ca
            if case == cases[1]:
                # Disables every collision involving these shoulder geoms,
                # not just the selected world pairs: DIAGNOSTIC ONLY.
                for g in shoulder_world_geoms:
                    model.geom_contype[g] = 0
                    model.geom_conaffinity[g] = 0
            elif case == cases[2]:
                model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            mujoco.mj_resetData(model, data)
            data.qpos[:] = initial
            data.qvel[:] = 0.
            mujoco.mj_forward(model, data)
            case_rows = []
            for step in range(1500):
                t = float(data.time)
                f = min(1., t / 2.2)
                smooth = f*f*(3. - 2.*f)
                target = float(limits[1]) + smooth*(float(limits[0])-float(limits[1]))
                data.qpos[armq] = initial[armq]
                data.qvel[armd] = 0.
                mujoco.mj_forward(model, data)
                command = float(np.clip(.15*(target-data.qpos[jawq])-.08*data.qvel[jawd],
                                        -.015, .015))
                data.qfrc_applied[:] = 0.
                data.qfrc_applied[jawd] = command
                mujoco.mj_step(model, data)
                constraint = float(data.qfrc_constraint[jawd])
                ncon = int(data.ncon)
                data.qpos[armq] = initial[armq]
                data.qvel[armd] = 0.
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError(f"{case} non-finite state at step {step}")
                row = {
                    "case": case, "time_s": float(data.time),
                    "j6_target_deg": math.degrees(target),
                    "j6_actual_deg": math.degrees(float(data.qpos[jawq])),
                    "command_nm": command,
                    "j6_constraint_nm": constraint,
                    "contact_count": ncon,
                }
                case_rows.append(row)
            traces.extend(case_rows)
            final = case_rows[-1]
            summaries.append({
                "case": case, "final_target_deg": final["j6_target_deg"],
                "final_actual_deg": final["j6_actual_deg"],
                "final_error_deg": final["j6_actual_deg"]-final["j6_target_deg"],
                "max_abs_j6_constraint_nm": max(abs(r["j6_constraint_nm"]) for r in case_rows),
                "steps_with_contact": sum(r["contact_count"] > 0 for r in case_rows),
            })
    finally:
        model.opt.disableflags = original_flags
        for g, (ct, ca) in original_masks.items():
            model.geom_contype[g] = ct
            model.geom_conaffinity[g] = ca

    args.out.mkdir(parents=True)
    with (args.out/"j6_contact_attribution.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(traces[0]))
        w.writeheader()
        w.writerows(traces)
    report = {
        "status": "J6_WORLD_SHOULDER_CONTACT_DIAGNOSTIC",
        "simulation_only": True,
        "urdf_modified": False,
        "contact_disabled_not_for_production": True,
        "shoulder_geom_ids_masked_for_diagnosis": sorted(shoulder_world_geoms),
        "contact_geometry_inventory": inventory,
        "initial_contact_pairs": initial_contacts,
        "cases": summaries,
        "note": ("The shoulder-geom mask disables more than just its world pairs. "
                 "This isolates causality approximately; it is NOT a production collision fix. "
                 "J1-J5 are kinematically overwritten each step."),
    }
    (args.out/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for case in cases:
            subset = [r for r in traces if r["case"] == case]
            ax.plot([r["time_s"] for r in subset],
                    [r["j6_actual_deg"] for r in subset], label=case)
        first = traces[:1500]
        ax.plot([r["time_s"] for r in first], [r["j6_target_deg"] for r in first],
                "--", color="black", label="target")
        ax.set(xlabel="Time (s)", ylabel="J6 angle (deg)",
               title="J6: normal contacts vs targeted diagnostic mask vs all off")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(args.out/"j6_contact_attribution.png", dpi=160)
        plt.close(fig)
    except ImportError:
        report["plot_note"] = "matplotlib unavailable; CSV is the authoritative trace"
        (args.out/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
