#!/usr/bin/env python3
"""J6 dynamics sensitivity check (simulation only; no object/table/hardware).

Compare armature and damping with otherwise identical torque-limited PD.
The output explicitly separates observed response from assumed motor physics.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--duration-s", type=float, default=3.0)
    p.add_argument("--close-s", type=float, default=2.2)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"output directory already exists: {args.out}")
    if not (0 < args.dt <= .01 and args.duration_s > args.close_s > 0):
        p.error("invalid timestep or duration")

    spec = mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    spec.compiler.discardvisual = False
    model = spec.compile()
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    arm_q = []
    arm_d = []
    for name in r2s.JOINTS:
        q, d, _ = bench.joint_addr(model, name)
        arm_q.append(q)
        arm_d.append(d)
        data.qpos[q] = 0
    data.qpos[arm_q[r2s.JOINTS.index("wrist_roll")]] = math.radians(-90)
    j6q, j6d, limits = bench.joint_addr(model, "gripper")
    opened, closed = float(limits[1]), float(limits[0])
    data.qpos[j6q] = opened
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    arm_pose = data.qpos[arm_q].copy()
    original_armature = float(model.dof_armature[j6d])
    original_damping = float(model.dof_damping[j6d])

    # These are exploratory bench parameters, not identified STS3215 motor data.
    cases = (
        ("baseline", .002, .060),
        ("lower_damping", .002, .006),
        ("lower_armature", .0002, .060),
        ("lower_both", .0002, .006),
    )
    kp, kd, torque_limit = .15, .08, .015
    rows, summary_rows = [], []
    for case, armature, damping in cases:
        model.dof_armature[j6d] = max(original_armature, armature)
        model.dof_damping[j6d] = max(original_damping, damping)
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        case_rows = []
        for step in range(math.ceil(args.duration_s / args.dt)):
            t = float(data.time)
            phase = min(1., t / args.close_s)
            s = phase * phase * (3 - 2 * phase)
            target = opened + s * (closed - opened)
            data.qpos[arm_q] = arm_pose
            data.qvel[arm_d] = 0
            mujoco.mj_forward(model, data)
            bias = float(data.qfrc_bias[j6d])
            feedback = kp * (target - data.qpos[j6q]) - kd * data.qvel[j6d]
            applied = float(np.clip(feedback, -torque_limit, torque_limit))
            data.qfrc_applied[:] = 0
            data.qfrc_applied[j6d] = applied
            mujoco.mj_step(model, data)
            passive = float(data.qfrc_passive[j6d])
            constraint = float(data.qfrc_constraint[j6d])
            acceleration = float(data.qacc[j6d])
            data.qpos[arm_q] = arm_pose
            data.qvel[arm_d] = 0
            mujoco.mj_forward(model, data)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                raise FloatingPointError(f"nonfinite state in {case}, step {step}")
            actual = math.degrees(float(data.qpos[j6q]))
            record = {
                "case": case, "time_s": float(data.time),
                "target_deg": math.degrees(target),
                "actual_deg": actual,
                "error_deg": actual - math.degrees(target),
                "velocity_rad_s": float(data.qvel[j6d]),
                "acceleration_rad_s2": acceleration,
                "torque_nm": applied,
                "pd_requested_nm": float(feedback),
                "bias_nm": bias,
                "passive_nm": passive,
                "constraint_nm": constraint,
                "ncon": int(data.ncon),
            }
            case_rows.append(record)
        rows.extend(case_rows)
        last = case_rows[-1]
        summary_rows.append({
            "case": case,
            "armature_kg_m2": float(model.dof_armature[j6d]),
            "damping_nm_s_rad": float(model.dof_damping[j6d]),
            "final_target_deg": last["target_deg"],
            "final_actual_deg": last["actual_deg"],
            "final_error_deg": last["error_deg"],
            "max_abs_error_deg": max(abs(r["error_deg"]) for r in case_rows),
            "torque_saturation_fraction": sum(
                abs(r["torque_nm"]) >= .99 * torque_limit
                for r in case_rows) / len(case_rows),
            "max_abs_constraint_nm": max(abs(r["constraint_nm"]) for r in case_rows),
            "max_abs_passive_nm": max(abs(r["passive_nm"]) for r in case_rows),
            "steps_with_any_contact": sum(r["ncon"] > 0 for r in case_rows),
            "tracks_within_5deg": abs(last["error_deg"]) <= 5,
        })

    args.out.mkdir(parents=True)
    for filename, records in (
        ("j6_parameter_sweep.csv", rows),
        ("j6_parameter_summary.csv", summary_rows),
    ):
        with (args.out / filename).open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    report = {
        "status": "J6_PARAMETER_DIAGNOSTIC_ONLY",
        "simulation_only": True,
        "no_muhi_table_or_wrist": True,
        "original_urdf_modified": False,
        "j5_deg": -90,
        "pd_kp": kp, "pd_kd": kd,
        "torque_limit_nm": torque_limit,
        "original_armature_kg_m2": original_armature,
        "original_damping_nm_s_rad": original_damping,
        "cases": summary_rows,
        "caution": "Simulation sensitivity only; not a validated physical motor model.",
    }
    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for case, _, _ in cases:
            cr = [r for r in rows if r["case"] == case]
            ax.plot([r["time_s"] for r in cr],
                    [r["actual_deg"] for r in cr], label=case)
        base = [r for r in rows if r["case"] == "baseline"]
        ax.plot([r["time_s"] for r in base],
                [r["target_deg"] for r in base],
                linestyle="--", color="black", label="target (all cases)")
        ax.set(xlabel="Time (s)", ylabel="J6 angle (deg)",
               title="J6 dynamics sensitivity: target vs actual")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "j6_angle_comparison.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        for case, _, _ in cases:
            cr = [r for r in rows if r["case"] == case]
            ax.plot([r["time_s"] for r in cr],
                    [r["constraint_nm"] for r in cr], label=case)
        ax.set(xlabel="Time (s)", ylabel="Constraint generalized force (Nm)",
               title="J6 joint constraint force (diagnostic)")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "j6_constraint_force.png", dpi=150)
        plt.close(fig)
    except ImportError:
        report["plot_note"] = "matplotlib missing; CSV and summary are complete"
        (args.out / "summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
