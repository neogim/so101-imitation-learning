#!/usr/bin/env python3
"""Opt-in MuJoCo physics workcell with saved Side/Wrist cameras.

The saved SAC/J5 rollout and all robot/source data remain untouched. Modes:
  drop: gravity-only Muhi settling test;
  push: known EXTERNAL force tests Muhi friction and table/cup contacts;
  replay: bounded joint PD follows recorded target positions with mj_step;
          NOT a SAC policy evaluation and NOT an autonomous grasp.
No robot, serial port or physical camera is accessed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import demo_real2sim_muhi_cup_reach as scene
import wrist_camera_sim as wrist

DYNAMIC_BODY = "r2s_physical_muhi"
DYNAMIC_JOINT = "r2s_physical_muhi_free"
DYNAMIC_GEOMS = ("r2s_physical_muhi_body", "r2s_physical_muhi_cap")
MUHI_LENGTH = .097
MUHI_RADIUS = .035 / 2


def enable_static_contacts_in_spec(spec):
    """Set world collision flags BEFORE compile for MuJoCo broadphase."""
    names = []
    for geom in spec.geoms:
        name = geom.name
        if (name == "r2s_preview_table" or
                name.startswith("r2s_table_front_") or
                name.startswith("r2s_cup_")):
            geom.contype = 1
            geom.conaffinity = 1
            names.append(name)
    if "r2s_preview_table" not in names or "r2s_cup_base" not in names:
        raise ValueError("Physical tabletop or cup not found in scene spec")
    return names


def install_dynamic_muhi(spec):
    """Create free-body bottle BEFORE model compile, separately from preview."""
    if spec.body(DYNAMIC_BODY) is not None:
        raise ValueError("Physical Muhi already installed")
    body = spec.worldbody.add_body(name=DYNAMIC_BODY, pos=[0, 0, .08])
    body.add_freejoint(name=DYNAMIC_JOINT)
    body.add_geom(name=DYNAMIC_GEOMS[0], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[0, 0, .016975], size=[MUHI_RADIUS, MUHI_LENGTH * .65 / 2, 0],
                  mass=.040, friction=[.8, .01, .001], rgba=[.95, .95, .94, 1],
                  contype=1, conaffinity=1)
    body.add_geom(name=DYNAMIC_GEOMS[1], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=[0, 0, -.031525], size=[MUHI_RADIUS, MUHI_LENGTH * .35 / 2, 0],
                  mass=.010, friction=[.8, .01, .001], rgba=[.82, .89, .96, 1],
                  contype=1, conaffinity=1)


def geom_id(model, name):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if index < 0:
        raise ValueError("Missing geom: " + name)
    return index


def configure_dynamic_muhi(model, data):
    """Copy original saved Muhi pose, hide static preview, enable contacts."""
    b = geom_id(model, "r2s_muhi_body")
    c = geom_id(model, "r2s_muhi_cap")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DYNAMIC_JOINT)
    if joint < 0:
        raise ValueError("Missing free joint")
    adr = int(model.jnt_qposadr[joint])
    # The original cylinders' common axis defines the body's local Z.
    center = .65 * model.geom_pos[b] + .35 * model.geom_pos[c]
    quat = np.asarray(model.geom_quat[b], float).copy()
    quat /= np.linalg.norm(quat)
    data.qpos[adr:adr+3] = center
    data.qpos[adr+3:adr+7] = quat
    for i in (b, c):
        model.geom_rgba[i, 3] = 0
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0
    return adr


def prepare_environment(model):
    """Enable table/cup collision. Old, hidden Muhi preview remains inactive."""
    enabled = wrist.enable_environment_contacts(model)
    for name in ("r2s_muhi_body", "r2s_muhi_cap"):
        i = geom_id(model, name)
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0
    for name in DYNAMIC_GEOMS:
        i = geom_id(model, name)
        if model.geom_contype[i] == 0:
            raise AssertionError("Dynamic bottle collider disabled")
    return [n for n in enabled if not n.startswith("r2s_muhi_")]


def summarize_contacts(model, data, object_body_id):
    """Classify Muhi and wrist-rig collision pairs without double counting."""
    categories = {"muhi_table": 0, "muhi_cup": 0, "muhi_robot": 0,
                  "wrist_table": 0, "wrist_cup": 0, "wrist_muhi": 0,
                  "wrist_robot_self": 0, "other": 0}
    minimum = None
    details = []
    wrist_ids = {
        i for n in wrist.COLLISION_NAMES
        if (i := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)) >= 0
    }
    for con in data.contact[:data.ncon]:
        a, b = int(con.geom1), int(con.geom2)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, x) or f"geom{x}"
                 for x in (a, b)]
        a_bottle = int(model.geom_bodyid[a]) == object_body_id
        b_bottle = int(model.geom_bodyid[b]) == object_body_id
        a_wrist, b_wrist = a in wrist_ids, b in wrist_ids
        if a_bottle or b_bottle:
            other = names[1] if a_bottle else names[0]
            if other == "r2s_preview_table" or other.startswith("r2s_table_front_"):
                category = "muhi_table"
            elif other.startswith("r2s_cup_"):
                category = "muhi_cup"
            else:
                category = "muhi_robot"
        elif a_wrist or b_wrist:
            other = names[1] if a_wrist else names[0]
            if other == "r2s_preview_table" or other.startswith("r2s_table_front_"):
                category = "wrist_table"
            elif other.startswith("r2s_cup_"):
                category = "wrist_cup"
            else:
                category = "wrist_robot_self"
        else:
            category = "other"
        categories[category] += 1
        if (a_wrist and b_bottle) or (b_wrist and a_bottle):
            categories["wrist_muhi"] += 1
        distance = float(con.dist) * 1000
        minimum = distance if minimum is None else min(minimum, distance)
        details.append({"geom1": names[0], "geom2": names[1],
                        "category": category, "distance_mm": distance,
                        "wrist_muhi": bool((a_wrist and b_bottle) or (b_wrist and a_bottle))})
    return categories, minimum, details

def read_replay(path):
    from render_wrist_camera_rollout import load_rollout
    return load_rollout(path, "normal")[1]


def run(args):
    if args.out.exists():
        raise FileExistsError("Output exists: " + str(args.out))
    if args.steps < 1 or args.render_every < 1 or args.width < 64 or args.height < 64:
        raise ValueError("Invalid steps/render settings")
    if args.mode == "replay" and not args.rollout:
        raise ValueError("--rollout required in replay mode")
    if args.mode == "replay" and args.rollout and not args.rollout.is_file():
        raise FileNotFoundError(args.rollout)
    args.physical_objects = True
    args.wrist_camera = True
    (model, data, qids, dids, low, high, tcp, side_camera, goals,
     _cup, _layout, _offset_path, _offset_cfg) = scene.create_scene(args)
    enabled = prepare_environment(model)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, DYNAMIC_BODY)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, DYNAMIC_JOINT)
    adr = int(model.jnt_qposadr[joint_id])
    object_dof = int(model.jnt_dofadr[joint_id])
    # Valid physics start, NOT the out-of-bounds visual recorded home.
    data.qpos[qids] = np.clip(data.qpos[qids], low, high)
    data.qvel[:] = 0
    if args.drop_height_mm:
        data.qpos[adr+2] += args.drop_height_mm / 1000
    mujoco.mj_forward(model, data)
    initial_xyz = data.qpos[adr:adr+3].copy()
    if args.mode == "replay":
        targets = read_replay(args.rollout)
    else:
        targets = None

    args.out.mkdir(parents=True, exist_ok=False)
    side_dir, wrist_dir = args.out/"side", args.out/"wrist"
    side_dir.mkdir()
    wrist_dir.mkdir()
    records = []
    render_option = mujoco.MjvOption()
    render_option.geomgroup[3] = 0
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    with mujoco.Renderer(model, width=args.width, height=args.height) as renderer:
        for step in range(args.steps):
            # xfrc_applied is a known EXTERNAL diagnostic force, not robot contact.
            data.xfrc_applied[:] = 0
            if args.mode == "push" and step < args.force_steps:
                data.xfrc_applied[object_id, 0] = args.force_n
            data.qfrc_applied[:] = 0
            if args.mode == "replay":
                target = np.clip(targets[min(step//args.control_substeps, len(targets)-1)], low, high)
                demand = args.kp*(target-data.qpos[qids]) - args.kd*data.qvel[dids] + data.qfrc_bias[dids]
                data.qfrc_applied[dids] = np.clip(demand, -args.torque, args.torque)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError("Nonfinite MuJoCo state; inspect self collision and gains")
            categories, min_dist, _ = summarize_contacts(model, data, object_id)
            point = np.asarray(data.qpos[adr:adr+3]).copy()
            rows = {
                "step": step, "time_s": float(data.time),
                "muhi_x_m": float(point[0]), "muhi_y_m": float(point[1]),
                "muhi_z_m": float(point[2]),
                "muhi_displacement_mm": float(1000*np.linalg.norm(point-initial_xyz)),
                "muhi_velocity_m_s": float(np.linalg.norm(data.qvel[object_dof:object_dof+3])),
                "contacts": int(data.ncon),
                "min_contact_distance_mm": min_dist,
                **categories,
            }
            records.append(rows)
            if step % args.render_every == 0 or step == args.steps-1:
                for name, cam, directory in (("side",side_camera,side_dir),
                                              ("wrist",wrist.CAMERA,wrist_dir)):
                    renderer.update_scene(data, camera=cam, scene_option=render_option)
                    Image.fromarray(renderer.render().copy()).save(
                        directory / f"{name}_{step:05d}.png")
    with (args.out/"physics_trace.csv").open("x", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot([r["time_s"] for r in records],
                 [r["muhi_displacement_mm"] for r in records])
    axes[0].set_ylabel("Muhi displacement (mm)")
    axes[1].plot([r["time_s"] for r in records],
                 [r["contacts"] for r in records], label="all contacts")
    axes[1].plot([r["time_s"] for r in records],
                 [r["muhi_table"]+r["muhi_cup"]+r["muhi_robot"] for r in records],
                 label="Muhi contacts")
    axes[1].set(xlabel="Simulation time (s)", ylabel="Contact records")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out/"physics_trace.png", dpi=160)
    plt.close(fig)
    summary = {
        "status": "PHYSICAL_MUHI_WORKCELL_EXPERIMENTAL",
        "mode": args.mode,
        "simulation_only": True,
        "source_scene": "saved Episode 4 real-to-sim approximation",
        "physical_time_s": float(data.time),
        "frames": len(records),
        "final_muhi_displacement_mm": records[-1]["muhi_displacement_mm"],
        "max_muhi_displacement_mm": max(r["muhi_displacement_mm"] for r in records),
        "min_contact_distance_mm": min(
            (r["min_contact_distance_mm"] for r in records
             if r["min_contact_distance_mm"] is not None), default=None),
        "steps_with_muhi_cup_contact": sum(r["muhi_cup"]>0 for r in records),
        "steps_with_muhi_table_contact": sum(r["muhi_table"]>0 for r in records),
        "steps_with_wrist_contact": sum((r["wrist_table"]+r["wrist_cup"]+r["wrist_muhi"]+r["wrist_robot_self"])>0 for r in records),
        "environment_geoms_enabled": enabled,
        "physical_body": DYNAMIC_BODY,
        "physical_geom_names": DYNAMIC_GEOMS,
        "object_mass_kg_provisional": .050,
        "wrist_mass_kg_provisional": .025,
        "model_timestep_s": float(model.opt.timestep),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "limits": [
            "No grasp mechanism or ACT inference; replay uses bounded PD targets, not SAC actions.",
            "Push mode applies an external diagnostic force to the bottle; it is not a robot push.",
            "Muhi mass, static cup geometry, wrist mount, friction and calibration are provisional.",
            "Original URDF has potentially spurious base/shoulder mesh contacts.",
            "Metrics are MuJoCo numeric simulation results, not real hardware observations.",
        ]
    }
    (args.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in
          ("mode","final_muhi_displacement_mm","steps_with_muhi_table_contact",
           "steps_with_muhi_cup_contact","steps_with_wrist_contact")}, indent=2))
    print("Saved:", args.out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("drop","push","replay"), default="drop")
    p.add_argument("--rollout", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--render-every", type=int, default=30)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--drop-height-mm", type=float, default=20)
    p.add_argument("--force-n", type=float, default=.8)
    p.add_argument("--force-steps", type=int, default=80)
    p.add_argument("--kp", type=float, default=12.)
    p.add_argument("--kd", type=float, default=1.1)
    p.add_argument("--torque", type=float, default=3.)
    p.add_argument("--control-substeps", type=int, default=10)
    p.add_argument("--dataset-root", type=Path, default=scene.DATASET)
    p.add_argument("--episode", type=int, default=4)
    p.add_argument("--object-layout-json", type=Path, default=scene.OBJECTS)
    p.add_argument("--cup-pose-json", type=Path, default=scene.CUP_POSE)
    p.add_argument("--extrinsics", type=Path, default=scene.EXTRINSICS)
    p.add_argument("--intrinsics", type=Path, default=scene.INTRINSICS)
    p.add_argument("--urdf", type=Path, default=scene.r2s.DEFAULT_URDF)
    p.add_argument("--robot-joint-offsets-json", type=Path)
    p.add_argument("--muhi-clearance-mm", type=float, default=75.)
    p.add_argument("--cup-clearance-mm", type=float, default=45.)
    p.add_argument("--wrist-mount-pos", type=float, nargs=3, default=wrist.DEFAULT_MOUNT_POS)
    p.add_argument("--wrist-mount-rpy-deg", type=float, nargs=3, default=wrist.DEFAULT_MOUNT_RPY_DEG)
    p.add_argument("--wrist-fovy", type=float, default=wrist.DEFAULT_FOVY_DEG)
    p.add_argument("--wrist-look-at", type=float, nargs=3, default=wrist.DEFAULT_LOOK_AT)
    args = p.parse_args()
    if not np.isfinite([args.drop_height_mm,args.force_n,args.kp,args.kd,args.torque]).all():
        p.error("Nonfinite physical parameters")
    if args.control_substeps < 1 or args.torque <= 0 or args.drop_height_mm < 0:
        p.error("Invalid physical parameters")
    run(args)


if __name__ == "__main__":
    main()
