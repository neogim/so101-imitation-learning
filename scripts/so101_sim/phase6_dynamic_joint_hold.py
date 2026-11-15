#!/usr/bin/env python3
"""Phase 6A: SO-101 J1-J5 physical position hold without Muhi.

No robot hardware, no bottle, no cameras or proxies. Actuators are active
FROM THE FIRST mj_step; qpos/qvel are never overwritten inside the loop.
This is a deliberately small baseline before 5-mm TCP IK and grasp.
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


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"Output exists: {args.out}")
    urdf = args.urdf.resolve(strict=True)
    names = (*r2s.JOINTS, "gripper")
    angles_deg = (*args.arm_deg, args.j6_deg)
    if not all(math.isfinite(v) for v in angles_deg):
        raise ValueError("Nonfinite start joint angle")

    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.compiler.discardvisual = False
    # No control-mode handover: position actuators exist before compile/step.
    for name in names:
        if spec.joint(name) is None:
            raise ValueError(f"Missing joint in original URDF: {name}")
        a = spec.add_actuator(name=f"phase6_hold_{name}", target=name)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = args.kp
        a.biasprm[1] = -args.kp
        a.biasprm[2] = -args.kd
        a.ctrllimited = False
        a.forcelimited = True
        a.forcerange[:] = [-args.max_torque_nm, args.max_torque_nm]

    model = spec.compile()
    # Phase6A has no workcell or grasp object. The original full-mesh
    # URDF starts with 12 robot/robot collision candidates at this pose.
    # Keep those intact in the historical mode for an A/B comparison;
    # isolate servo dynamics by disabling ONLY collision detection in the
    # object-free baseline (inertia, gravity, limits and PD stay active).
    if args.self_collision == "off":
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    ids = []
    qs = []
    vs = []
    acts = []
    goals = []
    limits = {}
    for name, value in zip(names, angles_deg):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"phase6_hold_{name}")
        if jid < 0 or aid < 0:
            raise RuntimeError(f"Joint/actuator compilation failed: {name}")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected a 1DOF hinge: {name}")
        goal = math.radians(value)
        if bool(model.jnt_limited[jid]):
            lo, hi = map(float, model.jnt_range[jid])
            if not(lo + 1e-5 < goal < hi - 1e-5):
                raise ValueError(
                    f"{name}: {value} deg outside safe interior "
                    f"({math.degrees(lo):.2f}, {math.degrees(hi):.2f})")
            limits[name] = [math.degrees(lo), math.degrees(hi)]
        ids.append(jid)
        qs.append(int(model.jnt_qposadr[jid]))
        vs.append(int(model.jnt_dofadr[jid]))
        acts.append(aid)
        goals.append(goal)

    # Single state initialization ONLY. All subsequent states are integrated.
    data.qpos[qs] = goals
    data.qvel[vs] = 0.
    data.ctrl[acts] = goals
    mujoco.mj_forward(model, data)
    initial_contacts = int(data.ncon)
    initial_poses = data.qpos[qs].copy()
    goals = np.array(goals, dtype=float)
    error_max = np.zeros(len(names),dtype=float)
    torque_max = np.zeros(len(names),dtype=float)
    speed_max = np.zeros(len(names),dtype=float)
    finite = True
    rows = []
    steps = int(round(args.duration_s / args.dt))

    args.out.mkdir(parents=True,exist_ok=False)
    for step in range(steps):
        # Targets are constant, present since the first step.
        # Do not write qpos, qvel, or bottle state here.
        mujoco.mj_step(model, data)
        pos = np.asarray(data.qpos[qs],dtype=float)
        vel = np.asarray(data.qvel[vs],dtype=float)
        force = np.asarray(data.actuator_force[acts],dtype=float)
        err = goals - pos
        if not(all(np.isfinite(pos)) and all(np.isfinite(vel))
               and all(np.isfinite(force))):
            finite = False
            break
        error_max = np.maximum(error_max,np.abs(err))
        torque_max = np.maximum(torque_max,np.abs(force))
        speed_max = np.maximum(speed_max,np.abs(vel))
        row = {"step":step+1,"time_s":float(data.time),
               "ncon":int(data.ncon)}
        for i,name in enumerate(names):
            row[f"{name}_target_deg"]=float(math.degrees(goals[i]))
            row[f"{name}_actual_deg"]=float(math.degrees(pos[i]))
            row[f"{name}_error_deg"]=float(math.degrees(err[i]))
            row[f"{name}_velocity_deg_s"]=float(math.degrees(vel[i]))
            row[f"{name}_actuator_torque_nm"]=float(force[i])
        rows.append(row)

    if rows:
        with (args.out/"joint_trace.csv").open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    status=("NUMERIC_FAILURE" if not finite else
            "SIMULATION_COMPLETED_UNVERIFIED_HOLD")
    # A completed simulation is not automatically a mechanically valid hold.
    summary={
        "experiment":"phase6a_j1_j5_dynamic_constant_pose_no_object",
        "status":status,
        "simulation_only":True,
        "physical_arm_joints":list(r2s.JOINTS),
        "j6_position_held_as_auxiliary":True,
        "source_urdf":str(urdf),
        "source_urdf_modified":False,
        "muhi_present":False,
        "cameras_required_for_this_test":False,
        "contact_proxy_present":False,
        "control_mode":"position_PD_actuators_from_first_physics_step",
        "no_kinematic_state_overwrite_during_loop":True,
        "no_midrun_controller_switch":True,
        "dt_s":args.dt,
        "duration_requested_s":args.duration_s,
        "physics_steps_requested":steps,
        "physics_steps_completed":len(rows),
        "initial_contacts_count":initial_contacts,
        "self_collision_mode":args.self_collision,
        "kp_nm_per_rad":args.kp,
        "kd_nms_per_rad":args.kd,
        "actuator_torque_cap_nm":args.max_torque_nm,
        "initial_deg":{name:float(deg) for name,deg in zip(names,angles_deg)},
        "initial_qpos_degrees_from_compiled_model":{
            name:math.degrees(float(q)) for name,q in zip(names,initial_poses)},
        "urdf_joint_limits_deg":limits,
        "max_abs_error_deg":{n:math.degrees(float(v)) for n,v in zip(names,error_max)},
        "max_abs_joint_speed_deg_s":{
            n:math.degrees(float(v)) for n,v in zip(names,speed_max)},
        "max_abs_actuator_torque_nm":{
            n:float(v) for n,v in zip(names,torque_max)},
        "final_abs_error_deg":{
            n:abs(float(rows[-1][f"{n}_error_deg"])) for n in names}
            if rows else None,
        "final_ncon":rows[-1]["ncon"] if rows else None,
        "limitation":[
            "PD gains and torque caps are initial simulation assumptions, not calibrated servos.",
            "A run that completes is not automatically stable; inspect CSV angular errors and oscillation.",
            "This is NOT a 5mm TCP movement or a physical grasp test.",
            "No wrist rendering changes and no camera calibration performed.",
            "Self-collision off isolates joint servo from overlapping full-mesh robot contacts, and does not validate later grasp collisions."
        ]
    }
    (args.out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":status,"steps":len(rows),
          "initial_contacts_count":initial_contacts,
          "max_abs_error_deg":summary["max_abs_error_deg"],
          "final_abs_error_deg":summary["final_abs_error_deg"],
          "max_abs_actuator_torque_nm":summary["max_abs_actuator_torque_nm"],
          "output":str(args.out)},ensure_ascii=False,indent=2))


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--arm-deg",type=float,nargs=5,
                   default=[-30.,-30.,45.,-20.,-90.],
                   metavar=("J1","J2","J3","J4","J5"))
    p.add_argument("--j6-deg",type=float,default=60.)
    p.add_argument("--duration-s",type=float,default=4.)
    p.add_argument("--dt",type=float,default=.002)
    p.add_argument("--kp",type=float,default=18.)
    p.add_argument("--kd",type=float,default=1.8)
    p.add_argument("--max-torque-nm",type=float,default=3.)
    p.add_argument("--self-collision",choices=("off","on"),default="off",
                   help="off: object-free actuator benchmark (default); on: old full-mesh contact behavior")
    args=p.parse_args()
    if not(.5<=args.duration_s<=10. and .001<=args.dt<=.01
           and 0<args.kp<=100 and 0<=args.kd<=20
           and 0<args.max_torque_nm<=10):
        p.error("Invalid physical hold parameters")
    run(args)


if __name__=="__main__":
    main()
