#!/usr/bin/env python3
"""Table-supported SO-101 J6 contact test (simulation-only, NO pickup).

An independent bench that retains the original URDF and dynamic Muhi model.
First chooses a nonpenetrating two-jaw approach; then constructs an ACTUAL
collidable table underneath the upright Muhi. The free object settles under
gravity, and J6 closes under limited torque while J1..J5 are kinematically
held. Never writes the object's pose after starting mj_step, and applies
NO virtual external "fixture" force. This verifies contacts on a table,
NOT Lift/Place and NOT an autonomous pickup from the saved Episode 4 scene.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

import physical_grasp_j6_bench as bench
import physical_workcell_sim as physical
import real_to_sim as r2s
import compare_id1_eye_to_hand as preview

TABLE = "j6_bench_table"
TABLE_HALF_THICKNESS_M = .012
OBJECT_HALF_LENGTH_M = physical.MUHI_LENGTH / 2.


def body_name(model, geom_id):
    body=int(model.geom_bodyid[int(geom_id)])
    return mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body) or str(body)


def build_model(args):
    """Select Muhi pose ONLY if its supporting table also clears the arm."""
    spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    spec.compiler.discardvisual=False
    physical.install_dynamic_muhi(spec)
    # Use an actual small tabletop, not a fake vertical holding force.
    # Its pose is adjusted during placement search (world geom, no freejoint).
    # A direct worldbody geom is compiled as a STATIC world geom. Editing
    # model.geom_pos later leaves its world contact pose stale in mjData.
    # Instead attach the table to a mocap body, and move that body only
    # during candidate selection. mj_forward then updates the contact pose.
    table_body_spec=spec.worldbody.add_body(
        name="j6_bench_table_mocap",mocap=True,pos=[0.,0.,-10.])
    table_body_spec.add_geom(
        name=TABLE,type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.,0.,0.],size=[.08,.08,TABLE_HALF_THICKNESS_M],
        rgba=[.47,.51,.56,1.],friction=[.9,.01,.001],
        contype=1,conaffinity=1)
    model=spec.compile()
    preview.apply_white_follower_preview(model)
    model.opt.timestep=.002
    data=mujoco.MjData(model)
    mujoco.mj_resetData(model,data)
    qids=[];dids=[];lows=[];highs=[]
    for name in r2s.JOINTS:
        q,d,limits=bench.joint_addr(model,name)
        if not limits[0] <= 0 <= limits[1]:
            raise ValueError(f"{name}: zero not inside URDF limits")
        data.qpos[q]=0.
        qids.append(q);dids.append(d)
        lows.append(float(limits[0]));highs.append(float(limits[1]))
    j5q=qids[r2s.JOINTS.index("wrist_roll")]
    j5rad=math.radians(args.j5_deg)
    if not lows[-1] <= j5rad <= highs[-1]:
        raise ValueError("J5 angle outside URDF")
    data.qpos[j5q]=j5rad
    jawq,jawd,jawlim=bench.joint_addr(model,"gripper")
    open_angle=float(jawlim[1]);closed_angle=float(jawlim[0])
    data.qpos[jawq]=open_angle
    fixed=bench.owned_collision_geoms(model,bench.FINGER_BODY)
    moving=bench.owned_collision_geoms(model,bench.MOVING_BODY)
    obj=tuple(physical.geom_id(model,n) for n in physical.DYNAMIC_GEOMS)
    table=physical.geom_id(model,TABLE)
    table_body_id=mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_BODY,"j6_bench_table_mocap")
    table_mocap_id=int(model.body_mocapid[table_body_id])
    if table_mocap_id < 0:
        raise ValueError("Bench table is not attached to a mocap body")
    if not fixed or not moving:
        raise ValueError("Missing jaw collision geoms")
    freejoint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    freeq=int(model.jnt_qposadr[freejoint])
    mujoco.mj_forward(model,data)
    # Exclude both the bottle and the table from arm collision candidates.
    robot_geoms=tuple(g for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) not in
            (0,table_body_id,int(model.geom_bodyid[obj[0]]))
        and (model.geom_contype[g] or model.geom_conaffinity[g]))
    rejected={"tested":0,"rejected":0,"max_rejected_clearance_mm":None}
    def table_for_candidate(center,quat,xyz):
        rejected["tested"]+=1
        table_top=float(center[2])-OBJECT_HALF_LENGTH_M-.0005
        data.mocap_pos[table_mocap_id]=[
            float(center[0]),float(center[1]),
            table_top-TABLE_HALF_THICKNESS_M]
        mujoco.mj_forward(model,data)
        gap=min((bench.signed_pair_gap(model,data,g,table)
                 for g in robot_geoms),default=float("inf"))
        if gap < args.table_clearance_mm/1000.:
            rejected["rejected"]+=1
            if rejected["max_rejected_clearance_mm"] is None or (
               gap*1000 > rejected["max_rejected_clearance_mm"]):
                rejected["max_rejected_clearance_mm"]=float(gap*1000)
            return False
        return True
    pose,search=bench.select_bench_pose(
        model,data,freeq,open_angle,closed_angle,jawq,
        fixed,moving,obj,candidate_validator=table_for_candidate)
    if pose is None:
        return None,{"status":"NO_TABLE_CLEAR_TWO_JAW_SETUP",
                     "search":search,"table_clearance_search":rejected,
                     "table_clearance_requirement_mm":args.table_clearance_mm,
                     "note":"No valid pose at fixed J1-J5. Do not insert "
                            "table through a robot or enlarge joint limits."}
    center,quat=pose[:2]
    if abs(float(quat[0])-1.)>1e-7 or np.linalg.norm(quat[1:])>1e-7:
        raise ValueError("Table test requires upright Muhi")
    # The last candidate evaluated in the grid is NOT necessarily the
    # winning candidate. Reposition table using the selected Muhi.
    table_top=float(center[2])-OBJECT_HALF_LENGTH_M-.0005
    data.mocap_pos[table_mocap_id]=[
        float(center[0]),float(center[1]),
        table_top-TABLE_HALF_THICKNESS_M]
    data.qpos[freeq:freeq+3]=center
    data.qpos[freeq+3:freeq+7]=quat
    data.qpos[jawq]=open_angle
    data.qvel[:]=0.
    mujoco.mj_forward(model,data)
    gap_fixed=bench.geometry_gaps(model,data,obj,fixed)
    gap_moving=bench.geometry_gaps(model,data,obj,moving)
    robot_gap=min((bench.signed_pair_gap(model,data,g,table)
                   for g in robot_geoms),default=float("inf"))
    expected_table_center=np.array(
        [float(center[0]),float(center[1]),
         table_top-TABLE_HALF_THICKNESS_M])
    actual_table_center=np.asarray(data.geom_xpos[table]).copy()
    if np.linalg.norm(actual_table_center-expected_table_center)>.0001:
        return None,{"status":"TABLE_KINEMATICS_INVALID",
                     "expected_table_center":expected_table_center.tolist(),
                     "actual_table_center":actual_table_center.tolist()}
    diag={"search":search,"table_clearance_search":rejected,
          "table_actual_world_center_xyz_m":actual_table_center.tolist(),
          "initial_fixed_gap_mm":float(gap_fixed*1000),
          "initial_moving_gap_mm":float(gap_moving*1000),
          "initial_robot_table_gap_mm":float(robot_gap*1000),
          "table_top_world_z_m":table_top,
          "muhi_center_world_xyz_m":list(map(float,center)),
          "object_bottom_world_z_m":float(center[2])-OBJECT_HALF_LENGTH_M,
          "table_support_gap_mm":.5}
    if min(gap_fixed,gap_moving)<.0005:
        return None,{"status":"INITIAL_JAW_INTERSECTION","diagnostic":diag}
    if robot_gap<args.table_clearance_mm/1000.:
        return None,{"status":"ROBOT_TABLE_INTERSECTION","diagnostic":diag}
    return (model,data,tuple(qids),tuple(dids),np.asarray(lows),np.asarray(highs),
            jawq,jawd,freeq,fixed,moving,obj,table,center,
            open_angle,closed_angle,robot_geoms),diag

def contact_observations(model,data,object_geoms,fixed,moving,table_id):
    """Report contact points, force, signed depth and support separately."""
    obj=set(object_geoms);fs=set(fixed);ms=set(moving)
    records=[]
    counts={"fixed":0,"moving":0,"table":0}
    for index,con in enumerate(data.contact[:data.ncon]):
        gs={int(con.geom1),int(con.geom2)}
        if not (gs&obj):
            continue
        other=gs-obj
        # A contact between bottle sub-geoms is not a jaw/table event.
        kind=("fixed" if other&fs else "moving" if other&ms else
              "table" if table_id in other else "other")
        force=np.zeros(6,dtype=float)
        mujoco.mj_contactForce(model,data,index,force)
        if kind in counts: counts[kind]+=1
        records.append(dict(
           kind=kind,geom1=int(con.geom1),geom2=int(con.geom2),
           point_xyz_m=[float(v) for v in con.pos],
           contact_normal_xyz=[float(v) for v in con.frame[:3]],
           penetration_mm=float(min(0.,con.dist)*1000.),
           normal_force_n=float(force[0]),
           tangential_force_n=float(np.linalg.norm(force[1:3]))))
    return records,counts


def execute(args):
    if args.out.exists():raise FileExistsError(args.out)
    if not(0<args.settle_s<args.duration_s<=10 and args.kp>0 and
           args.kd>=0 and args.max_torque>0 and args.render_every>=1):
        raise ValueError("Bad simulation parameters")
    built,diag=build_model(args)
    args.out.mkdir(parents=True,exist_ok=False)
    info={"experiment":"J6 tabletop supported Muhi contact","version":"v1",
          "simulation_only":True,
          "not_full_pick_place":True,
          "no_fixture_force":True,"no_object_pose_reset_during_steps":True,
          "table_is_mocap_fixed_during_simulation":True,
          "j5_deg":args.j5_deg,"scene":"standalone local table, not saved Episode4",
          "object_mass_provisional_kg":.05,
          "object_diameter_provisional_m":.035,
          "object_length_provisional_m":.097,
          "diagnostic":diag,"status":"NOT_STARTED"}
    if built is None:
        info["status"]=diag["status"]
        (args.out/"summary.json").write_text(json.dumps(info,indent=2,ensure_ascii=False)+"\n")
        print(json.dumps({"status":info["status"],"diagnostic":diag},ensure_ascii=False,indent=2))
        return
    model,data,qids,dids,lows,highs,jawq,jawd,freeq,fixed,moving,obj,table,center,open_angle,closed_angle,robot_geoms=built
    armhome=data.qpos[list(qids)].copy()
    # Approach with a fingertip-area reference, not the gripper motor body.
    # The J1-J4 move is simulation-only prescribed IK, never a servo command.
    tip_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"gripper_frame_link")
    if tip_body < 0:raise ValueError("gripper_frame_link is missing from URDF")
    tip_xyz=np.asarray(data.xpos[tip_body]).copy()
    delta_xy=np.asarray(center[:2],float)-tip_xyz[:2]
    distance_xy=float(np.linalg.norm(delta_xy))
    approach_distance=min(float(args.approach_mm)/1000.,max(0.,distance_xy-.012))
    approach_target=tip_xyz.copy()
    if distance_xy>1e-9:
        approach_target[:2]+=delta_xy/distance_xy*approach_distance
    approach_blocked_steps=0
    approach_max_mm=0.
    grip_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,bench.FINGER_BODY)
    mj_initial_gripper_z=float(data.xpos[grip_body,2])
    mj_initial_object_z=float(data.qpos[freeq+2])
    lift_requested=False
    lift_started_at=None
    lift_rejected_steps=0
    rows=[];contactrows=[]
    no_table_support=False
    frames=[]
    render=None
    cam=None
    if args.video:
        from PIL import Image,ImageDraw
        model.vis.global_.offwidth=max(480,model.vis.global_.offwidth)
        model.vis.global_.offheight=max(360,model.vis.global_.offheight)
        render=mujoco.Renderer(model,width=480,height=360)
        cam=mujoco.MjvCamera()
        cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]=np.array(center,dtype=float)
        cam.distance=.32
        cam.elevation=-20
    n=math.ceil(args.duration_s/model.opt.timestep)
    def try_approach_step(goal):
        """Bounded DLS joint step, requires robot-table AND robot-object clearance."""
        saved=data.qpos[list(qids)].copy()
        mujoco.mj_forward(model,data)
        jac=np.zeros((3,model.nv));jac_r=np.zeros((3,model.nv))
        mujoco.mj_jacBody(model,data,jac,jac_r,tip_body)
        J=jac[:,list(dids[:4])]
        err=np.asarray(goal,float)-np.asarray(data.xpos[tip_body]).copy()
        err[2]=0.
        dq=J.T@np.linalg.solve(J@J.T+.05**2*np.eye(3),err)
        dq=np.clip(dq,-.0012,.0012)
        trial=armhome.copy()
        trial[:4]=np.clip(trial[:4]+dq,lows[:4],highs[:4])
        data.qpos[list(qids)]=trial
        mujoco.mj_forward(model,data)
        table_gap=min((bench.signed_pair_gap(model,data,g,table)
                       for g in robot_geoms),default=float("inf"))
        object_gap=min((bench.signed_pair_gap(model,data,g,o)
                        for g in robot_geoms for o in obj),default=float("inf"))
        data.qpos[list(qids)]=saved
        mujoco.mj_forward(model,data)
        return trial,(table_gap>=args.table_clearance_mm/1000. and
                      object_gap>=args.approach_clearance_mm/1000.)
    try:
        for i in range(n):
            t=float(data.time)
            # J1-J5 are a rigid kinematic fixture ONLY. The object and J6
            # are fully dynamic; table is fixed and supports gravity.
            data.qpos[list(qids)]=armhome
            data.qvel[list(dids)]=0.
            data.xfrc_applied[:]=0.
            data.qfrc_applied[:]=0.
            if t<args.settle_s:
                stage="settle"
                demand=open_angle
            elif t < args.settle_s + args.approach_s:
                stage="approach"
                demand=open_angle
                v=min(1.,max(0.,(t-args.settle_s)/args.approach_s))
                smooth=v*v*(3.-2.*v)
                goal=tip_xyz+(approach_target-tip_xyz)*smooth
                trial,clear=try_approach_step(goal)
                if clear:
                    armhome=trial
                    data.qpos[list(qids)]=armhome
                    mujoco.mj_forward(model,data)
                    approach_max_mm=max(approach_max_mm,1000.*float(
                      np.linalg.norm(data.xpos[tip_body,:2]-tip_xyz[:2])))
                else:
                    approach_blocked_steps+=1
            else:
                u=min(1.,max(0.,(t-args.settle_s-args.approach_s)/args.close_s))
                smooth=u*u*(3.-2.*u)
                demand=open_angle+smooth*(closed_angle-open_angle)
                stage="close" if u<1. else "hold"
                # Only attempt a kinematically guided 5mm lift after
                # sustained TWO-JAW contact in recent on-table samples.
                # A table-supported contact alone never proves a pickup.
                if (not lift_requested and
                    t>args.settle_s+args.approach_s+args.close_s+.22 and
                    len(rows)>=60 and
                    sum(r["bilateral"] for r in rows[-60:])>=48 and
                    sum(r["table_contacts"]>0 for r in rows[-60:])>=48 and
                    rows[-1]["min_object_contact_depth_mm"]>-2.):
                    lift_requested=True
                    lift_started_at=t
                if lift_requested:
                    stage="lift"
                    progress=min(1.,(t-lift_started_at)/.7)
                    wanted_z=mj_initial_gripper_z+args.lift_mm/1000.*progress
                    mujoco.mj_forward(model,data)
                    jac=np.zeros((3,model.nv))
                    jac_r=np.zeros((3,model.nv))
                    mujoco.mj_jacBody(model,data,jac,jac_r,grip_body)
                    J=jac[:,list(dids[:4])]
                    error=np.array([0.,0.,wanted_z-float(data.xpos[grip_body,2])])
                    dq=J.T@np.linalg.solve(J@J.T+.035**2*np.eye(3),error)
                    dq=np.clip(dq,-.002,.002)
                    oldq=armhome.copy()
                    trial=armhome.copy()
                    trial[:4]=np.clip(trial[:4]+dq,lows[:4],highs[:4])
                    data.qpos[list(qids)]=trial
                    mujoco.mj_forward(model,data)
                    clearance=min((bench.signed_pair_gap(model,data,g,table)
                                  for g in robot_geoms),default=float("inf"))
                    if clearance>=args.table_clearance_mm/1000.:
                        armhome=trial
                    else:
                        lift_rejected_steps+=1
                        data.qpos[list(qids)]=oldq
                        mujoco.mj_forward(model,data)
            torque=args.kp*(demand-data.qpos[jawq])-args.kd*data.qvel[jawd]
            commanded_torque=float(np.clip(torque,-args.max_torque,args.max_torque))
            data.qfrc_applied[jawd]=commanded_torque
            mujoco.mj_step(model,data)
            data.qpos[list(qids)]=armhome
            data.qvel[list(dids)]=0.
            mujoco.mj_forward(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError(f"nonfinite state at step={i}")
            contacts,counts=contact_observations(model,data,obj,fixed,moving,table)
            for c in contacts:contactrows.append({"step":i,"time_s":float(data.time),**c})
            rotation=np.empty(9)
            mujoco.mju_quat2Mat(rotation,data.qpos[freeq+3:freeq+7])
            tilt=math.degrees(math.acos(float(np.clip(rotation[8],-1.,1.))))
            rows.append({
                "step":i,"time_s":float(data.time),"stage":stage,
                "j6_rad":float(data.qpos[jawq]),
                "tcp_gripper_z_m":float(data.xpos[grip_body,2]),
                "gripper_lift_mm":float((data.xpos[grip_body,2]-mj_initial_gripper_z)*1000.),
                "j6_target_rad":float(demand),
                "j6_torque_nm":float(data.qfrc_applied[jawd]),
                "j6_commanded_torque_nm":commanded_torque,
                "approach_tip_horizontal_mm":float(
                    np.linalg.norm(data.xpos[tip_body,:2]-tip_xyz[:2])*1000.),
                "object_x_m":float(data.qpos[freeq]),
                "object_y_m":float(data.qpos[freeq+1]),
                "object_z_m":float(data.qpos[freeq+2]),
                "object_tilt_deg":tilt,
                "object_speed_m_s":float(np.linalg.norm(data.qvel[model.jnt_dofadr[
                   mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)]:
                   model.jnt_dofadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)]+3])),
                "fixed_jaw_contacts":counts["fixed"],
                "moving_jaw_contacts":counts["moving"],
                "table_contacts":counts["table"],
                "bilateral":bool(counts["fixed"] and counts["moving"]),
                "min_object_contact_depth_mm":min((c["penetration_mm"] for c in contacts),default=0.),
            })
            if render is not None and (i%args.render_every==0 or i==n-1):
                pics=[]
                for az in (55,150):
                    cam.azimuth=az
                    render.update_scene(data,camera=cam)
                    pics.append(Image.fromarray(render.render().copy()))
                # Even height required for libx264/yuv420p.
                image=Image.new("RGB",(960,396),(20,24,30))
                image.paste(pics[0],(0,36));image.paste(pics[1],(480,36))
                ImageDraw.Draw(image).text(
                   (8,8),f"{stage.upper()} t={data.time:.2f}s | "
                   f"fixed={counts['fixed']} moving={counts['moving']} "
                   f"table={counts['table']} tilt={tilt:.1f}deg",fill="white")
                frames.append(image)
            # The Muhi starts only 0.5mm above the table. If no contact
            # occurs by 0.15s and it has already dropped 5mm below the
            # nominal starting position, this is a broken table collision
            # setup, not a failed grasp. Stop before wasting simulation time.
            if (stage=="settle" and t>=.15 and
                not any(row["table_contacts"]>0 for row in rows) and
                data.qpos[freeq+2]<mj_initial_object_z-.005):
                no_table_support=True
                break
    finally:
        if render is not None:render.close()
    with (args.out/"table_grasp_trace.csv").open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (args.out/"contact_points.csv").open("x",newline="",encoding="utf-8") as f:
        fields=["step","time_s","kind","geom1","geom2","point_xyz_m",
                "contact_normal_xyz","penetration_mm",
                "normal_force_n","tangential_force_n"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows(contactrows)
    hold=[r for r in rows if r["stage"]=="hold"]
    # This is a TABLE-CONTACT criterion only. NEVER classify this as
    # successful pickup: the object may still be resting on the table.
    simultaneous=sum(r["bilateral"] for r in hold)
    deepest=min(r["min_object_contact_depth_mm"] for r in rows)
    max_tilt=max(r["object_tilt_deg"] for r in rows)
    supported=sum(r["table_contacts"]>0 for r in rows if r["stage"]=="settle")
    max_lift_z=max((r["object_z_m"]-mj_initial_object_z)*1000.
                   for r in rows if r["stage"]=="lift") if lift_requested else 0.
    lift_samples=[r for r in rows if r["stage"]=="lift"]
    lift_pass=bool(lift_samples and
        max_lift_z>=.8*args.lift_mm and
        any(r["table_contacts"]==0 and r["bilateral"] for r in lift_samples) and
        deepest>-2.)
    result_status=("SHORT_LIFT_OBSERVED" if lift_pass else
                   "SHORT_LIFT_FAILED" if lift_requested else
                   "TABLE_CONTACT_OBSERVED" if simultaneous else
                   "TABLE_BILATERAL_NOT_OBSERVED")
    if no_table_support:
        result_status="NO_TABLE_SUPPORT_CONTACT"
    info.update({
        "status":result_status,
        "table_support_preflight_failed":no_table_support,
        "approach_requested_mm":float(args.approach_mm),
        "approach_executed_max_horizontal_mm":float(approach_max_mm),
        "approach_blocked_steps":int(approach_blocked_steps),
        "approach_method":"DLS J1-J4, check full robot/table and robot/object clearance",
        "j6_trajectory":"cubic smoothstep with velocity-zero endpoints",
        "short_lift_command_mm":args.lift_mm,
        "short_lift_attempted":lift_requested,
        "short_lift_success":lift_pass,
        "max_object_vertical_rise_mm":float(max_lift_z),
        "lift_table_clearance_rejected_steps":lift_rejected_steps,
        "arm_lift_control":"DLS kinematic J1-J4; J5 fixed, J6 torque PD; object free dynamics",
        "physics_steps":len(rows),
        "settle_table_contact_steps":supported,
        "hold_steps":len(hold),
        "hold_bilateral_contact_steps":simultaneous,
        "deepest_object_contact_mm":deepest,
        "max_object_tilt_deg":max_tilt,
        "final_object_xyz_m":[float(v) for v in data.qpos[freeq:freeq+3]],
        "final_bilateral":rows[-1]["bilateral"],
        "final_table_supported":bool(rows[-1]["table_contacts"]),
        "lifting_success":"SHORT_LIFT_OBSERVED" if lift_pass else "NOT_OBSERVED",
        "grasp_success":"SHORT_LIFT_OBSERVED" if lift_pass else "NOT_ESTABLISHED",
        "note":"Short lift is eligible only after sustained bilateral table contacts; "
               "arm is moved kinematically via DLS, not dynamic servo tracking. "
               "Not a calibrated real robot pickup.",
    })
    # Save numeric results BEFORE video encoding. ffmpeg errors must never
    # destroy the simulation's diagnostics or prevent summary.json output.
    (args.out/"summary.json").write_text(
        json.dumps(info,ensure_ascii=False,indent=2)+"\n")
    if frames:
        frame_dir=args.out/"frames";frame_dir.mkdir()
        for i,p in enumerate(frames):p.save(frame_dir/f"frame_{i:04d}.png")
        ffmpeg=shutil.which("ffmpeg")
        if ffmpeg:
            try:
                subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-y",
                    "-framerate",str(args.fps),
                    "-i",str(frame_dir/"frame_%04d.png"),
                    "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v","libx264","-pix_fmt","yuv420p",
                    "-movflags","+faststart",
                    str(args.out/"table_contact_views.mp4")],check=True)
                info["video_status"]="created"
            except subprocess.CalledProcessError as exc:
                info["video_status"]="ffmpeg_failed_frames_saved"
                info["ffmpeg_exit_code"]=int(exc.returncode)
                # A zero-byte MP4 is not a video; remove it.
                broken=args.out/"table_contact_views.mp4"
                if broken.exists() and broken.stat().st_size==0:
                    broken.unlink()
        else:info["video_status"]="ffmpeg_missing_frames_saved"
    else:info["video_status"]="not_requested"
    (args.out/"summary.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:info[k] for k in
          ("status","settle_table_contact_steps","hold_bilateral_contact_steps",
           "short_lift_attempted","short_lift_success",
           "deepest_object_contact_mm","max_object_tilt_deg")},ensure_ascii=False,indent=2))
    print("Output:",args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--j5-deg",type=float,default=-90.)
    p.add_argument("--duration-s",type=float,default=5.0)
    p.add_argument("--approach-mm",type=float,default=15.)
    p.add_argument("--approach-s",type=float,default=1.0)
    p.add_argument("--approach-clearance-mm",type=float,default=.5)
    p.add_argument("--table-clearance-mm",type=float,default=2.0)
    p.add_argument("--lift-mm",type=float,default=5.0)
    p.add_argument("--settle-s",type=float,default=.35)
    p.add_argument("--close-s",type=float,default=1.8)
    p.add_argument("--kp",type=float,default=.6)
    p.add_argument("--kd",type=float,default=.2)
    p.add_argument("--max-torque",type=float,default=.15)
    p.add_argument("--video",action="store_true")
    p.add_argument("--fps",type=float,default=12.)
    p.add_argument("--render-every",type=int,default=20)
    args=p.parse_args()
    if not all(math.isfinite(v) for v in
               (args.j5_deg,args.duration_s,args.settle_s,args.close_s,
                args.kp,args.kd,args.max_torque,args.fps,
                args.approach_mm,args.approach_s,args.approach_clearance_mm)):
        p.error("parameters must be finite")
    if (args.fps<=0 or args.table_clearance_mm<0 or
        not(0<args.lift_mm<=10) or not(0<=args.approach_mm<=25) or
        args.approach_s<=0 or args.approach_clearance_mm<0 or
        args.settle_s+args.approach_s+args.close_s>=args.duration_s):
        p.error("Invalid FPS, table clearance or lift mm")
    execute(args)


if __name__=="__main__":
    main()
