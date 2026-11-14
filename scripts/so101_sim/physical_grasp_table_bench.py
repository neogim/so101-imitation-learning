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
import wrist_camera_sim as wrist

TABLE = "j6_bench_table"
TIP_SITE = "j6_bench_gripper_tip_site"
# Exact fixed gripper_frame_joint translation from so101_new_calib.urdf.
# Fixed child links may be merged away by MuJoCo's URDF compiler.
GRIPPER_FRAME_LOCAL_POS = (-.0079, -.000218121, -.0981274)
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
    # Restore the same photographed, provisional wrist camera and bracket
    # from the previous real-to-sim/kinematic experiment.
    # User ruler measurements: 15/35/35mm bracket segments.
    # TCP is in gripper_link, while the camera installer expects a
    # rig-local target (rig origin offset is DEFAULT_MOUNT_POS).
    tcp_rig_local=tuple(
        GRIPPER_FRAME_LOCAL_POS[i]-wrist.DEFAULT_MOUNT_POS[i]
        for i in range(3))
    wrist.install(spec,bracket_profile="ruler_15_35_35",
                  camera_aim_at=tcp_rig_local,
                  camera_retract_mm=args.wrist_retract_mm)
    gripper_spec = spec.body("gripper_link")
    if gripper_spec is None:
        raise ValueError("URDF gripper_link missing: cannot install TCP site")
    gripper_spec.add_site(
        name=TIP_SITE, pos=list(GRIPPER_FRAME_LOCAL_POS),
        size=[.002, .002, .002], rgba=[.1, .9, .8, 1.])
    # Visual-only pale backdrop: reuse the color/intent of the legacy
    # Real-to-Sim add_backdrop(), but keep it invisible to collision.
    # Its mocap pose is aligned with the wrist viewing direction after
    # selecting the actual tabletop grasp pose.
    background=spec.worldbody.add_body(
        name="j6_wrist_visual_backdrop_mocap",mocap=True,
        pos=[0.,0.,-10.])
    background.add_geom(
        name="j6_wrist_visual_backdrop",type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[3.,3.,.01],pos=[0.,0.,0.],
        rgba=[.73,.76,.79,1.],contype=0,conaffinity=0,group=4)
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
    # Reduce camera near clipping ONLY in this diagnostic benchmark.
    wrist_near_original=float(model.vis.map.znear)
    model.vis.map.znear=min(wrist_near_original,args.wrist_near_ratio)
    wrist_camera_id=mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_CAMERA,wrist.CAMERA)
    if wrist_camera_id<0:
        raise ValueError("Previous wrist RGB camera missing after compile")
    # Scale all dynamic Muhi inertias with measured mass, if provided.
    # The original physical_workcell_sim cylinder is 50g in total.
    muhibid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,
                               physical.DYNAMIC_BODY)
    if muhibid<0:raise ValueError("Dynamic Muhi body missing")
    original_mass=float(model.body_mass[muhibid])
    if original_mass<=0:raise ValueError("Invalid Muhi inertial mass")
    mass_ratio=(float(args.muhi_mass_g)/1000.)/original_mass
    model.body_mass[muhibid]*=mass_ratio
    model.body_inertia[muhibid]*=mass_ratio
    preview.apply_white_follower_preview(model)
    model.opt.timestep=.002
    # J6's CAD inertia is extremely small, making direct torque PD
    # alternate violently between saturation limits at 2ms. Add explicit
    # rotor/gear inertia and joint damping to this BENCH model only.
    # These are numerical stabilization assumptions, not measured servos.
    j6_pre=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"gripper")
    j6_dof=int(model.jnt_dofadr[j6_pre])
    model.dof_armature[j6_dof]=max(float(model.dof_armature[j6_dof]),args.j6_armature)
    model.dof_damping[j6_dof]=max(float(model.dof_damping[j6_dof]),args.j6_joint_damping)
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
        # Initial jaw/open clearance must be restored before trial.
        if gap < args.table_clearance_mm/1000.:
            rejected["rejected"]+=1
            if rejected["max_rejected_clearance_mm"] is None or (
               gap*1000 > rejected["max_rejected_clearance_mm"]):
                rejected["max_rejected_clearance_mm"]=float(gap*1000)
            return False
        return True
    pose,search=bench.select_bench_pose(
        model,data,freeq,open_angle,closed_angle,jawq,
        fixed,moving,obj,candidate_validator=table_for_candidate,
        min_fixed_gap_m=args.min_initial_fixed_gap_mm/1000.,
        max_fixed_gap_m=args.max_initial_fixed_gap_mm/1000.,
        prefer_small_moving_gap=True)
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
          "wrist_near_ratio_original":wrist_near_original,
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
           geom1_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(con.geom1)),
           geom2_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(con.geom2)),
           efc_address=int(con.efc_address),
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
          "pose_selection":"priority small moving jaw gap, fixed jaw initial >=2mm",
          "j6_armature_kgm2_assumed":args.j6_armature,
          "j6_damping_nmsrad_assumed":args.j6_joint_damping,
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
    # build_model() has its own scope. Resolve the camera ID locally where
    # render-loop diagnostics actually use it.
    wrist_camera_id=mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_CAMERA,wrist.CAMERA)
    shelf_wrist_camera_id=mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_CAMERA,wrist.SHELF_DIAGNOSTIC_CAMERA)
    retracted_wrist_camera_id=mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_CAMERA,wrist.RETRACTED_DIAGNOSTIC_CAMERA)
    if wrist_camera_id<0 or shelf_wrist_camera_id<0 or (
            args.wrist_retract_mm>0 and retracted_wrist_camera_id<0):
        raise ValueError("Wrist TCP-aim, shelf-view, or retracted camera missing")
    backdrop_bid=mujoco.mj_name2id(
        model,mujoco.mjtObj.mjOBJ_BODY,"j6_wrist_visual_backdrop_mocap")
    backdrop_mid=int(model.body_mocapid[backdrop_bid])
    if backdrop_mid<0:
        raise ValueError("Wrist visual backdrop mocap body missing")
    # Visual backdrop is 1.8m along wrist optical -Z, perpendicular to
    # the first camera direction. Same pale gray as old Real-to-Sim.
    # It has NO contact, mass, or force on the robot/bottle.
    wrist_R=np.asarray(data.cam_xmat[wrist_camera_id]).reshape(3,3)
    data.mocap_pos[backdrop_mid]=(
        np.asarray(data.cam_xpos[wrist_camera_id])-1.8*wrist_R[:,2])
    wrist_quat=np.zeros(4,dtype=float)
    mujoco.mju_mat2Quat(wrist_quat,np.asarray(wrist_R).reshape(9))
    data.mocap_quat[backdrop_mid]=wrist_quat
    mujoco.mj_forward(model,data)
    armhome=data.qpos[list(qids)].copy()
    # Approach with a fingertip-area reference, not the gripper motor body.
    # The J1-J4 move is simulation-only prescribed IK, never a servo command.
    tip_site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,TIP_SITE)
    if tip_site < 0:
        raise ValueError("Explicit gripper TCP site missing from compiled MuJoCo model")
    tip_xyz=np.asarray(data.site_xpos[tip_site]).copy()
    delta_xy=np.asarray(center[:2],float)-tip_xyz[:2]
    distance_xy=float(np.linalg.norm(delta_xy))
    approach_distance=min(float(args.approach_mm)/1000.,max(0.,distance_xy-.012))
    approach_target=tip_xyz.copy()
    if distance_xy>1e-9:
        approach_target[:2]+=delta_xy/distance_xy*approach_distance
    approach_blocked_steps=0
    approach_max_mm=0.
    j1_near_root_rejections=0
    j1_nearest_unsafe_geom=None
    j1_nearest_unsafe_gap_mm=None
    j1_nearest_unsafe_tip_distance_mm=None
    j1_stop_reason=None
    # J2/J3/J4 first align the TCP with the bottle. J1 must never be
    # tasked with both approaching the bottle and correcting wrist geometry.
    j234_start_q=armhome[1:4].copy()
    j234_accepted_steps=0
    j234_rejected_steps=0
    j234_tcp_initial=np.asarray(data.site_xpos[tip_site]).copy()
    j234_tcp_target=j234_tcp_initial.copy()
    toward=np.asarray(center[:2])-j234_tcp_initial[:2]
    toward_norm=float(np.linalg.norm(toward))
    j234_tcp_target[:2]+=(
        toward/max(toward_norm,1e-9)
        *min(float(args.j234_reposition_mm)/1000.,
             max(0.,toward_norm-args.j234_tcp_object_clearance_mm/1000.)))
    j234_tcp_target[2]+=float(np.clip(
        center[2]-j234_tcp_initial[2],-.010,.010))
    j234_tcp_move_mm=0.
    j1_precontact_reached=False
    j1_precontact_reason="not_started"
    j1_safe_near_contact=False
    j1_safe_near_contact_gap_mm=None
    fixed_contact_witness_world_xyz_m=None
    fixed_contact_witness_tcp_distance_mm=None
    fixed_contact_witness_object_distance_mm=None
    closing_low_force_stop=False
    closing_low_force_reason=None
    closing_hold_target=None
    closing_peak_fixed_force_n=0.
    closing_peak_moving_force_n=0.
    j6_preclose_drift_max_deg=0.
    j6_open_hold_steps=0
    j1_precontact_start_rad=float(armhome[0])
    j1_precontact_direction=0
    j1_precontact_min_fixed_gap_mm=float("inf")
    j1_precontact_contact_steps=0
    j1_precontact_force_confirmed_steps=0
    j1_precontact_max_fixed_normal_force_n=0.
    j1_precontact_motion_steps=0
    j1_precontact_exceeded_tilt=False
    grip_body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,bench.FINGER_BODY)
    mj_initial_gripper_z=float(data.xpos[grip_body,2])
    mj_initial_object_z=float(data.qpos[freeq+2])
    lift_requested=False
    lift_started_at=None
    lift_rejected_steps=0
    rows=[];contactrows=[]
    no_table_support=False
    contact_overpenetration_abort=False
    contact_overpenetration_step=None
    frames=[]
    wrist_frames=[]
    shelf_wrist_frames=[]
    retracted_wrist_frames=[]
    wrist_camera_frames_with_object_center=0
    wrist_camera_frames_with_tcp_center=0
    wrist_camera_object_center_uv_last=None
    wrist_camera_tcp_uv_last=None
    contact_abort_reason=None
    contact_abort_step=None
    jaw_self_gap_min_mm=float("inf")
    max_j6_open_overshoot_deg=0.
    max_jaw_impact_force_n=0.
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
        render_option=mujoco.MjvOption()
        render_option.geomgroup[3]=0
        # Backdrop group 4 visible in wrist views but not side views,
        # so it cannot obscure the original side camera inspection.
        render_option.geomgroup[4]=0
        wrist_render_option=mujoco.MjvOption()
        wrist_render_option.geomgroup[3]=0
        wrist_render_option.geomgroup[4]=1
    jaw_self_gap_initial_mm=float(1000.*bench.geometry_gaps(
        model,data,fixed,moving))
    n=math.ceil(args.duration_s/model.opt.timestep)
    def reposition_j234(goal):
        """Constrained TCP prealignment with ONLY J2,J3,J4, no object teleport."""
        nonlocal armhome,j234_accepted_steps,j234_rejected_steps,j234_tcp_move_mm
        saved=data.qpos[list(qids)].copy()
        mujoco.mj_forward(model,data)
        jac=np.zeros((3,model.nv));jacr=np.zeros((3,model.nv))
        mujoco.mj_jacSite(model,data,jac,jacr,tip_site)
        J=jac[:,list(dids[1:4])]
        error=np.asarray(goal,float)-np.asarray(data.site_xpos[tip_site]).copy()
        dq=J.T@np.linalg.solve(J@J.T+(.05**2)*np.eye(3),error)
        dq=np.clip(dq,-.0008,.0008)
        proposed=armhome.copy()
        proposed[1:4]=np.clip(proposed[1:4]+dq,lows[1:4],highs[1:4])
        if np.max(np.abs(proposed[1:4]-armhome[1:4]))<1e-9:
            j234_rejected_steps+=1
            return
        data.qpos[list(qids)]=proposed
        mujoco.mj_forward(model,data)
        table_clearance=min((bench.signed_pair_gap(model,data,g,table)
                             for g in robot_geoms),default=float("inf"))
        bottle_clearance=min((bench.signed_pair_gap(model,data,g,o)
                              for g in robot_geoms for o in obj),default=float("inf"))
        tcp_before=float(np.linalg.norm(
            np.asarray(data.site_xpos[tip_site])-np.asarray(goal)))
        # Use geometric collision guard; never run J2/J3/J4 into the object.
        if (table_clearance>=args.table_clearance_mm/1000. and
            bottle_clearance>=args.j234_object_clearance_mm/1000. and
            np.isfinite(tcp_before)):
            armhome=proposed
            j234_accepted_steps+=1
            j234_tcp_move_mm=float(np.linalg.norm(
                np.asarray(data.site_xpos[tip_site])-j234_tcp_initial)*1000.)
        else:
            j234_rejected_steps+=1
        data.qpos[list(qids)]=saved
        mujoco.mj_forward(model,data)

    def try_approach_step(goal):
        """Bounded DLS joint step, requires robot-table AND robot-object clearance."""
        saved=data.qpos[list(qids)].copy()
        mujoco.mj_forward(model,data)
        jac=np.zeros((3,model.nv));jac_r=np.zeros((3,model.nv))
        mujoco.mj_jacSite(model,data,jac,jac_r,tip_site)
        J=jac[:,list(dids[:4])]
        err=np.asarray(goal,float)-np.asarray(data.site_xpos[tip_site]).copy()
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
    def near_root_obstacle():
        """Find first sub-mm robot/object approach away from the TCP.

        MuJoCo's gripper_link owns both servo/palm and fixed jaw mesh
        geometry. Body ownership alone cannot classify fingertip contacts.
        Witness points from mj_geomDistance locate where the geometry
        approaches the object. Return the closest non-tip pair.
        """
        nearest=None
        tip=np.asarray(data.site_xpos[tip_site],dtype=float)
        for g in robot_geoms:
            for o in obj:
                witness=np.zeros(6,dtype=float)
                gap=float(mujoco.mj_geomDistance(
                    model,data,int(g),int(o),.15,witness))
                from_tip=float(np.linalg.norm(witness[:3]-tip))
                if from_tip<=args.j1_fingertip_zone_mm/1000.:
                    continue
                if nearest is None or gap<nearest[0]:
                    nearest=(gap,int(g),int(o),from_tip)
        return nearest

    def safe_j1_contact_step():
        """Try only J1, choose sign that reduces fixed-jaw distance.

        The bottle stays fully dynamic and table-supported. Reject changes
        that penetrate the object or table; no fake contact or pose reset.
        """
        nonlocal armhome,j1_precontact_direction,j1_precontact_motion_steps
        nonlocal j1_near_root_rejections,j1_nearest_unsafe_geom
        nonlocal j1_nearest_unsafe_gap_mm,j1_nearest_unsafe_tip_distance_mm
        gap0=bench.geometry_gaps(model,data,obj,fixed)
        best=None
        for sign in (-1,1):
            candidate=armhome.copy()
            candidate[0]=float(np.clip(candidate[0]+sign*args.j1_step_deg*math.pi/180.,
                                       lows[0],highs[0]))
            if abs(candidate[0]-armhome[0])<1e-12:continue
            data.qpos[list(qids)]=candidate
            mujoco.mj_forward(model,data)
            fixed_gap=bench.geometry_gaps(model,data,obj,fixed)
            moving_gap=bench.geometry_gaps(model,data,obj,moving)
            table_gap=min((bench.signed_pair_gap(model,data,g,table)
                         for g in robot_geoms),default=float("inf"))
            unsafe=near_root_obstacle()
            root_blocked=(unsafe is not None and
                          unsafe[0]<args.j1_root_clearance_mm/1000.)
            if root_blocked:
                j1_near_root_rejections+=1
                j1_nearest_unsafe_geom=mujoco.mj_id2name(
                    model,mujoco.mjtObj.mjOBJ_GEOM,unsafe[1])
                j1_nearest_unsafe_gap_mm=float(unsafe[0]*1000.)
                j1_nearest_unsafe_tip_distance_mm=float(unsafe[3]*1000.)
            if (not root_blocked and
                fixed_gap >= -args.j1_max_penetration_mm/1000. and
                moving_gap >= args.j1_moving_clearance_mm/1000. and
                table_gap >= args.table_clearance_mm/1000. and
                fixed_gap < gap0-1e-7 and
                abs(math.degrees(candidate[0]-j1_precontact_start_rad)) <= args.j1_max_deg):
                if best is None or fixed_gap<best[0]:
                    best=(float(fixed_gap),candidate.copy(),sign)
        data.qpos[list(qids)]=armhome
        mujoco.mj_forward(model,data)
        if best is None:return False
        armhome=best[1]
        j1_precontact_direction=best[2]
        j1_precontact_motion_steps+=1
        data.qpos[list(qids)]=armhome
        mujoco.mj_forward(model,data)
        return True

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
            elif t < args.settle_s + args.j234_s:
                stage="j234_prealign"
                demand=open_angle
                u234=min(1.,max(0.,(t-args.settle_s)/args.j234_s))
                v234=u234*u234*(3.-2.*u234)
                goal234=j234_tcp_initial+(
                    j234_tcp_target-j234_tcp_initial)*v234
                reposition_j234(goal234)
            elif j234_tcp_move_mm < args.j234_min_progress_mm:
                stage="j234_prealign_failed"
                demand=open_angle
            elif t < args.settle_s + args.j234_s + args.approach_s:
                stage="j1_precontact"
                demand=open_angle
                mujoco.mj_forward(model,data)
                fixed_gap=bench.geometry_gaps(model,data,obj,fixed)
                # Use actual collision-mesh witness point, NOT gripper_link
                # body origin. This is a geometric preview, not image ML.
                best_fixed_witness=None
                for fg in fixed:
                    for og in obj:
                        pair=np.zeros(6)
                        g=bench.signed_pair_gap(model,data,fg,og)
                        if best_fixed_witness is None or g<best_fixed_witness[0]:
                            mujoco.mj_geomDistance(
                                model,data,int(fg),int(og),.15,pair)
                            best_fixed_witness=(g,pair.copy())
                if best_fixed_witness is not None:
                    xyz=best_fixed_witness[1][:3]
                    fixed_contact_witness_world_xyz_m=xyz.tolist()
                    fixed_contact_witness_tcp_distance_mm=float(
                        np.linalg.norm(xyz-data.site_xpos[tip_site])*1000.)
                    fixed_contact_witness_object_distance_mm=float(
                        best_fixed_witness[0]*1000.)
                tilt_now=math.degrees(math.acos(float(np.clip(
                    1.-2.*(data.qpos[freeq+4]**2+data.qpos[freeq+5]**2),
                    -1.,1.))))
                if tilt_now > args.j1_max_tilt_deg:
                    j1_precontact_exceeded_tilt=True
                    j1_precontact_reason="object_tilt_exceeded"
                elif (fixed_gap<=args.j1_stop_before_contact_mm/1000.
                      and fixed_gap>=-0.00001):
                    # Previous 0.057mm penetration generated a ~124N
                    # instantaneous impulse. Do NOT push further by
                    # prescribing J1 qpos. Leave a small gap so the J6
                    # moving jaw can move the bottle gently toward it.
                    j1_safe_near_contact=True
                    j1_safe_near_contact_gap_mm=float(fixed_gap*1000.)
                    j1_precontact_reached=True
                    j1_precontact_reason="fixed_jaw_safe_geometric_precontact"
                elif fixed_gap<=args.j1_contact_gap_mm/1000.:
                    j1_precontact_reason="geometric_target_reached_waiting_for_force"
                elif not j1_precontact_reached and not j1_precontact_exceeded_tilt:
                    moved=safe_j1_contact_step()
                    if not moved:
                        approach_blocked_steps+=1
                        j1_precontact_reason="no_safe_j1_direction"
                        if j1_near_root_rejections:
                            j1_stop_reason="root_collision_candidate_rejected"
                    else:
                        j1_precontact_reason="j1_approaching"
                j1_precontact_min_fixed_gap_mm=min(
                    j1_precontact_min_fixed_gap_mm,
                    1000.*bench.geometry_gaps(model,data,obj,fixed))
            elif not j1_precontact_reached or j1_precontact_exceeded_tilt:
                # Do not close J6 when the opposing jaw did not reach Muhi.
                stage="j1_precontact_failed"
                if (not j1_precontact_reached and
                    not j1_precontact_exceeded_tilt and
                    j1_precontact_reason in
                    ("geometric_target_reached_waiting_for_force",
                     "fixed_jaw_contact_without_force")):
                    j1_precontact_reason="no_effective_fixed_jaw_force"
                demand=open_angle
            else:
                u=min(1.,max(0.,(t-args.settle_s-args.j234_s-args.approach_s)/args.close_s))
                smooth=u*u*(3.-2.*u)
                demand=open_angle+smooth*(closed_angle-open_angle)
                if closing_low_force_stop and closing_hold_target is not None:
                    demand=closing_hold_target
                    stage="contact_hold"
                else:
                    stage="close" if u<1. else "hold"
                # Only attempt a kinematically guided 5mm lift after
                # sustained TWO-JAW contact in recent on-table samples.
                # A table-supported contact alone never proves a pickup.
                if (not lift_requested and
                    t>args.settle_s+args.j234_s+args.approach_s+args.close_s+.22 and
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
            # Hold J6 OPEN during prealignment as J1..J5 are also
            # kinematically prescribed. Old weak PD permitted ~50deg
            # closing drift before the closure phase even began.
            # Switch back to dynamic torque control during J6 closure.
            preclose=stage in ("settle","j234_prealign","j1_precontact")
            if preclose:
                j6_preclose_drift_max_deg=max(
                    j6_preclose_drift_max_deg,
                    abs(math.degrees(float(data.qpos[jawq]-open_angle))))
                data.qpos[jawq]=open_angle
                data.qvel[jawd]=0.
                j6_open_hold_steps+=1
                commanded_torque=0.
            else:
                torque=args.kp*(demand-data.qpos[jawq])-args.kd*data.qvel[jawd]
                torque_cap=(args.j6_precontact_max_torque
                            if j1_safe_near_contact else args.max_torque)
                commanded_torque=float(np.clip(
                    torque,-torque_cap,torque_cap))
                data.qfrc_applied[jawd]=commanded_torque
            mujoco.mj_step(model,data)
            # CRITICAL: mj_step has solved contacts for the *pre-integration*
            # configuration and stores their constraint impulses in efc_force.
            # Do not call mj_forward before snapshotting these: mj_forward
            # recomputes the contact list at post-step positions, erasing
            # the actual impact / normal-force record. This previously made
            # the 1.414s Muhi impulse appear to happen with ZERO jaw contact.
            contacts,counts=contact_observations(model,data,obj,fixed,moving,table)
            tcp_contact_reference=np.asarray(data.site_xpos[tip_site]).copy()
            for c in contacts:
                if c["kind"]=="fixed":
                    c["tip_distance_mm"]=1000.*float(np.linalg.norm(
                        np.asarray(c["point_xyz_m"],dtype=float)
                        -tcp_contact_reference))
                else:
                    c["tip_distance_mm"]=None
                c["solver_time_s"]=float(data.time-model.opt.timestep)
            data.qpos[list(qids)]=armhome
            data.qvel[list(dids)]=0.
            if preclose:
                data.qpos[jawq]=open_angle
                data.qvel[jawd]=0.
            mujoco.mj_forward(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError(f"nonfinite state at step={i}")
            jaw_impact_force_n=sum(
                max(0.,float(c["normal_force_n"])) for c in contacts
                if c["kind"] in ("fixed","moving"))
            max_jaw_impact_force_n=max(
                max_jaw_impact_force_n,jaw_impact_force_n)
            moving_normal_n=sum(
                max(0.,float(c["normal_force_n"])) for c in contacts
                if c["kind"]=="moving")
            closing_peak_moving_force_n=max(
                closing_peak_moving_force_n,moving_normal_n)
            fixed_normal_n=sum(
                max(0.,float(c["normal_force_n"])) for c in contacts
                if c["kind"]=="fixed" and
                c["tip_distance_mm"]<=args.fixed_contact_max_tip_distance_mm)
            root_normal_n=sum(
                max(0.,float(c["normal_force_n"])) for c in contacts
                if c["kind"]=="fixed" and
                c["tip_distance_mm"]>args.fixed_contact_max_tip_distance_mm)
            j1_precontact_max_fixed_normal_force_n=max(
                j1_precontact_max_fixed_normal_force_n,fixed_normal_n)
            closing_peak_fixed_force_n=max(
                closing_peak_fixed_force_n,fixed_normal_n)
            if stage=="close" and (
                (moving_normal_n>=args.j6_bilateral_min_force_n and
                 fixed_normal_n>=args.j6_bilateral_min_force_n) or
                max(moving_normal_n,fixed_normal_n)>=args.j6_hold_force_n):
                # Continue closing after the first gentle contact, unless
                # force exceeds the safe cap. Hold after bilateral load.
                closing_low_force_stop=True
                closing_low_force_reason=(
                    "bilateral_measured_contact" if
                    moving_normal_n>=args.j6_bilateral_min_force_n and
                    fixed_normal_n>=args.j6_bilateral_min_force_n else
                    "excessive_one_sided_force")
                closing_hold_target=float(data.qpos[jawq])
            if stage=="j1_precontact":
                if counts["fixed"]>0:
                    j1_precontact_contact_steps+=1
                if fixed_normal_n>=args.j1_min_fixed_normal_force_n:
                    j1_precontact_force_confirmed_steps+=1
                else:
                    j1_precontact_force_confirmed_steps=0
                    if counts["fixed"]>0 and not j1_precontact_reached:
                        j1_precontact_reason="fixed_jaw_contact_without_force"
                if j1_precontact_force_confirmed_steps>=args.j1_force_hold_steps:
                    j1_precontact_reached=True
                    j1_precontact_reason="fixed_jaw_force_confirmed"
            for c in contacts:contactrows.append({"step":i,"time_s":float(data.time),**c})
            rotation=np.empty(9)
            mujoco.mju_quat2Mat(rotation,data.qpos[freeq+3:freeq+7])
            tilt=math.degrees(math.acos(float(np.clip(rotation[8],-1.,1.))))
            rows.append({
                "step":i,"time_s":float(data.time),"stage":stage,
                "j1_rad":float(data.qpos[qids[0]]),
                "fixed_jaw_gap_mm":float(1000.*bench.geometry_gaps(model,data,obj,fixed)),
                "j1_precontact_reached":bool(j1_precontact_reached),
                "fixed_jaw_normal_force_n":float(fixed_normal_n),
                "moving_jaw_normal_force_n":float(moving_normal_n),
                "j6_closure_hold_on_contact":bool(closing_low_force_stop),
                "fixed_contact_witness_tcp_distance_mm":
                    fixed_contact_witness_tcp_distance_mm,
                "total_jaw_normal_force_n":float(jaw_impact_force_n),
                "root_contact_normal_force_n":float(root_normal_n),
                "jaw_self_gap_mm":float(1000.*bench.geometry_gaps(
                    model,data,fixed,moving)),
                "j6_rad":float(data.qpos[jawq]),
                "j6_open_held_kinematically":bool(preclose),
                "tcp_gripper_z_m":float(data.xpos[grip_body,2]),
                "gripper_lift_mm":float((data.xpos[grip_body,2]-mj_initial_gripper_z)*1000.),
                "j6_target_rad":float(demand),
                "j6_torque_nm":float(data.qfrc_applied[jawd]),
                "j6_commanded_torque_nm":commanded_torque,
                "approach_tip_horizontal_mm":float(
                    np.linalg.norm(data.site_xpos[tip_site,:2]-tip_xyz[:2])*1000.),
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
                    render.update_scene(data,camera=cam,
                                        scene_option=render_option)
                    pics.append(Image.fromarray(render.render().copy()))
                render.update_scene(data,camera=wrist.CAMERA,
                                    scene_option=wrist_render_option)
                wrist_image=Image.fromarray(render.render().copy())
                wrist_frames.append(wrist_image)
                render.update_scene(data,camera=wrist.SHELF_DIAGNOSTIC_CAMERA,
                                    scene_option=wrist_render_option)
                shelf_image=Image.fromarray(render.render().copy())
                shelf_wrist_frames.append(shelf_image)
                retracted_image=None
                if retracted_wrist_camera_id>=0:
                    render.update_scene(
                        data,camera=wrist.RETRACTED_DIAGNOSTIC_CAMERA,
                        scene_option=wrist_render_option)
                    retracted_image=Image.fromarray(render.render().copy())
                    if fixed_contact_witness_world_xyz_m is not None:
                        # Project the predicted fixed-jaw collision witness,
                        # not a vision-inferred grasp point, onto the wrist
                        # image. This aids user validation of target selection.
                        cid=retracted_wrist_camera_id
                        cp=np.asarray(data.cam_xpos[cid])
                        cr=np.asarray(data.cam_xmat[cid]).reshape(3,3)
                        local=cr.T@(np.asarray(
                            fixed_contact_witness_world_xyz_m)-cp)
                        dep=-float(local[2])
                        if dep>1e-6:
                            fy=math.tan(math.radians(
                                float(model.cam_fovy[cid]))/2.)
                            nx=float(local[0]/(dep*fy*(480/360)))
                            ny=float(local[1]/(dep*fy))
                            if abs(nx)<=1 and abs(ny)<=1:
                                u=(nx+1)*240.
                                v=(1-ny)*180.
                                d=ImageDraw.Draw(retracted_image)
                                d.ellipse((u-5,v-5,u+5,v+5),
                                          outline=(255,90,90),width=2)
                    retracted_wrist_frames.append(retracted_image)
                cam_pos=np.asarray(data.cam_xpos[wrist_camera_id])
                cam_rot=np.asarray(
                    data.cam_xmat[wrist_camera_id]).reshape(3,3)
                rel=cam_rot.T@(np.asarray(data.qpos[freeq:freeq+3])-cam_pos)
                depth=float(-rel[2])
                focal=math.tan(math.radians(
                    float(model.cam_fovy[wrist_camera_id]))/2.)
                if depth>1e-6:
                    nx=float(rel[0]/(depth*focal*(480./360.)))
                    ny=float(rel[1]/(depth*focal))
                    if abs(nx)<=1. and abs(ny)<=1.:
                        wrist_camera_frames_with_object_center+=1
                        wrist_camera_object_center_uv_last=[
                            float((nx+1.)*.5*480.),
                            float((1.-ny)*.5*360.)]
                        # Projection of 3D centre, NOT image segmentation.
                        x,y=wrist_camera_object_center_uv_last
                        ImageDraw.Draw(wrist_image).ellipse(
                            (x-3,y-3,x+3,y+3),
                            outline=(240,220,20),width=2)
                tcp_cam=cam_rot.T@(
                    np.asarray(data.site_xpos[tip_site])-cam_pos)
                tcp_depth=float(-tcp_cam[2])
                if tcp_depth>1e-6:
                    tx=float(tcp_cam[0]/(tcp_depth*focal*(480./360.)))
                    ty=float(tcp_cam[1]/(tcp_depth*focal))
                    if abs(tx)<=1 and abs(ty)<=1:
                        wrist_camera_frames_with_tcp_center+=1
                        wrist_camera_tcp_uv_last=[
                            float((tx+1.)*.5*480.),
                            float((1.-ty)*.5*360.)]
                        # Cyan mark is the 3D jaw/TCP projection, not
                        # image-recognition output or physical calibration.
                        x,y=wrist_camera_tcp_uv_last
                        d=ImageDraw.Draw(wrist_image)
                        d.line((x-5,y,x+5,y),fill=(20,225,225),width=2)
                        d.line((x,y-5,x,y+5),fill=(20,225,225),width=2)
                panels=(*pics,wrist_image,shelf_image)
                if retracted_image is not None:
                    panels=(*panels,retracted_image)
                image=Image.new("RGB",(480*len(panels),396),(20,24,30))
                for idx,p in enumerate(panels):
                    image.paste(p,(idx*480,36))
                draw=ImageDraw.Draw(image)
                draw.text((8,8),
                    f"SIDE A | {stage.upper()} t={data.time:.2f}s "
                    f"fixed={counts['fixed']} moving={counts['moving']} "
                    f"tilt={tilt:.1f}deg",fill="white")
                draw.text((488,8),"SIDE B | same step",fill="white")
                draw.text((968,8),"WRIST TCP AIM | provisional",fill="white")
                draw.text((1448,8),"WRIST SHELF AXIS | same lens",fill="white")
                if retracted_image is not None:
                    draw.text((1928,8),
                        f"WRIST RETRACTED | {args.wrist_retract_mm:g}mm",
                        fill="white")
                frames.append(image)
            # Stop on wrong (root) contact, bottle tip-over, physically
            # impossible J6 angle, or jaw mesh interpenetration. Never
            # continue a failed pre-contact until the Muhi falls 88 metres.
            max_j6_open_overshoot_deg=max(
                max_j6_open_overshoot_deg,
                math.degrees(max(0.,float(data.qpos[jawq]-open_angle))))
            jaw_self_gap_mm=float(1000.*bench.geometry_gaps(
                model,data,fixed,moving))
            jaw_self_gap_min_mm=min(jaw_self_gap_min_mm,jaw_self_gap_mm)
            if jaw_impact_force_n>args.jaw_force_spike_abort_n:
                contact_abort_reason="JAW_CONTACT_FORCE_SPIKE"
            elif stage=="j234_prealign_failed":
                contact_abort_reason="J234_PREALIGN_INSUFFICIENT"
            elif stage=="j1_precontact_failed":
                contact_abort_reason="J1_PRECONTACT_NOT_ESTABLISHED"
            elif root_normal_n>=args.root_contact_abort_force_n and stage in (
                    "j234_prealign","j1_precontact"):
                contact_abort_reason="ROOT_CONTACT_NOT_FINGERTIP"
            elif (stage!="settle" and tilt>args.j1_max_tilt_deg):
                contact_abort_reason="MUHI_TIP_OVER"
            elif (data.qpos[jawq] < closed_angle-args.j6_limit_margin_deg*math.pi/180. or
                  data.qpos[jawq] > open_angle+args.j6_limit_margin_deg*math.pi/180.):
                # MuJoCo joint limits are soft solver constraints. A small
                # 2deg overrun of an OPEN gripper is not a grasp-failure
                # reason to abort before the J1 approach completes.
                contact_abort_reason="J6_JOINT_LIMIT_EXCEEDED"
            elif (jaw_self_gap_mm<min(-1.,jaw_self_gap_initial_mm-1.)):
                contact_abort_reason="J6_JAWS_INTERPENETRATION"
            if contact_abort_reason:
                contact_abort_step=i
                break
            # Hard diagnostic stop: object-to-jaw deep overlap is a geometry
            # / contact solver failure, NOT progress toward grasp. Stop
            # rather than allow another 26mm interpenetration.
            jaw_depth=min((c["penetration_mm"] for c in contacts
                           if c["kind"] in ("fixed","moving")),default=0.)
            if jaw_depth < -args.abort_jaw_penetration_mm:
                contact_overpenetration_abort=True
                contact_overpenetration_step=i
                break
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
        fields=["step","time_s","solver_time_s","kind","geom1","geom2",
                "geom1_name","geom2_name","efc_address","tip_distance_mm",
                "point_xyz_m",
                "contact_normal_xyz","penetration_mm",
                "normal_force_n","tangential_force_n"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows(contactrows)
    hold=[r for r in rows if r["stage"]=="hold"]
    closing_rows=[r for r in rows if r["stage"] in ("close","hold","lift")]
    j6_actual_closure_deg=math.degrees(
        open_angle-min((r["j6_rad"] for r in closing_rows),default=open_angle))
    j6_command_saturation_fraction=(
        sum(abs(r["j6_commanded_torque_nm"])>=.99*args.max_torque
            for r in closing_rows)/len(closing_rows) if closing_rows else 0.)
    j6_not_closing=bool(closing_rows and j6_actual_closure_deg<10.)
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
    if contact_overpenetration_abort:
        result_status="JAW_OVERPENETRATION_ABORT"
    if contact_abort_reason:
        result_status=contact_abort_reason
    info.update({
        "status":result_status,
        "contact_logging":"snapshot immediately after mj_step before mj_forward; solver_time_s marks pre-integration contacts",
        "contact_abort_reason":contact_abort_reason,
        "contact_abort_step":contact_abort_step,
        "max_jaw_impact_force_n":float(max_jaw_impact_force_n),
        "jaw_force_spike_abort_n":float(args.jaw_force_spike_abort_n),
        "jaw_self_gap_open_mm":float(jaw_self_gap_initial_mm),
        "j6_limit_margin_deg":float(args.j6_limit_margin_deg),
        "j6_max_open_overshoot_deg":float(max_j6_open_overshoot_deg),
        "jaw_self_gap_min_mm":float(jaw_self_gap_min_mm),
        "muhi_mass_g":float(args.muhi_mass_g),
        "muhi_mass_source":"user-supplied value" if args.muhi_mass_g!=50. else "provisional default 50g",
        "fixed_contact_tip_distance_limit_mm":float(args.fixed_contact_max_tip_distance_mm),
        "table_support_preflight_failed":no_table_support,
        "jaw_overpenetration_abort":bool(contact_overpenetration_abort),
        "jaw_overpenetration_step":contact_overpenetration_step,
        "jaw_overpenetration_threshold_mm":float(args.abort_jaw_penetration_mm),
        "wrist_camera_installed":True,
        "wrist_camera_name":wrist.CAMERA,
        "wrist_camera_frame_count":len(wrist_frames),
        "wrist_shelf_camera_name":wrist.SHELF_DIAGNOSTIC_CAMERA,
        "wrist_shelf_camera_frame_count":len(shelf_wrist_frames),
        "wrist_retracted_camera_name":(
            wrist.RETRACTED_DIAGNOSTIC_CAMERA if args.wrist_retract_mm>0 else None),
        "wrist_retracted_camera_frame_count":len(retracted_wrist_frames),
        "wrist_retract_mm_virtual_diagnostic":float(args.wrist_retract_mm),
        "wrist_render_background":"legacy Real-to-Sim pale gray wall; visual only",
        "wrist_near_ratio_original":diag.get("wrist_near_ratio_original"),
        "wrist_near_ratio_applied":float(model.vis.map.znear),
        "wrist_comparison_note":"Same MuJoCo pose/lens/mount/FOV; only optical orientation differs",
        "wrist_object_center_projected_in_fov_frames":
            int(wrist_camera_frames_with_object_center),
        "wrist_tcp_projected_in_fov_frames":
            int(wrist_camera_frames_with_tcp_center),
        "wrist_tcp_projected_uv_last_px":wrist_camera_tcp_uv_last,
        "wrist_object_center_projected_uv_last_px":
            wrist_camera_object_center_uv_last,
        "wrist_camera_mount_source":"photo and user ruler 15/35/35mm; angles approximate +/-30deg, NOT calibrated",
        "wrist_bracket_profile":"ruler_15_35_35",
        "wrist_optical_aim":"sim gripper TCP from existing fixed gripper_frame_joint; provisional, NOT calibrated",
        "wrist_camera_control_mode":"diagnostic rendering only; grasp remains geometric/force based",
        "j234_prealign_enabled":True,
        "j234_prealign_joints":"J2,J3,J4 only; J1/J5 fixed during prealignment",
        "j234_prealign_target_world_xyz_m":j234_tcp_target.tolist(),
        "j234_min_progress_mm":float(args.j234_min_progress_mm),
        "j234_prealign_actual_tcp_move_mm":float(j234_tcp_move_mm),
        "j234_prealign_accepted_steps":int(j234_accepted_steps),
        "j234_prealign_rejected_steps":int(j234_rejected_steps),
        "j234_total_joint_change_deg":np.degrees(
            armhome[1:4]-j234_start_q).tolist(),
        "j1_near_root_rejections":int(j1_near_root_rejections),
        "j1_nearest_unsafe_geom":j1_nearest_unsafe_geom,
        "j1_nearest_unsafe_gap_mm":j1_nearest_unsafe_gap_mm,
        "j1_nearest_unsafe_tip_distance_mm":
            j1_nearest_unsafe_tip_distance_mm,
        "j1_stop_reason":j1_stop_reason,
        "j1_safe_near_contact":bool(j1_safe_near_contact),
        "j1_safe_near_contact_gap_mm":j1_safe_near_contact_gap_mm,
        "fixed_contact_witness_world_xyz_m":
            fixed_contact_witness_world_xyz_m,
        "fixed_target_overlay":"red circle on retracted wrist = MuJoCo collision witness, not RGB object detection",
        "fixed_contact_witness_tcp_distance_mm":
            fixed_contact_witness_tcp_distance_mm,
        "fixed_contact_witness_object_distance_mm":
            fixed_contact_witness_object_distance_mm,
        "j6_closure_hold_on_contact":bool(closing_low_force_stop),
        "j6_closure_hold_reason":closing_low_force_reason,
        "j6_peak_moving_normal_force_n":float(closing_peak_moving_force_n),
        "j6_peak_fixed_normal_force_n":float(closing_peak_fixed_force_n),
        "j1_precontact_enabled":True,
        "j1_control_note":"J1 advances gently to -0.25mm max geometry; J6 runs only after sustained measured fixed-jaw normal force; no force -> no J6.",
        "j1_precontact_reached":bool(j1_precontact_reached),
        "j1_precontact_reason":j1_precontact_reason,
        "j1_precontact_direction":int(j1_precontact_direction),
        "j1_precontact_rotation_deg":float(
            math.degrees(armhome[0]-j1_precontact_start_rad)),
        "j1_precontact_min_fixed_gap_mm":(
            float(j1_precontact_min_fixed_gap_mm)
            if math.isfinite(j1_precontact_min_fixed_gap_mm) else None),
        "j1_precontact_physical_contact_steps":int(j1_precontact_contact_steps),
        "j1_precontact_max_fixed_normal_force_n":
            float(j1_precontact_max_fixed_normal_force_n),
        "j1_precontact_force_confirmed_consecutive_steps":
            int(j1_precontact_force_confirmed_steps),
        "j1_min_fixed_normal_force_n":float(args.j1_min_fixed_normal_force_n),
        "j1_force_hold_steps":int(args.j1_force_hold_steps),
        "j1_precontact_motion_steps":int(j1_precontact_motion_steps),
        "j1_precontact_tilt_abort":bool(j1_precontact_exceeded_tilt),
        "approach_requested_mm":float(args.approach_mm),
        "approach_executed_max_horizontal_mm":float(approach_max_mm),
        "approach_blocked_steps":int(approach_blocked_steps),
        "approach_method":"DLS J1-J4, check full robot/table and robot/object clearance",
        "j6_trajectory":"cubic smoothstep with velocity-zero endpoints",
        "j6_preclose_open_hold":"kinematic until closure; dynamic torque PD afterward (sim-only)",
        "j6_preclose_open_hold_steps":int(j6_open_hold_steps),
        "j6_preclose_drift_max_deg_before_correction":
            float(j6_preclose_drift_max_deg),
        "j6_actual_closure_deg":float(j6_actual_closure_deg),
        "j6_torque_saturation_fraction_during_close_hold":
            float(j6_command_saturation_fraction),
        "j6_not_closing":j6_not_closing,
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
        wrist_dir=args.out/"wrist_frames"
        wrist_dir.mkdir()
        shelf_dir=args.out/"wrist_shelf_frames"
        shelf_dir.mkdir()
        retracted_dir=args.out/"wrist_retracted_frames"
        if retracted_wrist_frames:
            retracted_dir.mkdir()
        for i,p in enumerate(wrist_frames):
            p.save(wrist_dir/f"wrist_{i:04d}.png")
        for i,p in enumerate(shelf_wrist_frames):
            p.save(shelf_dir/f"shelf_{i:04d}.png")
        for i,p in enumerate(retracted_wrist_frames):
            p.save(retracted_dir/f"retracted_{i:04d}.png")
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
                try:
                    subprocess.run([
                        ffmpeg,"-hide_banner","-loglevel","error","-y",
                        "-framerate",str(args.fps),
                        "-i",str(wrist_dir/"wrist_%04d.png"),
                        "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v","libx264","-pix_fmt","yuv420p",
                        "-movflags","+faststart",
                        str(args.out/"wrist_camera.mp4")],check=True)
                    info["wrist_video_status"]="created"
                    subprocess.run([
                        ffmpeg,"-hide_banner","-loglevel","error","-y",
                        "-framerate",str(args.fps),
                        "-i",str(shelf_dir/"shelf_%04d.png"),
                        "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v","libx264","-pix_fmt","yuv420p",
                        "-movflags","+faststart",
                        str(args.out/"wrist_shelf_camera.mp4")],check=True)
                    info["wrist_shelf_video_status"]="created"
                    if retracted_wrist_frames:
                        subprocess.run([
                            ffmpeg,"-hide_banner","-loglevel","error","-y",
                            "-framerate",str(args.fps),
                            "-i",str(retracted_dir/"retracted_%04d.png"),
                            "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2",
                            "-c:v","libx264","-pix_fmt","yuv420p",
                            "-movflags","+faststart",
                            str(args.out/"wrist_retracted_camera.mp4")],
                            check=True)
                        info["wrist_retracted_video_status"]="created"
                except subprocess.CalledProcessError:
                    info["wrist_video_status"]="ffmpeg_failed_pngs_saved"
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
    p.add_argument("--wrist-near-ratio",type=float,default=.001,
                   help="MuJoCo model.vis.map.znear ratio for camera near clip")
    p.add_argument("--wrist-retract-mm",type=float,default=5.,
                   help="Virtual optical comparison only; physical mount unchanged")
    p.add_argument("--muhi-mass-g",type=float,default=50.,
                   help="Measured total mass in grams, defaults to provisional 50g")
    p.add_argument("--fixed-contact-max-tip-distance-mm",type=float,default=45.)
    p.add_argument("--root-contact-abort-force-n",type=float,default=.02)
    p.add_argument("--jaw-force-spike-abort-n",type=float,default=10.,
                   help="Stop after impact above 10N; diagnostics only, not preventive contact control")
    p.add_argument("--duration-s",type=float,default=6.0)
    p.add_argument("--approach-mm",type=float,default=0.,
                   help="Do not move arm during baseline. The Muhi is placed closer by search.")
    p.add_argument("--min-initial-fixed-gap-mm",type=float,default=2.)
    p.add_argument("--max-initial-fixed-gap-mm",type=float,default=25.)
    p.add_argument("--j6-armature",type=float,default=.002)
    p.add_argument("--j6-joint-damping",type=float,default=.06)
    p.add_argument("--j6-limit-margin-deg",type=float,default=6.,
                   help="Soft MuJoCo joint-limit diagnostic margin; abort beyond 6 degrees")
    p.add_argument("--j234-s",type=float,default=.8)
    p.add_argument("--j234-reposition-mm",type=float,default=15.)
    p.add_argument("--j234-min-progress-mm",type=float,default=3.,
                   help="Fail closed if J2-J4 TCP cannot move at least this far")
    p.add_argument("--j234-tcp-object-clearance-mm",type=float,default=22.)
    p.add_argument("--j234-object-clearance-mm",type=float,default=.8)
    p.add_argument("--approach-s",type=float,default=1.0,
                   help="Time reserved for gentle J1 fixed-jaw contact")
    p.add_argument("--j1-step-deg",type=float,default=.012)
    p.add_argument("--j1-root-clearance-mm",type=float,default=1.0)
    p.add_argument("--j1-fingertip-zone-mm",type=float,default=45.0)
    p.add_argument("--j1-max-deg",type=float,default=8.)
    p.add_argument("--j1-stop-before-contact-mm",type=float,default=.65,
                   help="Leave 0.65mm between fixed jaw and Muhi before J6 starts")
    p.add_argument("--j6-precontact-max-torque",type=float,default=.015,
                   help="Reduced Nm cap while gently closing around Muhi")
    p.add_argument("--j6-hold-force-n",type=float,default=1.,
                   help="Stop advancing J6 if any one jaw exceeds 1N")
    p.add_argument("--j6-bilateral-min-force-n",type=float,default=.02,
                   help="Hold J6 when both jaws sustain minimum measured force")
    p.add_argument("--j1-contact-gap-mm",type=float,default=-.25,
                   help="Guarded J1 geometric limit (signed distance, -0.25mm); contact force required for J6")
    p.add_argument("--j1-min-fixed-normal-force-n",type=float,default=.02,
                   help="Minimum actual fixed-jaw normal force to enable J6")
    p.add_argument("--j1-force-hold-steps",type=int,default=5)
    p.add_argument("--j1-max-penetration-mm",type=float,default=.4)
    p.add_argument("--j1-moving-clearance-mm",type=float,default=.5)
    p.add_argument("--j1-max-tilt-deg",type=float,default=8.)
    p.add_argument("--abort-jaw-penetration-mm",type=float,default=1.5)
    p.add_argument("--approach-clearance-mm",type=float,default=.5)
    p.add_argument("--table-clearance-mm",type=float,default=2.0)
    p.add_argument("--lift-mm",type=float,default=5.0)
    p.add_argument("--settle-s",type=float,default=.35)
    p.add_argument("--close-s",type=float,default=2.2)
    p.add_argument("--kp",type=float,default=.15)
    p.add_argument("--kd",type=float,default=.08)
    p.add_argument("--max-torque",type=float,default=.06)
    p.add_argument("--video",action="store_true")
    p.add_argument("--fps",type=float,default=12.)
    p.add_argument("--render-every",type=int,default=20)
    args=p.parse_args()
    if not all(math.isfinite(v) for v in
               (args.j5_deg,args.duration_s,args.settle_s,args.close_s,
                args.kp,args.kd,args.max_torque,args.fps,
                args.approach_mm,args.approach_s,args.approach_clearance_mm,
                args.j1_stop_before_contact_mm,
                args.j6_precontact_max_torque,args.j6_hold_force_n,
                args.j6_bilateral_min_force_n,
                args.j1_step_deg,args.j1_root_clearance_mm,
                args.j1_fingertip_zone_mm,
                args.j1_max_deg,args.j1_contact_gap_mm,
                args.j1_max_penetration_mm,args.j1_moving_clearance_mm,
                args.j1_max_tilt_deg,
                args.j1_min_fixed_normal_force_n,
                args.j234_s,args.j234_reposition_mm,
                args.j234_min_progress_mm,
                args.j234_tcp_object_clearance_mm,args.j234_object_clearance_mm,
                args.j6_limit_margin_deg,
                args.wrist_near_ratio,args.wrist_retract_mm,
                args.muhi_mass_g,args.fixed_contact_max_tip_distance_mm,
                args.root_contact_abort_force_n,args.jaw_force_spike_abort_n)):
        p.error("parameters must be finite")
    if (args.fps<=0 or args.table_clearance_mm<0 or
        not(0.0001<=args.wrist_near_ratio<=.01) or
        not(0<=args.wrist_retract_mm<=15) or
        not(5<=args.muhi_mass_g<=500) or
        not(10<=args.fixed_contact_max_tip_distance_mm<=100) or
        not(0<args.root_contact_abort_force_n<=5) or
        not(1<=args.jaw_force_spike_abort_n<=100) or
        args.j6_armature<0 or args.j6_joint_damping<0 or
        not(2<=args.j6_limit_margin_deg<=8) or
        not(0.5<=args.min_initial_fixed_gap_mm<args.max_initial_fixed_gap_mm<=35) or
        not(0<args.lift_mm<=10) or not(0<=args.approach_mm<=25) or
        args.approach_s<=0 or args.approach_clearance_mm<0 or
        not(.1<=args.j1_stop_before_contact_mm<=2.) or
        not(0<args.j6_precontact_max_torque<=args.max_torque) or
        not(.1<=args.j6_hold_force_n<=5.) or
        not(0<args.j6_bilateral_min_force_n<=args.j6_hold_force_n) or
        not(0<args.j1_step_deg<=.05) or
        not(0.2<=args.j1_root_clearance_mm<=5.) or
        not(10<=args.j1_fingertip_zone_mm<=100) or not(0<args.j1_max_deg<=15) or
        not(-.35<=args.j1_contact_gap_mm<=.5) or
        not(0<args.j1_min_fixed_normal_force_n<=1.) or
        not(1<=args.j1_force_hold_steps<=30) or
        not(0<=args.j1_max_penetration_mm<=.5) or
        args.j1_max_penetration_mm < -args.j1_contact_gap_mm or
        args.j1_moving_clearance_mm<0 or
        not(0<args.j1_max_tilt_deg<=15) or
        not(.5<=args.abort_jaw_penetration_mm<=3.) or
        args.j234_s<=0 or
        not(0<=args.j234_reposition_mm<=25) or
        not(0<=args.j234_min_progress_mm<=args.j234_reposition_mm) or
        args.j234_tcp_object_clearance_mm<0 or
        args.j234_object_clearance_mm<0 or
        args.settle_s+args.j234_s+args.approach_s+args.close_s>=args.duration_s):
        p.error("Invalid FPS, table clearance or lift mm")
    execute(args)


if __name__=="__main__":
    main()
