#!/usr/bin/env python3
"""SO-101 isolated Realistic Reach preflight (MuJoCo-only, no robot hardware).

Unlike physical_grasp_table_bench.py this script NEVER chooses the Muhi's
position from the gripper. The world-frame object/table pose and robot home
are independent inputs. It checks initial separation, plans a guarded
kinematic J1-J4 TCP approach, then replays it with gravity and actual
Muhi/table contacts. It does NOT close J6 or claim a physical grasp.
All poses here are SYNTHETIC / UNCALIBRATED until explicitly measured.
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
import base_shoulder_pair_filter as pair_filter
import wrist_camera_sim as wrist
import fixed_fingertip_contact_planner as fingertip

SITE = "realistic_reach_tcp"
TABLE = "realistic_reach_table"
TABLE_BODY = "realistic_reach_table_body"
TCP_LOCAL = (-.0079, -.000218121, -.0981274)


def finite(values):
    return all(math.isfinite(float(v)) for v in values)


def joint_info(model, name):
    q, d, lim = bench.joint_addr(model, name)
    return q, d, tuple(float(v) for v in lim)


def dist(model, data, x, y):
    return float(mujoco.mj_geomDistance(model, data, int(x), int(y),
                                       .25, np.zeros(6)))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--muhi-xy-m", type=float, nargs=2, default=[.30, -.09],
                    help="Independent synthetic WORLD x,y, not gripper local")
    ap.add_argument("--table-top-z-m", type=float, default=.16)
    ap.add_argument("--home-deg", type=float, nargs=5,
                    default=[0., -30., 45., -20., -90.])
    ap.add_argument("--auto-safe-home", action="store_true",
                    help="Search ONLY J1 home yaw candidates for a separated "
                         "initial scene, never relocate Muhi/table")
    ap.add_argument("--pregrasp-offset-m", type=float, nargs=3,
                    default=[0., .075, .015],
                    help="WORLD offset: approach from robot/home +Y side; TCP")
    ap.add_argument("--min-initial-clearance-mm", type=float, default=45.)
    ap.add_argument("--min-path-clearance-mm", type=float, default=4.)
    ap.add_argument("--tcp-tolerance-mm", type=float, default=12.)
    ap.add_argument("--max-iterations", type=int, default=400)
    ap.add_argument("--j4-policy", choices=("prefer-fixed","locked","free"),
                    default="prefer-fixed",
                    help="Weighted DLS penalizes J4 motion; locked excludes J4 "
                         "entirely; free reproduces the original DLS")
    ap.add_argument("--j4-weight", type=float, default=80.,
                    help="Penalty on J4 joint velocity when prefer-fixed")
    ap.add_argument("--reach-mode", choices=("staged", "direct"),
                    default="staged",
                    help="Staged: lift TCP, move over target, descend; "
                         "direct retains old diagnostic behavior")
    ap.add_argument("--via-height-mm", type=float, default=110.,
                    help="Increase TCP Z by this amount above home AND pregrasp "
                         "TCP heights before the horizontal segment")
    ap.add_argument("--render", action="store_true",
                    help="Save start, middle and final snapshots")
    ap.add_argument("--approach-fixed-jaw", action="store_true",
                    help="After Reach, plan guarded *fixed fingertip* geometric "
                         "precontact. Report physical force separately.")
    ap.add_argument("--grasp-depth-mm", type=float, default=18.,
                    help="Bottle-center depth from fixed fingertip toward palm in gripper coordinates")
    ap.add_argument("--grasp-depth-tolerance-mm",type=float,default=7.,
                    help="Accepted grasp depth error in millimeters")
    ap.add_argument("--fingertip-zone-mm", type=float, default=30.,
                    help="Max distance between Muhi/fixed-jaw witness and TCP")
    ap.add_argument("--fixed-jaw-target-gap-mm", type=float, default=-.02,
                    help="Simulation only: bounded -0.02mm fixed-tip overlap to test solver force")
    ap.add_argument("--video", action="store_true",
                    help="Render original 2-side + 3-wrist combined MP4 and separate camera MP4s")
    ap.add_argument("--wrist-retract-mm", type=float, default=5.,
                    help="Virtual retracted wrist optic comparison, millimeters")
    ap.add_argument("--fps", type=int, default=15,
                    help="Playback fps for diagnostic Reach video")
    ap.add_argument("--video-every-waypoints", type=int, default=2,
                    help="Sample every N accepted waypoints for video")
    ap.add_argument("--exclude-base-shoulder-contact", action="store_true",
                    help="Simulation diagnostic only; do not modify URDF")
    args = ap.parse_args()
    if args.out.exists():
        ap.error("Output folder exists; choose a fresh path")
    if not finite([*args.muhi_xy_m, args.table_top_z_m, *args.home_deg,
                   *args.pregrasp_offset_m, args.via_height_mm,
                   args.j4_weight, args.grasp_depth_mm,
                   args.grasp_depth_tolerance_mm, args.fingertip_zone_mm,
                   args.fixed_jaw_target_gap_mm,
                   args.min_initial_clearance_mm,
                   args.min_path_clearance_mm, args.tcp_tolerance_mm]):
        ap.error("Non-finite pose or tolerance")
    if not (0.08 <= args.table_top_z_m <= .45 and
            args.min_initial_clearance_mm >= 20. and
            1. <= args.min_path_clearance_mm <= 20. and
            1. <= args.tcp_tolerance_mm <= 30. and
            10 <= args.max_iterations <= 1500 and
            30 <= args.via_height_mm <= 250 and
            1 <= args.j4_weight <= 1000 and
            10 <= args.grasp_depth_mm <= 30 and
            2 <= args.grasp_depth_tolerance_mm <= 12 and
            15 <= args.fingertip_zone_mm <= 60 and
            -.04 <= args.fixed_jaw_target_gap_mm <= .5 and
            0 <= args.wrist_retract_mm <= 15 and
            1 <= args.fps <= 60 and
            1 <= args.video_every_waypoints <= 30):
        ap.error("Invalid guard thresholds")

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "scenario": "independently placed tabletop Muhi; home-to-pregrasp Reach",
        "simulation_only": True, "uncalibrated_synthetic_geometry": True,
        "muhi_is_never_selected_from_gripper_pose": True,
        "object_pose_never_teleported_during_replay": True,
        "physics_grasp_success": "NOT_TESTED",
        "j6_closure": "NOT_IMPLEMENTED_IN_THIS_STAGE",
        "status": "NOT_STARTED",
    }
    model = None
    data = None
    timeline = []
    trajectory = []

    try:
        spec = mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
        spec.compiler.discardvisual = False
        physical.install_dynamic_muhi(spec)
        # Reuse the same provisional camera, 15/35/35mm bent bracket and
        # collision proxies as physical_grasp_table_bench.py. The body is
        # mounted on gripper_link (J5 side) and follows the arm during Reach.
        tcp_rig_local=tuple(
            TCP_LOCAL[i]-wrist.DEFAULT_MOUNT_POS[i] for i in range(3))
        wrist.install(spec,bracket_profile="ruler_15_35_35",
                      camera_aim_at=tcp_rig_local,
                      camera_retract_mm=args.wrist_retract_mm)
        report["wrist_camera_mount"]={
            "installed":True,"camera_name":wrist.CAMERA,
            "bracket_profile":"ruler_15_35_35",
            "geometry":"visual and collision",
            "mount_calibration":"photo/ruler estimate, not calibrated",
            "virtual_retract_mm":float(args.wrist_retract_mm)}
        grip = spec.body("gripper_link")
        if grip is None:
            raise ValueError("Missing gripper_link")
        grip.add_site(name=SITE, pos=list(TCP_LOCAL),
                      size=[.002, .002, .002], rgba=[0.2, .9, .7, 1])
        center = np.array([*args.muhi_xy_m,
                           args.table_top_z_m + physical.MUHI_LENGTH / 2. + .0005])
        table_half = [.11, .11, .012]
        table = spec.worldbody.add_body(
            name=TABLE_BODY,
            pos=[args.muhi_xy_m[0], args.muhi_xy_m[1],
                 args.table_top_z_m-table_half[2]])
        table.add_geom(name=TABLE,
                       type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=table_half, pos=[0, 0, 0],
                       contype=1, conaffinity=1, rgba=[.43,.5,.56,1],
                       friction=[.9,.01,.001])
        model = spec.compile()
        model.opt.timestep = .002
        # Match the original tabletop bench's wrist near-plane setting:
        # a large near plane can clip the close-up gripper/mount.
        near_original=float(model.vis.map.znear)
        model.vis.map.znear=min(near_original,.001)
        report["wrist_near_ratio_original"]=near_original
        report["wrist_near_ratio_applied"]=float(model.vis.map.znear)
        data = mujoco.MjData(model)
        qs, ds, lims = [], [], []
        for name in r2s.JOINTS:
            q, d, lim = joint_info(model, name)
            qs.append(q)
            ds.append(d)
            lims.append(lim)
        jaw_q, jaw_d, jaw_lim = joint_info(model, "gripper")
        home = np.radians(np.asarray(args.home_deg, dtype=float))
        for i, (a, (lo, hi)) in enumerate(zip(home, lims)):
            if not lo <= a <= hi:
                raise ValueError(f"Home J{i+1} outside URDF limits: {args.home_deg[i]}")
        joint = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, physical.DYNAMIC_JOINT))
        if joint < 0:
            raise ValueError("Dynamic Muhi joint missing")
        free_q = int(model.jnt_qposadr[joint])
        body_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, physical.DYNAMIC_BODY))
        obj_geoms = tuple(physical.geom_id(model, n) for n in physical.DYNAMIC_GEOMS)
        table_geom = physical.geom_id(model, TABLE)
        tcp_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE))
        if tcp_id < 0:
            raise ValueError("Missing TCP site")
        wrist_cam_id=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_CAMERA,wrist.CAMERA))
        if wrist_cam_id < 0:
            raise ValueError("Wrist camera missing after MuJoCo compile")
        report["wrist_camera_id"]=wrist_cam_id
        shelf_cam_id=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_CAMERA,
            wrist.SHELF_DIAGNOSTIC_CAMERA))
        retract_cam_id=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_CAMERA,
            wrist.RETRACTED_DIAGNOSTIC_CAMERA))
        if shelf_cam_id<0 or (args.wrist_retract_mm>0 and retract_cam_id<0):
            raise ValueError("Required shelf/retracted wrist camera missing")
        report["video_layout"]="previous bench: side A/B above wrist TCP/shelf/retracted, 1440x780"
        if args.exclude_base_shoulder_contact:
            report["collision_policy"] = pair_filter.apply_base_shoulder_pair_filter(
                model, object_geoms=obj_geoms, table_geom=table_geom)
        else:
            report["collision_policy"] = {"mode": "original_contact_masks"}
        robot_geoms = tuple(i for i in range(model.ngeom)
                            if int(model.geom_bodyid[i]) not in
                            (0, body_id, int(model.geom_bodyid[table_geom])) and
                            (model.geom_contype[i] or model.geom_conaffinity[i]))
        if not robot_geoms:
            raise ValueError("No robot collidable geoms")
        mujoco.mj_resetData(model, data)
        data.qpos[jaw_q] = jaw_lim[1]
        data.qpos[free_q:free_q+3] = center
        data.qpos[free_q+3:free_q+7] = [1., 0., 0., 0.]
        data.qvel[:] = 0
        target = center + np.asarray(args.pregrasp_offset_m)
        robot_obj = lambda: min(
            dist(model, data, g, og)
            for g in robot_geoms for og in obj_geoms)
        robot_table = lambda: min(
            dist(model, data, g, table_geom)
            for g in robot_geoms)

        # The previously chosen fixed home [0,-30,45,-20,-90] placed an
        # arm *mesh* only 27.97 mm from Muhi, despite TCP-center distance
        # 101.56 mm. Select a separated HOME by rotating J1 ONLY.
        # The object's world pose and tabletop must never be moved.
        original_home_deg = list(map(float, args.home_deg))
        offsets_deg = [0.]
        if args.auto_safe_home:
            for magnitude in (15., 30., 45., 60., 75., 90., 105.):
                offsets_deg += [magnitude, -magnitude]
        safe_candidates = []
        chosen = None
        for offset in offsets_deg:
            candidate = home.copy()
            candidate[0] = home[0] + math.radians(offset)
            if not (lims[0][0] <= candidate[0] <= lims[0][1]):
                continue
            data.qpos[qs] = candidate
            mujoco.mj_forward(model, data)
            cg = float(robot_obj())
            tg = float(robot_table())
            tc = np.asarray(data.site_xpos[tcp_id]).copy()
            to_target = float(np.linalg.norm(tc - target)*1000.)
            eligible = (cg >= args.min_initial_clearance_mm/1000. and
                        tg >= args.min_path_clearance_mm/1000. and
                        to_target >= args.min_initial_clearance_mm)
            safe_candidates.append({
                "j1_deg": float(math.degrees(candidate[0])),
                "robot_object_gap_mm": cg*1000.,
                "robot_table_gap_mm": tg*1000.,
                "tcp_to_pregrasp_mm": to_target,
                "eligible": bool(eligible)})
            if eligible and chosen is None:
                chosen = candidate.copy()
                if args.auto_safe_home:
                    break
        report["requested_home_deg"] = original_home_deg
        report["auto_safe_home_enabled"] = bool(args.auto_safe_home)
        report["home_candidate_scan"] = safe_candidates
        if chosen is None:
            best_gap = max((c["robot_object_gap_mm"] for c in safe_candidates),
                           default=None)
            report["largest_candidate_robot_object_gap_mm"] = best_gap
            raise ValueError(
                "INITIAL_HOME_NOT_SEPARATED: no safe J1-only home candidate; "
                "see home_candidate_scan; no object/table relocation")
        home = chosen
        data.qpos[qs] = home
        mujoco.mj_forward(model, data)
        start_tcp = np.asarray(data.site_xpos[tcp_id]).copy()
        initial_object_gap = robot_obj()
        initial_table_gap = robot_table()
        report.update({
            "world_muhi_center_m": center.tolist(),
            "world_table_top_m": float(args.table_top_z_m),
            "world_pregrasp_tcp_target_m": target.tolist(),
            "robot_home_deg": np.degrees(home).tolist(),
            "home_pose_requested_deg": list(map(float, args.home_deg)),
            "home_pose_applied_deg": np.degrees(home).tolist(),
            "tcp_initial_world_m": start_tcp.tolist(),
            "tcp_initial_distance_to_muhi_mm":
                float(np.linalg.norm(start_tcp-center)*1000),
            "initial_robot_object_gap_mm": initial_object_gap*1000,
            "initial_robot_table_gap_mm": initial_table_gap*1000,
            "min_initial_clearance_mm": args.min_initial_clearance_mm,
            "pregrasp_offset_m": list(map(float, args.pregrasp_offset_m)),
        })
        if initial_object_gap < args.min_initial_clearance_mm/1000:
            raise ValueError(
                "INITIAL_ROBOT_TOO_CLOSE_TO_MUHI: model is not a separated start")
        if initial_table_gap < args.min_path_clearance_mm/1000:
            raise ValueError("INITIAL_ROBOT_TABLE_INTERSECTION")
        if np.linalg.norm(start_tcp-target) < args.min_initial_clearance_mm/1000:
            raise ValueError("INITIAL_TCP_TOO_CLOSE_TO_PREGRASP")
        report["status"] = "INITIAL_SCENE_VALID"

        # Position-only IK, with optional *world-frame* staged approach.
        # This is still not a global path planner. A rigid world target and
        # 4mm robot-table/object clearance are enforced throughout. Neither
        # Muhi nor the table is moved to make an unreachable stage succeed.
        pose = home.copy()
        trajectory.append(pose.copy())
        min_gap = initial_object_gap
        min_table_gap = initial_table_gap
        report["planned_tcp_position_only"] = True
        report["j4_policy"] = args.j4_policy
        report["j4_weight"] = float(args.j4_weight)
        report["j4_home_deg"] = float(math.degrees(home[3]))
        raised_z = max(float(start_tcp[2]), float(target[2])) + (
            args.via_height_mm / 1000.)
        if args.reach_mode == "staged":
            stages = (
                ("raise_above_table",
                 np.array([start_tcp[0], start_tcp[1], raised_z])),
                ("translate_above_table",
                 np.array([target[0], target[1], raised_z])),
                ("descend_to_pregrasp", target.copy()),
            )
        else:
            stages = (("direct_to_pregrasp", target.copy()),)
        report["reach_mode"] = args.reach_mode
        report["via_height_mm"] = float(args.via_height_mm)
        report["reach_stage_targets_world_m"] = [
            dict(stage=n, tcp_goal_m=t.tolist()) for n,t in stages]
        stage_summaries = []
        trajectory_stages = ["home"]
        overall_failure = None
        for stage_name, goal in stages:
            counters = {
                "accepted_steps": 0, "unsafe_object": 0,
                "unsafe_table": 0, "no_improvement": 0,
                "joint_limit_or_zero_step": 0,
            }
            stage_failure = None
            for _ in range(args.max_iterations):
                data.qpos[qs] = pose
                mujoco.mj_forward(model, data)
                tcp = np.asarray(data.site_xpos[tcp_id]).copy()
                e = goal-tcp
                distance = float(np.linalg.norm(e))
                if distance*1000. <= args.tcp_tolerance_mm:
                    break
                jacp = np.zeros((3,model.nv))
                jacr = np.zeros((3,model.nv))
                mujoco.mj_jacSite(model,data,jacp,jacr,tcp_id)
                # Weighted DLS: J1 rotates the base; J2/J3 provide most
                # reach. Penalize J4 wrist-flex unless it is really needed.
                # A user may lock J4 or reproduce the old unconstrained DLS.
                if args.j4_policy == "locked":
                    indices = (0,1,2)
                    weights = np.ones(3)
                else:
                    indices = (0,1,2,3)
                    weights = np.array([1.,1.,1.,
                        args.j4_weight if args.j4_policy == "prefer-fixed" else 1.])
                J = jacp[:,[ds[j] for j in indices]]
                weighted_jac = J / weights[np.newaxis,:]
                step_active = (J.T @ np.linalg.solve(
                    weighted_jac @ J.T + .04**2*np.eye(3),e))/weights
                step = np.zeros(4)
                for j,delta in zip(indices,step_active):
                    step[j]=delta
                if not finite(step):
                    stage_failure = "NONFINITE_DLS_UPDATE"
                    break
                step=np.clip(step,-.012,.012)
                if args.j4_policy == "prefer-fixed":
                    step[3]=np.clip(step[3],-.002,.002)
                best = None
                for scale in (1., .5, .25, .125, .0625):
                    proposed = pose.copy()
                    proposed[:4] = np.clip(
                        proposed[:4]+scale*step,
                        np.asarray(lims)[:4,0],
                        np.asarray(lims)[:4,1])
                    if np.linalg.norm(proposed-pose)<1e-8:
                        counters["joint_limit_or_zero_step"]+=1
                        continue
                    # Check the swept joint segment, not merely its endpoint:
                    # otherwise a single IK step can tunnel through Muhi/table.
                    candidate_ok = True
                    candidate_min_obj = float("inf")
                    candidate_min_table = float("inf")
                    for fraction in (.25,.5,.75,1.):
                        intermediate = pose + fraction*(proposed-pose)
                        data.qpos[qs] = intermediate
                        mujoco.mj_forward(model,data)
                        gap = robot_obj()
                        table_gap = robot_table()
                        candidate_min_obj = min(candidate_min_obj,gap)
                        candidate_min_table = min(candidate_min_table,table_gap)
                        if gap < args.min_path_clearance_mm/1000.:
                            counters["unsafe_object"]+=1
                            candidate_ok=False
                            break
                        if table_gap < args.min_path_clearance_mm/1000.:
                            counters["unsafe_table"]+=1
                            candidate_ok=False
                            break
                    if not candidate_ok:
                        continue
                    new_error = float(
                        np.linalg.norm(goal-data.site_xpos[tcp_id]))
                    if new_error >= distance-1e-8:
                        counters["no_improvement"]+=1
                        continue
                    if best is None or new_error < best[0]:
                        best=(new_error,proposed.copy(),
                              candidate_min_obj,candidate_min_table)
                if best is None:
                    stage_failure="DLS_STALLED_OR_CLEARANCE_REJECTED"
                    break
                pose=best[1]
                min_gap=min(min_gap,best[2])
                min_table_gap=min(min_table_gap,best[3])
                trajectory.append(pose.copy())
                trajectory_stages.append(stage_name)
                counters["accepted_steps"]+=1
            data.qpos[qs]=pose
            mujoco.mj_forward(model,data)
            final_stage_error_mm = float(
                np.linalg.norm(goal-data.site_xpos[tcp_id])*1000.)
            stage_ok = final_stage_error_mm <= args.tcp_tolerance_mm
            if not stage_ok and stage_failure is None:
                stage_failure="STAGE_ITERATION_LIMIT"
            stage_summaries.append({
                "stage": stage_name,
                "target_tcp_world_m": goal.tolist(),
                "end_tcp_world_m": np.asarray(
                    data.site_xpos[tcp_id]).tolist(),
                "final_error_mm": final_stage_error_mm,
                "success": bool(stage_ok),
                "failure_reason": stage_failure,
                **counters,
            })
            if not stage_ok:
                overall_failure=f"{stage_name}: {stage_failure}"
                break
        data.qpos[qs] = pose
        mujoco.mj_forward(model, data)
        reached_error_mm = float(
            np.linalg.norm(target-data.site_xpos[tcp_id])*1000.)
        # Identify the actual closest robot-vs-object witness when a stage
        # stalls. This is more actionable than a single global clearance.
        data.qpos[qs]=pose
        mujoco.mj_forward(model,data)
        closest=(float("inf"),None,None)
        for g in robot_geoms:
            for og in obj_geoms:
                gap=dist(model,data,g,og)
                if gap<closest[0]:
                    closest=(gap,int(g),int(og))
        def geom_name(g):
            return (mujoco.mj_id2name(
                model,mujoco.mjtObj.mjOBJ_GEOM,g) or f"geom_{g}")
        report["closest_robot_object_pair_at_end"]={
            "clearance_mm":closest[0]*1000.,
            "robot_geom_id":closest[1],
            "robot_geom_name":geom_name(closest[1]),
            "robot_body_name":(
                mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,
                                  int(model.geom_bodyid[closest[1]]))
                if closest[1] is not None else None),
            "muhi_geom_name":geom_name(closest[2]),
        }
        report.update({
            "reach_accepted_steps": len(trajectory)-1,
            "reach_stage_results": stage_summaries,
            "reach_final_tcp_error_mm": reached_error_mm,
            "reach_final_joint_deg": np.degrees(pose).tolist(),
            "j4_total_motion_deg":float(math.degrees(pose[3]-home[3])),
            "reach_min_robot_object_clearance_mm": min_gap*1000.,
            "reach_min_robot_table_clearance_mm": min_table_gap*1000.,
            "reach_failure_reason": overall_failure,
        })
        if overall_failure or reached_error_mm > args.tcp_tolerance_mm:
            report["status"]="REACH_NOT_ESTABLISHED"
        else:
            report["status"]="PREGRASP_POSITION_REACHED_NO_GRASP"

        # This was missing in the old "Reach success" definition.
        # A TCP 75mm from bottle center is NOT a fixed-fingertip contact.
        fixed_geoms=bench.owned_collision_geoms(model,bench.FINGER_BODY)
        moving_geoms=bench.owned_collision_geoms(model,bench.MOVING_BODY)
        grip_body=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_BODY,bench.FINGER_BODY))
        other_robot_geoms=tuple(g for g in robot_geoms if g not in fixed_geoms)
        report["fixed_fingertip_contact_phase"]="NOT_REQUESTED"
        if args.approach_fixed_jaw:
            if report["status"]!="PREGRASP_POSITION_REACHED_NO_GRASP":
                report["fixed_fingertip_contact_phase"]="REACH_FAILED_SKIP"
            elif not fixed_geoms:
                report["fixed_fingertip_contact_phase"]="NO_FIXED_GEOMS"
            else:
                subpath,metrics,approach=fingertip.plan_fixed_fingertip_approach(
                    model,data,pose,qs,obj_geoms,fixed_geoms,
                    other_robot_geoms,table_geom,tcp_id,lims,
                    target_gap_mm=args.fixed_jaw_target_gap_mm,
                    tip_zone_mm=args.fingertip_zone_mm,
                    clearance_mm=args.min_path_clearance_mm,
                    max_steps=400,
                    object_center_world=center,
                    grip_body_id=grip_body,
                    moving_geoms=moving_geoms,
                    jaw_qpos_id=jaw_q,
                    jaw_limits=jaw_lim,
                    target_depth_mm=args.grasp_depth_mm,
                    depth_tolerance_mm=args.grasp_depth_tolerance_mm)
                report["fixed_fingertip_contact_phase"]=approach["status"]
                report["fixed_fingertip_approach"]=approach
                write_csv(args.out/"fixed_fingertip_plan_trace.csv",metrics)
                for q in subpath[1:]:
                    trajectory.append(q.copy())
                    trajectory_stages.append("fixed_fingertip_approach")
                if approach["success"]:
                    report["status"]="FINGERTIP_PRECONTACT_GEOMETRY_ONLY"
                else:
                    report["status"]="FINGERTIP_APPROACH_NOT_ESTABLISHED"
        # Replay the path at fixed 10ms steps. Only object dynamics are free.
        # J1-J5 are kinematically prescribed; this is NOT servo-control proof.
        mujoco.mj_resetData(model,data)
        data.qpos[qs] = home
        data.qpos[jaw_q] = jaw_lim[1]
        data.qpos[free_q:free_q+3] = center
        data.qpos[free_q+3:free_q+7] = [1.,0.,0.,0.]
        data.qvel[:] = 0.
        mujoco.mj_forward(model,data)
        path = trajectory if len(trajectory)>1 else [home]
        snapshot_index = {0, len(path)//2, len(path)-1}
        fixed_contact_count=0
        fixed_contact_consecutive=0
        fixed_contact_max_consecutive=0
        max_fixed_contact_force_n=0.
        # MuJoCo's solver is the only authority for claiming force contact.
        # Being 0.03mm away geometrically does not prove force or grasp.
        # Same desktop comparison layout as physical_grasp_table_bench.py:
        # 2 side views (top), 3 wrist views (bottom), 1440x780.
        # Also emit each individual camera movie. Keep frame series synced.
        video_frame_count=0
        frame_dir=args.out/"_temporary_video_frames"
        names=("combined","side","side_b","wrist","shelf","retracted")
        dirs={name:frame_dir/name for name in names}
        if args.video:
            for directory in dirs.values():
                directory.mkdir(parents=True,exist_ok=True)
            report["video_status"]="PENDING"
            report["wrist_video_status"]="PENDING"
        renderer=None
        if args.render or args.video:
            try:
                model.vis.global_.offwidth=max(480,model.vis.global_.offwidth)
                model.vis.global_.offheight=max(360,model.vis.global_.offheight)
                renderer=mujoco.Renderer(model,width=480,height=360)
            except Exception as render_error:
                report["video_status"]="RENDERER_UNAVAILABLE"
                report["video_error"]=repr(render_error)
        side_camera=mujoco.MjvCamera()
        side_camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        side_camera.lookat[:]=center
        side_camera.distance=.75
        side_camera.azimuth=70
        side_camera.elevation=-20
        scene_option=mujoco.MjvOption()
        scene_option.geomgroup[3]=0
        for k, q in enumerate(path):
            for _ in range(5):
                data.qpos[qs] = q
                data.qvel[ds] = 0
                data.qpos[jaw_q] = jaw_lim[1]
                data.qvel[jaw_d] = 0
                mujoco.mj_step(model,data)
                data.qpos[qs] = q
                data.qvel[ds] = 0
                data.qpos[jaw_q] = jaw_lim[1]
                data.qvel[jaw_d] = 0
                mujoco.mj_forward(model,data)
                gap = robot_obj()
                table_gap = robot_table()
                force=0.
                obj_set=set(obj_geoms)
                fixed_set=set(fixed_geoms)
                for ci in range(data.ncon):
                    ct=data.contact[ci]
                    g1,g2=int(ct.geom1),int(ct.geom2)
                    if ((g1 in fixed_set and g2 in obj_set) or
                        (g2 in fixed_set and g1 in obj_set)):
                        wrench=np.zeros(6)
                        mujoco.mj_contactForce(model,data,ci,wrench)
                        force+=max(0.,float(wrench[0]))
                max_fixed_contact_force_n=max(max_fixed_contact_force_n,force)
                if force>=.02:
                    fixed_contact_count+=1
                    fixed_contact_consecutive+=1
                    fixed_contact_max_consecutive=max(
                        fixed_contact_max_consecutive,fixed_contact_consecutive)
                else:
                    fixed_contact_consecutive=0
                timeline.append(dict(
                    time_s=float(data.time), waypoint_index=k,
                    reach_stage=trajectory_stages[k],
                    tcp_x_m=float(data.site_xpos[tcp_id,0]),
                    tcp_y_m=float(data.site_xpos[tcp_id,1]),
                    tcp_z_m=float(data.site_xpos[tcp_id,2]),
                    robot_muhi_gap_mm=gap*1000,
                    robot_table_gap_mm=table_gap*1000,
                    fixed_jaw_force_n=force,
                    fixed_jaw_force_confirmed=(fixed_contact_consecutive>=5),
                    muhi_x_m=float(data.qpos[free_q]),
                    muhi_y_m=float(data.qpos[free_q+1]),
                    muhi_z_m=float(data.qpos[free_q+2])))
                # A sub-0.05mm fixed-jaw overlap is permitted ONLY in the
                # final contact attempt, with other robot geometries clear.
                # Anything more severe remains a hard failure.
                fixed_gap=bench.geometry_gaps(model,data,obj_geoms,fixed_geoms)
                other_gap=min((dist(model,data,g,og)
                    for g in other_robot_geoms for og in obj_geoms),
                    default=float("inf"))
                if (other_gap<args.min_path_clearance_mm/1000. or
                    table_gap<args.min_path_clearance_mm/1000. or
                    fixed_gap<-.00005 or force>2.):
                    report["status"]="REPLAY_COLLISION_OR_FORCE_LIMIT"
                    report["replay_abort_details"]={
                        "other_robot_gap_mm":other_gap*1000,
                        "fixed_jaw_gap_mm":fixed_gap*1000,
                        "robot_table_gap_mm":table_gap*1000,
                        "fixed_jaw_normal_force_n":force}
                    break
            if report["status"] in (
                    "REPLAY_COLLISION_DETECTED",
                    "REPLAY_COLLISION_OR_FORCE_LIMIT"):
                break
            need_snapshot=args.render and k in snapshot_index
            need_video=(args.video and
                (k%args.video_every_waypoints==0 or k==len(path)-1))
            if renderer is not None and (need_snapshot or need_video):
                try:
                    from PIL import Image,ImageDraw
                    renderer.update_scene(
                        data,camera=side_camera,scene_option=scene_option)
                    side_a=Image.fromarray(renderer.render().copy())
                    if need_snapshot:
                        side_a.save(args.out/f"reach_{k:04d}.png")
                    if need_video:
                        side_camera.azimuth=-70
                        renderer.update_scene(
                            data,camera=side_camera,scene_option=scene_option)
                        side_b=Image.fromarray(renderer.render().copy())
                        side_camera.azimuth=70
                        renderer.update_scene(
                            data,camera=wrist.CAMERA,
                            scene_option=scene_option)
                        w_tcp=Image.fromarray(renderer.render().copy())
                        renderer.update_scene(
                            data,camera=wrist.SHELF_DIAGNOSTIC_CAMERA,
                            scene_option=scene_option)
                        w_shelf=Image.fromarray(renderer.render().copy())
                        if retract_cam_id>=0:
                            renderer.update_scene(
                                data,camera=wrist.RETRACTED_DIAGNOSTIC_CAMERA,
                                scene_option=scene_option)
                            w_retracted=Image.fromarray(renderer.render().copy())
                        else:
                            w_retracted=Image.new("RGB",(480,360),(20,24,30))
                        # Preserve the prior output format, including labels.
                        combined=Image.new("RGB",(1440,780),(20,24,30))
                        combined.paste(side_a,(240,30))
                        combined.paste(side_b,(720,30))
                        combined.paste(w_tcp,(0,420))
                        combined.paste(w_shelf,(480,420))
                        combined.paste(w_retracted,(960,420))
                        draw=ImageDraw.Draw(combined)
                        stage=trajectory_stages[k]
                        draw.text((248,8),
                            f"SIDE A | {stage} | t={data.time:.2f}s",
                            fill="white")
                        draw.text((728,8),"SIDE B | SAME STEP",fill="white")
                        draw.text((8,398),"WRIST | TCP AIM",fill="white")
                        draw.text((488,398),"WRIST | SHELF AXIS",fill="white")
                        draw.text((968,398),
                            f"WRIST | RETRACTED {args.wrist_retract_mm:g}mm",
                            fill="white")
                        images={
                            "combined":combined,"side":side_a,
                            "side_b":side_b,"wrist":w_tcp,
                            "shelf":w_shelf,"retracted":w_retracted}
                        for name,img in images.items():
                            img.save(dirs[name]/
                                     f"frame_{video_frame_count:04d}.png")
                        video_frame_count+=1
                except Exception as render_error:
                    report.setdefault("render_errors",[]).append(
                        f"waypoint {k}: {render_error}")
                    if args.video:
                        report["video_status"]="FRAME_RENDER_FAILED"
                        break
        if renderer is not None:
            renderer.close()
        report["wrist_video_rendered_frames"]=int(video_frame_count)
        report["video_fps"]=int(args.fps)
        report["video_stride_waypoints"]=int(args.video_every_waypoints)
        if args.video and video_frame_count>0 and not report.get("render_errors"):
            ffmpeg=shutil.which("ffmpeg")
            if ffmpeg is None:
                report["video_status"]="FFMPEG_NOT_FOUND"
                report["wrist_video_status"]="FFMPEG_NOT_FOUND"
                report["video_error"]="Install ffmpeg in conda environment"
            else:
                outputs=(
                    ("combined","reach_combined.mp4"),
                    ("side","reach_side.mp4"),
                    ("side_b","reach_side_b.mp4"),
                    ("wrist","reach_wrist.mp4"),
                    ("shelf","reach_wrist_shelf.mp4"),
                    ("retracted","reach_wrist_retracted.mp4"))
                report["video_files"]=[]
                for source,filename in outputs:
                    try:
                        cmd=[ffmpeg,"-hide_banner","-loglevel","error","-y",
                             "-framerate",str(args.fps),"-i",
                             str(dirs[source]/"frame_%04d.png"),
                             "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2",
                             "-c:v","libx264","-pix_fmt","yuv420p",
                             "-movflags","+faststart",
                             str(args.out/filename)]
                        subprocess.run(
                            cmd,check=True,capture_output=True,text=True)
                        report["video_files"].append(filename)
                    except Exception as error:
                        report.setdefault("video_errors",[]).append(
                            f"{filename}: {error}")
                if len(report["video_files"])==len(outputs):
                    report["video_status"]="created"
                    report["wrist_video_status"]="created"
                    shutil.rmtree(frame_dir)
                else:
                    report["video_status"]="PARTIAL"
                    report["wrist_video_status"]="PARTIAL"
        elif args.video and report.get("video_status")=="PENDING":
            report["video_status"]="NO_FRAMES"
            report["wrist_video_status"]="NO_FRAMES"
        report["replay_frames"] = len(timeline)
        report["replay_min_robot_muhi_gap_mm"] = min(
            (r["robot_muhi_gap_mm"] for r in timeline),default=None)
        report["replay_min_robot_table_gap_mm"] = min(
            (r["robot_table_gap_mm"] for r in timeline),default=None)
        report["replay_final_muhi_center_m"] = (
            data.qpos[free_q:free_q+3].tolist())
        report["fixed_jaw_force_max_n"]=max_fixed_contact_force_n
        report["fixed_jaw_contact_steps"]=fixed_contact_count
        report["fixed_jaw_contact_max_consecutive_steps"]=(
            fixed_contact_max_consecutive)
        report["fixed_jaw_physical_contact_confirmed"]=(
            fixed_contact_max_consecutive>=5)
        report["physical_contact_phase"]=(
            "FIXED_JAW_FORCE_CONFIRMED" if
            fixed_contact_max_consecutive>=5 else
            "FIXED_JAW_FORCE_NOT_CONFIRMED")
        approach_result=report.get("fixed_fingertip_approach",{})
        verified=(fixed_contact_max_consecutive>=5 and
                  approach_result.get("success",False) and
                  approach_result.get("moving_jaw_crossing_angle_deg") is not None and
                  abs(approach_result.get("grasp_depth_mm",0.)-
                      args.grasp_depth_mm)<=args.grasp_depth_tolerance_mm and
                  approach_result.get("fixed_jaw_witness_tcp_distance_mm",
                                      float("inf"))<=args.fingertip_zone_mm)
        report["fixed_jaw_tip_contact_verified"]=bool(verified)
        if (args.approach_fixed_jaw and
                report["status"] not in (
                    "REPLAY_COLLISION_DETECTED",
                    "REPLAY_COLLISION_OR_FORCE_LIMIT") and
                not verified):
            report["status"]="FIXED_JAW_CONTACT_NOT_CONFIRMED"
        elif (args.approach_fixed_jaw and
              verified and
              report["status"] not in (
                  "REPLAY_COLLISION_DETECTED",
                  "REPLAY_COLLISION_OR_FORCE_LIMIT")):
            report["status"]="FIXED_JAW_CONTACT_CONFIRMED_NO_J6"
        # If the target TCP is still >tolerance, never report Reach success.
        if report["status"] == "PREGRASP_POSITION_REACHED_NO_GRASP" and (
                len(trajectory) < 2):
            report["status"] = "REACH_NOT_ESTABLISHED"
    except Exception as exc:
        report["status"] = "SCENE_PREFLIGHT_OR_REACH_FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if timeline:
            write_csv(args.out/"reach_replay_trace.csv",timeline)
        if trajectory:
            write_csv(args.out/"reach_joint_path_deg.csv",[
                {"waypoint":i,"reach_stage":trajectory_stages[i],
                 **{name:float(np.degrees(q[j]))
                                 for j,name in enumerate(r2s.JOINTS)}}
                for i,q in enumerate(trajectory)])
        (args.out/"summary.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
