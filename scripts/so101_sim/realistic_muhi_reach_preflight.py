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
from pathlib import Path

import mujoco
import numpy as np

import physical_grasp_j6_bench as bench
import physical_workcell_sim as physical
import real_to_sim as r2s
import base_shoulder_pair_filter as pair_filter
import wrist_camera_sim as wrist

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
                    default=[0., -.075, .015],
                    help="WORLD offset from Muhi center to TCP")
    ap.add_argument("--min-initial-clearance-mm", type=float, default=45.)
    ap.add_argument("--min-path-clearance-mm", type=float, default=4.)
    ap.add_argument("--tcp-tolerance-mm", type=float, default=12.)
    ap.add_argument("--max-iterations", type=int, default=400)
    ap.add_argument("--render", action="store_true",
                    help="Save snapshots if MuJoCo offscreen renderer is available")
    ap.add_argument("--exclude-base-shoulder-contact", action="store_true",
                    help="Simulation diagnostic only; do not modify URDF")
    args = ap.parse_args()
    if args.out.exists():
        ap.error("Output folder exists; choose a fresh path")
    if not finite([*args.muhi_xy_m, args.table_top_z_m, *args.home_deg,
                   *args.pregrasp_offset_m, args.min_initial_clearance_mm,
                   args.min_path_clearance_mm, args.tcp_tolerance_mm]):
        ap.error("Non-finite pose or tolerance")
    if not (0.08 <= args.table_top_z_m <= .45 and
            args.min_initial_clearance_mm >= 20. and
            1. <= args.min_path_clearance_mm <= 20. and
            1. <= args.tcp_tolerance_mm <= 30. and
            10 <= args.max_iterations <= 1500):
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
            "robot_home_deg": list(map(float, args.home_deg)),
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

        # Plan a collision-guarded position-only DLS TCP reach. J5 orientation
        # stays fixed; this is not full 6D grasp pose/orientation planning.
        pose = home.copy()
        trajectory.append(pose.copy())
        min_gap = initial_object_gap
        report["planned_tcp_position_only"] = True
        failure = None
        for it in range(args.max_iterations):
            data.qpos[qs] = pose
            mujoco.mj_forward(model, data)
            tcp = np.asarray(data.site_xpos[tcp_id]).copy()
            e = target-tcp
            error_mm = float(np.linalg.norm(e)*1000.)
            if error_mm <= args.tcp_tolerance_mm:
                break
            jacp = np.zeros((3,model.nv))
            jacr = np.zeros((3,model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, tcp_id)
            J = jacp[:,ds[:4]]
            step = J.T @ np.linalg.solve(J@J.T + .04**2*np.eye(3), e)
            if not finite(step):
                failure = "NONFINITE_DLS_UPDATE"
                break
            step = np.clip(step, -.012, .012)
            trial_ok = False
            original = pose.copy()
            for scale in (1., .5, .25, .125):
                proposed = original.copy()
                proposed[:4] = np.clip(
                    proposed[:4]+scale*step,
                    np.array(lims)[:4,0], np.array(lims)[:4,1])
                if np.linalg.norm(proposed-original)<1e-8:
                    continue
                data.qpos[qs] = proposed
                mujoco.mj_forward(model, data)
                gap = robot_obj()
                table_gap = robot_table()
                new_err = float(np.linalg.norm(target-data.site_xpos[tcp_id]))
                if (gap >= args.min_path_clearance_mm/1000. and
                    table_gap >= args.min_path_clearance_mm/1000. and
                    new_err < float(np.linalg.norm(e))-1e-8):
                    pose = proposed
                    min_gap = min(min_gap,gap)
                    trajectory.append(pose.copy())
                    trial_ok=True
                    break
            if not trial_ok:
                failure = "DLS_STALLED_OR_CLEARANCE_REJECTED"
                break
        data.qpos[qs] = pose
        mujoco.mj_forward(model, data)
        reached_error_mm = float(np.linalg.norm(target-data.site_xpos[tcp_id])*1000)
        report.update({
            "reach_accepted_steps": len(trajectory)-1,
            "reach_final_tcp_error_mm": reached_error_mm,
            "reach_final_joint_deg": np.degrees(pose).tolist(),
            "reach_min_robot_object_clearance_mm": min_gap*1000,
            "reach_failure_reason": failure,
        })
        if reached_error_mm > args.tcp_tolerance_mm:
            report["status"]="REACH_NOT_ESTABLISHED"
        else:
            report["status"]="PREGRASP_POSITION_REACHED_NO_GRASP"

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
        for k, q in enumerate(path):
            for _ in range(5):
                data.qpos[qs] = q
                data.qvel[ds] = 0
                data.qpos[jaw_q] = jaw_lim[1]
                data.qvel[jaw_d] = 0
                mujoco.mj_step(model,data)
                # Keep prescribed robot qpos at command after integration;
                # only the bottle is fully dynamic.
                data.qpos[qs] = q
                data.qvel[ds] = 0
                data.qpos[jaw_q] = jaw_lim[1]
                data.qvel[jaw_d] = 0
                mujoco.mj_forward(model,data)
                gap = robot_obj()
                table_gap = robot_table()
                timeline.append(dict(
                    time_s=float(data.time), waypoint_index=k,
                    tcp_x_m=float(data.site_xpos[tcp_id,0]),
                    tcp_y_m=float(data.site_xpos[tcp_id,1]),
                    tcp_z_m=float(data.site_xpos[tcp_id,2]),
                    robot_muhi_gap_mm=gap*1000,
                    robot_table_gap_mm=table_gap*1000,
                    muhi_x_m=float(data.qpos[free_q]),
                    muhi_y_m=float(data.qpos[free_q+1]),
                    muhi_z_m=float(data.qpos[free_q+2])))
                if gap < 0 or table_gap < 0:
                    report["status"] = "REPLAY_COLLISION_DETECTED"
                    break
            if report["status"] == "REPLAY_COLLISION_DETECTED":
                break
            if args.render and k in snapshot_index:
                try:
                    from PIL import Image
                    render = mujoco.Renderer(model,width=640,height=480)
                    camera = mujoco.MjvCamera()
                    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:]=center
                    camera.distance=.75
                    camera.azimuth=70
                    camera.elevation=-20
                    render.update_scene(data,camera=camera)
                    Image.fromarray(render.render().copy()).save(
                        args.out/f"reach_{k:04d}.png")
                    render.close()
                except Exception as err:
                    report.setdefault("render_errors",[]).append(str(err))
        report["replay_frames"] = len(timeline)
        report["replay_min_robot_muhi_gap_mm"] = min(
            (r["robot_muhi_gap_mm"] for r in timeline),default=None)
        report["replay_min_robot_table_gap_mm"] = min(
            (r["robot_table_gap_mm"] for r in timeline),default=None)
        report["replay_final_muhi_center_m"] = (
            data.qpos[free_q:free_q+3].tolist())
        report["physical_contact_phase"] = "NOT_STARTED"
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
                {"waypoint":i,**{name:float(np.degrees(q[j]))
                                 for j,name in enumerate(r2s.JOINTS)}}
                for i,q in enumerate(trajectory)])
        (args.out/"summary.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
