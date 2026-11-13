#!/usr/bin/env python3
"""Independent SO-101 J6 physical grasp bench (no table, no real hardware).

Reuses source URDF collision geometry and physical_workcell_sim's provisional
50g Muhi cylinder. J1..J5 are kinematically locked. A virtual setup fixture
holds the object during jaw closure and releases it for gravity/contact-only
retention; the hold is explicitly NOT a demonstrated pickup from a table.

No object weld/teleport is allowed AFTER release. If no collision-free
two-jaw setup is found, fail closed with a JSON report. No SAC/DLS/ACT.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

import physical_workcell_sim as physical
import real_to_sim as r2s

FINGER_BODY = "gripper_link"
MOVING_BODY = "moving_jaw_so101_v1_link"


def owned_collision_geoms(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body {body_name} missing")
    return tuple(i for i in range(model.ngeom)
                 if int(model.geom_bodyid[i]) == bid
                 and (model.geom_contype[i] or model.geom_conaffinity[i]))


def joint_addr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Missing joint {name}")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]), model.jnt_range[jid].copy()


def signed_pair_gap(model, data, a, b):
    return float(mujoco.mj_geomDistance(model, data, int(a), int(b),
                                       .15, np.zeros(6)))


def geometry_gaps(model, data, object_geoms, candidate_geoms):
    return min(signed_pair_gap(model, data, a, b)
               for a in object_geoms for b in candidate_geoms)


def select_bench_pose(model, data, object_adr, open_angle, open_gripper_adr,
                      fixed_geoms, moving_geoms, dynamic_geoms):
    """Bounded open-jaw placement scan: never create initial interpenetration.

    The jaw local XYZ grid is deliberately an exploratory setup fixture; it
    is NOT a reconstructed Muhi pose or a grasp planner.
    """
    grip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FINGER_BODY)
    data.qpos[open_gripper_adr] = open_angle
    mujoco.mj_forward(model, data)
    origin = np.asarray(data.xpos[grip_body]).copy()
    rot = np.asarray(data.xmat[grip_body]).reshape(3, 3).copy()
    quat = np.asarray(data.xquat[grip_body]).copy()
    other = tuple(i for i in range(model.ngeom)
                  if (model.geom_contype[i] or model.geom_conaffinity[i])
                  and int(model.geom_bodyid[i]) not in (
                      int(model.geom_bodyid[fixed_geoms[0]]),
                      int(model.geom_bodyid[moving_geoms[0]]),
                      int(model.geom_bodyid[dynamic_geoms[0]])))
    # Object cylinder long axis along local gripper Z (fingers extend -Z).
    # Search 0..5 cm either side plus positions along finger length.
    grid = (
        np.linspace(-.038, .038, 9),
        np.linspace(-.045, .045, 9),
        np.linspace(-.125, -.030, 11),
    )
    best, best_score, diagnostics = None, math.inf, {"evaluated": 0,
       "positive_clearance": 0, "near_both_jaws": 0,
       "closest_fixed_gap_mm": None, "closest_moving_gap_mm": None}
    for xyz in itertools.product(*grid):
        center = origin + rot @ np.asarray(xyz)
        data.qpos[object_adr:object_adr+3] = center
        data.qpos[object_adr+3:object_adr+7] = quat
        mujoco.mj_forward(model, data)
        gf = geometry_gaps(model, data, dynamic_geoms, fixed_geoms)
        gm = geometry_gaps(model, data, dynamic_geoms, moving_geoms)
        diagnostics["evaluated"] += 1
        if gf > .0007 and gm > .0007:
            diagnostics["positive_clearance"] += 1
        else:
            continue
        # Other body collisions are invalid, particularly the arm/palm.
        if any(geometry_gaps(model, data, dynamic_geoms, (gid,)) < .0007
               for gid in other):
            continue
        if gf > .024 or gm > .024:
            continue
        diagnostics["near_both_jaws"] += 1
        score = abs(gf-.003)+abs(gm-.003)+.03*abs(xyz[2]+.080)
        if score < best_score:
            best_score = score
            best = (center.copy(), quat.copy(), gf, gm, tuple(xyz))
    if best is not None:
        diagnostics["closest_fixed_gap_mm"] = best[2]*1000
        diagnostics["closest_moving_gap_mm"] = best[3]*1000
        diagnostics["object_gripper_local_xyz_m"] = list(best[4])
    return best, diagnostics


def jaw_contact_counts(model, data, object_geoms, fixed_geoms, moving_geoms):
    obj=set(object_geoms); fixed=set(fixed_geoms); moving=set(moving_geoms)
    f=m=0
    depths=[]
    for con in data.contact[:data.ncon]:
        ids={int(con.geom1),int(con.geom2)}
        if ids & obj and ids & fixed:
            f+=1; depths.append(float(con.dist))
        if ids & obj and ids & moving:
            m+=1; depths.append(float(con.dist))
    return f,m,min(depths,default=None)


def run(args):
    if args.out.exists():
        raise FileExistsError(f"Out path exists: {args.out}")
    if not (0 < args.fixture_s < args.duration_s <= 10):
        raise ValueError("Invalid fixture/duration seconds")
    if any(not math.isfinite(x) for x in
           (args.fixture_s,args.duration_s,args.kp,args.kd,args.max_torque)):
        raise ValueError("Nonfinite control")
    if args.kp <= 0 or args.kd < 0 or args.max_torque <= 0:
        raise ValueError("Invalid PD controller")
    spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    spec.compiler.discardvisual=False
    physical.install_dynamic_muhi(spec)
    model=spec.compile()
    model.opt.timestep=.002
    data=mujoco.MjData(model)
    mujoco.mj_resetData(model,data)
    # Isolated free-space bench (not saved Real-to-Sim tabletop). Hold arm
    # joints at their default zero positions, inside their URDF limits.
    qids=[];dids=[]
    for name in r2s.JOINTS:
        qa,da,limits=joint_addr(model,name)
        if not limits[0] <= 0 <= limits[1]:
            raise ValueError(f"Cannot hold {name} at zero")
        qids.append(qa);dids.append(da)
        data.qpos[qa]=0.
    jaw_q,jaw_d,jaw_limits=joint_addr(model,"gripper")
    jaw_open=float(jaw_limits[1]);jaw_closed=float(jaw_limits[0])
    if jaw_closed >= jaw_open: raise ValueError("J6 range malformed")
    data.qpos[jaw_q]=jaw_open
    fixed=owned_collision_geoms(model,FINGER_BODY)
    moving=owned_collision_geoms(model,MOVING_BODY)
    dyn=tuple(physical.geom_id(model,x) for x in physical.DYNAMIC_GEOMS)
    if not fixed or not moving:raise ValueError("Jaw collision meshes missing/disabled")
    free_j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    free_q=int(model.jnt_qposadr[free_j]);free_d=int(model.jnt_dofadr[free_j])
    mujoco.mj_forward(model,data)
    pose,search=select_bench_pose(model,data,free_q,jaw_open,jaw_q,
                                  fixed,moving,dyn)
    args.out.mkdir(parents=True,exist_ok=False)
    summary={"status":"SETUP_NOT_FOUND" if pose is None else "CONTACT_TEST_COMPLETED",
             "simulation_only":True,"experiment":"J6 isolated grasp fixture",
             "original_episode4_scene_used":False,
             "robot":str(args.urdf),"physics_object":physical.DYNAMIC_BODY,
             "object_mass_kg_provisional":.050,
             "object_dimensions_m_provisional":[2*physical.MUHI_RADIUS,
                                                  physical.MUHI_LENGTH],
             "j1_j5_mode":"kinematically fixed",
             "gripper_closed_rad":jaw_closed,"gripper_open_rad":jaw_open,
             "gripper_calibration":"URDF limits; physical close direction NOT verified",
             "fixture_hold_until_s":args.fixture_s,
             "fixture_warning":"Until release the object pose is explicitly held; "
                               "only post-release is a contact-and-gravity retention test.",
             "setup_search":search,
             "camera":"MuJoCo free-view closeups, not calibrated Side/Wrist",
             "not_tested":["Pickup from saved tabletop","Lift","Place","SAC","DLS",
                           "Robot arm reaction forces","Real hardware"],
             "video_status":"not_requested" if not args.video else "not_rendered"}
    if pose is None:
        summary["status"]="NO_COLLISION_FREE_TWO_JAW_START"
        (args.out/"summary.json").write_text(
            json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps({"status":summary["status"],"search":search},indent=2))
        return
    center,quat,_,_,_=pose
    data.qpos[free_q:free_q+3]=center
    data.qpos[free_q+3:free_q+7]=quat
    data.qpos[jaw_q]=jaw_open
    data.qvel[:]=0
    mujoco.mj_forward(model,data)
    init_contacts=jaw_contact_counts(model,data,dyn,fixed,moving)
    if init_contacts[0] or init_contacts[1]:
        raise ValueError("Invalid penetrating initial jaw contact")

    locked_q=data.qpos[np.asarray(qids,dtype=int)].copy()
    fixture_q=data.qpos[free_q:free_q+7].copy()
    rows=[];frames=[]
    width,height=480,360
    renderer=None
    if args.video:
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,width)
        model.vis.global_.offheight=max(model.vis.global_.offheight,height)
        renderer=mujoco.Renderer(model,width=width,height=height)
        from PIL import Image,ImageDraw
        cam=mujoco.MjvCamera()
        cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]=np.asarray(data.xpos[
            mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,FINGER_BODY)])
        cam.distance=.28
        cam.elevation=-18
    try:
        n=math.ceil(args.duration_s/model.opt.timestep)
        for step in range(n):
            clock=step*model.opt.timestep
            hold=clock < args.fixture_s
            # The arm is an immobile geometric fixture, only J6 dynamic.
            data.qpos[qids]=locked_q
            data.qvel[dids]=0
            if hold:
                data.qpos[free_q:free_q+7]=fixture_q
                data.qvel[free_d:free_d+6]=0
            mujoco.mj_forward(model,data)
            # Smoothly approach closed limit during the setup fixture phase.
            u=min(1.,clock/max(.1,args.fixture_s*.85))
            target=jaw_open + u*(jaw_closed-jaw_open)
            force=args.kp*(target-data.qpos[jaw_q])-args.kd*data.qvel[jaw_d]
            data.qfrc_applied[:]=0
            data.qfrc_applied[jaw_d]=float(np.clip(force,-args.max_torque,args.max_torque))
            mujoco.mj_step(model,data)
            data.qpos[qids]=locked_q;data.qvel[dids]=0
            if hold:
                data.qpos[free_q:free_q+7]=fixture_q
                data.qvel[free_d:free_d+6]=0
            mujoco.mj_forward(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError(f"Nonfinite MuJoCo state at step {step}")
            c1,c2,min_depth=jaw_contact_counts(model,data,dyn,fixed,moving)
            displacement=np.linalg.norm(data.qpos[free_q:free_q+3]-center)*1000
            rows.append({"step":step,"time_s":float(data.time),
                         "fixture_holding":hold,"jaw_angle_rad":float(data.qpos[jaw_q]),
                         "jaw_target_rad":float(target),
                         "jaw_applied_torque_nm":float(data.qfrc_applied[jaw_d]),
                         "fixed_jaw_contacts":c1,"moving_jaw_contacts":c2,
                         "jaw_bilateral_contact":bool(c1 and c2),
                         "min_jaw_contact_distance_mm":
                              min_depth*1000 if min_depth is not None else None,
                         "object_x_m":float(data.qpos[free_q]),
                         "object_y_m":float(data.qpos[free_q+1]),
                         "object_z_m":float(data.qpos[free_q+2]),
                         "object_displacement_mm":float(displacement),
                         "object_speed_m_s":float(np.linalg.norm(
                              data.qvel[free_d:free_d+3]))})
            if renderer is not None and (
                   step % args.render_every==0 or step==n-1):
                imgs=[]
                for azimuth in (40,145):
                    cam.azimuth=azimuth
                    renderer.update_scene(data,camera=cam)
                    imgs.append(Image.fromarray(renderer.render().copy()))
                panel=Image.new("RGB",(width*2,height+32),(20,25,34))
                panel.paste(imgs[0],(0,32));panel.paste(imgs[1],(width,32))
                dr=ImageDraw.Draw(panel)
                dr.text((8,9),f"J6 grasp | {'FIXTURE' if hold else 'RELEASED'} "
                              f"| {clock:.3f}s",fill="white")
                dr.text((width+8,9),f"Contacts fixed={c1}, moving={c2}",fill="white")
                frames.append(panel)
    finally:
        if renderer is not None:renderer.close()
    with (args.out/"grasp_trace.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    released=[row for row in rows if not row["fixture_holding"]]
    bilateral=sum(row["jaw_bilateral_contact"] for row in released)
    final=rows[-1]
    max_pen=min((row["min_jaw_contact_distance_mm"] for row in rows
                 if row["min_jaw_contact_distance_mm"] is not None),default=None)
    # Deep overlap while the virtual fixture held the object is a setup
    # failure: do not mistake forced interpenetration for gripping.
    deep_overlap=bool(max_pen is not None and max_pen < -2.)
    # A deliberately stringent provisional acceptance check, not calibration.
    success=(not deep_overlap and
             bilateral>=max(5,len(released)//4) and
             final["object_displacement_mm"] < 10. and
             final["jaw_bilateral_contact"])
    summary.update({
        "status":"GRASP_BENCH_PASS" if success else "GRASP_BENCH_NOT_YET_PASS",
        "physics_steps":len(rows),"duration_s":float(data.time),
        "post_release_steps":len(released),
        "post_release_bilateral_steps":bilateral,
        "min_jaw_contact_distance_mm_all_phases":max_pen,
        "deep_fixture_interpenetration_detected":deep_overlap,
        "final_object_displacement_mm":final["object_displacement_mm"],
        "final_bilateral_contact":final["jaw_bilateral_contact"],
        "criterion":"Post-release bilateral contact for >=25% of sampled "
                    "steps (min 5), final bilateral contact, displacement <10mm",
        "initial_jaw_contact_count":init_contacts[0]+init_contacts[1],
        "failed_due_to_forced_overlap":deep_overlap,
        "robot_or_table_contact_grasp":"not tested (no tabletop or arm movement)",
    })
    if frames:
        out_frames=args.out/"frames";out_frames.mkdir()
        for i,f in enumerate(frames):f.save(out_frames/f"frame_{i:04d}.png")
        ffmpeg=shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-y",
                 "-framerate",str(args.fps),
                 "-i",str(out_frames/"frame_%04d.png"),
                 "-c:v","libx264","-pix_fmt","yuv420p",
                 "-movflags","+faststart",
                 str(args.out/"grasp_side_views.mp4")],check=True)
            summary["video_status"]="created"
        else:summary["video_status"]="ffmpeg_missing_pngs_saved"
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,
                       ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in
          ("status","post_release_bilateral_steps","post_release_steps",
           "final_object_displacement_mm","video_status")},indent=2))
    print("Output:",args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--fixture-s",type=float,default=.9)
    p.add_argument("--duration-s",type=float,default=1.6)
    p.add_argument("--kp",type=float,default=2.0)
    p.add_argument("--kd",type=float,default=.08)
    p.add_argument("--max-torque",type=float,default=.4)
    p.add_argument("--video",action="store_true")
    p.add_argument("--fps",type=float,default=12.)
    p.add_argument("--render-every",type=int,default=30)
    args=p.parse_args()
    if args.render_every<1 or args.fps<=0:p.error("Invalid video settings")
    run(args)


if __name__=="__main__":
    main()
