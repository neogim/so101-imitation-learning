#!/usr/bin/env python3
"""Replay actual saved Real-to-Sim layout; DLS reach Muhi then clear cup.

Reads the SAME episode-4 object annotation, saved cup GUI correction,
candidate camera calibration and recorded start joint pose as the established
Real-to-Sim workflow. NO invented/repositioned objects, no grasp, no dynamics.
Uses the existing white-preview styling; source URDF is not modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import real_to_sim as r2s
import compare_real_sim_motion as movie
import align_cup_gui as cup_gui

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/so101/muhi-side-wrist-success-053-20261025-168x224"
OBJECTS = ROOT / "outputs/real2sim/episode4-objects-20261108-004859.json"
CUP_POSE = ROOT / "outputs/real2sim/episode4-cup-gui-20261108-131429.json"
EXTRINSICS = ROOT / "outputs/so101-eye-to-hand-id1-20261107/result.json"
INTRINSICS = ROOT / "assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
JOINTS = r2s.JOINTS


def _read_json(path):
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def load_joint_offset_candidate(args, cup):
    """Reuse saved robot overlay candidate, never invent or clip an offset.

    If not explicitly provided, discover a *unique* compatible candidate
    from the local Real-to-Sim outputs. Reject ambiguous candidates rather
    than silently selecting a different alignment.
    """
    from compare_real_sim_robot_overlay_gui import read_robot_pose, joint_config

    paths = ([args.robot_joint_offsets_json] if args.robot_joint_offsets_json
             else sorted((ROOT / "outputs/real2sim").rglob("*.json")))
    candidates = []
    for path in paths:
        path = Path(path)
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if args.robot_joint_offsets_json:
                raise
            continue
        if obj.get("schema") != "so101_robot_joint_offset_candidate_v1":
            if args.robot_joint_offsets_json:
                raise ValueError(f"Not a robot joint-offset candidate: {path}")
            continue
        pose = read_robot_pose(path)
        if pose.get("episode") != args.episode:
            continue
        if Path(pose.get("object_layout_json", "")).name != args.object_layout_json.name:
            continue
        if Path(pose.get("cup_pose_json", "")).name != args.cup_pose_json.name:
            continue
        table = pose.get("table", {})
        keys = ("table_extra_left_cm", "table_extra_front_cm",
                "table_base_x_cm", "table_base_y_cm", "table_base_z_cm",
                "table_yaw_delta_deg")
        if not all(key in table and np.isclose(
                float(table[key]), float(cup[key]), atol=1e-6)
                for key in keys):
            continue
        candidates.append((path, pose))
    if len(candidates) > 1:
        raise ValueError(
            "Multiple robot joint-offset candidates match this Real-to-Sim "
            "scene; pass --robot-joint-offsets-json explicitly: " +
            ", ".join(str(p) for p, _ in candidates))
    if not candidates:
        if args.robot_joint_offsets_json:
            raise ValueError(
                "The specified robot joint-offset JSON does not match the "
                "saved Real-to-Sim episode / table / object layout.")
        return None, {"joints": {"sign": {n: 1 for n in JOINTS},
                                  "offset_deg": {n: 0. for n in JOINTS}}}
    path, pose = candidates[0]
    config = joint_config([pose["joint_offset_deg"][n] for n in JOINTS])
    print(f"Using saved Real-to-Sim robot joint offsets: {path}")
    return path, config


def create_scene(args):
    """Rebuild saved Real-to-Sim without changing the object positions."""
    extras = _read_json(args.extrinsics)
    intr = _read_json(args.intrinsics)
    cup = cup_gui.read_cup_payload(args.cup_pose_json.resolve(strict=True))
    layout_path = args.object_layout_json.resolve(strict=True)
    if cup["episode"] != args.episode:
        raise ValueError("Cup JSON episode and requested source episode differ")
    if Path(cup["object_layout_json"]).name != layout_path.name:
        raise ValueError("Cup JSON refers to a different object-layout JSON")
    if (intr.get("width"), intr.get("height")) != (640, 480):
        raise ValueError("Expected Side-camera intrinsics at 640x480")
    muhi_px, cup_px, yaw = movie.read_object_targets(layout_path)
    layout = movie.video_object_layout(
        extras["T_base_camera"], intr, muhi_px, cup_px, yaw)

    model, side_camera = movie.make_calibrated_model(
        extras, intr, args.urdf, layout,
        table_extra_left_m=float(cup["table_extra_left_cm"])/100,
        table_extra_front_m=float(cup["table_extra_front_cm"])/100,
        physical_objects=getattr(args, "physical_objects", False),
        wrist_camera=getattr(args, "wrist_camera", False),
        wrist_mount_pos=getattr(args, "wrist_mount_pos", (0., .024, -.045)),
        wrist_mount_rpy_deg=getattr(args, "wrist_mount_rpy_deg", (0., 0., 0.)),
        wrist_fovy=getattr(args, "wrist_fovy", 70.),
        wrist_look_at=getattr(args, "wrist_look_at", (0.,-.024,-.112)))
    # This includes apply_white_follower_preview() inside the reused builder.
    # Reproduce the *saved* table translation/rotation and GUI cup correction.
    table_ids = movie.table_geom_ids(model)
    movie.set_table_pose_delta(
        model, model.geom_pos[table_ids].copy(),
        model.geom_quat[table_ids].copy(),
        [float(cup[k])/100 for k in
         ("table_base_x_cm","table_base_y_cm","table_base_z_cm")],
        float(cup["table_yaw_delta_deg"]))
    cup_ids = cup_gui.cup_geom_ids(model)
    cup_gui.set_cup_dimensions(
        model, cup_ids, model.geom_pos[cup_ids].copy(),
        model.geom_size[cup_ids].copy(),
        diameter_m=float(cup.get("cup_diameter_m", cup_gui.CUP_DIAMETER_M)),
        height_m=float(cup.get("cup_height_m", cup_gui.CUP_HEIGHT_M)),
        xyz_m=cup["cup_delta_base_m"])

    data = mujoco.MjData(model)
    episode = r2s.read_episode(args.dataset_root.resolve(strict=True),
                               args.episode, stride=1, max_frames=1)
    state = episode["rows"][0]["state"]
    robot_offset_path, robot_config = load_joint_offset_candidate(args, cup)
    joint_rad = r2s.joint_radians(state, robot_config)
    qids, dids, low, high = [], [], [], []
    for name in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0: raise ValueError("Missing URDF joint " + name)
        qids.append(int(model.jnt_qposadr[jid]))
        dids.append(int(model.jnt_dofadr[jid]))
        if model.jnt_limited[jid]:
            low.append(float(model.jnt_range[jid, 0]))
            high.append(float(model.jnt_range[jid, 1]))
        else:
            low.append(-math.pi/2); high.append(math.pi/2)
    qids = np.array(qids); dids = np.array(dids)
    q0 = np.array([joint_rad[name] for name in JOINTS])
    violations = [
        f"{name}: recorded+offset={np.degrees(q):.2f}deg, "
        f"URDF=[{np.degrees(lo):.2f}, {np.degrees(hi):.2f}]deg"
        for name, q, lo, hi in zip(JOINTS, q0, low, high)
        if q < lo - 1e-6 or q > hi + 1e-6
    ]
    if violations:
        print("WARNING: saved visual alignment starts outside URDF joint limits:")
        for message in violations:
            print("  " + message)
        print("The exact recorded initial pose is preserved for visualization. "
              "During the DLS demo, joints outside the URDF limits recover "
              "toward their bounds in steps of at most 0.04 rad. "
              "This is NOT a validated or hardware-safe movement.")

    data.qpos[:] = model.qpos0
    data.qpos[qids] = q0
    if getattr(args, "physical_objects", False):
        import physical_workcell_sim
        physical_workcell_sim.configure_dynamic_muhi(model, data)
    mujoco.mj_forward(model, data)
    # Gripper angle is only a display approximation, not a trained controller.
    movie.set_visual_gripper(model, data, state[5])
    gripper_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
    if gripper_id < 0: raise ValueError("Missing gripper_link")

    muhi_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("r2s_muhi_body", "r2s_muhi_cap")]
    if min(muhi_ids) < 0: raise ValueError("Real-to-Sim Muhi objects not found")
    # Weighted center along the actual saved capsule/cylinder orientation.
    muhi_center = .65*data.geom_xpos[muhi_ids[0]] + .35*data.geom_xpos[muhi_ids[1]]
    cup_base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_cup_base")
    if cup_base_id < 0: raise ValueError("Real-to-Sim cup not found")
    cup_bottom = np.asarray(data.geom_xpos[cup_base_id]).copy()
    goals = {
        "muhi": muhi_center + np.array([0, 0, args.muhi_clearance_mm/1000]),
        "cup": cup_bottom + np.array(
            [0, 0, float(cup.get("cup_height_m", cup_gui.CUP_HEIGHT_M))+args.cup_clearance_mm/1000])
    }
    return (model,data,qids,dids,np.asarray(low),np.asarray(high),gripper_id,
            side_camera,goals,cup,layout,robot_offset_path,robot_config)


def dls_two_stages(args, model, data, qids, dids, low, high, tcp_id, goals):
    frames=[]; stages=[]
    for stage in ("muhi","cup"):
        goal=goals[stage]
        initial = float(np.linalg.norm(goal-data.xpos[tcp_id])*1000)
        success = False  # require position AND joint-limit conditions
        for step in range(args.max_steps+1):
            tcp=np.asarray(data.xpos[tcp_id]).copy()
            distance=float(np.linalg.norm(goal-tcp)*1000)
            q_now=np.asarray(data.qpos[qids], dtype=float).copy()
            excess=np.maximum(low-q_now,0)+np.maximum(q_now-high,0)
            limits_ok=bool(np.max(excess)<=1e-7)
            row={"frame":len(frames),"stage":stage,"stage_step":step,
                 "tcp_x":float(tcp[0]),"tcp_y":float(tcp[1]),"tcp_z":float(tcp[2]),
                 "target_x":float(goal[0]),"target_y":float(goal[1]),
                 "target_z":float(goal[2]),"tcp_error_mm":distance,
                 "joint_limits_ok":limits_ok,
                 "max_joint_limit_excess_deg":float(np.degrees(np.max(excess)))}
            row.update({f"joint_{i+1}":float(data.qpos[qids[i]]) for i in range(5)})
            frames.append(row)
            if distance <= args.success_mm and limits_ok:
                success=True
                break
            if step==args.max_steps:break
            jac=np.zeros((3,model.nv));jac_r=np.zeros((3,model.nv))
            mujoco.mj_jacBody(model,data,jac,jac_r,tcp_id)
            J=jac[:,dids]
            delta=J.T@np.linalg.solve(J@J.T+.02**2*np.eye(3),
                                      .75*(goal-tcp))
            delta=np.clip(delta,-.04,.04)
            # Keep the recorded visual initial pose unchanged. Outside-limit
            # joints return toward the closest URDF bound gradually: no
            # instant clipping / teleport in the rendered trajectory.
            next_q=np.clip(q_now+delta,low,high)
            below=q_now<low
            above=q_now>high
            next_q[below]=q_now[below]+np.minimum(
                .04,low[below]-q_now[below])
            next_q[above]=q_now[above]-np.minimum(
                .04,q_now[above]-high[above])
            if np.any(np.abs(next_q-q_now)>.04000001):
                raise AssertionError("Joint step exceeded 0.04 rad")
            data.qpos[qids]=next_q
            mujoco.mj_forward(model,data)
        stages.append({"stage":stage,"initial_error_mm":initial,
                       "final_error_mm":frames[-1]["tcp_error_mm"],
                       "success":success,"steps":step,
                       "initial_joint_limit_excess_deg":
                           frames[-(step+1)]["max_joint_limit_excess_deg"],
                       "final_joint_limits_ok":frames[-1]["joint_limits_ok"]})
        # Stage 2 begins from the current terminal joints, even if stage 1 fails.
    return frames, stages

def return_home(args, model, data, qids, tcp_id, original_q, original_tcp,
                frames, stages):
    """Return to recorded visual pose by bounded quintic interpolation.

    IMPORTANT: the recorded pose is outside the imported URDF joint limits.
    Therefore this is a VISUAL-ONLY home, NOT a hardware-safe home trajectory.
    """
    q_start=np.asarray(data.qpos[qids],float).copy()
    q_goal=np.asarray(original_q,float).copy()
    delta=q_goal-q_start
    steps=max(args.home_steps,math.ceil(1.875*float(np.max(np.abs(delta)))/.04))
    if steps<1:raise ValueError("home_steps must be positive")
    bounds=[]
    for joint in JOINTS:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,joint)
        bounds.append(model.jnt_range[jid].copy())
    limits=np.asarray(bounds)
    start_err=float(np.linalg.norm(data.xpos[tcp_id]-original_tcp)*1000)
    last=q_start
    for step in range(1,steps+1):
        t=step/steps
        weight=10*t**3-15*t**4+6*t**5
        q=q_start+weight*delta
        if float(np.max(np.abs(q-last)))>.040001:
            raise AssertionError("Home joint step exceeded 0.04rad")
        data.qpos[qids]=q
        mujoco.mj_forward(model,data)
        xyz=np.asarray(data.xpos[tcp_id]).copy()
        violation=np.maximum(limits[:,0]-q,0)+np.maximum(q-limits[:,1],0)
        row={"frame":len(frames),"stage":"home","stage_step":step,
             "tcp_x":float(xyz[0]),"tcp_y":float(xyz[1]),"tcp_z":float(xyz[2]),
             "target_x":float(original_tcp[0]),
             "target_y":float(original_tcp[1]),
             "target_z":float(original_tcp[2]),
             "tcp_error_mm":float(np.linalg.norm(xyz-original_tcp)*1000),
             "joint_limits_ok":bool(np.max(violation)<=1e-7),
             "max_joint_limit_excess_deg":float(np.degrees(np.max(violation)))}
        row.update({f"joint_{i+1}":float(q[i]) for i in range(5)})
        frames.append(row)
        last=q
    final_q_err=float(np.max(np.abs(data.qpos[qids]-q_goal)))
    stages.append({"stage":"home","method":"quintic_joint_interpolation",
                   "initial_error_mm":start_err,
                   "final_error_mm":frames[-1]["tcp_error_mm"],
                   "steps":steps,"success":final_q_err<1e-8,
                   "max_joint_home_error_rad":final_q_err,
                   "final_joint_limits_ok":frames[-1]["joint_limits_ok"],
                   "warning":"Exact recorded home violates URDF limits: visual-only."})
    return frames,stages


def save_results(out, args, frames, stages, goals, cup, layout,
                 robot_offset_path, robot_config):
    if out.exists():raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    with (out/"rollout.csv").open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(frames)
    summary={
        "status":"REAL2SIM_SAVED_LAYOUT_KINEMATIC_REACH_ONLY",
        "calibration_status":"PREVIOUS_VISUAL_ALIGNMENT_CANDIDATE_NOT_METRICALLY_VERIFIED",
        "sources":{
            "episode":args.episode,"dataset":str(args.dataset_root),
            "object_layout_json":str(args.object_layout_json),
            "cup_pose_json":str(args.cup_pose_json),
            "extrinsics":str(args.extrinsics),"intrinsics":str(args.intrinsics),
            "urdf":str(args.urdf),
            "robot_joint_offsets_json":(
                str(robot_offset_path) if robot_offset_path else None)},
        "joint_offset_deg": robot_config["joints"]["offset_deg"],
        "initial_joint_limit_discrepancy": {
            "status": "OUTSIDE_LIMIT_VISUAL_POSE_NOT_PHYSICAL_TRAJECTORY"
                if frames[0]["max_joint_limit_excess_deg"]>1e-6
                else "WITHIN_URDF_LIMITS",
            "max_excess_deg":frames[0]["max_joint_limit_excess_deg"],
            "recovery_method": "stepwise_at_most_0.04rad_not_instant_clipping",
        },
        "joint_offset_status": ("VISUAL_CANDIDATE_NOT_CALIBRATED" if robot_offset_path
                                else "UNVERIFIED_ZERO_DEFAULT"),
        "saved_real2sim_layout":layout,
        "cup_delta_base_m":cup["cup_delta_base_m"],
        "goal_xyz_m":{k:v.tolist() for k,v in goals.items()},
        "muhi_clearance_mm":args.muhi_clearance_mm,
        "cup_clearance_mm":args.cup_clearance_mm,
        "success_radius_mm":args.success_mm,
        "home_method":"visual quintic joint-space interpolation to recorded pose",
        "home_warning":"Recorded pose violates URDF joint limits; not hardware-safe",
        "stages":stages,
        "limitations":"No grasp, no carrying Muhi, no contact/collision physics. "
                      "Targets are points above saved objects, not measured grasp poses."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    report="# SO-101 saved Real-to-Sim: Muhi → cup Reach (DLS)\n\n"
    report+="Sources: saved episode-4 Muhi annotation, cup GUI JSON, camera extrinsics/intrinsics.\n\n"
    report+="| Stage | Start error (mm) | End error (mm) | Steps | Success <= radius |\n|---|---:|---:|---:|---|\n"
    for s in stages:
        report+=f"| {s['stage']} | {s['initial_error_mm']:.2f} | {s['final_error_mm']:.2f} | {s['steps']} | {s['success']} |\n"
    report+="\nNo grasp/contact/transport; positions are saved visual-alignment estimates.\n"
    report+=("\nVisual calibration joint-limit excess at initial pose: "
             f"{frames[0]['max_joint_limit_excess_deg']:.3f} degrees. "
             "Step-limited recovery in kinematic demo only.\n")
    (out/"report.md").write_text(report,encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,4.8))
    colors={"muhi":"tab:red","cup":"tab:blue","home":"tab:green"}
    for stage in ("muhi","cup","home"):
        sub=[r for r in frames if r["stage"]==stage]
        ax.plot([r["frame"] for r in sub],[r["tcp_error_mm"] for r in sub],
                label=stage,color=colors[stage],lw=2)
    ax.axhline(args.success_mm,ls="--",color="gray",label="10 mm target")
    ax.set(xlabel="Frame",ylabel="TCP position error (mm)",
           title="Real-to-Sim: Muhi -> cup -> recorded home")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/"tcp_error_vs_step.png",dpi=170);plt.close(fig)


def camera_for(model,goals,use_side,side_name):
    if use_side:return side_name
    cam=mujoco.MjvCamera()
    cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=(goals["muhi"]+goals["cup"])/2
    cam.distance=0.90
    cam.azimuth=155;cam.elevation=-35
    return cam


def extras(scene,frames,index,goals):
    colors={"muhi":[.95,.25,.20,1],"cup":[.15,.75,.40,1],"home":[.16,.67,.34,1]}
    for k,v in goals.items():
        if scene.ngeom>=scene.maxgeom:raise RuntimeError("Too many scene geoms")
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([.009,.009,.009]),v,np.eye(3).ravel(),
                            np.array(colors[k],dtype=np.float32))
        scene.ngeom+=1
    for a,b in zip(frames[:index],frames[1:index+1]):
        if scene.ngeom>=scene.maxgeom:raise RuntimeError("Too many trajectory segments")
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3),np.zeros(3),np.eye(3).ravel(),
                            np.array(colors[b["stage"]],dtype=np.float32))
        p=np.array([a["tcp_"+x] for x in "xyz"]);q=np.array([b["tcp_"+x] for x in "xyz"])
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,2,p,q)
        scene.ngeom+=1


def save_video(args,model,data,qids,frames,goals,camera):
    try:from PIL import Image,ImageDraw,ImageFont
    except ImportError as e:raise RuntimeError("Install Pillow in conda environment") from e
    ffmpeg=shutil.which("ffmpeg")
    if ffmpeg is None:raise RuntimeError("ffmpeg is required: brew install ffmpeg")
    dest=args.save_video
    if dest.exists():raise FileExistsError(f"Video exists: {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True)
    width,height=1280,720
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,width)
    model.vis.global_.offheight=max(model.vis.global_.offheight,height)
    cmd=[ffmpeg,"-hide_banner","-loglevel","error","-f","rawvideo",
         "-pix_fmt","rgb24","-s:v",f"{width}x{height}","-r",str(args.video_fps),
         "-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p",
         "-movflags","+faststart",str(dest)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=height,width=width) as renderer:
            try:font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf",23)
            except OSError:font=ImageFont.load_default()
            n=math.ceil((len(frames)-1)*args.video_fps/args.fps)+1
            for i in range(n+int(round(args.hold_seconds*args.video_fps))):
                idx=min(len(frames)-1,int(i*args.fps/args.video_fps))
                row=frames[idx]
                data.qpos[qids]=[row[f"joint_{k+1}"] for k in range(5)]
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera)
                extras(renderer.scene,frames,idx,goals)
                im=Image.fromarray(renderer.render())
                dr=ImageDraw.Draw(im)
                dr.rectangle((0,0,width,72),fill=(19,29,40))
                dr.text((18,10),"SO-101 Real-to-Sim | DLS: Muhi -> cup -> home",
                        font=font,fill="white")
                dr.text((18,41),f"{row['stage']}  step {idx}  error {row['tcp_error_mm']:.1f} mm",
                        font=font,fill="white")
                dr.rectangle((0,684,width,720),fill=(19,29,40))
                dr.text((18,687),"Visual alignment candidate | no grasp or object movement",
                        font=font,fill="white")
                proc.stdin.write(np.asarray(im,np.uint8).tobytes())
        proc.stdin.close();proc.stdin=None
        stderr=proc.communicate()[1]
        if proc.returncode:raise RuntimeError(stderr.decode(errors="replace"))
    except Exception:
        if proc.poll() is None:proc.kill();proc.communicate()
        if dest.exists():dest.unlink()
        raise
    print("Saved",dest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,default=DATASET)
    p.add_argument("--object-layout-json",type=Path,default=OBJECTS)
    p.add_argument("--cup-pose-json",type=Path,default=CUP_POSE)
    p.add_argument("--extrinsics",type=Path,default=EXTRINSICS)
    p.add_argument("--intrinsics",type=Path,default=INTRINSICS)
    p.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument("--episode",type=int,default=4)
    p.add_argument("--robot-joint-offsets-json",type=Path,default=None,
                   help="saved robot overlay offsets; if omitted, auto-detect "
                        "a unique matching candidate under outputs/real2sim")
    p.add_argument("--muhi-clearance-mm",type=float,default=75)
    p.add_argument("--cup-clearance-mm",type=float,default=45)
    p.add_argument("--success-mm",type=float,default=10)
    p.add_argument("--max-steps",type=int,default=120)
    p.add_argument("--home-steps",type=int,default=80,
                   help="minimum frames for smooth visual return to recorded pose")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--save-video",type=Path)
    p.add_argument("--video-fps",type=int,default=30)
    p.add_argument("--fps",type=float,default=8)
    p.add_argument("--hold-seconds",type=float,default=2)
    p.add_argument("--side-camera",action="store_true",
                   help="render from saved calibrated candidate Side camera")
    args=p.parse_args()
    if (args.muhi_clearance_mm<=0 or args.cup_clearance_mm<=0
        or args.success_mm<=0 or args.max_steps<=0 or args.home_steps<=0
        or args.fps<=0 or args.video_fps<=0 or args.hold_seconds<0):
        p.error("All dimensions, speeds and step limits must be valid")
    for f in (args.dataset_root,args.object_layout_json,args.cup_pose_json,
              args.extrinsics,args.intrinsics,args.urdf):
        if not f.exists():p.error(f"Missing Real-to-Sim input: {f}")
    (model,data,qids,dids,low,high,tcp,cam_name,goals,cup,layout,
     offset_path,offset_config)=create_scene(args)
    recorded_q=np.asarray(data.qpos[qids],float).copy()
    recorded_tcp=np.asarray(data.xpos[tcp],float).copy()
    frames,stages=dls_two_stages(args,model,data,qids,dids,low,high,tcp,goals)
    frames,stages=return_home(args,model,data,qids,tcp,recorded_q,recorded_tcp,frames,stages)
    save_results(args.out,args,frames,stages,goals,cup,layout,
                 offset_path,offset_config)
    for s in stages:print(f"{s['stage']} success={s['success']} steps={s['steps']} "
                           f"initial_mm={s['initial_error_mm']:.2f} "
                           f"final_mm={s['final_error_mm']:.2f}")
    print("White SO-101 via existing preview styling; original URDF unchanged.")
    print("Report:",args.out/"summary.json")
    if args.save_video:
        camera=camera_for(model,goals,args.side_camera,cam_name)
        save_video(args,model,data,qids,frames,goals,camera)


if __name__=="__main__":
    main()
