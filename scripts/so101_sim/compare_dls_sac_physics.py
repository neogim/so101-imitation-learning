#!/usr/bin/env python3
"""Physical PD tracking of the SAVED Episode-4 DLS vs SAC kinematic trajectories.

This is NOT closed-loop SAC or a new DLS solve. It subjects both previously
generated joint target sequences to the SAME MuJoCo physics and capped PD,
including a dynamic Muhi, collidable cup/table, and Wrist camera attachment.
Neither source rollout nor checkpoints nor real robot are modified.
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

import demo_real2sim_muhi_cup_reach as scene
import physical_workcell_sim as phys
import wrist_camera_sim as wrist

POLICIES = ("dls", "sac")
JOINT_COLUMNS = tuple(f"joint_{i}" for i in range(1, 6))


def load_comparison(path):
    """Require both labeled policies and finite five-joint targets."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"policy", "stage", *JOINT_COLUMNS} <= set(rows[0]):
        raise ValueError("Expected DLS/SAC comparison_rollout.csv with stage and five joint columns")
    result = {}
    for policy in POLICIES:
        selected = [r for r in rows if r["policy"] == policy]
        if not selected:
            raise ValueError(f"Missing policy {policy}")
        q = np.array([[float(row[k]) for k in JOINT_COLUMNS] for row in selected], dtype=float)
        if not np.isfinite(q).all():
            raise ValueError(f"Nonfinite joint targets in {policy}")
        result[policy] = (selected, q)
    for policy in POLICIES:
        stages = {r["stage"] for r in result[policy][0]}
        if not {"muhi", "cup"} <= stages:
            raise ValueError(f"{policy} missing Muhi or cup stage: {stages}")
    np.testing.assert_allclose(result["dls"][1][0], result["sac"][1][0],
                               atol=1e-7, rtol=0, err_msg="Original DLS and SAC did not start identically")
    return result


def initial_pose_audit(source_q, low, high, *, tolerance_deg=0.1):
    """Report exact joint mismatch rather than quietly altering the baseline."""
    src=np.asarray(source_q,float)
    clipped=np.clip(src,low,high)
    delta=np.degrees(clipped-src)
    return {
        "source_joint_deg":np.degrees(src).round(4).tolist(),
        "physics_initial_joint_deg":np.degrees(clipped).round(4).tolist(),
        "clip_delta_deg":np.round(delta,4).tolist(),
        "max_clip_delta_deg":float(np.max(np.abs(delta))),
        "matches_visual_initial_pose":bool(np.max(np.abs(delta))<=tolerance_deg),
    }


def assert_initial_pose_fidelity(audit, *, allow_clipped_diagnostic=False):
    """Do not compare unlike initial states under a 'same scene' label."""
    if not audit["matches_visual_initial_pose"] and not allow_clipped_diagnostic:
        raise ValueError(
            "PHYSICS BASELINE REJECTED: the saved visual initial pose lies "
            f"outside the URDF joint limits (max clip {audit['max_clip_delta_deg']:.2f} deg). "
            "Instant clipping changes the robot's appearance and invalidates a direct "
            "DLS vs SAC physics extension. Check the real-to-sim joint zero/sign/limits "
            "and derive a physically valid shared initial pose. For exploratory "
            "NON-COMPARABLE debugging only, use --allow-clipped-diagnostic; this "
            "will visibly watermark the video and preserve the audit.")
    return audit


def target_sequence(rollout, index, low, high):
    rows, joint_targets = rollout
    source_index = min(index, len(rows)-1)
    return rows[source_index], np.clip(joint_targets[source_index], low, high), source_index


def physics_substeps_for_frame(frame, timestep, *, reference_fps=8., fixed_substeps=0):
    """Match original nominal kinematic frame period, never invent 20ms timing."""
    if fixed_substeps:
        return fixed_substeps
    previous=round(frame/(reference_fps*timestep))
    current=round((frame+1)/(reference_fps*timestep))
    return max(1,current-previous)


def pd_step(model, data, qids, dids, target, *, kp, kd, max_torque, gripper_qid, gripper_did, gripper_goal):
    """Apply joint torques (not direct five-joint qpos writes)."""
    data.qfrc_applied[:] = 0
    demand = kp*(target-data.qpos[qids]) - kd*data.qvel[dids] + data.qfrc_bias[dids]
    data.qfrc_applied[dids] = np.clip(demand, -max_torque, max_torque)
    if gripper_qid is not None:
        demand_j6 = kp*(gripper_goal-data.qpos[gripper_qid]) - kd*data.qvel[gripper_did] + data.qfrc_bias[gripper_did]
        data.qfrc_applied[gripper_did] = float(np.clip(demand_j6, -max_torque, max_torque))
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError("Nonfinite physics state (check initial self-collisions and PD gains)")


def add_reference_markers(render_scene, goals, tcp_trail=None):
    """Reuse earlier visual comparison goal cues; geoms are render-only."""
    palette={"muhi":[.95,.25,.20,1.],"cup":[.15,.75,.40,1.]}
    for name,xyz in goals.items():
        if name not in palette or render_scene.ngeom>=render_scene.maxgeom:
            continue
        g=render_scene.geoms[render_scene.ngeom]
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([.010,.010,.010]),
                           np.asarray(xyz,float),np.eye(3).ravel(),
                           np.asarray(palette[name],np.float32))
        render_scene.ngeom+=1
    if tcp_trail:
        for a,b in zip(tcp_trail[:-1],tcp_trail[1:]):
            if render_scene.ngeom>=render_scene.maxgeom:break
            g=render_scene.geoms[render_scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_LINE,
                               np.zeros(3),np.zeros(3),np.eye(3).ravel(),
                               np.asarray([.99,.8,.25,1],np.float32))
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,2,
                                np.asarray(a,float),np.asarray(b,float))
            render_scene.ngeom+=1


def frame_image(renderer, data, camera, option, label, *, goals=None, tcp_trail=None, extra_label=None):
    from PIL import Image, ImageDraw
    renderer.update_scene(data, camera=camera, scene_option=option)
    if goals is not None:
        add_reference_markers(renderer.scene,goals,tcp_trail=tcp_trail)
    raw = Image.fromarray(renderer.render().copy())
    out = Image.new("RGB", (raw.width, raw.height+56), (20,28,40))
    out.paste(raw, (0,56))
    d=ImageDraw.Draw(out)
    d.text((8,6), label, fill="white")
    if extra_label:
        d.text((8,30), extra_label, fill="#ffd080")
    return out


def write_video(images, out, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return "ffmpeg_missing"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", str(fps), "-i", str(images/"comparison_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out)],
                   check=True)
    return "created"


def build_start(args):
    args.physical_objects = True
    args.wrist_camera = True
    (model, init, qids, dids, low, high, tcp, side, goals,
     *_remainder) = scene.create_scene(args)
    phys.prepare_environment(model)
    original_q = init.qpos[qids].copy()
    pose_audit = initial_pose_audit(original_q, low, high)
    assert_initial_pose_fidelity(
        pose_audit, allow_clipped_diagnostic=args.allow_clipped_diagnostic)
    init.qpos[qids] = np.clip(init.qpos[qids], low, high)
    init.qvel[:] = 0
    mujoco.mj_forward(model, init)
    return model, init, qids, dids, low, high, tcp, side, goals, original_q, pose_audit


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    comparison = load_comparison(args.rollout)
    model, init, qids, dids, low, high, tcp, side, goals, original_q, pose_audit = build_start(args)
    reference_camera = scene.camera_for(model, goals, False, side)
    overview_camera = reference_camera if args.overview_camera == "reference" else side
    source_initial = comparison["dls"][1][0]
    if np.max(np.abs(source_initial-original_q)) > args.max_initial_delta_rad:
        raise ValueError("Input trajectory initial joint pose mismatches saved scene: "
                         "verify this is Episode-4 DLS/SAC comparison_rollout.csv")
    states = {}
    for policy in POLICIES:
        data = mujoco.MjData(model)
        data.qpos[:] = init.qpos
        data.qvel[:] = init.qvel
        mujoco.mj_forward(model, data)
        states[policy] = data
    gripper_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    gripper_qid = int(model.jnt_qposadr[gripper_joint]) if gripper_joint >= 0 else None
    gripper_did = int(model.jnt_dofadr[gripper_joint]) if gripper_joint >= 0 else None
    gripper_goal = float(init.qpos[gripper_qid]) if gripper_qid is not None else None
    muhi_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, phys.DYNAMIC_BODY)
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, phys.DYNAMIC_JOINT)
    object_qid = int(model.jnt_qposadr[object_joint_id])
    total_frames = max(len(v[0]) for v in comparison.values())
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)
    args.out.mkdir(parents=True, exist_ok=False)
    image_dir = args.out / "comparison_frames"
    image_dir.mkdir()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    render_option = mujoco.MjvOption()
    render_option.geomgroup[3] = 0
    results = []
    trail = {p:[] for p in POLICIES}
    render_count = 0
    from PIL import Image
    with mujoco.Renderer(model, width=args.width, height=args.height) as renderer:
        for frame in range(total_frames):
            frame_record = {}
            steps_this_frame = physics_substeps_for_frame(
                frame, model.opt.timestep, reference_fps=args.reference_fps,
                fixed_substeps=args.substeps)
            panels = {}
            for policy in POLICIES:
                data = states[policy]
                row, target, source_index = target_sequence(comparison[policy], frame, low, high)
                for _ in range(steps_this_frame):
                    pd_step(model, data, qids, dids, target,
                            kp=args.kp, kd=args.kd, max_torque=args.max_torque,
                            gripper_qid=gripper_qid, gripper_did=gripper_did, gripper_goal=gripper_goal)
                mujoco.mj_forward(model, data)
                target_xyz = np.array(goals[row["stage"]],float) if row["stage"] in goals else None
                tcp_xyz = data.xpos[tcp].copy()
                trail[policy].append(tcp_xyz.copy())
                distance_mm = float(np.linalg.norm(tcp_xyz-target_xyz)*1000) if target_xyz is not None else None
                contact_groups, min_dist, contacts = phys.summarize_contacts(model,data,muhi_body_id)
                measured = {
                    "policy": policy,
                    "frame": frame, "source_frame": source_index,
                    "stage": row["stage"], "time_s": float(data.time),
                    "tcp_error_mm": distance_mm,
                    "joint_tracking_error_deg": float(np.degrees(np.linalg.norm(data.qpos[qids]-target))),
                    "target_clipped": bool(np.max(np.abs(comparison[policy][1][source_index]-target))>1e-8),
                    "muhi_z_m": float(data.qpos[object_qid+2]),
                    "contact_total": int(data.ncon),
                    "min_contact_dist_mm": min_dist,
                    **contact_groups
                }
                results.append(measured)
                frame_record[policy] = measured
                if frame % args.render_every == 0 or frame == total_frames-1:
                    for cam, cam_name in ((overview_camera,"SIDE"),(wrist.CAMERA,"WRIST")):
                        label = f"{policy.upper()} {cam_name} | {row['stage']} | {data.time:.2f}s"
                        warning = ("CLIPPED START: NOT SAME AS ORIGINAL" if not pose_audit["matches_visual_initial_pose"]
                                   else "PHYSICS: PD + CONTACT (NO GRASP)")
                        err = "HOME (visual reference)" if distance_mm is None else f"TCP error {distance_mm:.1f} mm"
                        panels[policy,cam_name] = frame_image(
                            renderer,data,cam,render_option,label,
                            goals=goals if cam_name=="SIDE" else None,
                            tcp_trail=trail[policy] if cam_name=="SIDE" else None,
                            extra_label=f"{err} | {warning}")
            if panels:
                canvas = Image.new("RGB",(2*args.width,2*(args.height+56)),(28,36,45))
                for x,policy in enumerate(POLICIES):
                    for y,cam_name in enumerate(("SIDE","WRIST")):
                        canvas.paste(panels[policy,cam_name],
                                     (x*args.width,y*(args.height+56)))
                canvas.save(image_dir/f"comparison_{render_count:05d}.png")
                render_count += 1
    with (args.out/"physics_comparison.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    for policy in POLICIES:
        rows=[r for r in results if r["policy"]==policy]
        ax.plot([r["time_s"] for r in rows],
                [float("nan") if r["tcp_error_mm"] is None else r["tcp_error_mm"] for r in rows],
                label=policy.upper())
    ax.axhline(10,ls="--",color="gray",label="10 mm position-only threshold")
    ax.set(xlabel="Sim time (s)",ylabel="TCP error to current stage target (mm)")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(args.out/"physical_tcp_error.png",dpi=160);plt.close(fig)
    video_status = write_video(image_dir,args.out/"dls_vs_sac_physics_side_wrist.mp4",args.video_fps) if args.save_video else "disabled"
    summary={
        "status":"DLS_VS_SAC_SAVED_JOINT_TRAJECTORY_PD_PHYSICS_NOT_CLOSED_LOOP",
        "source_rollout":args.rollout.name,
        "policies":list(POLICIES),
        "same_start_between_policies":True,
        "same_start_as_original_visual":pose_audit["matches_visual_initial_pose"],
        "initial_pose_audit":pose_audit,
        "comparison_validity":("PRELIMINARY_PHYSICAL_TRACKING" if pose_audit["matches_visual_initial_pose"] else "NON_COMPARABLE_CLIPPED_DIAGNOSTIC_ONLY"),
        "mode":"capped_torque_PD_mj_step",
        "overview_camera_mode":args.overview_camera,
        "kp":args.kp,"kd":args.kd,"max_torque_nm":args.max_torque,
        "substeps_per_source_frame":(args.substeps if args.substeps else "derived_from_reference_fps"),
        "reference_fps":args.reference_fps,
        "reference_period_s":1/args.reference_fps,
        "sim_timestep_s":float(model.opt.timestep),
        "source_frames":total_frames,
        "rendered_frames":render_count,"video_status":video_status,
        "original_initial_pose_clipped_for_urdf":bool(np.max(np.abs(source_initial-init.qpos[qids]))>1e-8),
        "by_policy":{},
        "caveats":[
            "Both controllers' saved kinematic joint targets are tracked open-loop with equal PD gains; SAC and DLS are NOT re-evaluated as closed-loop controllers.",
            "Both start from the same URDF-clipped initial pose. Original visual home can exceed the joint limits.",
            "Muhi moves under contact but stable grasp/lift/release is not implemented; cup is a static collision proxy.",
            "Wrist camera and bracket proxies, object inertial/friction parameters and base-shoulder collision geometry are provisional.",
            "All numerical outcomes are simulation-only; no actual motor commands."
        ]
    }
    for policy in POLICIES:
        items=[r for r in results if r["policy"]==policy]
        stage_last={}
        for stage in ("muhi","cup"):
            hits=[r for r in items if r["stage"]==stage]
            if hits: stage_last[stage]=hits[-1]["tcp_error_mm"]
        summary["by_policy"][policy]={
            "final_tcp_error_by_stage_mm":stage_last,
            "max_joint_tracking_error_deg":max(x["joint_tracking_error_deg"] for x in items),
            "steps_with_muhi_robot_contact":sum(x["muhi_robot"]>0 for x in items),
            "steps_with_wrist_external_contact":sum(
                x["wrist_table"]+x["wrist_cup"]+x["wrist_muhi"]>0 for x in items),
            "total_contact_events":sum(x["contact_total"] for x in items)
        }
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Saved",args.out)
    print(json.dumps(summary["by_policy"],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout",type=Path,required=True,
                   help="comparison_rollout.csv from previous Episode-4 DLS vs SAC")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--substeps",type=int,default=0,
                   help="Override timing for diagnostics; 0 uses reference FPS")
    p.add_argument("--reference-fps",type=float,default=8.,
                   help="Nominal frame sampling rate of original DLS/SAC comparison")
    p.add_argument("--max-frames",type=int,default=0,
                   help="0 uses entire saved trajectory")
    p.add_argument("--render-every",type=int,default=4)
    p.add_argument("--width",type=int,default=640)
    p.add_argument("--height",type=int,default=480)
    p.add_argument("--overview-camera",choices=("reference","side"),default="reference",
                   help="Reference matches original kinematic DLS/SAC camera; side uses calibrated Side sensor")
    p.add_argument("--video-fps",type=float,default=8.)
    p.add_argument("--no-video",action="store_false",dest="save_video")
    p.add_argument("--kp",type=float,default=12.)
    p.add_argument("--kd",type=float,default=1.1)
    p.add_argument("--max-torque",type=float,default=3.)
    p.add_argument("--max-initial-delta-rad",type=float,default=.10)
    p.add_argument("--allow-clipped-diagnostic",action="store_true",
                   help="Only for debugging. Original out-of-limit pose is clipped, not comparable.")
    p.add_argument("--dataset-root",type=Path,default=scene.DATASET)
    p.add_argument("--episode",type=int,default=4)
    p.add_argument("--object-layout-json",type=Path,default=scene.OBJECTS)
    p.add_argument("--cup-pose-json",type=Path,default=scene.CUP_POSE)
    p.add_argument("--extrinsics",type=Path,default=scene.EXTRINSICS)
    p.add_argument("--intrinsics",type=Path,default=scene.INTRINSICS)
    p.add_argument("--urdf",type=Path,default=scene.r2s.DEFAULT_URDF)
    p.add_argument("--robot-joint-offsets-json",type=Path)
    p.add_argument("--muhi-clearance-mm",type=float,default=75)
    p.add_argument("--cup-clearance-mm",type=float,default=45)
    p.add_argument("--wrist-mount-pos",type=float,nargs=3,default=wrist.DEFAULT_MOUNT_POS)
    p.add_argument("--wrist-mount-rpy-deg",type=float,nargs=3,default=wrist.DEFAULT_MOUNT_RPY_DEG)
    p.add_argument("--wrist-fovy",type=float,default=wrist.DEFAULT_FOVY_DEG)
    p.add_argument("--wrist-look-at",type=float,nargs=3,default=wrist.DEFAULT_LOOK_AT)
    args=p.parse_args()
    if not args.rollout.is_file():
        p.error("Missing comparison_rollout.csv")
    if args.out.exists():p.error("Output exists")
    if (args.substeps<0 or args.render_every<1 or args.width<64 or args.height<64
            or args.max_frames<0 or not np.isfinite([args.kp,args.kd,args.max_torque,args.video_fps,args.reference_fps]).all()
            or args.reference_fps<=0
            or args.kp<=0 or args.kd<0 or args.max_torque<=0 or args.video_fps<=0):
        p.error("Invalid physics, render or timing parameters")
    run(args)


if __name__=="__main__":
    main()
