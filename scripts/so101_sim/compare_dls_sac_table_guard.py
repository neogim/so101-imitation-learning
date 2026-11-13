#!/usr/bin/env python3
"""Fresh kinematic DLS vs SAC comparison with a common table-collision guard.

Original scripts / checkpoints / rollout CSV are not modified. Both controllers
start from the SAME explicit URDF-limit-clipped saved pose; this is a NEW
baseline, not a reproduction of the old out-of-limit visual home.
The guard checks interpolated link geometry against tabletop and cup before
applying an action. It may shorten or stop actions but does not replan around
obstacles. NO MuJoCo dynamic integration or physical grasp is claimed.
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
import table_collision_guard as collision
import wrist_camera_sim as wrist


def record_row(policy, stage, step, q, xyz, target, detail, start_clip):
    d=float(np.linalg.norm(target-xyz)*1000.)
    result=dict(policy=policy,stage=stage,stage_step=step,
                tcp_x=float(xyz[0]),tcp_y=float(xyz[1]),tcp_z=float(xyz[2]),
                target_x=float(target[0]),target_y=float(target[1]),target_z=float(target[2]),
                tcp_error_mm=d,guard_status=detail["status"],
                table_distance_mm=detail["minimum_distance_mm"],
                nearest_pair=str(detail.get("nearest_pair")),
                initial_pose_adjusted=bool(start_clip))
    for i,v in enumerate(q):
        result[f"joint_{i+1}"]=float(v)
    return result


def run_policy(name, model, data, guard, qids, dids, low, high, tcp, goals, start, args, agent):
    data.qpos[qids]=start
    mujoco.mj_forward(model,data)
    output=[];stats=[]
    for stage in ("muhi","cup"):
        goal=np.asarray(goals[stage],float)
        blockers=0
        for t in range(args.max_steps+1):
            q=np.asarray(data.qpos[qids],float).copy()
            xyz=np.asarray(data.xpos[tcp],float).copy()
            detail=guard.inspect(q)
            detail={"status":"initial" if t==0 else "at_position",**detail}
            if t==0:
                output.append(record_row(name,stage,t,q,xyz,goal,detail,args.start_adjusted))
            distance=float(np.linalg.norm(goal-xyz)*1000)
            if distance<=args.success_mm:
                break
            if t>=args.max_steps:break
            jac=np.zeros((3,model.nv));jac_r=np.zeros((3,model.nv))
            if name=="dls":
                mujoco.mj_jacBody(model,data,jac,jac_r,tcp)
                J=jac[:,dids]
                delta=J.T@np.linalg.solve(
                    J@J.T+.02**2*np.eye(3),.75*(goal-xyz))
                delta=np.clip(delta,-.04,.04)
            else:
                error=goal-xyz
                obs=np.r_[q,error,np.linalg.norm(error)].astype(np.float32)
                action,_=agent.predict(obs,deterministic=True)
                delta=np.clip(np.asarray(action,float),-1,1)*.04
            proposed=np.clip(q+delta,low,high)
            next_q,detail=guard.filter_step(q,proposed)
            if detail["status"] in ("blocked","unsafe_start"):
                blockers+=1
            else:
                blockers=0
            data.qpos[qids]=next_q
            mujoco.mj_forward(model,data)
            output.append(record_row(name,stage,t+1,next_q,
                          np.asarray(data.xpos[tcp]).copy(),goal,
                          detail,args.start_adjusted))
            if blockers>=args.blocked_patience:
                break
        stage_rows=[r for r in output if r["stage"]==stage]
        stats.append({"stage":stage,
                      "final_tcp_error_mm":stage_rows[-1]["tcp_error_mm"],
                      "success":stage_rows[-1]["tcp_error_mm"]<=args.success_mm,
                      "steps":len(stage_rows)-1,
                      "blocked_steps":sum(r["guard_status"] in ("blocked","unsafe_start") for r in stage_rows),
                      "shortened_steps":sum(r["guard_status"]=="shortened" for r in stage_rows)})
    # Common home interpolation is guard checked too. The baseline pose is
    # now physically within joint limits (unlike old visual home).
    q_home=np.asarray(start,float)
    q_begin=np.asarray(data.qpos[qids],float)
    home_xyz=np.zeros(3)
    data.qpos[qids]=q_home
    mujoco.mj_forward(model,data)
    home_xyz[:]=data.xpos[tcp]
    data.qpos[qids]=q_begin
    mujoco.mj_forward(model,data)
    for t in range(1,args.home_steps+1):
        q=np.asarray(data.qpos[qids],float).copy()
        z=t/args.home_steps
        w=10*z**3-15*z**4+6*z**5
        wished=q_begin+w*(q_home-q_begin)
        desired=q+np.clip(wished-q,-.04,.04)
        next_q,detail=guard.filter_step(q,desired)
        data.qpos[qids]=next_q
        mujoco.mj_forward(model,data)
        output.append(record_row(name,"home",t,next_q,
                      np.asarray(data.xpos[tcp]).copy(),home_xyz,
                      detail,args.start_adjusted))
    return output,stats


def make_video(args, model, data, qids, goals, side, rows_by_policy, out):
    from PIL import Image, ImageDraw
    ffmpeg=shutil.which("ffmpeg")
    if ffmpeg is None:
        return "ffmpeg_missing"
    width,height=args.width,args.height
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,width)
    model.vis.global_.offheight=max(model.vis.global_.offheight,height)
    camera=scene.camera_for(model,goals,False,side)
    option=mujoco.MjvOption()
    option.geomgroup[3]=0
    from compare_dls_sac_physics import add_reference_markers
    video=out/"dls_sac_table_guard_side_wrist.mp4"
    command=[ffmpeg,"-hide_banner","-loglevel","error","-y","-f","rawvideo",
             "-pix_fmt","rgb24","-s:v",f"{2*width}x{2*(height+48)}",
             "-r",str(args.video_fps),"-i","-","-an","-c:v","libx264",
             "-pix_fmt","yuv420p","-movflags","+faststart",str(video)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    count=max(len(a) for a in rows_by_policy.values())
    try:
        with mujoco.Renderer(model,width=width,height=height) as renderer:
            for index in range(0,count,args.render_every):
                canvas=Image.new("RGB",(2*width,2*(height+48)),(20,29,39))
                draw=ImageDraw.Draw(canvas)
                for col,name in enumerate(("dls","sac")):
                    rows=rows_by_policy[name]
                    row=rows[min(index,len(rows)-1)]
                    data.qpos[qids]=[row[f"joint_{j}"] for j in range(1,6)]
                    mujoco.mj_forward(model,data)
                    for rr,cam in enumerate((camera,wrist.CAMERA)):
                        renderer.update_scene(data,camera=cam,scene_option=option)
                        if rr==0:
                            trail=[np.array([item["tcp_x"],item["tcp_y"],item["tcp_z"]],float)
                                   for item in rows[:min(index,len(rows)-1)+1]]
                            add_reference_markers(renderer.scene,goals,tcp_trail=trail)
                        image=Image.fromarray(renderer.render().copy())
                        x,y=width*col, (height+48)*rr
                        canvas.paste(image,(x,y+48))
                        label=f"{name.upper()} | {'OVERVIEW' if rr==0 else 'WRIST'} | {row['stage']}  {row['tcp_error_mm']:.1f}mm"
                        draw.text((x+8,y+10),label,fill="white")
                        draw.text((x+8,y+28),f"Guard={row['guard_status']} | gap={row['table_distance_mm']:.1f}mm",
                                  fill=(255,199,112))
                process.stdin.write(np.asarray(canvas,np.uint8).tobytes())
        process.stdin.close();process.stdin=None
        stderr=process.communicate()[1]
        if process.returncode:raise RuntimeError(stderr.decode(errors="replace"))
    except Exception:
        if process.poll() is None:process.kill();process.communicate()
        if video.exists():video.unlink()
        raise
    return "created"


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    if not args.sac_model.is_file():
        raise FileNotFoundError(args.sac_model)
    # Same saved Real-to-Sim layout; no new object placements.
    args.physical_objects=False
    args.wrist_camera=True
    (model,data,qids,dids,low,high,tcp,side,goals,
     *_remainder)=scene.create_scene(args)
    source=np.asarray(data.qpos[qids],float).copy()
    start=np.clip(source,low,high)
    delta=np.degrees(start-source)
    args.start_adjusted=bool(np.max(np.abs(delta))>1e-6)
    guard=collision.TableCollisionGuard(
        model,data,qids,clearance_mm=args.clearance_mm,
        max_delta_rad=args.check_resolution_rad,
        include_cup=args.include_cup)
    initial_check=guard.inspect(start)
    if not initial_check["clearance_satisfied"]:
        # A preflight failure is an actual experiment result: persist the
        # diagnostic BEFORE raising, rather than leaving an absent OUT folder.
        report={
            "status":"TABLE_CLEARANCE_BASELINE_REJECTED",
            "meaning":"Geometric tabletop clearance failure, not proof of real-world penetration",
            "clearance_requested_mm":args.clearance_mm,
            "original_visual_joint_deg":np.degrees(source).round(5).tolist(),
            "clipped_common_joint_deg":np.degrees(start).round(5).tolist(),
            "clip_delta_deg":delta.round(5).tolist(),
            "original_visual_pose_table":guard.inspect(source),
            "clipped_pose_table":initial_check,
            "nearest_pairs_clipped":guard.nearest_pairs(start),
            "nearest_pairs_original":guard.nearest_pairs(source),
            "notes":[
                "The saved Episode-4 visual joint calibration is not a verified physical calibration.",
                "Robot meshes may use conservative convex collision geometry.",
                "Do not disable the guard or infer a true 9.34mm physical penetration.",
                "Neither SAC nor DLS was run; correct table/base/gripper geometry first."
            ],
        }
        args.out.mkdir(parents=True,exist_ok=False)
        report_file=args.out/"preflight_diagnostic.json"
        report_file.write_text(
            json.dumps(report,indent=2,ensure_ascii=False)+"\\n",encoding="utf-8")
        pair=report["nearest_pairs_clipped"][0]
        raise ValueError(
            "TABLE CLEARANCE BASELINE REJECTED: saved scene has a "
            f"{initial_check['minimum_distance_mm']:.2f}mm signed gap "
            f"for geom {pair['robot_geom_id']} "
            f"({pair['robot_geom_name'] or '<unnamed>'}, body={pair['robot_body']}) "
            f"vs {pair['table_geom_name']}. "
            f"Report saved to {report_file}. "
            "The run is blocked; no SAC/DLS trajectories or MP4 were generated.")
    from stable_baselines3 import SAC
    agent=SAC.load(str(args.sac_model),device="cpu")
    if tuple(agent.observation_space.shape)!=(9,) or tuple(agent.action_space.shape)!=(5,):
        raise ValueError("Expected 9D SAC observations and 5D actions")
    results={};stages={}
    for name in ("dls","sac"):
        results[name],stages[name]=run_policy(
            name,model,data,guard,qids,dids,low,high,tcp,goals,start,args,agent)
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/"guarded_rollout.csv").open("x",newline="",encoding="utf-8") as fp:
        fields=list(results["dls"][0])
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        for policy in ("dls","sac"):writer.writerows(results[policy])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,8))
    for idx,stage in enumerate(("muhi","cup")):
        for name in ("dls","sac"):
            rows=[row for row in results[name] if row["stage"]==stage]
            axes[idx].plot(range(len(rows)),[row["tcp_error_mm"] for row in rows],label=name.upper())
        axes[idx].axhline(args.success_mm,color="gray",linestyle="--")
        axes[idx].set(title=stage.upper(),xlabel="Kinematic joint action",ylabel="TCP error (mm)")
        axes[idx].grid(alpha=.2)
        axes[idx].legend()
    fig.tight_layout();fig.savefig(args.out/"guarded_tcp_error.png",dpi=160);plt.close(fig)
    video_status=make_video(args,model,data,qids,goals,side,results,args.out) if args.save_video else "disabled"
    summary={
        "status":"KINEMATIC_DLS_SAC_TABLE_COLLISION_GUARD_EXPERIMENT",
        "physical_dynamics":False,
        "guard_method":"swept geom signed-distance, backtracking (not obstacle replanning)",
        "clearance_mm":args.clearance_mm,
        "include_cup":args.include_cup,
        "source_initial_joint_deg":np.degrees(source).tolist(),
        "common_in_limit_start_joint_deg":np.degrees(start).tolist(),
        "start_adjustment_deg":delta.tolist(),
        "same_as_original_visual_start":not args.start_adjusted,
        "initial_table_clearance":initial_check,
        "goals_xyz_m":{k:v.tolist() for k,v in goals.items()},
        "stages":stages,
        "video_status":video_status,
        "caveats":[
            "New shared in-limit initial pose; not identical to the previous visual-only trajectory.",
            "The same collision backtracking is applied to DLS and SAC, which have different proposed actions.",
            "No grasp, no physics, no guarantee of obstacle avoidance if no safe local step exists.",
            "Only link-vs-table and optionally cup distances; self-collision and camera pose remain unverified.",
            "Mesh convex hulls may give conservative false positives. A failing preflight is not proof of real hardware interference.",
        ]
    }
    (args.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"stages":stages,"video_status":video_status,
                      "start_adjustment_deg":delta.tolist()},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sac-model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--clearance-mm",type=float,default=3.)
    p.add_argument("--check-resolution-rad",type=float,default=.01)
    p.add_argument("--include-cup",action="store_true")
    p.add_argument("--max-steps",type=int,default=120)
    p.add_argument("--home-steps",type=int,default=80)
    p.add_argument("--blocked-patience",type=int,default=5)
    p.add_argument("--success-mm",type=float,default=10)
    p.add_argument("--render-every",type=int,default=4)
    p.add_argument("--video-fps",type=float,default=8.)
    p.add_argument("--width",type=int,default=640)
    p.add_argument("--height",type=int,default=480)
    p.add_argument("--no-video",dest="save_video",action="store_false")
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
    if (args.max_steps<1 or args.home_steps<1 or args.blocked_patience<1
            or args.render_every<1 or args.width<64 or args.height<64
            or not math.isfinite(args.video_fps) or args.video_fps<=0):
        p.error("invalid run parameters")
    run(args)


if __name__=="__main__":
    main()
