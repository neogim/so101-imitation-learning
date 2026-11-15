#!/usr/bin/env python3
"""Side-by-side DLS vs SAC on the SAME saved Real-to-Sim Muhi/cup positions.

Both start from the same recorded joint pose (which exceeds some URDF limits).
Each independently reaches Muhi, then cup, then returns to EXACT visual home.
The home movement is quintic JOINT interpolation, not SAC/DLS; both use the
same return algorithm. This is kinematic visualization ONLY, not safe robot
motion. Evaluation uses 9D SAC observation [q5, target - TCP, distance].
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
from PIL import Image,ImageDraw,ImageFont

import demo_real2sim_muhi_cup_reach as scene

JOINTS=scene.JOINTS


def sac_rollout(args, model, data, qids, tcp_id, goals, original_q, original_tcp, *, freeze_j5=False):
    from stable_baselines3 import SAC

    agent=SAC.load(str(args.sac_model),device="cpu")
    if tuple(agent.observation_space.shape)!=(9,) or tuple(agent.action_space.shape)!=(5,):
        raise ValueError("SAC checkpoint does not use 9D state / 5D action")
    limits=np.array([
        model.jnt_range[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)]
        for name in JOINTS])
    low,high=limits[:,0],limits[:,1]
    frames=[];stages=[]
    for stage in ("muhi","cup"):
        target=np.asarray(goals[stage],float)
        initial=float(np.linalg.norm(target-data.xpos[tcp_id])*1000)
        success=False
        for step in range(args.max_steps+1):
            current=np.asarray(data.xpos[tcp_id],float).copy()
            err=target-current
            dist=float(np.linalg.norm(err))
            q=np.asarray(data.qpos[qids],float).copy()
            excess=np.maximum(low-q,0)+np.maximum(q-high,0)
            in_limits=bool(np.max(excess)<=1e-7)
            row={"frame":len(frames),"stage":stage,"stage_step":step,
                 "tcp_x":float(current[0]),"tcp_y":float(current[1]),
                 "tcp_z":float(current[2]),"target_x":float(target[0]),
                 "target_y":float(target[1]),"target_z":float(target[2]),
                 "tcp_error_mm":dist*1000,"joint_limits_ok":in_limits,
                 "max_joint_limit_excess_deg":float(np.degrees(np.max(excess)))}
            row.update({f"joint_{i+1}":float(q[i]) for i in range(5)})
            frames.append(row)
            if dist*1000<=args.success_mm and in_limits:
                success=True
                break
            if step==args.max_steps:break
            obs=np.r_[q,err,dist].astype(np.float32)
            action,_=agent.predict(obs,deterministic=True)
            delta=np.clip(np.asarray(action,float),-1,1)*.04
            # Ablation: suppress only the J5 action AFTER policy inference,
            # preserving all five joint observations and the trained model.
            if freeze_j5:
                delta[4]=0.0
            # Same gradual URDF-bound recovery as the DLS demo.
            next_q=np.clip(q+delta,low,high)
            below=q<low;above=q>high
            next_q[below]=q[below]+np.minimum(.04,low[below]-q[below])
            next_q[above]=q[above]-np.minimum(.04,q[above]-high[above])
            if np.max(np.abs(next_q-q))>.040001:
                raise AssertionError("Joint step exceeded 0.04 rad")
            data.qpos[qids]=next_q
            mujoco.mj_forward(model,data)
        stages.append({"stage":stage,"success":success,"steps":step,
                       "initial_error_mm":initial,
                       "final_error_mm":frames[-1]["tcp_error_mm"],
                       "final_joint_limits_ok":frames[-1]["joint_limits_ok"]})
    return scene.return_home(args,model,data,qids,tcp_id,
                             original_q,original_tcp,frames,stages)


def _camera(model,goals,side,camera_name):
    return scene.camera_for(model,goals,side,camera_name)


def _draw_trail(renderer,rows,step,goals):
    # Show recorded TCP trajectory only, with stage-colored segments.
    scene.extras(renderer.scene,rows,step,goals)


def export_video(args,model,data,qids,goals,camera,rollouts):
    ffmpeg=shutil.which("ffmpeg")
    if ffmpeg is None:raise RuntimeError("Install ffmpeg: brew install ffmpeg")
    if args.save_video.exists():
        raise FileExistsError(f"MP4 already exists: {args.save_video}")
    args.save_video.parent.mkdir(parents=True,exist_ok=True)
    width,height=640,480
    model.vis.global_.offwidth=max(width,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(height,model.vis.global_.offheight)
    cmd=[ffmpeg,"-hide_banner","-loglevel","error","-f","rawvideo",
         "-pix_fmt","rgb24","-s:v","1280x720","-r",str(args.video_fps),
         "-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p",
         "-movflags","+faststart",str(args.save_video)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=height,width=width) as renderer:
            try:font=ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf",22)
            except OSError:font=ImageFont.load_default()
            n=max(len(v) for v in rollouts.values())
            steps=math.ceil((n-1)*args.video_fps/args.fps)+1
            for k in range(steps+round(args.hold_seconds*args.video_fps)):
                index=min(n-1,int(k*args.fps/args.video_fps))
                canvas=Image.new("RGB",(1280,720),(23,30,39))
                draw=ImageDraw.Draw(canvas)
                draw.text((22,16),"SO-101 Real-to-Sim: Muhi -> cup -> HOME",
                          font=font,fill="white")
                for c,policy in enumerate(("dls","sac")):
                    rows=rollouts[policy]
                    j=min(index,len(rows)-1)
                    row=rows[j]
                    data.qpos[qids]=[row[f"joint_{z+1}"] for z in range(5)]
                    mujoco.mj_forward(model,data)
                    renderer.update_scene(data,camera=camera)
                    _draw_trail(renderer,rows,j,goals)
                    picture=Image.fromarray(renderer.render().copy())
                    canvas.paste(picture,(640*c,80))
                    label="DLS" if policy=="dls" else "SAC (10mm trained)"
                    color=(240,154,66) if policy=="dls" else (91,181,249)
                    draw.text((20+640*c,54),label,font=font,fill=color)
                    draw.text((20+640*c,575),
                              f"{row['stage']}: error {row['tcp_error_mm']:.1f}mm | "
                              f"step {row['stage_step']}",
                              font=font,fill=color)
                    if j==len(rows)-1:
                        draw.text((20+640*c,612),"Finished (visual home)",
                                  font=font,fill="white")
                draw.text((22,666),
                          "Identical saved positions and initial pose | no grasp, collision, dynamics",
                          font=font,fill="white")
                draw.text((22,694),
                          "Initial visual pose exceeds URDF joint limits; NOT hardware-safe",
                          font=font,fill=(247,168,137))
                process.stdin.write(np.asarray(canvas,np.uint8).tobytes())
        process.stdin.close()
        process.stdin=None
        stderr=process.communicate()[1]
        if process.returncode:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")
    except Exception:
        if process.poll() is None:
            process.kill()
            process.communicate()
        if args.save_video.exists():
            args.save_video.unlink()
        raise
    print("Saved comparison MP4:",args.save_video)


def write_files(out,args,rollouts,summaries,goals):
    if out.exists():raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    with (out/"comparison_rollout.csv").open("x",newline="",encoding="utf-8") as fp:
        fields=["policy"]+list(rollouts["dls"][0])
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        for policy,rows in rollouts.items():
            for row in rows:writer.writerow({"policy":policy,**row})
    data={
        "status":"REAL2SIM_TWO_POLICY_KINEMATIC_ONLY",
        "sac_model":str(args.sac_model),
        "goal_xyz_m":{k:np.asarray(v).tolist() for k,v in goals.items()},
        "policies":summaries,
        "home_stage":"both use identical quintic joint interpolation",
        "warnings":[
            "Visual alignment candidate, not a validated 3D calibration.",
            "Initial home joint pose violates imported URDF limits.",
            "SAC trained on FK-generated goals; these are saved Real-to-Sim objects.",
            "No grasp/collision/physics; success is position-only."
        ]}
    (out/"summary.json").write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(10,5.4))
    for policy,rows in rollouts.items():
        ax.plot([r["frame"] for r in rows],
                [r["tcp_error_mm"] for r in rows],label=policy)
    ax.axhline(args.success_mm,ls="--",color="gray",label="10mm success")
    ax.set(xlabel="Local policy frame (stage switch differs)",
           ylabel="TCP error to current stage target (mm)",
           title="DLS vs SAC, saved Real-to-Sim positions")
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(out/"dls_vs_sac_tcp_error.png",dpi=175);plt.close(fig)
    print("Saved summary and graph:",out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--sac-model",type=Path,required=True)
    p.add_argument("--save-video",type=Path,default=None)
    p.add_argument("--fps",type=float,default=8)
    p.add_argument("--video-fps",type=int,default=30)
    p.add_argument("--hold-seconds",type=float,default=2)
    p.add_argument("--home-steps",type=int,default=80)
    p.add_argument("--max-steps",type=int,default=120)
    p.add_argument("--success-mm",type=float,default=10)
    p.add_argument("--side-camera",action="store_true")
    p.add_argument("--episode",type=int,default=4)
    p.add_argument("--dataset-root",type=Path,default=scene.DATASET)
    p.add_argument("--object-layout-json",type=Path,default=scene.OBJECTS)
    p.add_argument("--cup-pose-json",type=Path,default=scene.CUP_POSE)
    p.add_argument("--extrinsics",type=Path,default=scene.EXTRINSICS)
    p.add_argument("--intrinsics",type=Path,default=scene.INTRINSICS)
    p.add_argument("--urdf",type=Path,default=scene.r2s.DEFAULT_URDF)
    p.add_argument("--robot-joint-offsets-json",type=Path,default=None)
    p.add_argument("--muhi-clearance-mm",type=float,default=75)
    p.add_argument("--cup-clearance-mm",type=float,default=45)
    args=p.parse_args()
    if not args.sac_model.is_file():
        p.error(f"Missing SAC model: {args.sac_model}")
    if args.home_steps<=0 or args.max_steps<=0 or args.fps<=0 or args.video_fps<=0:
        p.error("steps/fps must be positive")
    for file in (args.dataset_root,args.object_layout_json,args.cup_pose_json,
                 args.extrinsics,args.intrinsics,args.urdf):
        if not file.exists():p.error(f"Missing input: {file}")
    (model,data,qids,dids,low,high,tcp,cam_name,goals,cup,layout,
     offset_path,offset_config)=scene.create_scene(args)
    initial_q=np.asarray(data.qpos[qids],float).copy()
    initial_tcp=np.asarray(data.xpos[tcp],float).copy()
    dls_frames,dls_stages=scene.dls_two_stages(
        args,model,data,qids,dids,low,high,tcp,goals)
    dls_frames,dls_stages=scene.return_home(
        args,model,data,qids,tcp,initial_q,initial_tcp,
        dls_frames,dls_stages)
    data.qpos[qids]=initial_q
    mujoco.mj_forward(model,data)
    sac_frames,sac_stages=sac_rollout(
        args,model,data,qids,tcp,goals,initial_q,initial_tcp)
    rollouts={"dls":dls_frames,"sac":sac_frames}
    summaries={"dls":dls_stages,"sac":sac_stages}
    write_files(args.out,args,rollouts,summaries,goals)
    for name,stages in summaries.items():
        print(name," | ".join(f"{s['stage']}={s['success']} "
                              f"{s['final_error_mm']:.2f}mm ({s['steps']} steps)"
                              for s in stages))
    if args.save_video:
        camera=_camera(model,goals,args.side_camera,cam_name)
        export_video(args,model,data,qids,goals,camera,rollouts)


if __name__=="__main__":
    main()
