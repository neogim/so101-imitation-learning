#!/usr/bin/env python3
"""Ablation: same trained SAC, normal J5 vs J5 action fixed to zero.

Uses the saved Episode-4 Real-to-Sim layout, exact recorded initial joint
state, and unchanged trained policy. Both runs return to the recorded visual
home using identical quintic interpolation. NO grasp, dynamics, or real robot.
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
from PIL import Image, ImageDraw, ImageFont

import demo_real2sim_muhi_cup_reach as scene
import compare_real2sim_muhi_cup_dls_sac as compare

LABELS={"normal":"SAC normal", "j5_fixed":"SAC J5 action = 0"}
COLORS={"normal":"tab:blue","j5_fixed":"tab:orange"}


def metrics(rows, stages):
    per_stage={}
    for stage in ("muhi","cup"):
        subset=[row for row in rows if row["stage"]==stage]
        coords=np.asarray([[row[f"tcp_{axis}"] for axis in "xyz"] for row in subset])
        q=np.asarray([[row[f"joint_{i}"] for i in range(1,6)] for row in subset])
        travel=float(np.linalg.norm(np.diff(coords,axis=0),axis=1).sum()*1000)
        j5_delta=np.diff(q[:,4])
        per_stage[stage]={
            "success":next(s["success"] for s in stages if s["stage"]==stage),
            "steps":len(subset)-1,
            "final_tcp_error_mm":float(subset[-1]["tcp_error_mm"]),
            "tcp_path_mm":travel,
            "j5_total_rotation_deg":float(np.degrees(np.abs(j5_delta).sum())),
            "j5_net_rotation_deg":float(np.degrees(q[-1,4]-q[0,4])),
            "j5_max_step_deg":float(np.degrees(np.max(np.abs(j5_delta)))) if j5_delta.size else 0.0,
        }
    return per_stage


def save_outputs(args, rollouts, stages, goals):
    if args.out.exists():
        raise FileExistsError(f"Output directory already exists: {args.out}")
    args.out.mkdir(parents=True)
    with (args.out/"rollout.csv").open("x",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=["policy"]+list(rollouts["normal"][0]))
        writer.writeheader()
        for name,rows in rollouts.items():
            for row in rows:
                writer.writerow({"policy":name,**row})
    results={name:metrics(rollouts[name],stages[name]) for name in LABELS}
    summary={
        "status":"SAC_J5_INFERENCE_ABLATION_ON_SAVED_REAL2SIM_LAYOUT",
        "model":str(args.sac_model),
        "goals_xyz_m":{key:value.tolist() for key,value in goals.items()},
        "success_radius_mm":args.success_mm,
        "results":results,
        "home_return":"same recorded pose via quintic interpolation on both",
        "caveats":[
            "J5 suppressed only at inference; SAC checkpoint unchanged.",
            "Initial visual home joint pose is outside URDF limits: NOT hardware-safe.",
            "Home return is excluded from metrics and not learned by SAC.",
            "Kinematic TCP position-only reach; no grasp, orientation goal, contact or physics.",
            "Only one saved workcell/initial pose: no generalization claim."
        ]
    }
    (args.out/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=False)
    for name,rows in rollouts.items():
        for stage,ax in zip(("muhi","cup"),axes):
            selected=[r for r in rows if r["stage"]==stage]
            ax.plot([r["stage_step"] for r in selected],
                    [r["tcp_error_mm"] for r in selected],
                    label=LABELS[name],color=COLORS[name])
            ax.set(ylabel="TCP error (mm)",title=f"Stage: {stage}")
            ax.axhline(args.success_mm,ls="--",color="gray")
            ax.grid(alpha=.25)
    axes[-1].set_xlabel("Stage step")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.out/"tcp_error_comparison.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=False)
    for name,rows in rollouts.items():
        for stage,ax in zip(("muhi","cup"),axes):
            selected=[r for r in rows if r["stage"]==stage]
            ax.plot([r["stage_step"] for r in selected],
                    [np.degrees(r["joint_5"]) for r in selected],
                    label=LABELS[name],color=COLORS[name])
            ax.set(ylabel="J5 angle (deg)",title=f"Stage: {stage}")
            ax.grid(alpha=.25)
    axes[-1].set_xlabel("Stage step")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.out/"j5_angle_comparison.png",dpi=170)
    plt.close(fig)
    for name,detail in results.items():
        for stage,m in detail.items():
            print(f"{LABELS[name]} {stage}: success={m['success']} "
                  f"steps={m['steps']} final={m['final_tcp_error_mm']:.2f}mm "
                  f"path={m['tcp_path_mm']:.1f}mm "
                  f"J5_total={m['j5_total_rotation_deg']:.2f}deg")
    print("Saved:",args.out)
    return summary


def save_video(args, model, data, qids, goals, camera, rollouts):
    if args.save_video.exists():
        raise FileExistsError(f"Video already exists: {args.save_video}")
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; brew install ffmpeg")
    args.save_video.parent.mkdir(parents=True,exist_ok=True)
    width,height=640,480
    model.vis.global_.offwidth=max(width,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(height,model.vis.global_.offheight)
    command=[ffmpeg,"-hide_banner","-loglevel","error","-f","rawvideo",
             "-pix_fmt","rgb24","-s:v","1280x720","-r",str(args.video_fps),
             "-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p",
             "-movflags","+faststart",str(args.save_video)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,width=width,height=height) as renderer:
            try:
                font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf",20)
            except OSError:
                font=ImageFont.load_default()
            total=max(map(len,rollouts.values()))
            total_frames=math.ceil((total-1)*args.video_fps/args.fps)+1
            for frame in range(total_frames+int(args.hold_seconds*args.video_fps)):
                index=min(total-1,int(frame*args.fps/args.video_fps))
                canvas=Image.new("RGB",(1280,720),(24,32,42))
                draw=ImageDraw.Draw(canvas)
                draw.text((18,16),"SAC J5 ablation: Muhi -> cup -> recorded visual home",
                          font=font,fill="white")
                for c,name in enumerate(LABELS):
                    rows=rollouts[name]
                    j=min(index,len(rows)-1)
                    row=rows[j]
                    data.qpos[qids]=[row[f"joint_{i}"] for i in range(1,6)]
                    mujoco.mj_forward(model,data)
                    renderer.update_scene(data,camera=camera)
                    scene.extras(renderer.scene,rows,j,goals)
                    canvas.paste(Image.fromarray(renderer.render().copy()),(c*640,80))
                    tint=(120,191,255) if name=="normal" else (255,178,108)
                    draw.text((20+c*640,53),LABELS[name],font=font,fill=tint)
                    draw.text((20+c*640,571),
                              f"{row['stage']} | TCP {row['tcp_error_mm']:.1f}mm "
                              f"| J5 {math.degrees(row['joint_5']):.1f} deg",
                              font=font,fill=tint)
                draw.text((18,661),
                          "Same saved layout / initial pose / trained SAC; only J5 action differs",
                          font=font,fill="white")
                draw.text((18,691),
                          "Visual-only; recorded home violates URDF limits. No grasp/physics.",
                          font=font,fill=(255,175,145))
                proc.stdin.write(np.asarray(canvas,np.uint8).tobytes())
        proc.stdin.close()
        proc.stdin=None
        errors=proc.communicate()[1]
        if proc.returncode:
            raise RuntimeError(errors.decode(errors="replace"))
    except Exception:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        if args.save_video.exists():
            args.save_video.unlink()
        raise
    print("Saved MP4:",args.save_video)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sac-model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--save-video",type=Path)
    p.add_argument("--side-camera",action="store_true")
    p.add_argument("--fps",type=float,default=8)
    p.add_argument("--video-fps",type=int,default=30)
    p.add_argument("--hold-seconds",type=float,default=2)
    p.add_argument("--max-steps",type=int,default=120)
    p.add_argument("--home-steps",type=int,default=80)
    p.add_argument("--success-mm",type=float,default=10)
    p.add_argument("--muhi-clearance-mm",type=float,default=75)
    p.add_argument("--cup-clearance-mm",type=float,default=45)
    p.add_argument("--dataset-root",type=Path,default=scene.DATASET)
    p.add_argument("--episode",type=int,default=4)
    p.add_argument("--object-layout-json",type=Path,default=scene.OBJECTS)
    p.add_argument("--cup-pose-json",type=Path,default=scene.CUP_POSE)
    p.add_argument("--extrinsics",type=Path,default=scene.EXTRINSICS)
    p.add_argument("--intrinsics",type=Path,default=scene.INTRINSICS)
    p.add_argument("--urdf",type=Path,default=scene.r2s.DEFAULT_URDF)
    p.add_argument("--robot-joint-offsets-json",type=Path)
    args=p.parse_args()
    if (args.max_steps<=0 or args.home_steps<=0 or args.success_mm<=0
            or args.fps<=0 or args.video_fps<=0 or args.hold_seconds<0):
        p.error("Step limits, tolerance and frame rates must be positive")
    for path in (args.sac_model,args.dataset_root,args.object_layout_json,
                 args.cup_pose_json,args.extrinsics,args.intrinsics,args.urdf):
        if not path.is_file() and not(path==args.dataset_root and path.is_dir()):
            p.error(f"Missing required input: {path}")
    (model,data,qids,dids,low,high,tcp,camera_name,goals,cup,layout,
     offset_path,offset_cfg)=scene.create_scene(args)
    initial_q=np.asarray(data.qpos[qids],float).copy()
    initial_tcp=np.asarray(data.xpos[tcp],float).copy()
    rollouts={};stages={}
    for name in LABELS:
        data.qpos[qids]=initial_q
        mujoco.mj_forward(model,data)
        rows,detail=compare.sac_rollout(
            args,model,data,qids,tcp,goals,initial_q,initial_tcp,
            freeze_j5=(name=="j5_fixed"))
        rollouts[name]=rows
        stages[name]=detail
    save_outputs(args,rollouts,stages,goals)
    if args.save_video:
        camera=scene.camera_for(model,goals,args.side_camera,camera_name)
        save_video(args,model,data,qids,goals,camera,rollouts)


if __name__=="__main__":
    main()
