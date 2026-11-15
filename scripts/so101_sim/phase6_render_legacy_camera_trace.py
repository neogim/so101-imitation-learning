#!/usr/bin/env python3
"""Phase 6A visual playback of an ALREADY RECORDED dynamic-joint CSV.

Reuses the established Real-to-Sim URDF/MjSpec, wrist_camera_sim.install,
render_wrist_camera_rollout.jaw_centers_in_wrist_view, and
real_to_sim.render_image. This is separate offline kinematic VISUALIZATION:
it never changes the saved physical simulation or controls hardware.

There is no Muhi and no invented gripper-camera substitute. Wrist optical
axes are those installed by the historic original mount without a
per-frame "look at gripper" override.
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
from PIL import Image, ImageDraw

import real_to_sim as r2s
import wrist_camera_sim as wrist
from render_wrist_camera_rollout import jaw_centers_in_wrist_view


NAMES = (*r2s.JOINTS,"gripper")


def render(args):
    source=args.trace.resolve(strict=True)
    with source.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty joint trace")
    keys=[f"{name}_actual_deg" for name in NAMES]
    if not all(k in rows[0] for k in keys):
        raise ValueError("Trace is not phase6_dynamic_joint_hold.py format")
    if args.out.exists():
        raise FileExistsError(args.out)

    spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    spec.compiler.discardvisual=False
    # Historical Real-to-Sim wrist-camera scene construction; NOT a new
    # camera replacing the mounted optic.
    wrist.install(spec,bracket_profile=args.bracket_profile)
    config=r2s.load_config(args.camera_config)
    side=config["cameras"]["side"]
    spec.worldbody.add_camera(
        name="phase6_legacy_side",
        pos=list(side["pos_m"]),
        quat=list(side["quat_wxyz"]),
        fovy=float(side["fovy_deg"]))
    model=spec.compile()
    ids=[]
    for name in NAMES:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid<0:
            raise RuntimeError("Missing joint in rendered source URDF: "+name)
        ids.append(int(model.jnt_qposadr[jid]))
    for cam in (wrist.CAMERA,wrist.MOUNT_INSPECTION_CAMERA,"phase6_legacy_side"):
        if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,cam)<0:
            raise RuntimeError("Missing old camera "+cam)
    model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),args.width)
    model.vis.global_.offheight=max(int(model.vis.global_.offheight),args.height)
    data=mujoco.MjData(model)
    args.out.mkdir(parents=True,exist_ok=False)
    frame_dir=args.out/"frames"
    frame_dir.mkdir()
    scene_option=mujoco.MjvOption()
    # Same wrist collision-proxy filtering as old render_wrist_camera_rollout.py.
    scene_option.geomgroup[3]=0
    # This script reads the old camera pipeline; it does not add any
    # changing model physics, visual-only floor or virtual jaw camera.
    count=0
    report=[]
    with mujoco.Renderer(model,width=args.width,height=args.height) as renderer:
        for row_idx in range(0,len(rows),args.stride):
            row=rows[row_idx]
            positions=np.array([math.radians(float(row[k])) for k in keys])
            if not np.isfinite(positions).all():
                raise ValueError("Non-finite recorded pose")
            data.qpos[ids]=positions
            mujoco.mj_forward(model,data)
            views=[]
            for name in ("phase6_legacy_side",wrist.CAMERA,wrist.MOUNT_INSPECTION_CAMERA):
                renderer.update_scene(data,camera=name,scene_option=scene_option)
                views.append(Image.fromarray(renderer.render().copy()))
            panel=Image.new("RGB",(3*args.width,args.height+30),(25,30,40))
            for i,im in enumerate(views):
                panel.paste(im,(i*args.width,30))
            draw=ImageDraw.Draw(panel)
            draw.text((8,8),f"SIDE (config example) | t={float(row['time_s']):.3f}s",fill="white")
            draw.text((args.width+8,8),"WRIST (original mount optic)",fill="white")
            draw.text((2*args.width+8,8),"WRIST RIG INSPECTION (old)",fill="white")
            panel.save(frame_dir/f"frame_{count:04d}.png")
            jaw_fov=jaw_centers_in_wrist_view(model,data,args.width,args.height)
            report.append({"source_row":row_idx,"time_s":float(row["time_s"]),
                           "jaw_geom_centers_in_wrist_fov":len(jaw_fov)})
            count+=1
    video_status="not_requested"
    if args.video:
        ffmpeg=shutil.which("ffmpeg")
        if ffmpeg:
            cmd=[ffmpeg,"-hide_banner","-loglevel","error","-y",
                 "-framerate",str(args.fps),"-i",
                 str(frame_dir/"frame_%04d.png"),"-c:v","libx264",
                 "-pix_fmt","yuv420p",str(args.out/"phase6_legacy_three_views.mp4")]
            subprocess.run(cmd,check=True)
            video_status="created"
        else:
            video_status="ffmpeg_missing_png_frames_saved"
    metadata={"mode":"OFFLINE_VISUALIZATION_NOT_PHYSICS",
        "source_trace":str(source),"frames":count,"video_status":video_status,
        "wrist_camera_source":"wrist_camera_sim.install, original camera optics",
        "inspection_camera_source":"wrist_camera_sim.MOUNT_INSPECTION_CAMERA",
        "side_camera_source":str(args.camera_config),
        "side_camera_is_calibrated":bool(config["calibrated"]),
        "bracket_profile":args.bracket_profile,
        "no_physical_control_or_scene_changes":True,
        "jaw_nominal_fov_diagnostics":report,
        "limitation":"Joint CSV playback cannot establish mechanical stability, and original camera optics may not show the fingertips"}
    (args.out/"render_summary.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:metadata[k] for k in (
        "mode","frames","video_status","bracket_profile")},indent=2))
    print("Output:",args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument("--camera-config",type=Path,
                   default=Path(__file__).with_name("workcell.example.json"))
    p.add_argument("--bracket-profile",choices=("legacy","ruler_15_35_35"),
                   default="ruler_15_35_35")
    p.add_argument("--width",type=int,default=480)
    p.add_argument("--height",type=int,default=360)
    p.add_argument("--stride",type=int,default=100)
    p.add_argument("--fps",type=int,default=5)
    p.add_argument("--video",action="store_true")
    a=p.parse_args()
    if a.width<128 or a.height<128 or a.stride<=0 or a.fps<=0:
        p.error("invalid rendering arguments")
    render(a)


if __name__=="__main__":
    main()
