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
import demo_real2sim_muhi_cup_reach as scene
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

    if args.scene=="experiment16":
        # Reuse the SAME saved visual scene as compare_dls_sac_table_guard.py:
        # white follower, pale backdrop, studio light, table, Muhi, cup,
        # measured side camera, original wrist camera.
        args.physical_objects=False
        args.wrist_camera=True
        (model,data,_qids,_dids,_low,_high,_tcp,side_camera,goals,
         *_extras)=scene.create_scene(args)
        overview=scene.camera_for(model,goals,False,side_camera)
        scene_source="experiment16: demo_real2sim_muhi_cup_reach.create_scene"
        calibrated=True
    else:
        spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
        spec.compiler.discardvisual=False
        wrist.install(spec,bracket_profile=args.bracket_profile)
        config=r2s.load_config(args.camera_config)
        side=config["cameras"]["side"]
        spec.worldbody.add_camera(
            name="phase6_legacy_side",
            pos=list(side["pos_m"]),
            quat=list(side["quat_wxyz"]),
            fovy=float(side["fovy_deg"]))
        model=spec.compile()
        data=mujoco.MjData(model)
        scene_source="bare URDF (no workcell or lighting)"
        calibrated=bool(config["calibrated"])
        overview=None
    ids=[]
    for name in NAMES:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid<0:
            raise RuntimeError("Missing joint in rendered source URDF: "+name)
        ids.append(int(model.jnt_qposadr[jid]))
    for cam in ((wrist.CAMERA,) if args.scene=="experiment16" else (wrist.CAMERA,wrist.MOUNT_INSPECTION_CAMERA,"phase6_legacy_side")):
        if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,cam)<0:
            raise RuntimeError("Missing old camera "+cam)
    model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),args.width)
    model.vis.global_.offheight=max(int(model.vis.global_.offheight),args.height)
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
    if overview is None:
        # Bare fallback only; the experiment16 mode uses its real saved targets.
        data.qpos[ids]=[math.radians(float(rows[0][k])) for k in keys]
        mujoco.mj_forward(model,data)
        gripper_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"gripper_link")
        if gripper_id<0:
            raise RuntimeError("Original gripper_link not compiled")
        anchor=np.asarray(data.xpos[gripper_id],dtype=float).copy()
        overview=scene.camera_for(model,{"muhi":anchor,"cup":anchor},False,"phase6_legacy_side")
    reference=np.array([math.radians(float(rows[0][n+"_target_deg"])) for n in NAMES])
    if not np.isfinite(reference).all():
        raise ValueError("Non-finite reference target")
    # Historic compare_dls_sac_table_guard.make_video() arrangement:
    # each column is a condition, top row overview, bottom row mounted wrist.
    # Here REF is the fixed joint target and ACTUAL is the physics CSV.
    # Same MuJoCo model, same renderer, same two camera objects, 2x2.
    from PIL import ImageDraw
    with mujoco.Renderer(model,width=args.width,height=args.height) as renderer:
        for row_idx in range(0,len(rows),args.stride):
            row=rows[row_idx]
            positions=np.array([math.radians(float(row[k])) for k in keys])
            if not np.isfinite(positions).all():
                raise ValueError("Non-finite recorded pose")
            canvas=Image.new("RGB",(2*args.width,2*(args.height+48)),(20,29,39))
            draw=ImageDraw.Draw(canvas)
            for col,(column_label,q) in enumerate((
                    ("REFERENCE TARGET",reference),("PHYSICAL ACTUAL CSV",positions))):
                data.qpos[ids]=q
                mujoco.mj_forward(model,data)
                for rr,(cam,view_label) in enumerate((
                        (overview,"OVERVIEW"),(wrist.CAMERA,"WRIST"))):
                    renderer.update_scene(data,camera=cam,scene_option=scene_option)
                    im=Image.fromarray(renderer.render().copy())
                    x,y=col*args.width,rr*(args.height+48)
                    canvas.paste(im,(x,y+48))
                    draw.text((x+8,y+10),column_label+" | "+view_label,fill="white")
                    draw.text((x+8,y+28),f"t={float(row['time_s']):.3f}s",fill="white")
            canvas.save(frame_dir/f"frame_{count:04d}.png")
            data.qpos[ids]=positions
            mujoco.mj_forward(model,data)
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
                 "-pix_fmt","yuv420p",str(args.out/"phase6_dls_sac_style_four_views.mp4")]
            subprocess.run(cmd,check=True)
            video_status="created"
        else:
            video_status="ffmpeg_missing_png_frames_saved"
    metadata={"mode":"OFFLINE_VISUALIZATION_NOT_PHYSICS",
        "source_trace":str(source),"frames":count,"video_status":video_status,
        "reference_render_source":"compare_dls_sac_table_guard.make_video (2026-11-14 experiment 16)",
        "layout":"2 columns (target, physical actual) x 2 rows (overview, wrist)",
        "overview_camera_source":"demo_real2sim_muhi_cup_reach.camera_for",
        "scene_source":scene_source,
        "overview_lookat":"saved Episode-4 Muhi/cup goals" if args.scene=="experiment16" else "initial gripper_link",
        "wrist_camera_source":"wrist_camera_sim.install, original camera optics",
        "side_camera_source":str(args.extrinsics) if args.scene=="experiment16" else str(args.camera_config),
        "side_camera_is_calibrated":calibrated,
        "bracket_profile":args.bracket_profile,
        "no_physical_control_changes":True,
        "visual_scene_from_original_experiment16":args.scene=="experiment16",
        "jaw_nominal_fov_diagnostics":report,
        "limitation":"CSV playback cannot establish mechanical stability. Saved Episode-4 workcell is visual context only and was not simulated during Phase6A physics."}
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
    p.add_argument("--scene",choices=("experiment16","bare"),default="experiment16")
    p.add_argument("--dataset-root",type=Path,default=scene.DATASET)
    p.add_argument("--episode",type=int,default=4)
    p.add_argument("--object-layout-json",type=Path,default=scene.OBJECTS)
    p.add_argument("--cup-pose-json",type=Path,default=scene.CUP_POSE)
    p.add_argument("--extrinsics",type=Path,default=scene.EXTRINSICS)
    p.add_argument("--intrinsics",type=Path,default=scene.INTRINSICS)
    p.add_argument("--robot-joint-offsets-json",type=Path)
    p.add_argument("--muhi-clearance-mm",type=float,default=75)
    p.add_argument("--cup-clearance-mm",type=float,default=45)
    p.add_argument("--wrist-mount-pos",type=float,nargs=3,default=wrist.DEFAULT_MOUNT_POS)
    p.add_argument("--wrist-mount-rpy-deg",type=float,nargs=3,default=wrist.DEFAULT_MOUNT_RPY_DEG)
    p.add_argument("--wrist-fovy",type=float,default=wrist.DEFAULT_FOVY_DEG)
    p.add_argument("--wrist-look-at",type=float,nargs=3,default=wrist.DEFAULT_LOOK_AT)
    p.add_argument("--camera-config",type=Path,
                   default=Path(__file__).with_name("workcell.example.json"))
    p.add_argument("--bracket-profile",choices=("legacy","ruler_15_35_35"),
                   default="legacy")
    p.add_argument("--width",type=int,default=640)
    p.add_argument("--height",type=int,default=480)
    p.add_argument("--stride",type=int,default=100)
    p.add_argument("--fps",type=int,default=5)
    p.add_argument("--video",action="store_true")
    a=p.parse_args()
    if a.width<128 or a.height<128 or a.stride<=0 or a.fps<=0:
        p.error("invalid rendering arguments")
    render(a)


if __name__=="__main__":
    main()
