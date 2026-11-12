#!/usr/bin/env python3
"""Render saved SO-101 trajectories from an optionally collidable wrist camera.

Off-line only; no motor/camera hardware. The original SAC/J5 rollout and source
URDF are read-only. Default mode is kinematic and produces contact *detection*.
--mode physics is an experimental PD/mj_step replay (not an ACT/SAC evaluation).
The saved Muhi/cup are static preview geoms even when contact masks are enabled.
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

import demo_real2sim_muhi_cup_reach as scene
import wrist_camera_sim as wrist

JOINT_COLS = tuple(f"joint_{i}" for i in range(1, 6))


def load_rollout(path, policy):
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows or not all(k in rows[0] for k in JOINT_COLS):
        raise ValueError("Expected nonempty saved J5-ablation rollout.csv")
    if "policy" in rows[0]:
        rows = [row for row in rows if row["policy"] == policy]
    if not rows:
        raise ValueError(f"No frames for policy={policy}")
    q = np.asarray([[float(r[k]) for k in JOINT_COLS] for r in rows])
    if not np.isfinite(q).all():
        raise ValueError("Non-finite recorded joint values")
    return rows, q


def physical_track(model, data, qids, dids, target, steps, kp, kd, torque):
    """Bounded PD with bias compensation: deliberately no teleport after start."""
    for _ in range(steps):
        err = target - data.qpos[qids]
        qvel = data.qvel[dids]
        demand = kp * err - kd * qvel + data.qfrc_bias[dids]
        data.qfrc_applied[:] = 0
        data.qfrc_applied[dids] = np.clip(demand, -torque, torque)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("Nonfinite physics state: inspect self-collisions/PD")
    data.qfrc_applied[:] = 0


def run(args):
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")
    if args.stride < 1 or args.substeps < 1 or not 0 < args.torque <= 10:
        raise ValueError("Invalid stride/substeps/torque")
    if args.width < 64 or args.height < 64 or args.fps <= 0:
        raise ValueError("Invalid render resolution/fps")
    rows, q = load_rollout(args.rollout, args.policy)
    args.wrist_camera = True
    (model, data, qids, dids, low, high, tcp, side_camera, goals,
     _cup, _layout, _offset_path, _offset_cfg) = scene.create_scene(args)
    wrist_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in wrist.COLLISION_NAMES
    ]
    if min(wrist_geom_ids) < 0:
        raise ValueError("Wrist geom missing after compile")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, wrist.CAMERA)
    if cid < 0:
        raise ValueError("Wrist RGB camera missing after compile")
    enabled = wrist.enable_environment_contacts(model)
    max_steps = math.ceil(len(rows) / args.stride)
    args.out.mkdir(parents=True, exist_ok=False)
    image_dir = args.out / "wrist_frames"
    image_dir.mkdir()
    side_dir = args.out / "side_frames"
    side_dir.mkdir()
    comparison_dir = args.out / "side_wrist_frames"
    comparison_dir.mkdir()
    report_rows = []
    if args.mode == "physics":
        # The recorded visual home is intentionally outside URDF constraints.
        # Start from the nearest in-range pose; do not claim exact replay.
        data.qpos[qids] = np.clip(q[0], low, high)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    with mujoco.Renderer(model, width=args.width, height=args.height) as renderer:
        for i, row_idx in enumerate(range(0, len(rows), args.stride)):
            target = q[row_idx]
            if args.mode == "kinematic":
                data.qpos[qids] = target
                mujoco.mj_forward(model, data)
            else:
                physical_track(model, data, qids, dids,
                               np.clip(target, low, high),
                               args.substeps, args.kp, args.kd, args.torque)
            contacts = wrist.contact_summary(model, data)
            renderer.update_scene(data, camera=wrist.CAMERA)
            pixels = renderer.render().copy()
            wrist_image = Image.fromarray(pixels)
            wrist_image.save(image_dir / f"wrist_{i:04d}.png")
            # Render the SAME pose through the known side view; crucial for
            # checking whether the camera/brace floats or intersects the arm.
            renderer.update_scene(data, camera=side_camera)
            side_image = Image.fromarray(renderer.render().copy())
            side_image.save(side_dir / f"side_{i:04d}.png")
            comparison = Image.new("RGB", (2 * args.width, args.height + 28), (25, 30, 40))
            comparison.paste(side_image, (0, 28))
            comparison.paste(wrist_image, (args.width, 28))
            draw = ImageDraw.Draw(comparison)
            draw.text((6, 7), f"SIDE | {rows[row_idx].get('stage', '')} | frame {row_idx}", fill="white")
            draw.text((args.width + 6, 7), "WRIST | provisional pose", fill="white")
            comparison.save(comparison_dir / f"side_wrist_{i:04d}.png")
            report_rows.append({
                "frame": i, "source_frame": row_idx,
                "stage": rows[row_idx].get("stage"),
                "wrist_contacts": len(contacts),
                "penetrating_contacts": sum(c["distance_mm"] < 0 for c in contacts),
                "most_negative_distance_mm": min(
                    (c["distance_mm"] for c in contacts), default=None),
                "joint_tracking_error_deg": float(np.degrees(
                    np.linalg.norm(data.qpos[qids] - target))),
                "contacts": contacts,
            })
    if args.save_video:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for --save-video")
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(args.fps), "-i",
            str(image_dir / "wrist_%04d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.out / "wrist_camera.mp4")
        ], check=True)
        for pattern, filename in [
            ("side_%04d.png", "side_camera.mp4"),
            ("side_wrist_%04d.png", "side_wrist_comparison.mp4"),
        ]:
            subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(args.fps), "-i",
                str((side_dir if pattern.startswith("side_") and
                     pattern != "side_wrist_%04d.png" else comparison_dir) / pattern),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(args.out / filename)
            ], check=True)
    summary = {
        "status": "WRIST_CAMERA_PROVISIONAL_SIM",
        "mode": args.mode, "camera": wrist.CAMERA,
        "side_camera": side_camera,
        "views": ["side", "wrist", "side_wrist_comparison"],
        "attachment_link": "gripper_link",
        "fovy_deg": args.wrist_fovy,
        "width": args.width, "height": args.height,
        "mount_pos_m": list(args.wrist_mount_pos),
        "mount_rpy_deg": list(args.wrist_mount_rpy_deg),
        "rig_mass_kg_provisional": 0.025,
        "frames": len(report_rows),
        "frames_with_wrist_contact": sum(r["wrist_contacts"] > 0 for r in report_rows),
        "frames_with_wrist_penetration": sum(
            r["penetrating_contacts"] > 0 for r in report_rows),
        "enabled_static_environment_geoms": enabled,
        "source_rollout": args.rollout.name, "policy": args.policy,
        "limitations": [
            "Camera mount, orientation, mass and FOV are provisional, not calibrated.",
            "Bracket/PCB/lens are collision proxies, not a verified CAD match.",
            "Saved Muhi and cup are static preview geometry; they cannot tumble.",
            "Kinematic mode detects contact but cannot generate contact response.",
            "Physics mode uses capped PD and mj_step; it does not reproduce the learned SAC action semantics.",
            "Recorded home pose violates URDF joint limits; physics mode clips its initial pose.",
            "Initial base/shoulder collision hull artifacts may influence dynamics.",
            "No ACT inference, validated grasp, safe real-world behavior, or hardware control.",
        ],
        "frames_detail": report_rows,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out / "contact_frames.csv").open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame", "source_frame", "stage", "wrist_contacts",
            "penetrating_contacts", "most_negative_distance_mm",
            "joint_tracking_error_deg"])
        w.writeheader()
        for r in report_rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"Saved {max_steps} wrist RGB frames and contact report to {args.out}")
    print("Contact frames:", summary["frames_with_wrist_contact"])
    print("Mode:", args.mode, " | wrist rig dimensions / pose are provisional")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout", type=Path, required=True,
                   help="Saved J5-ablation rollout.csv")
    p.add_argument("--policy", choices=("normal", "j5_fixed"), default="normal")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("kinematic", "physics"), default="kinematic")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=float, default=20)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--substeps", type=int, default=10)
    p.add_argument("--kp", type=float, default=12.)
    p.add_argument("--kd", type=float, default=1.1)
    p.add_argument("--torque", type=float, default=3.)
    p.add_argument("--wrist-mount-pos", type=float, nargs=3,
                   default=(0., 0., -.048), metavar=("X", "Y", "Z"))
    p.add_argument("--wrist-mount-rpy-deg", type=float, nargs=3,
                   default=(0., 0., 0.), metavar=("R", "P", "Y"))
    p.add_argument("--wrist-fovy", type=float, default=75.)
    p.add_argument("--dataset-root", type=Path, default=scene.DATASET)
    p.add_argument("--episode", type=int, default=4)
    p.add_argument("--object-layout-json", type=Path, default=scene.OBJECTS)
    p.add_argument("--cup-pose-json", type=Path, default=scene.CUP_POSE)
    p.add_argument("--extrinsics", type=Path, default=scene.EXTRINSICS)
    p.add_argument("--intrinsics", type=Path, default=scene.INTRINSICS)
    p.add_argument("--urdf", type=Path, default=scene.r2s.DEFAULT_URDF)
    p.add_argument("--robot-joint-offsets-json", type=Path)
    p.add_argument("--muhi-clearance-mm", type=float, default=75.)
    p.add_argument("--cup-clearance-mm", type=float, default=45.)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
