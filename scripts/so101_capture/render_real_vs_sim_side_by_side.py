#!/usr/bin/env python3
"""Real side video vs MuJoCo URDF replay: illuminated, adjustable camera (offline only).

This is a *visual* comparison. Camera extrinsics and joint mapping are uncalibrated.
No robot connection, motor command, URDF edits or contact simulation is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "so101_sim"))
import real_to_sim as r2s  # noqa: E402

SIDE = "r2s_side"


def recorded_video(dataset: Path, episode: int):
    """Map a v3 episode to its timestamped side-video shard."""
    info = json.loads((dataset / "meta" / "info.json").read_text())
    metas = sorted((dataset / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not metas:
        raise FileNotFoundError("Missing v3 episode metadata parquet")
    tab = pd.concat((pd.read_parquet(p) for p in metas), ignore_index=True)
    matches = tab[tab["episode_index"] == episode]
    if len(matches) != 1:
        raise ValueError(f"Expected one episode {episode} metadata row; got {len(matches)}")
    row = matches.iloc[0]
    key = "observation.images.side"
    video = dataset / info["video_path"].format(
        video_key=key,
        chunk_index=int(row[f"videos/{key}/chunk_index"]),
        file_index=int(row[f"videos/{key}/file_index"]),
    )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open recorded video {video}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        cap.release()
        raise ValueError("Video FPS is invalid")
    frame_start = int(round(float(row[f"videos/{key}/from_timestamp"]) * video_fps))
    return cap, video, frame_start, video_fps


def frame_rgb(cap, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Could not decode recorded video at frame {idx}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def brighten_sim(model, *, ambient: float):
    """Change only MuJoCo *visual* illumination, never physical properties."""
    model.vis.headlight.ambient[:] = (ambient, ambient, ambient)
    model.vis.headlight.diffuse[:] = (0.95, 0.95, 0.95)
    model.vis.headlight.specular[:] = (0.25, 0.25, 0.25)
    # Do not assign model.vis.rgba.skybox: this field does not exist on
    # mjVisual/Rgba in some supported MuJoCo Python bindings. Background
    # configuration is intentionally unchanged; headlight brightens the arm.


def panel(real, sim, title: str, fovy: float) -> Image.Image:
    height = 360
    def scale(rgb):
        img = Image.fromarray(rgb).convert("RGB")
        return img.resize((round(img.width * height / img.height), height))
    real, sim = scale(real), scale(sim)
    width = real.width + sim.width + 48
    output = Image.new("RGB", (width, height + 94), "#ffffff")
    d = ImageDraw.Draw(output)
    d.text((16, 10), title, fill="#202b38")
    d.text((16, 33), "LEFT: recorded REAL camera   RIGHT: MuJoCo (lit visualization)", fill="#254e77")
    d.text((16, 54), f"Sim vertical FOV={fovy:.1f} deg (approximate camera pose; NOT calibrated)", fill="#74592b")
    output.paste(real, (16, 88))
    output.paste(sim, (32 + real.width, 88))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 355, 374, 405, 494])
    parser.add_argument("--sim-fovy-deg", type=float, default=None,
                        help="Override sim vertical field-of-view; default uses config.")
    parser.add_argument("--ambient", type=float, default=0.75,
                        help="Visual headlight ambient (0..1), default 0.75.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory exists; preserve previous evidence")
    if not 0 <= args.ambient <= 1:
        parser.error("--ambient must be between 0 and 1")
    if args.sim_fovy_deg is not None and not 10 <= args.sim_fovy_deg <= 150:
        parser.error("--sim-fovy-deg must be 10..150")
    dataset, urdf = args.dataset.resolve(strict=True), args.urdf.resolve(strict=True)
    output = args.output.resolve()
    if output == dataset or dataset in output.parents:
        parser.error("Output must be outside the source dataset")
    config = json.loads(args.config.read_text())
    model = r2s.build_model(urdf, config)
    camid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, SIDE)
    if camid < 0:
        raise ValueError(f"Missing sim camera {SIDE}")
    original_fovy = float(model.cam_fovy[camid])
    if args.sim_fovy_deg is not None:
        model.cam_fovy[camid] = args.sim_fovy_deg
    fovy = float(model.cam_fovy[camid])
    brighten_sim(model, ambient=args.ambient)
    data = mujoco.MjData(model)
    ep = r2s.read_episode(dataset, args.episode, stride=1, max_frames=100000)
    recorded = {int(row["frame_index"]): row for row in ep["rows"]}
    cap, video_path, start, video_fps = recorded_video(dataset, args.episode)
    output.mkdir(parents=True, exist_ok=False)
    panels = []
    summary_rows = []
    try:
        for frame in args.frames:
            if frame not in recorded:
                raise ValueError(f"Frame {frame} not in episode {args.episode}")
            row = recorded[frame]
            r2s.set_pose(model, data, row["state"], config)
            sim = r2s.render_image(model, data, SIDE, 640, 480)
            real = frame_rgb(cap, start + frame)
            title = f"Episode {args.episode} frame {frame} / t={row['timestamp']:.2f}s"
            img = panel(real, sim, title, fovy)
            img.save(output / f"compare_frame_{frame:04d}.png")
            panels.append(img)
            summary_rows.append([frame, row["timestamp"], *row["state"]])
    finally:
        cap.release()
    if panels:
        thumb_width = min(1000, max(im.width for im in panels))
        thumbs = []
        for im in panels:
            if im.width > thumb_width:
                im = im.resize((thumb_width, round(im.height * thumb_width / im.width)))
            thumbs.append(im)
        h = sum(im.height for im in thumbs) + 12 * (len(thumbs) - 1)
        sheet = Image.new("RGB", (thumb_width, h), "white")
        y = 0
        for im in thumbs:
            sheet.paste(im, (0, y))
            y += im.height + 12
        sheet.save(output / "real_vs_sim_contact_sheet.png")
    with (output / "frame_summary.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_s", *r2s.STATE_NAMES])
        writer.writerows(summary_rows)
    summary = {
        "status": "READ_ONLY_UNCALIBRATED_VISUAL_COMPARISON",
        "dataset": str(dataset), "video": str(video_path),
        "episode": args.episode, "frames": args.frames,
        "video_fps": video_fps, "video_start": start,
        "original_sim_fovy_deg": original_fovy,
        "used_sim_fovy_deg": fovy, "ambient": args.ambient,
        "warnings": [
            "Visual settings change only light/background and optionally FOV; physical dynamics unchanged.",
            "Camera position and orientation are STILL the approximate side_baseline config.",
            "Camera mismatch alone can mimic joint sign and zero errors; do not decide those from these screenshots.",
            "Joint signs/offsets remain those in config, not independently verified.",
            "Do not treat this as physical collision, calibrated TCP or real grasp validation."
        ]
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
