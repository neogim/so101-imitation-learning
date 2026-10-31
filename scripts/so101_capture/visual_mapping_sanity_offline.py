#!/usr/bin/env python3
"""Create uncalibrated side-video versus hypothetical URDF pose mosaics offline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from audit_joint_zero_direction_episode import axis_angle, parse_urdf, rpy_matrix
HYPOTHESES = {
    "shoulder+_elbow+": np.array([1, 1, 1, 1, 1], dtype=float),
    "shoulder-_elbow+": np.array([1, -1, 1, 1, 1], dtype=float),
    "shoulder+_elbow-": np.array([1, 1, -1, 1, 1], dtype=float),
    "shoulder-_elbow-": np.array([1, -1, -1, 1, 1], dtype=float),
}


def motor_to_urdf_degrees(motor_degrees: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return np.asarray(motor_degrees, dtype=float) * signs


def skeleton(joints: list[dict], degrees: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    points = [transform[:3, 3].copy()]
    for spec, degree in zip(joints, degrees):
        origin = np.eye(4)
        origin[:3, :3] = rpy_matrix(" ".join(map(str, spec["origin_rpy"])))
        origin[:3, 3] = spec["origin_xyz"]
        transform = transform @ origin
        points.append(transform[:3, 3].copy())
        rotation = np.eye(4)
        rotation[:3, :3] = axis_angle(np.asarray(spec["axis_local"]), math.radians(float(degree)))
        transform = transform @ rotation
    points.append(transform[:3, 3] + transform[:3, :3] @ np.array([0.0, -0.08, 0.0]))
    return np.asarray(points)


def render_pose(joints: list[dict], q_urdf: np.ndarray, title: str, destination: Path) -> None:
    points = skeleton(joints, q_urdf)
    fig = plt.figure(figsize=(4, 4))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(points[:, 0], points[:, 1], points[:, 2], "o-", linewidth=4)
    axis.set(xlim=(-0.35, 0.15), ylim=(-0.25, 0.25), zlim=(-0.15, 0.35))
    axis.view_init(elev=20, azim=-55)
    axis.set_title(title, fontsize=8)
    fig.text(0.5, 0.01, "fixed virtual camera; NOT calibrated to video", ha="center", color="red", fontsize=7)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.dataset.resolve() in args.output.resolve().parents:
        parser.error("output must be new and outside dataset")
    info = json.loads((args.dataset / "meta/info.json").read_text())
    table = pq.read_table(next((args.dataset / "data").rglob("*.parquet")), columns=["episode_index", "frame_index", "observation.state"])
    rows = {(int(row["episode_index"]), int(row["frame_index"])): row for row in table.to_pylist()}
    episode_meta = pq.read_table(next((args.dataset / "meta/episodes").rglob("*.parquet"))).to_pylist()
    meta = next(row for row in episode_meta if int(row["episode_index"]) == args.episode)
    key = "observation.images.side"
    video_start = int(round(float(meta[f"videos/{key}/from_timestamp"]) * float(info["fps"])))
    video = args.dataset / info["video_path"].format(
        video_key=key,
        chunk_index=int(meta[f"videos/{key}/chunk_index"]),
        file_index=int(meta[f"videos/{key}/file_index"]),
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("side video could not be opened")
    args.output.mkdir(parents=True)
    joints = parse_urdf(args.urdf)
    index = []
    for frame in args.frames:
        row = rows[(args.episode, frame)]
        state = np.asarray(row["observation.state"][:5], dtype=float)
        capture.set(cv2.CAP_PROP_POS_FRAMES, video_start + frame)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode frame {frame}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        source = Image.fromarray(rgb).resize((480, 360))
        canvas = Image.new("RGB", (480 * 5, 420), "white")
        canvas.paste(source, (0, 60))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"recorded side | ep{args.episode} frame {frame} | State motor deg", fill="black")
        draw.text((8, 30), " ".join(f"{x:+.1f}" for x in state), fill="black")
        for column, (name, signs) in enumerate(HYPOTHESES.items(), start=1):
            q = motor_to_urdf_degrees(state, signs)
            temporary = args.output / f".frame_{frame}_{name}.png"
            render_pose(joints, q, f"{name}\nqU=s*qM, b=0", temporary)
            rendered = Image.open(temporary).convert("RGB").resize((480, 420))
            canvas.paste(rendered, (column * 480, 0))
            temporary.unlink()
        filename = f"frame_{frame:04d}_mapping_mosaic.png"
        canvas.save(args.output / filename)
        index.append({"episode": args.episode, "frame": frame, "video_frame": video_start + frame, "file": filename})
    capture.release()
    (args.output / "index.json").write_text(json.dumps({
        "classification": "VISUAL_SCREENING",
        "camera": "fixed_virtual_camera_not_calibrated_to_video",
        "items": index,
        "warning": "Do not rank mappings by pixel alignment or infer physical joint angles.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
