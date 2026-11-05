#!/usr/bin/env python3
"""Offline fixed-view comparison of SO-101 joint mapping hypotheses.

The generated virtual free-camera viewpoints are IDENTICAL for both candidate
joint maps. They are NOT calibrated to the real Side or Wrist cameras.
No real robot or camera hardware is used; original data is read-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

import real_to_sim as r2s

VIEWS = {
    "oblique": {"lookat": [0.13, 0.0, 0.10], "distance": 0.70,
                "azimuth": 135, "elevation": -24},
    "opposite": {"lookat": [0.13, 0.0, 0.10], "distance": 0.70,
                 "azimuth": -45, "elevation": -24},
    "overhead": {"lookat": [0.13, 0.0, 0.10], "distance": 0.70,
                 "azimuth": 135, "elevation": -80},
}
SIGN_SETS = {"plus": (1, 1, 1, 1, 1), "j2j3_minus": (1, -1, -1, 1, 1)}


def viewpoint(name: str):
    spec = VIEWS[name]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = spec["lookat"]
    camera.distance = spec["distance"]
    camera.azimuth = spec["azimuth"]
    camera.elevation = spec["elevation"]
    return camera


def mapping(base: dict, hypothesis: str) -> dict:
    config = copy.deepcopy(base)
    config["joints"]["sign"] = dict(zip(r2s.JOINTS, SIGN_SETS[hypothesis]))
    config["calibrated"] = False
    return config


def actual_joint_angles(state: list[float], config: dict) -> dict[str, float]:
    return {name: math.degrees(value) for name, value in
            r2s.joint_radians(state, config).items()}


def title_image(frame: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (frame.width, frame.height + 28), (30, 30, 30))
    canvas.paste(frame.convert("RGB"), (0, 28))
    ImageDraw.Draw(canvas).text((8, 7), label, fill="white")
    return canvas


def compare(root: Path, output: Path, episode: int, frames: list[int],
            width: int, height: int, urdf: Path, config_path: Path,
            views: list[str]) -> dict:
    if not frames or any(f < 0 for f in frames) or len(set(frames)) != len(frames):
        raise ValueError("frames must be unique non-negative indices")
    if width < 64 or height < 64:
        raise ValueError("width/height must be >=64")
    if output.exists():
        raise ValueError(f"Refusing to overwrite: {output}")
    root = root.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the source dataset")
    config = r2s.load_config(config_path.expanduser().resolve(strict=True))
    ep = r2s.read_episode(root, episode, stride=1, max_frames=max(frames)+1)
    rows_by_frame = {row["frame_index"]: row for row in ep["rows"]}
    if not set(frames).issubset(rows_by_frame):
        raise ValueError("Requested frame exceeds episode length")
    model = r2s.build_model(urdf, config)
    data = mujoco.MjData(model)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "mode": "OFFLINE_UNCALIBRATED_FREE_VIEW_MAPPING_COMPARISON",
        "episode": episode, "views": {k: VIEWS[k] for k in views},
        "candidate_signs": {k: list(v) for k, v in SIGN_SETS.items()},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "state_is_observed_not_action": True,
        "note": "Free views are identical across hypotheses, but are NOT real camera calibration. "+ 
                "Joint zero offsets assumed from config. Do not infer physical mapping from appearance alone.",
        "frames": [],
    }
    cameras = {view: viewpoint(view) for view in views}
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for frame in frames:
            row = rows_by_frame[frame]
            candidates = {}
            for name in SIGN_SETS:
                hypothesis = mapping(config, name)
                r2s.set_pose(model, data, row["state"], hypothesis)
                candidates[name] = {
                    "joint_angles_deg": actual_joint_angles(row["state"], hypothesis),
                    **r2s.diagnostics(model, data),
                }
                for view in views:
                    renderer.update_scene(data, camera=cameras[view])
                    rgb = renderer.render().copy()
                    candidates[name][view] = Image.fromarray(rgb)
            for view in views:
                left = title_image(candidates["plus"][view], "+++++ | uncalibrated")
                right = title_image(candidates["j2j3_minus"][view], "+--++ | uncalibrated")
                canvas = Image.new("RGB", (width*2, height+28))
                canvas.paste(left, (0, 0))
                canvas.paste(right, (width, 0))
                canvas.save(output / f"frame_{frame:06d}_{view}_side_by_side.png")
            result["frames"].append({
                "frame_index": frame, "timestamp": row["timestamp"],
                "state_deg": row["state"],
                "hypotheses": {
                    name: {k: v for k, v in item.items() if k not in views}
                    for name, item in candidates.items()
                },
            })
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 80, 160, 240, 320])
    parser.add_argument("--views", nargs="+", choices=VIEWS.keys(), default=list(VIEWS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("workcell.example.json"))
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    args = parser.parse_args()
    try:
        report = compare(args.dataset_root, args.output, args.episode, args.frames,
                         args.width, args.height, args.urdf, args.config, args.views)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"saved_frames": len(report["frames"]), "output": str(args.output),
                      "mode": report["mode"]}, indent=2))


if __name__ == "__main__":
    main()
