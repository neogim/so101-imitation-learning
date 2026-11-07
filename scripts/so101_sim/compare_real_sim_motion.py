#!/usr/bin/env python3
"""Offline recorded-side-video and MuJoCo FK side-by-side comparison.

There are two deliberately separate inputs:
- poses: 5 static photos from the ArUco manifest (no implied temporal order)
- motion: recorded LeRobot episode Side MP4 and logged joint states
The motion mode never executes a policy or drives hardware.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import compare_id1_eye_to_hand as comparison
import real_to_sim as r2s

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRINSICS = ROOT / "assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
DEFAULT_URDF = ROOT / "data/urdf/so101/so101_new_calib.urdf"


def sample_evenly(samples, count):
    if count < 1:
        raise ValueError("count must be positive")
    if not samples:
        raise ValueError("No valid static poses")
    n = min(count, len(samples))
    return [samples[i] for i in sorted(set(round(float(x)) for x in np.linspace(0, len(samples)-1, n)))]


def make_calibrated_model(extrinsics, intrinsics, urdf):
    import mujoco

    position, quat = comparison.mujoco_camera_pose(extrinsics["T_base_camera"])
    fovy = comparison.fovy_from_intrinsics(float(intrinsics["fy"]), int(intrinsics["height"]))
    spec = mujoco.MjSpec.from_file(str(urdf.resolve()))
    spec.compiler.discardvisual = False
    shift = comparison.workcell_shift_left_front(extrinsics["T_base_camera"], .08, .10)
    extra = comparison.workcell_shift_left_front(extrinsics["T_base_camera"], .04, .05)
    comparison.add_approximate_workcell(spec, shift, extra, 15.)
    comparison.add_backdrop(spec, extrinsics["T_base_camera"])
    comparison.add_simple_studio_light(spec)
    camera = "r2s_side_calibrated_candidate"
    spec.worldbody.add_camera(name=camera, pos=position, quat=quat, fovy=fovy)
    model = spec.compile()
    comparison.apply_white_follower_preview(model)
    return model, camera


def comparison_panel(real, sim, subtitle=""):
    width, height = 640, 480
    out = Image.new("RGB", (1280, 508), (25, 25, 25))
    out.paste(real.convert("RGB").resize((width, height)), (0, 28))
    out.paste(sim.convert("RGB").resize((width, height)), (width, 28))
    label = ImageDraw.Draw(out)
    label.text((8, 7), f"REAL: Side {subtitle}", fill="white")
    label.text((width+8, 7), "SIM: preliminary calibrated Side / FK", fill="white")
    return out


def run_poses(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    valid = [s for s in manifest["samples"] if any(int(m["id"]) == 1 for m in s["markers"])]
    poses = sample_evenly(valid, args.count)
    if args.output.exists():
        raise ValueError("Output exists; use a new directory")
    args.output.mkdir(parents=True, exist_ok=False)
    panels, results = [], []
    for i, selected in enumerate(poses):
        frame = int(selected["frame_index"])
        folder = args.output / f"pose_{i+1:02d}_frame_{frame:06d}"
        comparison.render_one(args.manifest, args.extrinsics, args.intrinsics,
                              args.urdf, folder, sample_number=frame)
        path = folder / "real_vs_mujoco.png"
        with Image.open(path) as picture:
            panels.append(picture.convert("RGB"))
        results.append({"frame_index": frame, "image": str(path)})
    combined = Image.new("RGB", (panels[0].width, len(panels)*panels[0].height))
    for i, picture in enumerate(panels):
        combined.paste(picture, (0, i*picture.height))
    output = args.output / "five_pose_comparison.png"
    combined.save(output)
    (args.output / "report.json").write_text(json.dumps({
        "mode": "STATIC_POSE_COMPARISON_NOT_CONTINUOUS_MOTION",
        "calibration": "provisional", "frames": results,
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return output


def side_video_source(root, episode):
    info, metadata = episode["info"], episode["metadata"]
    key = "observation.images.side"
    if key not in info.get("features", {}):
        raise ValueError("Dataset has no observation.images.side")
    if info.get("codebase_version", "").startswith("v3"):
        if metadata is None:
            raise ValueError("v3 episode metadata missing")
        prefix = f"videos/{key}/"
        keys = [prefix+s for s in ("chunk_index","file_index","from_timestamp","to_timestamp")]
        if not all(k in metadata for k in keys):
            raise ValueError("Missing v3 Side video offsets")
        path = root / info["video_path"].format(
            video_key=key, chunk_index=int(metadata[keys[0]]),
            file_index=int(metadata[keys[1]]))
        start_index = round(float(metadata[keys[2]])*float(info["fps"]))
    else:
        found = list(root.glob(f"videos/{key}/chunk-*/episode_{episode['episode']:06d}.mp4"))
        if len(found) != 1:
            raise ValueError(f"Expected 1 Side video for episode; got {len(found)}")
        path, start_index = found[0], 0
    if not path.is_file():
        raise FileNotFoundError(f"Missing Side video: {path}")
    return path, start_index


def read_frames_once(path, indices):
    import av
    needed, selected = set(indices), {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in needed:
                selected[i] = frame.to_image()
                if len(selected) == len(needed):
                    break
    if set(selected) != needed:
        raise ValueError(f"Missing Side frames: {sorted(needed - set(selected))[:5]}")
    return selected


def run_motion(args):
    import av
    import mujoco

    if args.output.exists():
        raise ValueError("Output exists; use a new directory")
    if args.stride <= 0 or args.max_frames <= 0:
        raise ValueError("stride and max-frames must be positive")
    dataset = args.dataset_root.resolve(strict=True)
    if args.output.resolve() == dataset or dataset in args.output.resolve().parents:
        raise ValueError("Output must be outside the raw dataset")
    ep = r2s.read_episode(dataset, args.episode, args.stride, args.max_frames)
    video_path, start = side_video_source(dataset, ep)
    frame_ids = [start + int(x["frame_index"]) for x in ep["rows"]]
    reals = read_frames_once(video_path, frame_ids)

    extrinsics = json.loads(args.extrinsics.read_text(encoding="utf-8"))
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    if (intrinsics.get("width"), intrinsics.get("height")) != (640, 480):
        raise ValueError("Expected 640x480 Side candidate intrinsics")
    model, camera = make_calibrated_model(extrinsics, intrinsics, args.urdf)
    data = mujoco.MjData(model)
    config = {"joints": {"sign": {name: 1 for name in r2s.JOINTS},
                         "offset_deg": {name: 0. for name in r2s.JOINTS}}}
    source_fps = float(ep["info"]["fps"])
    fps_out = source_fps / args.stride
    if not (0 < fps_out < 121):
        raise ValueError("Invalid source FPS / stride")
    # Fixed integer frame rate is sufficient for offline visual inspection;
    # original source frame indices are retained in the overlay and report.
    encoded_fps = max(1, round(fps_out))
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "real_vs_sim_episode.mp4"
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=encoded_fps)
        stream.width, stream.height = 1280, 508
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "23"}
        with mujoco.Renderer(model, width=640, height=480) as renderer:
            for row, source_index in zip(ep["rows"], frame_ids):
                r2s.set_pose(model, data, row["state"], config)
                renderer.update_scene(data, camera=camera)
                simulation = Image.fromarray(renderer.render().copy()).convert("RGB")
                panel = comparison_panel(reals[source_index], simulation,
                                         subtitle=f"ep={args.episode} frame={row['frame_index']}")
                rgb = av.VideoFrame.from_ndarray(np.asarray(panel), format="rgb24")
                for packet in stream.encode(rgb):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    (args.output / "report.json").write_text(json.dumps({
        "mode": "RECORDED_REAL_SIDE_VS_MUJOCO_FK_NOT_PHYSICS_REPLAY",
        "output": str(output), "video_source": str(video_path),
        "episode": args.episode, "stride": args.stride, "frames": len(ep["rows"]),
        "source_fps": source_fps, "video_fps": encoded_fps,
        "limitations": [
            "The Side calibration dates to 2026-11-07 and may not match the recording setup.",
            "Joint angles are replayed by FK; normalized gripper is not mapped.",
            "Static illustrative Muhi/cup do not physically respond to the arm.",
            "There is no ACT inference and no hardware connection.",
        ],
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="mode", required=True)
    for mode in ("poses", "motion"):
        sub = actions.add_parser(mode)
        sub.add_argument("--extrinsics", type=Path, required=True)
        sub.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
        sub.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
        sub.add_argument("--output", type=Path, required=True)
        if mode == "poses":
            sub.add_argument("--manifest", type=Path, required=True)
            sub.add_argument("--count", type=int, default=5)
        else:
            sub.add_argument("--dataset-root", type=Path, required=True)
            sub.add_argument("--episode", type=int, default=0)
            sub.add_argument("--stride", type=int, default=5)
            sub.add_argument("--max-frames", type=int, default=60)
    args = parser.parse_args()
    print("Saved:", run_poses(args) if args.mode == "poses" else run_motion(args))


if __name__ == "__main__":
    main()
