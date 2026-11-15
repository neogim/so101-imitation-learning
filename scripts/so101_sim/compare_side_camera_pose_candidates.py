#!/usr/bin/env python3
"""Offline Side-camera pose candidate comparison, BEFORE joint-sign auditing.

Find already computed 4x4 T_base_camera camera poses in outputs (no new
guesswork or motor/camera access), render the same recorded episode frames,
and create aligned-size REAL / SIM panels. Candidate poses may be dated after
the recording and are NEVER silently declared calibrated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "so101_sim"))
import real_to_sim as r2s  # noqa: E402
import compare_id1_eye_to_hand as camera_tools  # noqa: E402

SIDE = "r2s_side"
DEFAULT_INTRINSICS = REPO / "assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"


def extract_camera_matrices(obj):
    """Accept explicit candidate schema only; no blind parsing of random matrices."""
    if not isinstance(obj, dict):
        return []
    found = []
    if "T_base_camera" in obj:
        found.append(obj["T_base_camera"])
    if isinstance(obj.get("extrinsics"), dict) and "T_base_camera" in obj["extrinsics"]:
        found.append(obj["extrinsics"]["T_base_camera"])
    return found


def valid_camera_transform(candidate):
    try:
        mat = np.asarray(candidate, dtype=float)
        camera_tools.mujoco_camera_pose(mat)
        return mat
    except (TypeError, ValueError, KeyError, np.linalg.LinAlgError):
        return None


def discover(root: Path, explicit: list[Path], limit: int):
    paths = []
    if explicit:
        paths = [p.resolve(strict=True) for p in explicit]
    else:
        if not root.exists():
            raise FileNotFoundError(f"Calibration search directory missing: {root}")
        # Search candidate output JSONs only; no private calibration JSON dumps.
        paths = sorted(root.rglob("*.json"), key=lambda p: str(p))
    results = []
    seen = set()
    for path in paths:
        try:
            if path.stat().st_size > 2_000_000:
                continue
            obj = json.loads(path.read_text(encoding="utf-8"))
            matrices = extract_camera_matrices(obj)
        except (ValueError, OSError, UnicodeError):
            continue
        for mat_value in matrices:
            mat = valid_camera_transform(mat_value)
            if mat is None:
                continue
            key = tuple(np.round(mat.ravel(), 7))
            if key in seen:
                continue
            seen.add(key)
            results.append((path, mat))
        if not explicit and len(results) >= limit:
            break
    return results[:limit]


def recorded_video(dataset: Path, episode: int):
    info = json.loads((dataset / "meta/info.json").read_text())
    meta = sorted((dataset / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not meta:
        raise FileNotFoundError("No LeRobot v3 episode metadata")
    table = pd.concat((pd.read_parquet(p) for p in meta), ignore_index=True)
    rows = table[table["episode_index"] == episode]
    if len(rows) != 1:
        raise ValueError(f"Episode {episode} has {len(rows)} metadata rows")
    row = rows.iloc[0]
    key = "observation.images.side"
    video_path = dataset / info["video_path"].format(
        video_key=key,
        chunk_index=int(row[f"videos/{key}/chunk_index"]),
        file_index=int(row[f"videos/{key}/file_index"]))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open side video {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise ValueError("Invalid video FPS")
    return cap, int(round(float(row[f"videos/{key}/from_timestamp"]) * fps)), video_path


def make_camera_model(urdf: Path, original_config: dict, pose: np.ndarray, intrinsics: dict):
    config = json.loads(json.dumps(original_config))
    pos, quat = camera_tools.mujoco_camera_pose(pose)
    config["cameras"]["side"]["pos_m"] = pos
    config["cameras"]["side"]["quat_wxyz"] = quat
    fovy = camera_tools.fovy_from_intrinsics(float(intrinsics["fy"]), int(intrinsics["height"]))
    config["cameras"]["side"]["fovy_deg"] = fovy
    # Only the side camera is changed; all joint mapping remains as supplied in config.
    model = r2s.build_model(urdf, config)
    model.vis.headlight.ambient[:] = (0.8, 0.8, 0.8)
    model.vis.headlight.diffuse[:] = (0.95, 0.95, 0.95)
    camera_tools.apply_white_follower_preview(model)
    return model, config, fovy


def side_by_side(real: np.ndarray, sim: np.ndarray, label: str, fovy: float):
    a, b = Image.fromarray(real).convert("RGB"), Image.fromarray(sim).convert("RGB")
    # Compare at *identical pixel dimensions*, not arbitrarily resized crop.
    if a.size != b.size:
        raise ValueError(f"Real and sim resolutions differ: {a.size} vs {b.size}")
    w, h = a.size
    canvas = Image.new("RGB", (2*w+24, h+78), "white")
    canvas.paste(a, (8, 70))
    canvas.paste(b, (w+16, 70))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), label, fill="black")
    draw.text((8, 27), "REAL side camera", fill="#18548c")
    draw.text((w+16, 27), f"SIM camera, FOV {fovy:.2f} deg", fill="#197048")
    draw.text((8, 49), "Candidate extrinsic only; NOT verified against original recording", fill="#916020")
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    p.add_argument("--extrinsics", type=Path, nargs="*", default=[],
                   help="Optional exact prior camera-pose JSON files; otherwise scan --search-dir")
    p.add_argument("--search-dir", type=Path, default=REPO/"outputs")
    p.add_argument("--max-candidates", type=int, default=8)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 374, 405])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists; do not overwrite evidence")
    if not 1 <= a.max_candidates <= 30:
        p.error("--max-candidates must be 1..30")
    intr = json.loads(a.intrinsics.read_text(encoding="utf-8"))
    iw, ih = int(intr["width"]), int(intr["height"])
    if iw <= 0 or ih <= 0:
        p.error("Invalid calibration image dimensions")
    for k in ("fx", "fy", "cx", "cy"):
        if not np.isfinite(float(intr[k])):
            p.error(f"Invalid {k}")
    if float(intr["fy"]) <= 0:
        p.error("fy must be positive")
    candidates = discover(a.search_dir, a.extrinsics, a.max_candidates)
    if not candidates:
        print("NO CAMERA POSE CANDIDATES FOUND. No J1/J2/J3 conclusion possible.")
        print("Prior camera extrinsic JSON must contain a 4x4 'T_base_camera'.")
        print("Try locating previous outputs with:")
        print("  find outputs -type f -name '*.json' | grep -Ei 'extrinsic|camera|eye|aruco|pnp|pose'")
        return 2
    dataset = a.dataset.resolve(strict=True)
    urdf = a.urdf.resolve(strict=True)
    base_config = json.loads(a.config.read_text(encoding="utf-8"))
    ep = r2s.read_episode(dataset, a.episode, stride=1, max_frames=100000)
    frames = {int(r["frame_index"]): r for r in ep["rows"]}
    cap, video_start, video_path = recorded_video(dataset, a.episode)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_width <= 0 or video_height <= 0:
        cap.release()
        raise ValueError("Cannot read side video dimensions")
    # A uniform video resize preserves focal field of view. Different aspect
    # ratio may signal crop/pad/OBS transformation and must not be guessed.
    ratio_x = video_width / iw
    ratio_y = video_height / ih
    if not np.isclose(ratio_x, ratio_y, rtol=0.002, atol=0.002):
        cap.release()
        raise ValueError(f"Side video {video_width}x{video_height} is not a uniform resize of "
                         f"intrinsics {iw}x{ih}; crop/OBS geometry must be checked")
    native_intr = dict(intr)
    native_intr.update(width=video_width, height=video_height,
                       fx=float(intr["fx"])*ratio_x, fy=float(intr["fy"])*ratio_y,
                       cx=float(intr["cx"])*ratio_x, cy=float(intr["cy"])*ratio_y)
    try:
        real = {}
        for frame in a.frames:
            if frame not in frames:
                raise ValueError(f"Frame {frame} not in episode {a.episode}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_start + frame)
            ok, frame_bgr = cap.read()
            if not ok:
                raise ValueError(f"Failed to decode side frame {frame}")
            real[frame] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if real[frame].shape[:2] != (video_height, video_width):
                raise ValueError(f"Unexpected side video frame shape {real[frame].shape[:2]}")
    finally:
        cap.release()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {"status": "CAMERA_EXTRINSIC_CANDIDATES_UNVALIDATED",
               "episode": a.episode, "frames": a.frames,
               "source_video": str(video_path),
               "intrinsics": str(a.intrinsics),
               "calibration_resolution": [iw, ih],
               "video_resolution": [video_width, video_height],
               "effective_intrinsics_at_video_resolution": {k: native_intr[k] for k in ("fx", "fy", "cx", "cy")},
               "warnings": [
                   "These candidates have no verified same-session relationship to the October 25 recording.",
                   "Only uniform resize of intrinsics is supported; OBS cropping must be separately checked.",
                   "Only vertical FOV is derived from scaled fy. MuJoCo projection does not model principal-point offsets or lens distortion in this preview.",
                   "Camera candidates from other sessions may not match the October 25 capture.",
                   "Real joint state mapping and model workcell objects are not validated.",
                   "Never infer J1 sign or cup-placement height before camera matching."
               ],
               "candidates": []}
    with (out/"candidates.csv").open("x", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["candidate", "path", "fovy_deg", "pos_x", "pos_y", "pos_z"])
        for idx, (path, pose) in enumerate(candidates):
            model, cfg, fovy = make_camera_model(urdf, base_config, pose, native_intr)
            data = mujoco.MjData(model)
            panels=[]
            for frame in a.frames:
                r2s.set_pose(model, data, frames[frame]["state"], cfg)
                # Render at calibration resolution, then resample uniformly to the
                # recorded-video resolution; do not stretch or crop.
                sim_cal = r2s.render_image(model, data, SIDE, iw, ih)
                sim = np.asarray(Image.fromarray(sim_cal).resize(
                    (video_width, video_height), Image.Resampling.LANCZOS))
                im = side_by_side(real[frame], sim, f"Candidate {idx+1} / frame {frame} / {path.name}", fovy)
                im.save(out/f"candidate_{idx+1:02d}_frame_{frame:04d}.png")
                panels.append(im)
            if panels:
                sheet = Image.new("RGB", (panels[0].width, len(panels)*panels[0].height), "white")
                for i, im in enumerate(panels):
                    sheet.paste(im, (0, i*im.height))
                sheet.save(out/f"candidate_{idx+1:02d}_contact_sheet.png")
            writer.writerow([idx+1,str(path),round(fovy,4),*pose[:3,3].tolist()])
            summary["candidates"].append({"id":idx+1,"path":str(path),"pos_m":pose[:3,3].tolist(),
                                          "fovy_deg":fovy})
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(f"Rendered {len(candidates)} camera pose candidates with same recorded frames: {out}")
    print("Compare matching robot base / shoulder / frame edges, not the illustrative cup geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
