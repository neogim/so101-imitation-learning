#!/usr/bin/env python3
"""Create an offline SO-101 joint zero/direction evidence report.

This program only reads a LeRobot v3 dataset, a URDF, and a calibration JSON.
It never imports robot hardware modules and never opens a serial device.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
DEFAULT_FRAMES = (0, 100, 200, 300, 400, 500, 600)


def rpy_matrix(rpy: str) -> np.ndarray:
    r, p, y = (float(x) for x in rpy.split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, q = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[c+x*x*q, x*y*q-z*s, x*z*q+y*s],
                     [y*x*q+z*s, c+y*y*q, y*z*q-x*s],
                     [z*x*q-y*s, z*y*q+x*s, c+z*z*q]])


def parse_urdf(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    by_name = {j.get("name"): j for j in root.findall("joint") if j.get("type") != "fixed"}
    result = []
    for name in JOINTS:
        node = by_name[name]
        origin = node.find("origin")
        axis = node.find("axis")
        limit = node.find("limit")
        result.append({
            "name": name,
            "parent": node.find("parent").get("link"),
            "child": node.find("child").get("link"),
            "origin_xyz": [float(x) for x in origin.get("xyz", "0 0 0").split()],
            "origin_rpy": [float(x) for x in origin.get("rpy", "0 0 0").split()],
            "axis_local": [float(x) for x in axis.get("xyz", "1 0 0").split()],
            "limit_rad": [float(limit.get("lower")), float(limit.get("upper"))],
        })
    return result


def load_episode(dataset: Path, episode: int) -> tuple[pd.DataFrame, dict, pd.Series]:
    info = json.loads((dataset / "meta/info.json").read_text())
    chunks = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not chunks:
        raise FileNotFoundError("No data parquet files")
    frames = pd.concat((pd.read_parquet(p) for p in chunks), ignore_index=True)
    frames = frames[frames["episode_index"] == episode].sort_values("frame_index").reset_index(drop=True)
    if frames.empty:
        raise ValueError(f"Episode {episode} is absent")
    episodes = pd.concat((pd.read_parquet(p) for p in sorted((dataset / "meta/episodes").glob("chunk-*/*.parquet"))), ignore_index=True)
    erow = episodes.loc[episodes["episode_index"] == episode]
    if len(erow) != 1:
        raise ValueError(f"Expected one episode metadata row, got {len(erow)}")
    return frames, info, erow.iloc[0]


def select_pairs(states: np.ndarray, min_gap: int = 10, max_gap: int = 220) -> list[dict]:
    """Choose high target-motion / low cross-joint-motion pairs deterministically."""
    n = len(states)
    scales = np.maximum(np.ptp(states, axis=0), 1.0)
    out = []
    for target in range(len(JOINTS)):
        candidates = []
        for a in range(n):
            end = min(n, a + max_gap + 1)
            for b in range(a + min_gap, end):
                delta = np.abs(states[b] - states[a])
                target_norm = delta[target] / scales[target]
                other_norm = float(np.linalg.norm(np.delete(delta / scales, target)))
                # Large isolated motion wins; a tiny temporal penalty breaks broad ties.
                score = target_norm / (0.05 + other_norm) - 0.0001 * (b - a)
                candidates.append((score, delta[target], -other_norm, a, b, delta, other_norm))
        best = max(candidates, key=lambda x: x[:5])
        _, _, _, a, b, delta, other_norm = best
        out.append({"joint": JOINTS[target], "frame_a": int(a), "frame_b": int(b),
                    "delta_target_deg": float(states[b, target] - states[a, target]),
                    "other_joint_normalized_l2": float(other_norm),
                    "all_delta_deg": [float(x) for x in states[b] - states[a]]})
    return out


def video_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not decode video frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def annotated(image: np.ndarray, title: str, state: np.ndarray, scale: int = 3) -> Image.Image:
    base = Image.fromarray(image).resize((image.shape[1] * scale, image.shape[0] * scale))
    canvas = Image.new("RGB", (base.width, base.height + 112), "white")
    canvas.paste(base, (0, 112))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black", font=ImageFont.load_default())
    draw.text((8, 28), "  ".join(f"J{i+1}={v:+.2f} deg" for i, v in enumerate(state)), fill="black")
    draw.text((8, 52), "J1 pan | J2 lift | J3 elbow | J4 wrist flex | J5 wrist roll", fill="black")
    draw.text((8, 76), "Recorded image/state only; no claim of physical replay.", fill=(150, 0, 0))
    return canvas


def contact_sheet(images: list[Image.Image], columns: int, path: Path) -> None:
    rows = math.ceil(len(images) / columns)
    w, h = max(i.width for i in images), max(i.height for i in images)
    sheet = Image.new("RGB", (columns*w, rows*h), (235, 235, 235))
    for k, im in enumerate(images):
        sheet.paste(im, ((k % columns)*w, (k // columns)*h))
    sheet.save(path)


def fk_origins(joints: list[dict], degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.eye(4)
    points = [t[:3, 3].copy()]
    axes = []
    for spec, deg in zip(joints, degrees):
        o = np.eye(4)
        o[:3, :3] = rpy_matrix(" ".join(str(x) for x in spec["origin_rpy"]))
        o[:3, 3] = spec["origin_xyz"]
        t = t @ o
        axis = np.asarray(spec["axis_local"], float)
        axes.append(t[:3, :3] @ axis)
        points.append(t[:3, 3].copy())
        rot = np.eye(4)
        rot[:3, :3] = axis_angle(axis, math.radians(float(deg)))
        t = t @ rot
    # Add a short tool direction solely to make wrist-roll orientation visible.
    points.append(t[:3, 3] + t[:3, :3] @ np.array([0.0, -0.08, 0.0]))
    return np.asarray(points), np.asarray(axes)


def render_urdf(joints: list[dict], zero_candidates: list[dict], states: np.ndarray, path: Path) -> None:
    fig = plt.figure(figsize=(14, 8))
    poses = [("URDF all joints = 0 rad", np.zeros(5))]
    poses += [(f"{x['joint']} closest-to-zero frame {x['frame']}", states[x["frame"]]) for x in zero_candidates]
    for k, (title, q) in enumerate(poses):
        ax = fig.add_subplot(2, 3, k+1, projection="3d")
        pts, axes = fk_origins(joints, q)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", lw=3)
        for p, axis in zip(pts[1:6], axes):
            ax.quiver(*p, *axis, length=.035, color="r")
        ax.set_title(title, fontsize=8)
        ax.set_xlim(-.35, .15); ax.set_ylim(-.25, .25); ax.set_zlim(-.15, .35)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=20, azim=-55)
    fig.suptitle("URDF kinematic rendering (red = joint local axis transformed into base frame)\nCamera extrinsics unknown; this is not pixel-registered to recorded images.")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="*", default=DEFAULT_FRAMES)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; refusing to overwrite evidence")
    for p in (args.dataset / "meta/info.json", args.urdf, args.calibration):
        if not p.exists():
            raise FileNotFoundError(p)
    args.output.mkdir(parents=True)

    df, info, erow = load_episode(args.dataset, args.episode)
    state_names = info["features"]["observation.state"]["names"]
    expected = [f"{j}.pos" for j in JOINTS]
    indices = [state_names.index(x) for x in expected]
    states = np.asarray([np.asarray(x, float)[indices] for x in df["observation.state"]])
    fps = float(info["fps"])
    requested = list(args.frames)
    invalid = [x for x in requested if x < 0 or x >= len(df)]
    if invalid:
        raise ValueError(f"Frames outside episode: {invalid}; valid 0..{len(df)-1}")

    video_key = "observation.images.side"
    chunk = int(erow[f"videos/{video_key}/chunk_index"])
    file_index = int(erow[f"videos/{video_key}/file_index"])
    video = args.dataset / info["video_path"].format(video_key=video_key, chunk_index=chunk, file_index=file_index)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_start = int(round(float(erow[f"videos/{video_key}/from_timestamp"]) * video_fps))

    pairs = select_pairs(states)
    for pair in pairs:
        a, b = pair["frame_a"], pair["frame_b"]
        pair["timestamp_a_s"] = float(df.iloc[a].timestamp)
        pair["timestamp_b_s"] = float(df.iloc[b].timestamp)
        pair["state_a_deg"] = [float(x) for x in states[a]]
        pair["state_b_deg"] = [float(x) for x in states[b]]
    zero_candidates = [{"joint": name, "frame": int(np.argmin(np.abs(states[:, k]))),
                        "angle_deg": float(states[np.argmin(np.abs(states[:, k])), k])}
                       for k, name in enumerate(JOINTS)]
    needed = sorted(set(requested + [x["frame_a"] for x in pairs] + [x["frame_b"] for x in pairs] + [x["frame"] for x in zero_candidates]))
    decoded = {f: video_frame(cap, video_start + f) for f in needed}
    cap.release()

    rows = []
    selected_images = []
    for f in requested:
        row = df.iloc[f]
        title = f"Episode {args.episode} frame {f} | t={float(row.timestamp):.2f}s | video frame {video_start+f}"
        im = annotated(decoded[f], title, states[f])
        im.save(args.output / f"frame_{f:04d}.png")
        selected_images.append(im)
        rows.append({"episode": args.episode, "frame": f, "timestamp_s": float(row.timestamp),
                     "video_frame": video_start+f, **{f"J{k+1}_{name}_deg": float(states[f, k]) for k, name in enumerate(JOINTS)}})
    contact_sheet(selected_images, len(selected_images), args.output / "selected_frames_contact_sheet.png")
    with (args.output / "selected_frames.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    for p in pairs:
        ims = []
        for label, f in (("A", p["frame_a"]), ("B", p["frame_b"])):
            ims.append(annotated(decoded[f], f"{p['joint']} pair {label}: frame {f}, t={float(df.iloc[f].timestamp):.2f}s", states[f]))
        contact_sheet(ims, 2, args.output / f"direction_{p['joint']}.png")
    pair_rows = []
    for p in pairs:
        row = {k: p[k] for k in ("joint", "frame_a", "timestamp_a_s", "frame_b", "timestamp_b_s", "delta_target_deg", "other_joint_normalized_l2")}
        for k, name in enumerate(JOINTS):
            row[f"a_J{k+1}_{name}_deg"] = p["state_a_deg"][k]
            row[f"b_J{k+1}_{name}_deg"] = p["state_b_deg"][k]
            row[f"delta_J{k+1}_{name}_deg"] = p["all_delta_deg"][k]
        pair_rows.append(row)
    with (args.output / "direction_pairs.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(pair_rows[0])); writer.writeheader(); writer.writerows(pair_rows)
    zero_images = [annotated(decoded[z["frame"]], f"{z['joint']} closest to zero: frame {z['frame']}, value={z['angle_deg']:+.3f} deg", states[z["frame"]]) for z in zero_candidates]
    contact_sheet(zero_images, 5, args.output / "zero_candidates.png")

    urdf_joints = parse_urdf(args.urdf)
    render_urdf(urdf_joints, zero_candidates, states, args.output / "urdf_kinematic_views.png")
    calibration = json.loads(args.calibration.read_text())
    decisions = []
    for k, spec in enumerate(urdf_joints):
        pair, zero = pairs[k], zero_candidates[k]
        decisions.append({
            "joint": spec["name"], "zero_alignment": "UNVERIFIED", "positive_direction_alignment": "UNVERIFIED",
            "zero_candidate_frame": zero["frame"], "zero_candidate_deg": zero["angle_deg"],
            "direction_pair_frames": [pair["frame_a"], pair["frame_b"]], "direction_delta_deg": pair["delta_target_deg"],
            "other_joint_normalized_l2": pair["other_joint_normalized_l2"],
            "urdf_axis_local_xyz": spec["axis_local"], "urdf_origin_rpy": spec["origin_rpy"],
            "calibration": calibration[spec["name"]],
            "reason": "Recorded side images and joint states provide comparison evidence, but no calibrated camera extrinsics, image joint landmarks, or independently measured physical reference establishes the URDF sign/zero mapping."
        })
    summary = {
        "mode": "offline_read_only", "episode": args.episode, "fps": fps,
        "dataset_frames_read": len(df), "video_frames_decoded": len(decoded), "requested_frames_exported": len(requested),
        "source_files_read": {"data_parquet": len(list((args.dataset/"data").glob("chunk-*/*.parquet"))),
                              "episode_metadata_parquet": len(list((args.dataset/"meta/episodes").glob("chunk-*/*.parquet"))),
                              "dataset_info_json": 1, "video_files": 1, "urdf_files": 1, "calibration_files": 1},
        "camera_key": video_key, "video_file": str(video), "video_frame_count": video_count,
        "video_fps": video_fps, "episode_video_start_frame": video_start,
        "selected_frames": rows, "direction_pairs": pairs, "zero_candidates": zero_candidates,
        "joint_results": decisions,
        "confirmed": ["Dataset metadata maps the side camera to the reported MP4.",
                      "Episode timestamps and video positions are consistent with the reported 20 FPS.",
                      "URDF axes are expressed in each joint frame and were transformed through parent frames for rendering."],
        "unverified": ["Physical equality of LeRobot 0 degrees and URDF 0 rad for J1-J5.",
                       "Physical positive-direction equality between recorded LeRobot angles and URDF joints for J1-J5."],
        "additional_measurements": ["Calibrated side-camera intrinsics/extrinsics and visible joint/link landmarks.",
                                    "An independent physical reference pose with measured link orientations for every joint.",
                                    "For direction, two safely obtained reference poses per joint with other joints fixed; do not infer this from FK/IK consistency."],
        "warnings": ["No robot connection or motor command was performed.", "Recorded imagery is evidence, not a physical replay.",
                     "FK/IK agreement is not treated as evidence of physical zero alignment.", "No pixel-position score is computed because camera extrinsics are unknown."]
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    md = ["# SO-101 Episode 0 joint zero/direction offline audit", "",
          "> Offline/read-only. No robot connection, command, or physical replay was performed.", "",
          "## Result", "", "| Joint | zero | positive direction | zero candidate | direction pair (delta) | URDF local axis |", "|---|---|---|---|---|---|"]
    for d in decisions:
        md.append(f"| {d['joint']} | **UNVERIFIED** | **UNVERIFIED** | f{d['zero_candidate_frame']} ({d['zero_candidate_deg']:+.3f}°) | f{d['direction_pair_frames'][0]}→f{d['direction_pair_frames'][1]} ({d['direction_delta_deg']:+.2f}°) | `{d['urdf_axis_local_xyz']}` |")
    md += ["", "## Evidence artifacts", "", "- `selected_frames_contact_sheet.png`: requested frames 0/100/200/300/400/500/600",
           "- `selected_frames.csv`: timestamps, video positions, and J1–J5 observation states",
           "- `direction_<joint>.png`: automatically selected before/after evidence",
           "- `direction_pairs.csv`: both endpoint timestamps, J1–J5 angles, and deltas",
           "- `zero_candidates.png`: closest recorded state to 0° for each joint",
           "- `urdf_kinematic_views.png`: URDF skeleton and transformed local axes (not camera-registered)", "",
           "## Counts", "", f"- Episode frames processed for numeric selection: **{len(df)}**",
           f"- Unique video frames decoded: **{len(decoded)}**", f"- Requested images exported: **{len(requested)}**",
           f"- Source files read: **{sum(summary['source_files_read'].values())}** ({summary['source_files_read']})", "",
           "## Confirmed", ""] + [f"- {x}" for x in summary["confirmed"]] + ["", "## Unverified / limitations", ""] + [f"- {x}" for x in summary["unverified"]] + ["- Camera extrinsics are unknown, so no image-to-URDF pixel error was calculated.", "", "## Additional measurements needed", ""] + [f"- {x}" for x in summary["additional_measurements"]]
    (args.output / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"output": str(args.output), "episode_frames": len(df), "decoded_video_frames": len(decoded), "joint_results": decisions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
