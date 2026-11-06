#!/usr/bin/env python3
"""Extract stationary ArUco/Follower pairs from an existing LeRobot capture.

Offline only: never opens robot serial ports, camera devices, or ACT checkpoints.
Keep source data unchanged. Each output sample is one stationary pose, not a
claim that robot-to-camera extrinsics have already been calibrated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper")


def stationary_candidates(states, *, window=12, tolerance_deg=0.7,
                          distinct_deg=3.0, min_gap_frames=12):
    """Return representative frames for sufficiently stable, distinct poses.

    The first five axes are angles in degrees. J6 is deliberately excluded:
    its LeRobot values are normalized, not degrees. The entire window must
    stay within tolerance of its first state, and candidates are nonoverlapping.
    """
    import numpy as np
    a = np.asarray(states, dtype=float)
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError("states must be a finite Nx6 matrix")
    if window < 3 or tolerance_deg <= 0 or distinct_deg <= 0 or min_gap_frames < 0:
        raise ValueError("Invalid stationary-selection configuration")
    selected = []
    start = 0
    while start + window <= len(a):
        segment = a[start:start + window, :5]
        span = np.ptp(segment, axis=0)
        if float(np.max(span)) > tolerance_deg:
            start += 1
            continue
        mid = start + window // 2
        if not selected or float(np.max(np.abs(a[mid, :5] - a[selected[-1], :5]))) >= distinct_deg:
            selected.append(mid)
        start += max(window, min_gap_frames)
    return selected


def detect_markers(image, *, dictionary_name="DICT_4X4_50"):
    import cv2
    import numpy as np
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco support required (opencv-contrib-python)")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    found = []
    if ids is not None:
        for pts, row in zip(corners, ids):
            marker_id = int(row[0])
            if not 0 <= marker_id <= 5:
                continue
            xy = np.asarray(pts, dtype=float).reshape(4, 2)
            lengths = [float(np.linalg.norm(xy[(j + 1) % 4] - xy[j])) for j in range(4)]
            found.append(dict(id=marker_id, corners_px=xy.round(3).tolist(),
                              min_edge_px=round(min(lengths), 3)))
    return sorted(found, key=lambda item: item["id"])


def extract(source, output, *, window=12, tolerance_deg=0.7, distinct_deg=3.0,
            min_edge_px=8.0, max_poses=30):
    import av
    import cv2
    import numpy as np
    import pyarrow.parquet as pq

    if output.exists() or source == output or source in output.parents:
        raise ValueError("Output must be a new directory outside the source dataset")
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    key = "observation.images.side"
    if key not in features:
        raise ValueError("Dataset must contain observation.images.side")
    if "observation.state" not in features:
        raise ValueError("Dataset has no measured observation.state")
    fps = float(info.get("fps", 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source FPS")

    parquet_files = sorted((source / "data").rglob("*.parquet"))
    video_files = sorted((source / "videos" / key).rglob("*.mp4"))
    if not parquet_files or not video_files:
        raise ValueError("Missing LeRobot data parquet or Side video")
    rows = []
    for path in parquet_files:
        rows.extend(pq.read_table(path, columns=["episode_index", "frame_index", "timestamp",
                                                   "observation.state"]).to_pylist())
    # Dataset video is encoded per-episode. Validate frame order before joining.
    rows_by_episode = {}
    for row in rows:
        ep = int(row["episode_index"])
        rows_by_episode.setdefault(ep, []).append(row)
    if len(video_files) != len(rows_by_episode):
        raise ValueError("Side video count does not match episode count")
    if len({int(r["episode_index"]) for r in rows}) != len(rows_by_episode):
        raise ValueError("Duplicate episode mapping")
    if len(video_files) != int(info["total_episodes"]):
        raise ValueError("Episode count differs from metadata")

    output.mkdir(parents=True)
    images_dir = output / "images"
    images_dir.mkdir()
    samples = []
    rejected = []
    previous_state = None
    try:
        for ep, video_file in zip(sorted(rows_by_episode), video_files):
            episode_rows = sorted(rows_by_episode[ep], key=lambda r: int(r["frame_index"]))
            for index, row in enumerate(episode_rows):
                if int(row["frame_index"]) != index:
                    raise ValueError(f"Episode {ep}: non-contiguous frame indices")
            states = np.asarray([row["observation.state"] for row in episode_rows], dtype=float)
            candidates = set(stationary_candidates(states, window=window,
                                                   tolerance_deg=tolerance_deg,
                                                   distinct_deg=distinct_deg))
            with av.open(str(video_file)) as video:
                idx = 0
                for frame in video.decode(video=0):
                    if idx >= len(episode_rows):
                        raise ValueError(f"Episode {ep}: extra video frames")
                    if idx in candidates and len(samples) < max_poses:
                        row = episode_rows[idx]
                        state = states[idx]
                        if previous_state is not None and np.max(np.abs(state[:5] - previous_state[:5])) < distinct_deg:
                            idx += 1
                            continue
                        rgb = frame.to_ndarray(format="rgb24")
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        detected = detect_markers(bgr)
                        usable = [m for m in detected if m["min_edge_px"] >= min_edge_px]
                        if not usable:
                            rejected.append({"episode": ep, "frame": idx, "reason": "no_sufficient_marker",
                                             "detected_ids": [m["id"] for m in detected]})
                        else:
                            filename = f"pose_{len(samples):03d}_ep{ep:03d}_frame{idx:05d}.png"
                            if not cv2.imwrite(str(images_dir / filename), bgr):
                                raise OSError("Failed to save image")
                            samples.append({
                                "image": "images/" + filename,
                                "episode_index": ep, "frame_index": idx,
                                "timestamp_s": float(row["timestamp"]),
                                "joints": {name: float(value) for name, value in zip(JOINT_NAMES, state)},
                                "markers": usable,
                                "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
                            })
                            previous_state = state.copy()
                    idx += 1
                if idx != len(episode_rows):
                    raise ValueError(f"Episode {ep}: video frames={idx}, parquet={len(episode_rows)}")
        result = {
            "kind": "so101_aruco_eye_to_hand_observations_v1",
            "source_dataset_name": source.name,
            "dictionary": "DICT_4X4_50",
            "marker_ids": list(range(6)),
            "marker_black_outer_edge_m": 0.014,
            "marker_print_total_edge_m": 0.018,
            "fps": fps,
            "stationary_window_frames": window,
            "stationary_max_range_deg": tolerance_deg,
            "distinct_pose_min_max_axis_delta_deg": distinct_deg,
            "min_detected_edge_px": min_edge_px,
            "samples": samples,
            "rejected": rejected,
            "warnings": [
                "Images and joint telemetry are from the same recorded frame; device exposure time is not known.",
                "Markers on different moving links are not a common rigid board.",
                "Marker-to-link transforms and robot FK (including J2/J3 conventions) must be validated before extrinsics fitting.",
                "Small detected markers may give unstable 6DoF pose estimates.",
                "These observations do not constitute an Eye-to-Hand calibration result.",
            ],
        }
        (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
        print(f"Saved {len(samples)} stationary poses to {output}; rejected={len(rejected)}")
    except BaseException:
        # Keep partial diagnostic output, but never touch the source.
        raise
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window", type=int, default=12, help="Stationary frames at dataset FPS")
    p.add_argument("--tolerance-deg", type=float, default=0.7)
    p.add_argument("--distinct-deg", type=float, default=3.0)
    p.add_argument("--min-edge-px", type=float, default=8.0)
    p.add_argument("--max-poses", type=int, default=30)
    a = p.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in
           (a.tolerance_deg, a.distinct_deg, a.min_edge_px)) or a.max_poses < 1:
        p.error("Invalid numerical threshold")
    extract(a.dataset.expanduser().resolve(), a.out_dir.expanduser().resolve(),
            window=a.window, tolerance_deg=a.tolerance_deg, distinct_deg=a.distinct_deg,
            min_edge_px=a.min_edge_px, max_poses=a.max_poses)


if __name__ == "__main__":
    main()
