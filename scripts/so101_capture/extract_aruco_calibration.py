#!/usr/bin/env python3
"""Extract stationary ArUco/Follower pairs from existing LeRobot captures.

Offline only. No robot/camera connections or changes to the source dataset.
Search every eligible frame within each stationary region, then save the best
ArUco observation (one observation per distinct stationary pose).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper")

DEFAULT_MARKER_SIZES_MM = {i: 14.0 for i in range(6)}
MIXED_MARKER_SIZES_MM = {0: 14.0, 1: 25.0, 2: 14.0, 3: 25.0, 4: 25.0, 5: 25.0}


def parse_marker_sizes_mm(value):
    """Parse complete per-ID size mapping, e.g. '0:14,1:25,...,5:25'."""
    if value is None:
        return dict(DEFAULT_MARKER_SIZES_MM)
    result = {}
    for entry in value.split(","):
        parts = entry.split(":")
        if len(parts) != 2:
            raise ValueError("Expected comma-separated ID:mm pairs")
        try:
            marker_id = int(parts[0])
            side_mm = float(parts[1])
        except ValueError as exc:
            raise ValueError("Invalid marker ID or dimension") from exc
        if marker_id not in range(6) or marker_id in result:
            raise ValueError("Marker ID must be unique and between 0 and 5")
        if not math.isfinite(side_mm) or side_mm <= 0:
            raise ValueError("Marker side must be finite and >0 mm")
        result[marker_id] = side_mm
    if set(result) != set(range(6)):
        raise ValueError("Specify sizes for all IDs 0..5")
    return result


def stationary_regions(states, *, window=12, tolerance_deg=0.7):
    """Return groups of candidate *frame indices* within stationary windows.

    A frame is eligible when it belongs to a length-'window' observation whose
    maximum range along each rotary axis (J1-J5) is <= tolerance_deg.
    Contiguous eligible frames form one stationary region. J6 is a normalized
    gripper value, not an angle. Never mix its units with J1-J5.
    """
    import numpy as np
    a = np.asarray(states, dtype=float)
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError("states must be a finite Nx6 matrix")
    if window < 3 or tolerance_deg <= 0 or not math.isfinite(tolerance_deg):
        raise ValueError("Invalid stationary detection settings")
    eligible = np.zeros(len(a), dtype=bool)
    for start in range(len(a) - window + 1):
        if float(np.max(np.ptp(a[start:start + window, :5], axis=0))) <= tolerance_deg:
            eligible[start:start + window] = True
    runs = []
    start = None
    for idx in range(len(eligible) + 1):
        is_eligible = idx < len(eligible) and bool(eligible[idx])
        if is_eligible and start is None:
            start = idx
        elif not is_eligible and start is not None:
            runs.append(list(range(start, idx)))
            start = None
    return runs


def stationary_candidates(states, *, window=12, tolerance_deg=0.7,
                          distinct_deg=3.0, min_gap_frames=12):
    """Backward-compatible compact representatives used by legacy tests."""
    import numpy as np
    if not math.isfinite(distinct_deg) or distinct_deg <= 0 or min_gap_frames < 0:
        raise ValueError("Invalid stationary selection settings")
    a = np.asarray(states, dtype=float)
    regions = stationary_regions(a, window=window, tolerance_deg=tolerance_deg)
    chosen = []
    for region in regions:
        center = region[len(region) // 2]
        if chosen and center - chosen[-1] < min_gap_frames:
            continue
        if chosen and np.max(np.abs(a[center, :5] - a[chosen[-1], :5])) < distinct_deg:
            continue
        chosen.append(center)
    return chosen


def detect_markers(image, *, dictionary_name="DICT_4X4_50"):
    import cv2
    import numpy as np
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco support required")
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


def marker_quality(markers, *, min_edge_px):
    """Quality for a frame. Prefer more usable IDs, then larger marker edges.

    Return None for frames without usable IDs. Markers from different links are
    counted only for visibility, never interpreted as a rigid multi-marker board.
    """
    usable = [m for m in markers if m["min_edge_px"] >= min_edge_px]
    if not usable:
        return None
    return (len(usable), round(sum(m["min_edge_px"] for m in usable), 3),
            round(min(m["min_edge_px"] for m in usable), 3))


def extract(source, output, *, window=12, tolerance_deg=0.7, distinct_deg=3.0,
            min_edge_px=8.0, max_poses=30, marker_sizes_mm=None):
    import av
    import cv2
    import numpy as np
    import pyarrow.parquet as pq

    if marker_sizes_mm is None:
        marker_sizes_mm = dict(DEFAULT_MARKER_SIZES_MM)
    if set(marker_sizes_mm) != set(range(6)) or any(
            not math.isfinite(v) or v <= 0 for v in marker_sizes_mm.values()):
        raise ValueError("Provide a positive measured size for every marker ID")

    if (window < 3 or max_poses < 1 or
            not all(math.isfinite(v) and v > 0 for v in
                    (tolerance_deg, distinct_deg, min_edge_px))):
        raise ValueError("Invalid extraction parameters")
    if output.exists() or source == output or source in output.parents:
        raise ValueError("Output must be a new directory outside source dataset")
    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    side_key = "observation.images.side"
    if side_key not in features or "observation.state" not in features:
        raise ValueError("Dataset must contain Side video and measured state")
    fps = float(info.get("fps", 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid dataset FPS")

    parquet_files = sorted((source / "data").rglob("*.parquet"))
    video_files = sorted((source / "videos" / side_key).rglob("*.mp4"))
    if not parquet_files or not video_files:
        raise ValueError("Missing Parquet or Side video")
    rows = []
    for file in parquet_files:
        rows.extend(pq.read_table(file, columns=["episode_index", "frame_index", "timestamp",
                                                  "observation.state"]).to_pylist())
    by_episode = {}
    for row in rows:
        by_episode.setdefault(int(row["episode_index"]), []).append(row)
    if len(video_files) != len(by_episode) or len(video_files) != int(info["total_episodes"]):
        raise ValueError("Video and episode counts differ")
    # This code supports the capture_session.py convention of one video per
    # episode, ordered in lexicographic episode index order.
    if sorted(by_episode) != list(range(len(video_files))):
        raise ValueError("Non-contiguous episode indices")

    output.mkdir(parents=True)
    images_dir = output / "images"
    images_dir.mkdir()
    samples, rejected = [], []
    previous_state = None
    examined_frames = 0
    stable_regions_count = 0
    for ep, video_file in zip(sorted(by_episode), video_files):
        episode_rows = sorted(by_episode[ep], key=lambda row: int(row["frame_index"]))
        for i, row in enumerate(episode_rows):
            if int(row["frame_index"]) != i:
                raise ValueError(f"Episode {ep}: non-contiguous frame indices")
        states = np.asarray([row["observation.state"] for row in episode_rows], dtype=float)
        regions = stationary_regions(states, window=window, tolerance_deg=tolerance_deg)
        stable_regions_count += len(regions)
        region_for_frame = {}
        for region_index, indices in enumerate(regions):
            for idx in indices:
                region_for_frame[idx] = region_index
        best_by_region = {}
        diagnostics = [{
            "episode": ep,
            "region": j,
            "start_frame": indices[0],
            "end_frame": indices[-1],
            "frames_checked": 0,
            "frames_with_any_marker": 0,
            "frames_with_usable_marker": 0,
            "ids_seen": set(),
            "max_detected_edge_px": 0.0,
        } for j, indices in enumerate(regions)]

        with av.open(str(video_file)) as video:
            n = 0
            for frame in video.decode(video=0):
                if n >= len(episode_rows):
                    raise ValueError(f"Episode {ep}: too many video frames")
                region_index = region_for_frame.get(n)
                if region_index is not None:
                    examined_frames += 1
                    diag = diagnostics[region_index]
                    diag["frames_checked"] += 1
                    bgr = frame.to_ndarray(format="bgr24")
                    markers = detect_markers(bgr)
                    if markers:
                        diag["frames_with_any_marker"] += 1
                        diag["ids_seen"].update(m["id"] for m in markers)
                        diag["max_detected_edge_px"] = max(
                            diag["max_detected_edge_px"],
                            max(m["min_edge_px"] for m in markers))
                    quality = marker_quality(markers, min_edge_px=min_edge_px)
                    if quality is not None:
                        diag["frames_with_usable_marker"] += 1
                        usable = [m for m in markers if m["min_edge_px"] >= min_edge_px]
                        # Earlier frame breaks quality ties, deterministically.
                        rank = (*quality, -n)
                        prev = best_by_region.get(region_index)
                        if prev is None or rank > prev["rank"]:
                            best_by_region[region_index] = dict(
                                frame_index=n, markers=usable, rank=rank, image=bgr.copy())
                n += 1
            if n != len(episode_rows):
                raise ValueError(f"Episode {ep}: video frames={n}, parquet={len(episode_rows)}")

        for idx, diag in enumerate(diagnostics):
            diag["ids_seen"] = sorted(diag["ids_seen"])
            diag["max_detected_edge_px"] = round(diag["max_detected_edge_px"], 3)
            best = best_by_region.get(idx)
            if best is None:
                rejected.append({**diag, "reason": "no_usable_aruco_in_stationary_region"})
                continue
            frame_idx = best["frame_index"]
            state = states[frame_idx]
            if previous_state is not None and float(
                    np.max(np.abs(state[:5] - previous_state[:5]))) < distinct_deg:
                rejected.append({**diag, "reason": "duplicate_pose",
                                 "best_frame": frame_idx,
                                 "best_ids": [m["id"] for m in best["markers"]]})
                continue
            if len(samples) >= max_poses:
                rejected.append({**diag, "reason": "max_poses_reached",
                                 "best_frame": frame_idx})
                continue
            filename = f"pose_{len(samples):03d}_ep{ep:03d}_frame{frame_idx:05d}.png"
            if not cv2.imwrite(str(images_dir / filename), best["image"]):
                raise OSError(f"Failed to write {filename}")
            row = episode_rows[frame_idx]
            samples.append({
                "image": "images/" + filename,
                "episode_index": ep,
                "frame_index": frame_idx,
                "stationary_region_frames": [diag["start_frame"], diag["end_frame"]],
                "stationary_frames_checked": diag["frames_checked"],
                "timestamp_s": float(row["timestamp"]),
                "joints": {name: float(v) for name, v in zip(JOINT_NAMES, state)},
                "markers": [
                    {**marker, "marker_black_outer_edge_m":
                     marker_sizes_mm[marker["id"]] / 1000.0}
                    for marker in best["markers"]
                ],
                "image_size": [int(best["image"].shape[1]), int(best["image"].shape[0])],
                "selection_quality": {
                    "usable_marker_count": best["rank"][0],
                    "sum_min_edge_px": best["rank"][1],
                },
            })
            previous_state = state.copy()
    result = {
        "kind": "so101_aruco_eye_to_hand_observations_v2",
        "source_dataset_name": source.name,
        "dictionary": "DICT_4X4_50",
        "marker_ids": list(range(6)),
        "marker_black_outer_edge_m": (
            next(iter(set(marker_sizes_mm.values()))) / 1000.0
            if len(set(marker_sizes_mm.values())) == 1 else None),
        "marker_print_total_edge_m": None,
        "marker_sizes_mm_by_id": {str(i): marker_sizes_mm[i] for i in range(6)},
        "fps": fps,
        "stationary_window_frames": window,
        "stationary_max_range_deg": tolerance_deg,
        "distinct_pose_min_max_axis_delta_deg": distinct_deg,
        "min_detected_edge_px": min_edge_px,
        "stationary_regions": stable_regions_count,
        "eligible_frames_examined": examined_frames,
        "samples": samples,
        "rejected": rejected,
        "warnings": [
            "Select one observation per distinct stationary pose, not independent poses from one hold.",
            "Video frames are paired to LeRobot observation.state by episode/frame index; optical exposure timestamp is unknown.",
            "Markers from moving links do not form a common rigid calibration board.",
            "Camera intrinsics must match the exact Side image coordinate system.",
            "Marker-to-link rigid transforms and FK J2/J3 conventions must be validated before Eye-to-Hand fitting.",
            "Detection of small markers does not guarantee stable 6DoF pose estimation.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(f"Saved {len(samples)} stationary poses to {output}; "
          f"regions={stable_regions_count}, frames_checked={examined_frames}, rejected={len(rejected)}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--tolerance-deg", type=float, default=0.7)
    p.add_argument("--distinct-deg", type=float, default=3.0)
    p.add_argument("--min-edge-px", type=float, default=8.0)
    p.add_argument("--max-poses", type=int, default=30)
    p.add_argument("--marker-sizes-mm", type=str, default=None,
                   help="All six ID:outer-black-edge-mm pairs; e.g. 0:14,1:25,2:14,3:25,4:25,5:25")
    a = p.parse_args()
    extract(a.dataset.expanduser().resolve(), a.out_dir.expanduser().resolve(),
            window=a.window, tolerance_deg=a.tolerance_deg, distinct_deg=a.distinct_deg,
            min_edge_px=a.min_edge_px, max_poses=a.max_poses,
            marker_sizes_mm=parse_marker_sizes_mm(a.marker_sizes_mm))


if __name__ == "__main__":
    main()
