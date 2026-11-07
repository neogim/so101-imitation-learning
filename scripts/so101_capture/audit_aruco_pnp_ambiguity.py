#!/usr/bin/env python3
"""Read-only per-marker ArUco PnP ambiguity audit, NOT Eye-to-Hand extrinsics.

Input: extracted stationary ArUco manifest + provisional intrinsics.
Each four-corner square produces two IPPE pose candidates. Both are retained
because a tiny near-frontoparallel marker can have ambiguous 3D orientation.
Never connects to robot or cameras. Does not mutate source files.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def marker_object_points(side_m):
    if not math.isfinite(side_m) or side_m <= 0:
        raise ValueError("Marker side length must be positive")
    h = side_m / 2
    return np.array([[-h, h, 0], [h, h, 0],
                     [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def candidate_poses(corners, camera_matrix, distortion, side_m):
    image = np.asarray(corners, np.float32).reshape(4, 2)
    obj = marker_object_points(side_m)
    if not np.isfinite(image).all():
        raise ValueError("Non-finite ArUco image corners")
    solved = cv2.solvePnPGeneric(
        obj, image, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    valid, rvecs, tvecs = solved[:3]
    if not valid or not rvecs:
        return []
    solutions = []
    for rv, tv in zip(rvecs, tvecs):
        projected = cv2.projectPoints(obj, rv, tv, camera_matrix, distortion)[0]
        err = np.linalg.norm(projected.reshape(4, 2) - image, axis=1)
        t = tv.reshape(3)
        solutions.append({
            "rvec": rv.reshape(3).tolist(),
            "tvec_m": t.tolist(),
            "distance_m": float(np.linalg.norm(t)),
            "reprojection_rms_px": float(np.sqrt(np.mean(err**2))),
            "reprojection_max_px": float(np.max(err)),
        })
    return sorted(solutions, key=lambda x: x["reprojection_rms_px"])


def audit(manifest, intrinsics, marker_mm=14.0):
    samples = json.loads(Path(manifest).read_text(encoding="utf-8"))["samples"]
    cam = json.loads(Path(intrinsics).read_text(encoding="utf-8"))
    if (cam.get("width"), cam.get("height")) != (640, 480):
        raise ValueError("Intrinsics must match the 640x480 ArUco observations")
    for k in ("fx", "fy", "cx", "cy", "distortion"):
        if k not in cam:
            raise ValueError("Missing camera parameter " + k)
    K = np.array([[cam["fx"], 0, cam["cx"]],
                  [0, cam["fy"], cam["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    D = np.asarray(cam["distortion"], dtype=np.float64)
    if not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("Nonfinite camera parameters")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid focal lengths")
    rows = []
    for s in samples:
        if s.get("image_size") != [640, 480]:
            raise ValueError(f"ArUco image size differs at frame {s.get('frame_index')}")
        for marker in s.get("markers", []):
            candidate = candidate_poses(
                marker["corners_px"], K, D, marker_mm / 1000)
            if not candidate:
                continue
            best = candidate[0]
            second = candidate[1] if len(candidate) > 1 else None
            rows.append({
                "frame_index": int(s["frame_index"]),
                "marker_id": int(marker["id"]),
                "min_edge_px": float(marker["min_edge_px"]),
                "best": best,
                "alternative": second,
                "rms_gap_px": (None if second is None
                               else second["reprojection_rms_px"]
                               - best["reprojection_rms_px"]),
            })
    by_id = {}
    for mid in sorted({x["marker_id"] for x in rows}):
        subset = [x for x in rows if x["marker_id"] == mid]
        by_id[str(mid)] = {
            "observations": len(subset),
            "min_edge_px": min(x["min_edge_px"] for x in subset),
            "median_best_reprojection_rms_px": float(np.median(
                [x["best"]["reprojection_rms_px"] for x in subset])),
            "median_two_solution_rms_gap_px": float(np.median(
                [x["rms_gap_px"] for x in subset if x["rms_gap_px"] is not None]))
                if any(x["rms_gap_px"] is not None for x in subset) else None,
        }
    return {
        "kind": "aruco_ippe_square_pose_ambiguity_audit_v1",
        "status": "diagnostic_only_no_camera_extrinsics",
        "manifest": str(manifest), "intrinsics": str(intrinsics),
        "marker_side_mm": marker_mm, "per_marker": by_id,
        "observations": rows,
        "notes": [
            "Marker poses are camera->marker observations, not Base->Camera.",
            "Two IPPE solutions may be nearly indistinguishable for small markers.",
            "No independent validation of image geometry or calibrated joint zero/sign.",
            "IDs 0/1/2 link assignments are hypotheses; others remain unassigned.",
            "The input intrinsics are provisional; do not apply as final MuJoCo camera poses."
        ],
    }


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path,
                   default=root / "assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json")
    p.add_argument("--marker-mm", type=float, default=14)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"Output already exists: {args.out}")
    if not args.out.parent.is_dir():
        p.error("Output parent directory must already exist")
    report = audit(args.manifest, args.intrinsics, args.marker_mm)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for mid, data in report["per_marker"].items():
        print(f"ID{mid}: n={data['observations']}, "
              f"best RMS median={data['median_best_reprojection_rms_px']:.4f}px, "
              f"IPPE gap median={data['median_two_solution_rms_gap_px']}")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
