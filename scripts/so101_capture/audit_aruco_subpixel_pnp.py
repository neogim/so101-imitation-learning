#!/usr/bin/env python3
"""Read-only ArUco subpixel corner refinement and IPPE PnP comparison.

Reads existing 640x480 PNGs referenced by a stationary-pose manifest.
OpenCV cornerSubPix is initialized from the saved integer-rounded corners.
Rejects suspiciously large shifts. No motor/camera connections or source edits.
A lower 4-corner in-sample RMS does NOT establish better true 3D accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audit_aruco_pnp_ambiguity import candidate_poses


def refine_corners(gray, corners, *, half_window=3, max_shift_px=2.0):
    initial = np.asarray(corners, np.float32).reshape(4, 2)
    h, w = gray.shape
    if not np.isfinite(initial).all():
        raise ValueError("Nonfinite original corners")
    if np.any(initial[:, 0] < half_window + 2) or np.any(initial[:, 0] >= w - half_window - 2) or (
        np.any(initial[:, 1] < half_window + 2) or np.any(initial[:, 1] >= h - half_window - 2)
    ):
        return initial, False, "corner_near_image_boundary"
    refined = cv2.cornerSubPix(
        gray, initial.reshape(4, 1, 2).copy(),
        (half_window, half_window), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001),
    ).reshape(4, 2)
    shifts = np.linalg.norm(refined - initial, axis=1)
    if not np.isfinite(refined).all() or max(shifts) > max_shift_px:
        return initial, False, "large_or_invalid_corner_shift"
    return refined, True, None


def summarize(rows):
    result = {}
    for mid in sorted({int(r["marker_id"]) for r in rows}):
        subset = [r for r in rows if r["marker_id"] == mid]
        valid = [r for r in subset if r["refinement_accepted"]]
        result[str(mid)] = {
            "observations": len(subset),
            "refinement_accepted": len(valid),
            "median_corner_shift_px": float(np.median(
                [r["mean_corner_shift_px"] for r in valid])) if valid else None,
            "median_original_ippe_rms_px": float(np.median(
                [r["original_best"]["reprojection_rms_px"] for r in subset])),
            "median_refined_ippe_rms_px": float(np.median(
                [r["refined_best"]["reprojection_rms_px"] for r in subset])),
        }
    return result


def displacement_pair(rows, mid, a, b):
    by_frame = {r["frame_index"]: r for r in rows if r["marker_id"] == mid}
    if a not in by_frame or b not in by_frame:
        return {"frames": [a, b], "status": "missing_observation"}
    aa, bb = by_frame[a], by_frame[b]
    def dist(key):
        x, y = np.array(aa[key]["tvec_m"]), np.array(bb[key]["tvec_m"])
        return float(np.linalg.norm(x - y) * 1000)
    return {
        "frames": [a, b],
        "original_center_delta_mm": dist("original_best"),
        "refined_center_delta_mm": dist("refined_best"),
        "both_refinements_accepted": bool(
            aa["refinement_accepted"] and bb["refinement_accepted"]),
    }


def run(manifest_path, intrinsics_path, marker_mm, window, max_shift):
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    camera = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    if (camera.get("width"), camera.get("height")) != (640, 480):
        raise ValueError("Expected 640x480 intrinsics")
    K = np.array([[camera["fx"], 0, camera["cx"]],
                  [0, camera["fy"], camera["cy"]], [0, 0, 1.]], np.float64)
    D = np.asarray(camera["distortion"], np.float64)
    if not (np.isfinite(K).all() and np.isfinite(D).all()):
        raise ValueError("Nonfinite intrinsics")
    rows = []
    for sample in data["samples"]:
        path = manifest_path.parent / sample["image"]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Missing original image {path}")
        if gray.shape != (480, 640) or sample.get("image_size") != [640, 480]:
            raise ValueError(f"Image geometry mismatch: {path}")
        for marker in sample["markers"]:
            original = np.asarray(marker["corners_px"], np.float32).reshape(4, 2)
            refined, accepted, reason = refine_corners(
                gray, original, half_window=window, max_shift_px=max_shift)
            old_pose = candidate_poses(original, K, D, marker_mm / 1000)
            new_pose = candidate_poses(refined, K, D, marker_mm / 1000)
            if not old_pose or not new_pose:
                raise ValueError(f"PnP failed for marker {marker['id']} in {path}")
            diff = np.linalg.norm(refined - original, axis=1)
            rows.append({
                "frame_index": int(sample["frame_index"]),
                "marker_id": int(marker["id"]),
                "file": sample["image"],
                "refinement_accepted": accepted,
                "rejection_reason": reason,
                "original_corners_px": original.tolist(),
                "refined_corners_px": refined.tolist(),
                "mean_corner_shift_px": float(np.mean(diff)),
                "max_corner_shift_px": float(np.max(diff)),
                "original_best": old_pose[0],
                "refined_best": new_pose[0],
                "original_second": old_pose[1] if len(old_pose) > 1 else None,
                "refined_second": new_pose[1] if len(new_pose) > 1 else None,
            })
    return {
        "kind": "so101_aruco_subpixel_pnp_audit_v1",
        "status": "diagnostic_only_no_eye_to_hand",
        "manifest": str(manifest_path),
        "intrinsics": str(intrinsics_path),
        "marker_mm": marker_mm,
        "refinement": {
            "method": "cv2.cornerSubPix initialized from rounded original ArUco corner coordinates",
            "half_window_px": window,
            "max_accepted_corner_shift_px": max_shift,
        },
        "observations": rows,
        "per_marker": summarize(rows),
        "id2_near_repeat_pairs": [
            displacement_pair(rows, 2, 631, 680),
            displacement_pair(rows, 2, 1451, 1510),
        ],
        "limitations": [
            "Only the corner positions are refined; no new ArUco detection or marker ID assignment.",
            "Optimizing a 4-point PnP reprojection residual is an in-sample test, not independent 3D accuracy.",
            "Small 14mm markers retain planar IPPE 2-solution rotational ambiguity.",
            "Eye-to-Hand camera extrinsics remain unknown; FK joint zero/sign and rigid marker mounting unverified.",
            "Compare near-repeat center displacement and FK consistency, not reprojection RMS alone.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--marker-mm", type=float, default=14.0)
    p.add_argument("--corner-window", type=int, default=3)
    p.add_argument("--max-shift-px", type=float, default=2.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.marker_mm <= 0 or args.corner_window < 2 or args.corner_window > 8 or args.max_shift_px <= 0:
        p.error("Invalid marker dimensions or refinement settings")
    if args.out.exists():
        p.error(f"Output already exists: {args.out}")
    if not args.out.parent.is_dir():
        p.error(f"Create output parent directory first: {args.out.parent}")
    report = run(args.manifest, args.intrinsics, args.marker_mm,
                 args.corner_window, args.max_shift_px)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for mid, v in report["per_marker"].items():
        print(f"ID{mid}: accepted={v['refinement_accepted']}/{v['observations']} "
              f"corner shift median={v['median_corner_shift_px']:.3f}px "
              f"PnP RMS before/after={v['median_original_ippe_rms_px']:.3f}/"
              f"{v['median_refined_ippe_rms_px']:.3f}px")
    for pair in report["id2_near_repeat_pairs"]:
        if "original_center_delta_mm" in pair:
            print(f"ID2 frames {pair['frames']}: center displacement "
                  f"{pair['original_center_delta_mm']:.2f} -> "
                  f"{pair['refined_center_delta_mm']:.2f} mm")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
