#!/usr/bin/env python3
"""Compare original ArUco corners vs fresh OpenCV detections and guarded exclusions.

Read-only: scans existing 640x480 PNGs and manifest; no camera/robot calls.
Always evaluates ALL common observations before considering exclusions.
Do not equate small per-square PnP reprojection error with geometric accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from audit_aruco_pnp_ambiguity import candidate_poses


METHODS = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}


def redetect(gray, method):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = METHODS[method]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    result = {}
    for xy, marker_id in zip(corners, ids.reshape(-1)):
        marker_id = int(marker_id)
        if marker_id in result:  # Ambiguous ID: skip this observation.
            result[marker_id] = None
        else:
            result[marker_id] = np.asarray(xy, np.float32).reshape(4, 2)
    return {k: v for k, v in result.items() if v is not None}


def camera_arrays(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if (cfg.get("width"), cfg.get("height")) != (640, 480):
        raise ValueError("Expected 640x480 intrinsics")
    K = np.array([[cfg["fx"], 0, cfg["cx"]],
                  [0, cfg["fy"], cfg["cy"]], [0, 0, 1]], np.float64)
    D = np.asarray(cfg["distortion"], np.float64)
    if not np.isfinite(K).all() or not np.isfinite(D).all() or K[0,0] <= 0 or K[1,1] <= 0:
        raise ValueError("Invalid intrinsics")
    return K, D


def statistics(values):
    if not values:
        return None
    a = np.asarray(values, float)
    return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(np.max(a))}


def run(manifest_path, intrinsics_path, marker_mm=14., max_corner_displacement_px=3.):
    if marker_mm <= 0 or max_corner_displacement_px <= 0:
        raise ValueError("Dimensions and displacement threshold must be > 0")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if abs(data.get("marker_black_outer_edge_m", 0) * 1000 - marker_mm) > 1e-6:
        raise ValueError("Marker size differs from manifest's black outer edge")
    K, D = camera_arrays(intrinsics_path)
    rows = []
    for sample in data["samples"]:
        path = manifest_path.parent / sample["image"]
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(str(path))
        if frame.shape[:2] != (480, 640) or sample.get("image_size") != [640, 480]:
            raise ValueError("Image geometry mismatch: " + str(path))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = {mode: redetect(gray, mode) for mode in METHODS}
        for marker in sample["markers"]:
            mid = int(marker["id"])
            reference = np.asarray(marker["corners_px"], np.float32).reshape(4, 2)
            variants = {"saved": {
                "found": True, "accepted": True,
                "corners_px": reference.tolist(),
                "mean_corner_displacement_px": 0.0,
            }}
            for mode in METHODS:
                detected = detections[mode].get(mid)
                if detected is None:
                    variants[mode] = {"found": False, "accepted": False,
                                      "reason": "not_redetected"}
                    continue
                shift = np.linalg.norm(detected - reference, axis=1)
                if float(np.max(shift)) > max_corner_displacement_px:
                    variants[mode] = {"found": True, "accepted": False,
                                      "reason": "detector_disagrees_with_saved_corners",
                                      "max_corner_displacement_px": float(np.max(shift))}
                    continue
                variants[mode] = {
                    "found": True, "accepted": True,
                    "corners_px": detected.tolist(),
                    "mean_corner_displacement_px": float(np.mean(shift)),
                }
            for name, v in variants.items():
                if not v["accepted"]:
                    continue
                solutions = candidate_poses(v["corners_px"], K, D, marker_mm / 1000)
                if not solutions:
                    v["accepted"] = False
                    v["reason"] = "pnp_failed"
                    continue
                v["best"] = solutions[0]
                v["second"] = solutions[1] if len(solutions) > 1 else None
                v["rms_gap_px"] = (None if len(solutions) < 2 else
                                   solutions[1]["reprojection_rms_px"]
                                   - solutions[0]["reprojection_rms_px"])
            rows.append({"frame_index": int(sample["frame_index"]),
                         "marker_id": mid, "image": sample["image"],
                         "min_edge_px": float(marker["min_edge_px"]),
                         "variants": variants})

    by_id = {}
    for mid in sorted({r["marker_id"] for r in rows}):
        subset = [r for r in rows if r["marker_id"] == mid]
        by_id[str(mid)] = {"observations": len(subset)}
        for mode in ("saved", *METHODS):
            accepted = [r["variants"][mode] for r in subset
                        if r["variants"][mode]["accepted"]]
            by_id[str(mid)][mode] = {
                "accepted": len(accepted),
                "pnp_reprojection_rms_px": statistics(
                    [v["best"]["reprojection_rms_px"] for v in accepted]),
                "camera_z_m": statistics([v["best"]["tvec_m"][2] for v in accepted]),
                "corner_displacement_px": statistics(
                    [v["mean_corner_displacement_px"] for v in accepted]),
            }

    # Structural consistency diagnostic on pairs whose upstream joints are
    # practically unchanged; these are not independent ground-truth labels.
    joint_by_frame = {int(s["frame_index"]): s["joints"] for s in data["samples"]}
    near_pairs = []
    id2 = [r for r in rows if r["marker_id"] == 2]
    for i, a in enumerate(id2):
        for b in id2[i + 1:]:
            upstream = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
            delta = max(abs(joint_by_frame[a["frame_index"]][name]
                            - joint_by_frame[b["frame_index"]][name])
                        for name in upstream)
            if delta > 3.0:
                continue
            item = {"frames": [a["frame_index"], b["frame_index"]],
                    "max_j1_to_j4_delta_deg": float(delta)}
            for mode in ("saved", *METHODS):
                va, vb = a["variants"][mode], b["variants"][mode]
                item[mode + "_center_distance_mm"] = (
                    float(np.linalg.norm(np.asarray(va["best"]["tvec_m"])
                        - np.asarray(vb["best"]["tvec_m"])) * 1000)
                    if va["accepted"] and vb["accepted"] else None)
            near_pairs.append(item)

    # Flag potential high reprojection RMS only as a REVIEW LIST. Never discard
    # based on this or on a fit residual computed on the same samples.
    review = []
    for row in rows:
        v = row["variants"]["saved"]
        if v["best"]["reprojection_rms_px"] > 0.5:
            review.append({"frame": row["frame_index"], "marker_id": row["marker_id"],
                           "reason": "saved_four_point_rms_gt_0_5px",
                           "saved_rms_px": v["best"]["reprojection_rms_px"]})
    return {
        "kind": "so101_aruco_redetection_comparison_v1",
        "status": "no_extrinsics_and_no_automatic_exclusion",
        "source_manifest": str(manifest_path),
        "source_intrinsics": str(intrinsics_path),
        "opencv": cv2.__version__, "methods": ["saved", *METHODS],
        "max_corner_displacement_px": max_corner_displacement_px,
        "rows": rows, "per_marker": by_id,
        "id2_near_upstream_pose_pairs": near_pairs,
        "review_only_not_excluded": review,
        "exclusion_policy": [
            "Do not drop frames based only on in-sample PnP RMS.",
            "First investigate marker ID, corner order, blur, partial occlusion, and pose ambiguity.",
            "Compare candidate methods on identical frame/ID subsets (coverage differs).",
            "For future FK residual pruning, estimate outlier thresholds ONLY on training poses; evaluate held-out poses separately.",
            "Keep J1-J4 and camera-frame spatial diversity; never discard all difficult views.",
            "Do not treat ID4 as static ground truth unless its physical attachment has been checked.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--marker-mm", type=float, default=14.)
    p.add_argument("--max-corner-displacement-px", type=float, default=3.)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"Output exists: {args.out}")
    if not args.out.parent.is_dir():
        p.error("Create output parent directory first")
    report = run(args.manifest, args.intrinsics, args.marker_mm,
                 args.max_corner_displacement_px)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for mid, summary in report["per_marker"].items():
        print(f"ID{mid} (n={summary['observations']}):")
        for mode in report["methods"]:
            info = summary[mode]
            rms = info["pnp_reprojection_rms_px"]
            print(f"  {mode:8s}: n={info['accepted']:2d} "
                  f"PnP_RMS_median={'NA' if rms is None else format(rms["median"], ".3f")} px")
    print("ID2 near-matching J1-J4:")
    for pair in report["id2_near_upstream_pose_pairs"]:
        values = ", ".join(f"{mode}={pair[mode+'_center_distance_mm']:.2f}mm"
                           for mode in report["methods"]
                           if pair[mode + "_center_distance_mm"] is not None)
        print(f"  {pair['frames']}: {values}")
    print(f"Review-only observations: {len(report['review_only_not_excluded'])}")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
