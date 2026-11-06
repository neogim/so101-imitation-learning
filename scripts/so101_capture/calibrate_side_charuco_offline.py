#!/usr/bin/env python3
"""Offline ChArUco calibration for SO-101 Side camera frames (no hardware I/O).

Separate clean capture images from screenshots with bars/crops.
Detect on actual pixel data; fit on training frames and report held-out
reprojection errors without re-fitting camera intrinsics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

BOARD = dict(squares_x=7, squares_y=9, square_length_mm=20., marker_length_mm=14.,
             dictionary="DICT_4X4_50")


def make_board(spec=BOARD):
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        raise RuntimeError("OpenCV with cv2.aruco.CharucoBoard is required")
    sx, sy = int(spec["squares_x"]), int(spec["squares_y"])
    square, marker = float(spec["square_length_mm"]), float(spec["marker_length_mm"])
    if sx < 2 or sy < 2 or not (0 < marker < square):
        raise ValueError("Invalid ChArUco board dimensions")
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, spec["dictionary"]))
    return cv2.aruco.CharucoBoard((sx, sy), square / 1000., marker / 1000., dictionary)


def detect(board, image, min_corners=10):
    if not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError("OpenCV cv2.aruco.CharucoDetector required")
    cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if cc is None or ci is None:
        return None
    cc = np.asarray(cc, np.float32).reshape(-1, 1, 2)
    ci = np.asarray(ci, np.int32).reshape(-1, 1)
    if len(ci) < min_corners or np.linalg.matrix_rank(cc.reshape(-1, 2) - cc.reshape(-1, 2).mean(axis=0), tol=1e-4) < 2:
        return None
    return (cc, ci)


def calibrate(board, training, image_size):
    if len(training) < 5:
        raise ValueError("At least 5 calibration views required")
    board_points = np.asarray(board.getChessboardCorners(), np.float32)
    object_points = [board_points[s["ids"].reshape(-1)].reshape(-1, 1, 3)
                     for s in training]
    image_points = [s["corners"] for s in training]
    result = cv2.calibrateCameraExtended(
        object_points, image_points, image_size, None, None)
    rms, k, distortion, _, _, _, _, per_view = result
    if not np.isfinite(k).all() or not np.isfinite(distortion).all() or not np.isfinite(rms):
        raise ValueError("Calibration has invalid parameters")
    return float(rms), k, distortion, np.asarray(per_view).reshape(-1)


def holdout_error(board, observations, camera_matrix, distortion):
    results = []
    board_points = board.getChessboardCorners()
    for item in observations:
        object_points = np.asarray(board_points, np.float32)[item["ids"].reshape(-1)]
        image_points = item["corners"].reshape(-1, 2)
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            results.append({"image": item["image"], "status": "pnp_failed"})
            continue
        predicted, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion)
        errors = np.linalg.norm(predicted.reshape(-1, 2) - image_points, axis=1)
        results.append({
            "image": item["image"],
            "status": "ok",
            "corner_count": len(image_points),
            "reprojection_rmse_px": round(float(np.sqrt(np.mean(errors**2))), 5),
            "reprojection_median_px": round(float(np.median(errors)), 5),
            "reprojection_max_px": round(float(np.max(errors)), 5),
        })
    return results


def run(image_dir, out_dir, *, min_corners=10, min_train=12):
    board = make_board()
    files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in
                   (".png", ".jpg", ".jpeg"))
    if not files:
        raise ValueError("No images in image directory")
    observed, rejected, size = [], [], None
    detector = cv2.aruco.CharucoDetector(board)
    for file in files:
        im = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if im is None:
            rejected.append({"image": file.name, "reason": "unreadable"})
            continue
        current_size = (int(im.shape[1]), int(im.shape[0]))
        if size is None:
            size = current_size
        if size != current_size:
            raise ValueError(f"Mixed image sizes: expected {size}, got {current_size}: {file}")
        corners, ids, _, _ = detector.detectBoard(im)
        if corners is None or ids is None or len(ids) < min_corners or np.linalg.matrix_rank(np.asarray(corners).reshape(-1, 2) - np.asarray(corners).reshape(-1, 2).mean(axis=0), tol=1e-4) < 2:
            rejected.append({"image": file.name, "reason": "insufficient_or_collinear_corners"})
            continue
        observed.append({"image": file.name,
                         "corners": np.asarray(corners, np.float32).reshape(-1, 1, 2),
                         "ids": np.asarray(ids, np.int32).reshape(-1, 1)})
    # Deterministic holdout: every fifth accepted image, starting at position 4.
    training = [v for i, v in enumerate(observed) if i % 5 != 4]
    holdout = [v for i, v in enumerate(observed) if i % 5 == 4]
    if len(training) < min_train or len(holdout) < 3:
        raise ValueError(f"Insufficient views: train={len(training)}, holdout={len(holdout)}; need train>={min_train}, holdout>=3")
    rms, k, dist, per_view = calibrate(board, training, size)
    heldout = holdout_error(board, holdout, k, dist)
    out_dir.mkdir(parents=True, exist_ok=False)
    intrinsics = {
        "width": size[0], "height": size[1],
        "fx": float(k[0, 0]), "fy": float(k[1, 1]),
        "cx": float(k[0, 2]), "cy": float(k[1, 2]),
        "distortion": dist.reshape(-1).tolist(),
        "calibration_source": "ChArUco OBS final frames",
        "board": BOARD,
        "note": "Candidate only: validate camera image geometry and independent physical accuracy before use.",
    }
    report = {
        "board": BOARD, "image_size": list(size), "input_images": len(files),
        "accepted_images": len(observed), "train_views": len(training),
        "holdout_views": len(holdout), "rejected": rejected,
        "training_rms_px": rms,
        "training_per_view_rms_px": dict(zip(
            [s["image"] for s in training], [float(v) for v in per_view])),
        "holdout_reprojection": heldout,
        "important_limitations": [
            "Holdout PnP estimates board pose per frame using fixed K; this is not independent 3D camera accuracy.",
            "Overlapping video frames or poor viewpoint diversity may overstate confidence.",
            "The calibration applies only to frames sharing exactly the same pixel geometry as these images.",
        ],
    }
    (out_dir / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2) + "\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return intrinsics, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True,
                   help="Directory of 640x480 Side-image files (no OBS window borders)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Must NOT exist, protects previous calibration")
    p.add_argument("--min-corners", type=int, default=10)
    p.add_argument("--min-train", type=int, default=12)
    args = p.parse_args()
    if args.min_corners < 4 or args.min_train < 5:
        p.error("min-corners must be >=4 and min-train >=5")
    values, report = run(args.images, args.out_dir,
                         min_corners=args.min_corners, min_train=args.min_train)
    print("Result:", args.out_dir)
    print("Size:", values["width"], "x", values["height"])
    print("training views:", report["train_views"], "holdout:", report["holdout_views"])
    print("training RMS(px):", report["training_rms_px"])
    print("intrinsics:", values["fx"], values["fy"], values["cx"], values["cy"])
    for item in report["holdout_reprojection"]:
        print("holdout:", item)


if __name__ == "__main__":
    main()
