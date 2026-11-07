#!/usr/bin/env python3
"""Recheck Side ChArUco intrinsics from two original capture ZIPs, offline only.

No camera/robot access. Excludes __MACOSX metadata, checks 640x480, saves a
fresh report. Keeps the two sessions separate for cross-session validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

BOARD = {
    "dictionary": "DICT_4X4_50", "squares_x": 7, "squares_y": 9,
    "square_mm": 20.0, "marker_mm": 14.0,
}
MODELS = {
    "five_parameter": 0,
    "fix_k3_zero_tangential": cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
}


def read_session(archive, board):
    detector = cv2.aruco.CharucoDetector(board)
    samples, detections = [], []
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".png")
                       and "__MACOSX/" not in n)
        for name in names:
            image = cv2.imdecode(np.frombuffer(z.read(name), np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (480, 640):
                raise ValueError(f"Image missing or not 640x480: {name}")
            corners, ids, _, _ = detector.detectBoard(image)
            count = 0 if ids is None else len(ids)
            entry = {"file": Path(name).name, "corners": count,
                     "accepted": bool(count >= 10)}
            detections.append(entry)
            if count < 10:
                continue
            coords = np.asarray(corners, np.float32).reshape(-1, 2)
            if np.linalg.matrix_rank(coords - coords.mean(axis=0), tol=1e-4) < 2:
                entry["accepted"] = False
                entry["reason"] = "collinear"
                continue
            samples.append({"file": Path(name).name,
                            "ids": np.asarray(ids, np.int32).reshape(-1),
                            "corners": coords})
    return samples, detections


def fit(board, samples, flags):
    if len(samples) < 5:
        raise ValueError("At least 5 views required")
    points = np.asarray(board.getChessboardCorners(), np.float32)
    object_points = [points[s["ids"]].reshape(-1, 1, 3) for s in samples]
    image_points = [s["corners"].reshape(-1, 1, 2) for s in samples]
    rms, K, D, *_ = cv2.calibrateCameraExtended(
        object_points, image_points, (640, 480), None, None, flags=flags)
    return float(rms), K, D


def evaluate(board, samples, K, D):
    object_points = np.asarray(board.getChessboardCorners(), np.float32)
    all_squared = []
    per_image = []
    for sample in samples:
        o = object_points[sample["ids"]]
        observed = sample["corners"]
        ok, rvec, tvec = cv2.solvePnP(o, observed, K, D,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("PnP failed for " + sample["file"])
        predicted = cv2.projectPoints(o, rvec, tvec, K, D)[0].reshape(-1, 2)
        errors_sq = np.sum((predicted - observed) ** 2, axis=1)
        all_squared.extend(errors_sq.tolist())
        per_image.append({"file": sample["file"], "corner_count": len(o),
                          "rms_px": float(np.sqrt(np.mean(errors_sq)))})
    return {"pooled_rms_px": float(np.sqrt(np.mean(all_squared))),
            "per_image": per_image, "total_corners": len(all_squared)}


def model_report(board, first, edge, flags):
    all_samples = first + edge
    training = [s for i, s in enumerate(all_samples) if i % 5 != 4]
    heldout = [s for i, s in enumerate(all_samples) if i % 5 == 4]
    rms, K, D = fit(board, training, flags)
    first_rms, first_K, first_D = fit(board, first, flags)
    return {
        "combined_training_views": len(training),
        "combined_holdout_views": len(heldout),
        "combined_training_rms_px": rms,
        "combined_holdout": evaluate(board, heldout, K, D),
        "combined_candidate": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "distortion": D.reshape(-1).tolist()},
        "first_session_training_rms_px": first_rms,
        "first_session_to_edge_holdout": evaluate(board, edge, first_K, first_D),
        "warning": "PnP re-fits board pose in each held-out frame, not an independent 3D position test.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-zip", type=Path, required=True)
    parser.add_argument("--edge-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("Output already exists; never overwrite previous result")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((7, 9), .020, .014, dictionary)
    first, first_detections = read_session(args.first_zip, board)
    edge, edge_detections = read_session(args.edge_zip, board)
    if len(first) < 15 or len(edge) < 10:
        raise ValueError(f"Insufficient calibration views: {len(first)}, {len(edge)}")
    result = {
        "status": "candidate_not_validated",
        "board": BOARD,
        "image_width": 640, "image_height": 480,
        "opencv": cv2.__version__,
        "sources": {
            "first_zip_sha256": hashlib.sha256(args.first_zip.read_bytes()).hexdigest(),
            "edge_zip_sha256": hashlib.sha256(args.edge_zip.read_bytes()).hexdigest()},
        "detections": {"first": first_detections, "edge": edge_detections},
        "models": {name: model_report(board, first, edge, flags)
                   for name, flags in MODELS.items()},
        "limitations": [
            "Camera and OBS geometry must match both sessions and application data.",
            "Check printed board flatness, real square size, camera focus, and crop.",
            "Holdout board pose is re-estimated by solvePnP on each test frame.",
            "Do not label Eye-to-Hand or MuJoCo camera pose calibrated by this report."],
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    output = args.out_dir / "combined_report.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"Accepted {len(first)}/{len(first_detections)} initial and "
          f"{len(edge)}/{len(edge_detections)} edge images")
    for key, values in result["models"].items():
        print(key, "train", round(values["combined_training_rms_px"], 4),
              "heldout", round(values["combined_holdout"]["pooled_rms_px"], 4),
              "old-to-new", round(values["first_session_to_edge_holdout"]["pooled_rms_px"], 4))
    print("Report:", output)


if __name__ == "__main__":
    main()
