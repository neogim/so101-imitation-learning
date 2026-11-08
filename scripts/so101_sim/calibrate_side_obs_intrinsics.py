#!/usr/bin/env python3
"""Calibrate the *effective* Side camera from existing OBS-output checkerboard images.

Offline-only. All input PNG/JPG images are read-only; output is a new directory.
Use the SAME OBS crop/scale/rotation as the ACT recording pipeline.
No access to camera hardware, OBS control, or robot motors.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg"}


def collect_images(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTENSIONS)


def calibrate(paths: list[Path], columns: int, rows: int, square_mm: float,
              minimum: int = 12) -> dict:
    if columns < 3 or rows < 3 or square_mm <= 0 or minimum < 3:
        raise ValueError("Invalid chessboard dimensions/square length/minimum")
    if len(paths) < minimum:
        raise ValueError(f"Need at least {minimum} images; found {len(paths)}")
    pattern = (columns, rows)
    obj_template = np.zeros((columns*rows, 3), dtype=np.float32)
    obj_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)*square_mm/1000.
    objects, pixels, accepted, rejected = [], [], [], []
    dims = None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rejected.append({"file": path.name, "reason": "unreadable"})
            continue
        h, w = gray.shape[:2]
        if dims is None:
            dims = (w, h)
        if dims != (w, h):
            raise ValueError("Image size mismatch; check OBS output settings")
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            rejected.append({"file": path.name, "reason": "board_not_found"})
            continue
        objects.append(obj_template.copy())
        pixels.append(corners.astype(np.float32))
        accepted.append(path.name)
    if len(accepted) < minimum:
        raise ValueError(f"Only {len(accepted)} usable chessboard images; need >= {minimum}")
    if dims is None:
        raise ValueError("No readable images")
    rms, K, dist, rv, tv = cv2.calibrateCamera(
        objects, pixels, dims, None, None)
    if not np.isfinite([rms, *K.flatten(), *dist.flatten()]).all():
        raise ValueError("Calibration returned nonfinite values")
    return {
        "intrinsics": {
            "width": dims[0], "height": dims[1],
            "fx": float(K[0,0]), "fy": float(K[1,1]),
            "cx": float(K[0,2]), "cy": float(K[1,2]),
            "distortion": [float(v) for v in dist.flatten()],
        },
        "report": {
            "status": "INTRINSICS_ESTIMATE_NOT_INDEPENDENTLY_VALIDATED",
            "image_size": list(dims), "accepted": accepted, "rejected": rejected,
            "pattern_inner_corners": [columns, rows],
            "square_size_mm": square_mm,
            "reprojection_rms_px": float(rms),
            "notes": [
                "Checkerboard columns and rows mean INNER corners, not square count.",
                "Move the board, never the fixed Side camera, while collecting calibration views.",
                "Capture through the same OBS scene transform and final resize/crop as ACT.",
                "Do not interpret good reprojection RMS as verified physical extrinsics.",
                "Use held-out images to validate calibration and review coverage of the full FOV.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--columns", type=int, required=True, help="checkerboard INNER corner columns")
    parser.add_argument("--rows", type=int, required=True, help="checkerboard INNER corner rows")
    parser.add_argument("--square-mm", type=float, required=True, help="measured printed square side in mm")
    parser.add_argument("--min-images", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        folder = args.images.expanduser().resolve(strict=True)
        if not folder.is_dir():
            raise ValueError("--images must be a directory")
        out = args.output.expanduser().resolve()
        if out.exists():
            raise ValueError(f"Refusing to overwrite output directory: {out}")
        if out == folder or folder in out.parents:
            raise ValueError("Output must be outside input directory")
        result = calibrate(collect_images(folder), args.columns, args.rows,
                           args.square_mm, args.min_images)
        out.mkdir(parents=True)
        (out/"intrinsics.json").write_text(json.dumps(result["intrinsics"],indent=2)+"\n")
        (out/"report.json").write_text(json.dumps(result["report"],indent=2)+"\n")
        print(json.dumps({"output":str(out), "image_size":result["report"]["image_size"],
                          "accepted":len(result["report"]["accepted"]),
                          "rms_px":result["report"]["reprojection_rms_px"]}, indent=2))
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    main()
