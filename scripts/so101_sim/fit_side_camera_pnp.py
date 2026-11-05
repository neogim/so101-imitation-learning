#!/usr/bin/env python3
"""Estimate fixed Side camera pose from independently measured 3D-to-2D pairs.

Offline-only: does not touch the follower, cameras, URDF, or raw dataset.
Requires 6+ noncoplanar correspondences in the URDF/MuJoCo world frame.
Do NOT fabricate coordinates from approximate workcell rendering.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_points(path: Path):
    required = ("point_id", "x_m", "y_m", "z_m", "u_px", "v_px")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError("CSV columns required: " + ", ".join(required))
        rows = list(reader)
    if len(rows) < 6 or len({r["point_id"] for r in rows}) != len(rows):
        raise ValueError("At least 6 unique correspondences are required")
    xyz = np.array([[float(r[k]) for k in ("x_m", "y_m", "z_m")] for r in rows], np.float64)
    uv = np.array([[float(r[k]) for k in ("u_px", "v_px")] for r in rows], np.float64)
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("Coordinates must be finite")
    if np.linalg.matrix_rank(xyz - xyz.mean(axis=0), tol=1e-6) != 3:
        raise ValueError("3D points are coplanar or degenerate; use independently known noncoplanar points")
    return rows, xyz, uv


def read_intrinsics(path: Path):
    intrinsics = json.loads(path.read_text(encoding="utf-8"))
    required = ("width", "height", "fx", "fy", "cx", "cy")
    if not all(k in intrinsics for k in required):
        raise ValueError("Missing camera intrinsic value; expected width,height,fx,fy,cx,cy")
    w, h = int(intrinsics["width"]), int(intrinsics["height"])
    fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
    if w <= 0 or h <= 0 or fx <= 0 or fy <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
        raise ValueError("Invalid camera intrinsics")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    dist = np.array(intrinsics.get("distortion", [0., 0., 0., 0., 0.]), np.float64)
    if len(dist) not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
        raise ValueError("Invalid distortion array")
    return w, h, K, dist.reshape(-1, 1)


def solve(world_points, pixels, K, dist, threshold_px=4.0):
    if threshold_px <= 0:
        raise ValueError("threshold_px must be positive")
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        world_points, pixels, K, dist, flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=1000, reprojectionError=threshold_px, confidence=0.999)
    if not ok or inliers is None or len(inliers) < 6:
        raise ValueError("PnP failed or fewer than six RANSAC inliers")
    indices = inliers.flatten()
    rvec, tvec = cv2.solvePnPRefineLM(
        world_points[indices], pixels[indices], K, dist, rvec, tvec)
    projected = cv2.projectPoints(world_points, rvec, tvec, K, dist)[0].reshape(-1, 2)
    errors = np.linalg.norm(projected-pixels, axis=1)
    R_world_to_cv = cv2.Rodrigues(rvec)[0]
    translation_world = (-R_world_to_cv.T @ tvec).flatten()
    # CV frame: +X right, +Y down, +Z forward
    # MuJoCo camera: +X right, +Y up, -Z forward
    R_mujoco_to_world = R_world_to_cv.T @ np.diag([1., -1., -1.])
    q_xyzw = Rotation.from_matrix(R_mujoco_to_world).as_quat()
    q_wxyz = np.roll(q_xyzw, 1)
    if q_wxyz[0] < 0:
        q_wxyz *= -1
    depth = (R_world_to_cv @ world_points.T + tvec).T[:, 2]
    if (depth[indices] <= 0).any():
        raise ValueError("Some inlier landmarks lie behind camera")
    return {
        "pos_m": translation_world.tolist(), "quat_wxyz": q_wxyz.tolist(),
        "inliers": indices.tolist(),
        "rmse_inlier_px": float(np.sqrt(np.mean(errors[indices]**2))),
        "rmse_all_px": float(np.sqrt(np.mean(errors**2))),
        "errors_px": errors.tolist(), "projected_px": projected.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=Path, required=True, help="independently measured 3D/2D correspondences CSV")
    p.add_argument("--intrinsics", type=Path, required=True, help="calibrated camera intrinsics at source resolution")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("workcell.example.json"))
    p.add_argument("--image", type=Path, help="optional existing Side still image to visualize residuals")
    p.add_argument("--output", type=Path, required=True, help="new output directory outside source dataset")
    p.add_argument("--ransac-threshold-px", type=float, default=4.0)
    args = p.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"Refusing to overwrite existing output: {args.output}")
        rows, world, pixels = read_points(args.points)
        w, h, K, dist = read_intrinsics(args.intrinsics)
        if not ((pixels[:,0]>=0).all() and (pixels[:,0]<w).all() and
                (pixels[:,1]>=0).all() and (pixels[:,1]<h).all()):
            raise ValueError("Points outside image dimensions")
        if args.image:
            from PIL import Image
            with Image.open(args.image) as im:
                if im.size != (w, h):
                    raise ValueError(f"Image size {im.size} != intrinsics size {(w,h)}")
        result = solve(world, pixels, K, dist, args.ransac_threshold_px)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config["cameras"]["side"]["pos_m"] = result["pos_m"]
        config["cameras"]["side"]["quat_wxyz"] = result["quat_wxyz"]
        config["cameras"]["side"]["fovy_deg"] = math.degrees(2*math.atan(h/(2*K[1,1])))
        config["cameras"]["side"]["calibration"] = "PNP_ESTIMATE_NOT_INDEPENDENTLY_VALIDATED"
        config["calibrated"] = False
        args.output.mkdir(parents=True)
        (args.output/"side_candidate.json").write_text(json.dumps(config, indent=2)+"\n")
        (args.output/"report.json").write_text(json.dumps({
            "status": "PNP_ESTIMATE_NOT_PHYSICALLY_VALIDATED",
            "point_ids": [row["point_id"] for row in rows],
            "source_intrinsics": str(args.intrinsics),
            "warning": "Real 3D measurements, camera intrinsics, coordinate alignment and held-out reprojection must be validated independently",
            "result": result,
        }, indent=2)+"\n")
        if args.image:
            from PIL import Image, ImageDraw
            with Image.open(args.image) as im:
                picture = im.convert("RGB")
            painter = ImageDraw.Draw(picture)
            for row, measured, predicted in zip(rows, pixels, result["projected_px"]):
                x, y = measured; u, v = predicted
                painter.ellipse((x-3,y-3,x+3,y+3), fill="yellow")
                painter.ellipse((u-3,v-3,u+3,v+3), outline="red", width=2)
                painter.text((x+5,y+5), row["point_id"], fill="white")
            picture.save(args.output/"overlay.png")
        print(json.dumps({"output": str(args.output),
                          "inliers": len(result["inliers"]),
                          "rmse_inlier_px": result["rmse_inlier_px"],
                          "calibrated": False}, indent=2))
    except (OSError, ValueError, KeyError) as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
