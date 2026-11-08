#!/usr/bin/env python3
"""Fit an initial fixed Side-camera pose using one rigidly mounted ArUco center.

Model: p_camera = T_camera_base @ T_base_gripper(q) @ p_gripper_marker.
This center-only fit avoids relying on a small square's ambiguous PnP orientation.
Read-only inputs; no robot/camera interaction or MuJoCo modification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from audit_fk_aruco import JOINTS, forward_kinematics, parse_urdf
from audit_aruco_pnp_ambiguity import candidate_poses


def camera_arrays(info):
    if (info.get("width"), info.get("height")) != (640, 480):
        raise ValueError("Expected calibrated Side image geometry of 640x480")
    K = np.array([[info["fx"], 0, info["cx"]],
                  [0, info["fy"], info["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    D = np.asarray(info["distortion"], dtype=np.float64)
    if not (np.isfinite(K).all() and np.isfinite(D).all()) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    return K, D


def marker_side_m(manifest, marker_id):
    dims = manifest.get("marker_sizes_mm_by_id")
    if not isinstance(dims, dict) or str(marker_id) not in dims:
        raise ValueError("Manifest must provide marker_sizes_mm_by_id for mixed-size capture")
    mm = float(dims[str(marker_id)])
    if not math.isfinite(mm) or mm <= 0:
        raise ValueError("Invalid marker side length")
    return mm / 1000.


def model_centers(x, gripper_t):
    """x=[camera<-base Rodrigues, translation, gripper marker-center xyz] (metres)."""
    r = Rotation.from_rotvec(x[:3]).as_matrix()
    pbase = np.einsum("nij,j->ni", gripper_t[:, :3, :3], x[6:9]) + gripper_t[:, :3, 3]
    return (r @ pbase.T).T + x[3:6]


def solve_center_fit(gripper_t, observed_cam_xyz, *, holdout_interval=5):
    gripper_t = np.asarray(gripper_t, float)
    observed_cam_xyz = np.asarray(observed_cam_xyz, float)
    if gripper_t.shape[1:] != (4, 4) or observed_cam_xyz.shape != (len(gripper_t), 3):
        raise ValueError("Mismatched observation dimensions")
    if len(gripper_t) < 8 or not np.isfinite(gripper_t).all() or not np.isfinite(observed_cam_xyz).all():
        raise ValueError("Need >=8 finite ID1 observations")
    test = np.arange(len(gripper_t)) % holdout_interval == holdout_interval - 1
    train = ~test

    def residual(x, idx):
        return (model_centers(x, gripper_t[idx]) - observed_cam_xyz[idx]).ravel()

    starts = []
    rotations = [np.zeros(3), [math.pi, 0, 0], [0, math.pi, 0],
                 [0, 0, math.pi], [math.pi / 2, 0, 0],
                 [0, math.pi / 2, 0], [0, 0, math.pi / 2],
                 [-math.pi / 2, 0, 0], [0, -math.pi / 2, 0],
                 [0, 0, -math.pi / 2]]
    for initial_rotation in rotations:
        r = Rotation.from_rotvec(initial_rotation).as_matrix()
        mean_pbase = np.mean(gripper_t[train, :3, 3], axis=0)
        initial_t = np.mean(observed_cam_xyz[train], axis=0) - r @ mean_pbase
        x0 = np.r_[initial_rotation, initial_t, [0., 0., 0.]]
        fitted = least_squares(lambda x: residual(x, train), x0,
                               loss="soft_l1", f_scale=0.008,
                               max_nfev=1200, x_scale="jac")
        score = np.median(np.linalg.norm(residual(fitted.x, train).reshape(-1, 3), axis=1))
        starts.append((float(score), fitted))
    best_score, best = min(starts, key=lambda item: item[0])
    x = best.x
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    T_camera_base = np.eye(4)
    T_camera_base[:3, :3] = R
    T_camera_base[:3, 3] = x[3:6]
    T_base_camera = np.linalg.inv(T_camera_base)
    prediction = model_centers(x, gripper_t)
    err_mm = np.linalg.norm(prediction - observed_cam_xyz, axis=1) * 1000
    rank = int(np.linalg.matrix_rank(best.jac, tol=1e-7))
    return {
        "T_camera_base": T_camera_base.tolist(),
        "T_base_camera": T_base_camera.tolist(),
        "p_gripper_marker_center_m": x[6:9].tolist(),
        "train_indices": np.where(train)[0].tolist(),
        "holdout_indices": np.where(test)[0].tolist(),
        "training_error_mm": {
            "median": float(np.median(err_mm[train])),
            "rms": float(np.sqrt(np.mean(err_mm[train] ** 2))),
            "max": float(np.max(err_mm[train])),
        },
        "holdout_error_mm": {
            "median": float(np.median(err_mm[test])),
            "rms": float(np.sqrt(np.mean(err_mm[test] ** 2))),
            "max": float(np.max(err_mm[test])),
        },
        "observations_error_mm": err_mm.tolist(),
        "jacobian_rank": rank,
        "identifiable_9dof": rank == 9,
        "optimizer_success": bool(best.success),
        "candidate_starts": len(starts),
    }


def run(manifest_path, intrinsics_path, urdf_path, marker_id=1, link="gripper_link"):
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    cam = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    K, D = camera_arrays(cam)
    if data.get("dictionary") != "DICT_4X4_50":
        raise ValueError("Unexpected ArUco dictionary")
    marker_m = marker_side_m(data, marker_id)
    graph, _ = parse_urdf(urdf_path)
    robot_poses, centers, image_frames, pnp_rms = [], [], [], []
    for sample in data["samples"]:
        observed = [m for m in sample["markers"] if int(m["id"]) == marker_id]
        if not observed:
            continue
        if len(observed) != 1:
            raise ValueError("Duplicated marker ID within frame")
        marker = observed[0]
        measured_m = marker.get("marker_black_outer_edge_m")
        if measured_m is not None and not np.isclose(float(measured_m), marker_m, atol=1e-7):
            raise ValueError("Per-observation marker size disagrees with manifest")
        if sample["image_size"] != [640, 480]:
            raise ValueError("Image size differs from intrinsics")
        solutions = candidate_poses(marker["corners_px"], K, D, marker_m)
        if not solutions:
            continue
        q = {name: float(sample["joints"][name]) for name in JOINTS}
        robot_poses.append(forward_kinematics(graph, q, link))
        centers.append(solutions[0]["tvec_m"])
        pnp_rms.append(solutions[0]["reprojection_rms_px"])
        image_frames.append(int(sample["frame_index"]))
    result = solve_center_fit(np.asarray(robot_poses), np.asarray(centers))
    result.update({
        "kind": "so101_initial_eye_to_hand_marker_center_v1",
        "status": "provisional_not_applied_to_mujoco",
        "base_frame": "base_link",
        "camera_frame": "side_camera_optical",
        "marker_link": link,
        "marker_id": marker_id,
        "marker_black_outer_edge_mm": marker_m * 1000,
        "num_observations": len(centers),
        "frame_indices": image_frames,
        "PnP_best_solution_reprojection_rms_px": pnp_rms,
        "inputs": {"manifest": str(manifest_path), "intrinsics": str(intrinsics_path), "urdf": str(urdf_path)},
        "assumptions": [
            "Measured LeRobot joint degrees match this URDF's zero offsets/signs.",
            "Marker remains rigidly attached to the declared link.",
            "OBS Side camera and intrinsics use unchanged 640x480 image geometry.",
            "Only ArUco center translation is fitted; marker orientation is not fitted.",
            "A low error does not independently verify physical 6DoF camera calibration.",
        ],
    })
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, default=root / "assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json")
    p.add_argument("--urdf", type=Path, default=root / "data/urdf/so101/so101_new_calib.urdf")
    p.add_argument("--marker-id", type=int, default=1)
    p.add_argument("--link", default="gripper_link")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error(f"Already exists: {a.out}")
    if not a.out.parent.is_dir():
        p.error("Create output directory first")
    result = run(a.manifest, a.intrinsics, a.urdf, marker_id=a.marker_id, link=a.link)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ID{a.marker_id}: {result['num_observations']} observations")
    print(f"training RMS: {result['training_error_mm']['rms']:.2f}mm; holdout RMS: {result['holdout_error_mm']['rms']:.2f}mm")
    print(f"Jacobian rank: {result['jacobian_rank']}/9 (9 means numerically identifiable)")
    print("T_base_camera:")
    for row in result["T_base_camera"]:
        print("  ", " ".join(f"{v: .5f}" for v in row))
    print("Saved:", a.out)
    if not result["identifiable_9dof"]:
        print("WARNING: insufficient geometric diversity; do not apply extrinsic")


if __name__ == "__main__":
    main()
