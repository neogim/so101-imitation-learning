#!/usr/bin/env python3
"""Offline Cartesian7 delta -> regularized 5-joint IK. No hardware imports/writes.

Input JSON: {"joint_deg":[J1..J5], "cartesian7":[dx_m,dy_m,dz_m,rx_rad,ry_rad,rz_rad,gripper]}
Cartesian rotation is a base-frame rotation vector: R_target = Exp(rotvec) @ R_current.
This evaluates ONE predicted action only; no collision or trajectory validation.
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def limits_from_urdf(path):
    root = ET.parse(path).getroot()
    nodes = {j.get("name"): j for j in root.findall("joint")}
    return np.array([[float(nodes[n].find("limit").get(k)) for k in ("lower", "upper")] for n in JOINTS])


def solve(fk, limits, joint_deg, action7, rot_weight_mm_per_rad=25.0,
          joint_weight_mm_per_deg=0.05, max_nfev=150):
    joint_deg = np.asarray(joint_deg, float)
    action7 = np.asarray(action7, float)
    limits = np.asarray(limits, float)
    if joint_deg.shape != (5,) or action7.shape != (7,) or limits.shape != (5, 2):
        raise ValueError("Expected joint_deg[5], cartesian7[7], limits[5,2]")
    if not all(np.isfinite(x).all() for x in (joint_deg, action7, limits)):
        raise ValueError("Nonfinite input")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Invalid URDF bounds")
    q0 = np.deg2rad(joint_deg)
    if np.any(q0 < limits[:, 0] - 1e-8) or np.any(q0 > limits[:, 1] + 1e-8):
        raise ValueError("Initial joint pose outside URDF bounds; do not clamp silently")
    T0 = np.asarray(fk(q0), float)
    if T0.shape != (4, 4):
        raise ValueError("FK must return 4x4")
    target_xyz = T0[:3, 3] + action7[:3]
    target_R = Rotation.from_rotvec(action7[3:6]).as_matrix() @ T0[:3, :3]

    def residual(q):
        pose = np.asarray(fk(q), float)
        pos_mm = (pose[:3, 3] - target_xyz) * 1000
        rot_error = Rotation.from_matrix(target_R @ pose[:3, :3].T).as_rotvec()
        return np.r_[pos_mm, rot_error * rot_weight_mm_per_rad,
                     np.rad2deg(q - q0) * joint_weight_mm_per_deg]

    fit = least_squares(residual, q0, bounds=(limits[:, 0], limits[:, 1]),
                        max_nfev=max_nfev)
    achieved = np.asarray(fk(fit.x), float)
    pos_mm = float(np.linalg.norm(achieved[:3, 3] - target_xyz) * 1000)
    rot_deg = float(np.rad2deg(Rotation.from_matrix(
        target_R @ achieved[:3, :3].T).magnitude()))
    return {
        "optimizer_success": bool(fit.success),
        "position_error_mm": pos_mm,
        "orientation_error_deg": rot_deg,
        "joint_goal_deg": np.rad2deg(fit.x).tolist(),
        "gripper_goal": float(action7[6]),
        "within_urdf_limits": bool(np.all(fit.x >= limits[:, 0] - 1e-8)
                                    and np.all(fit.x <= limits[:, 1] + 1e-8)),
        "interpretation": "One-shot offline 5-DOF IK; not a safe executable trajectory",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, default=Path("data/urdf/so101/so101_new_calib.urdf"))
    a = ap.parse_args()
    import placo
    data = json.loads(a.input.read_text())
    robot = placo.RobotWrapper(str(a.urdf.resolve()))
    def fk(q):
        for name, angle in zip(JOINTS, q):
            robot.set_joint(name, float(angle))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"), float).copy()
    result = solve(fk, limits_from_urdf(a.urdf), data["joint_deg"], data["cartesian7"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
