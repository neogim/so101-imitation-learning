#!/usr/bin/env python3
"""Offline SO-101 URDF forward-kinematics audit from an ArUco manifest.

Uses observation.state J1..J5 (degrees) literally as URDF joint angles, with
no inferred offsets or sign changes. This is a *hypothesis check*, NOT a
validation against physical marker coordinates or a camera extrinsic fit.
Only reads input files; prints JSON to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LINKS = ("wrist_link", "gripper_link")


def homogeneous_rotation(rpy):
    """URDF extrinsic fixed-axis R=Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_rotation(axis, theta):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid joint axis")
    x, y, z = axis / norm
    c, s = math.cos(theta), math.sin(theta)
    return c * np.eye(3) + (1 - c) * np.outer([x, y, z], [x, y, z]) + s * np.array(
        [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    )


def to_matrix(rotation, translation=(0., 0., 0.)):
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def parse_urdf(urdf):
    root = ET.parse(urdf).getroot()
    graph = {}
    limits = {}
    for node in root.findall("joint"):
        name = node.get("name")
        kind = node.get("type")
        if kind not in ("fixed", "revolute", "continuous"):
            raise ValueError(f"Unsupported URDF joint type: {name}: {kind}")
        p = node.find("parent")
        c = node.find("child")
        if p is None or c is None:
            raise ValueError(f"Missing parent/child for joint {name}")
        parent, child = p.get("link"), c.get("link")
        if child in graph:
            raise ValueError(f"Multiple URDF parents for {child}")
        origin = node.find("origin")
        xyz = [0., 0., 0.] if origin is None else [
            float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [0., 0., 0.] if origin is None else [
            float(v) for v in origin.get("rpy", "0 0 0").split()]
        if len(xyz) != 3 or len(rpy) != 3:
            raise ValueError(f"Invalid origin for {name}")
        axis_node = node.find("axis")
        axis = [1., 0., 0.] if axis_node is None else [
            float(v) for v in axis_node.get("xyz", "1 0 0").split()]
        if len(axis) != 3:
            raise ValueError(f"Invalid axis for {name}")
        graph[child] = dict(parent=parent, name=name, type=kind,
                            origin=to_matrix(homogeneous_rotation(rpy), xyz), axis=axis)
        if kind == "revolute":
            limit = node.find("limit")
            if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                raise ValueError(f"Missing revolute limits for {name}")
            limits[name] = (math.degrees(float(limit.get("lower"))),
                            math.degrees(float(limit.get("upper"))))
    if graph.get("gripper_link", {}).get("name") != "wrist_roll":
        raise ValueError("Expected J5 child gripper_link")
    if graph["gripper_link"]["parent"] != "wrist_link":
        raise ValueError("Expected J5 parent wrist_link")
    return graph, limits


def forward_kinematics(graph, angles_degrees, link):
    """Compute Base->link. q is keyed by *URDF joint name* in degrees."""
    chain = []
    visited = set()
    current = link
    while current != "base_link":
        if current in visited:
            raise ValueError("URDF joint cycle")
        visited.add(current)
        joint = graph.get(current)
        if joint is None:
            raise ValueError(f"No kinematic path base_link -> {link}: missing {current}")
        chain.append(joint)
        current = joint["parent"]
    t = np.eye(4)
    for joint in reversed(chain):
        t = t @ joint["origin"]
        if joint["type"] in ("revolute", "continuous"):
            if joint["name"] not in angles_degrees:
                raise ValueError(f"Missing angle {joint['name']}")
            t = t @ to_matrix(axis_rotation(
                joint["axis"], math.radians(angles_degrees[joint["name"]])))
    return t


def rotation_distance_deg(a, b):
    r = a[:3, :3].T @ b[:3, :3]
    return math.degrees(math.acos(float(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def audit(urdf, manifest, *, perturb_j5_deg=10.):
    if not math.isfinite(perturb_j5_deg) or perturb_j5_deg <= 0:
        raise ValueError("perturb_j5_deg must be positive")
    graph, limits = parse_urdf(urdf)
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("ArUco manifest has no saved samples")
    rows = []
    qvalues = []
    for sample in samples:
        angles = {j: float(sample["joints"][j]) for j in JOINTS}
        if not all(math.isfinite(v) for v in angles.values()):
            raise ValueError("Nonfinite joint data")
        qvalues.append(angles)
        w = forward_kinematics(graph, angles, "wrist_link")
        g = forward_kinematics(graph, angles, "gripper_link")
        modified = dict(angles, wrist_roll=angles["wrist_roll"] + perturb_j5_deg)
        w2 = forward_kinematics(graph, modified, "wrist_link")
        g2 = forward_kinematics(graph, modified, "gripper_link")
        observations = {int(m["id"]) for m in sample["markers"]}
        rows.append({
            "episode": int(sample["episode_index"]),
            "frame": int(sample["frame_index"]),
            "ids": sorted(observations),
            "base_to_wrist_xyz_m": np.round(w[:3, 3], 6).tolist(),
            "base_to_gripper_xyz_m": np.round(g[:3, 3], 6).tolist(),
            "j5_perturbation": {
                "wrist_translation_delta_mm": round(float(np.linalg.norm(w2[:3, 3] - w[:3, 3]) * 1000), 8),
                "wrist_rotation_delta_deg": round(rotation_distance_deg(w, w2), 8),
                "gripper_translation_delta_mm": round(float(np.linalg.norm(g2[:3, 3] - g[:3, 3]) * 1000), 8),
                "gripper_rotation_delta_deg": round(rotation_distance_deg(g, g2), 8),
            },
        })
    limit_checks = []
    for j in JOINTS:
        values = [q[j] for q in qvalues]
        lo, hi = limits[j]
        violations = [
            {"episode": int(s["episode_index"]), "frame": int(s["frame_index"]),
             "value_deg": round(float(v), 4),
             "outside_deg": round(max(lo - v, v - hi, 0.), 4)}
            for s, v in zip(samples, values) if v < lo or v > hi
        ]
        limit_checks.append({
            "joint": j,
            "urdf_limits_deg": [round(lo, 4), round(hi, 4)],
            "observed_range_deg": [round(min(values), 4), round(max(values), 4)],
            "violations": violations
        })
    wrist_points = np.array([r["base_to_wrist_xyz_m"] for r in rows])
    gripper_points = np.array([r["base_to_gripper_xyz_m"] for r in rows])
    return {
        "kind": "so101_urdf_fk_offline_audit_v1",
        "urdf": str(urdf),
        "manifest": str(manifest),
        "samples": len(rows),
        "assumption": "LeRobot joint degrees are applied directly to URDF radians without offsets/sign corrections.",
        "j5_perturbation_deg": perturb_j5_deg,
        "j5_check": {
            "max_wrist_translation_delta_mm": max(r["j5_perturbation"]["wrist_translation_delta_mm"] for r in rows),
            "max_wrist_rotation_delta_deg": max(r["j5_perturbation"]["wrist_rotation_delta_deg"] for r in rows),
            "min_gripper_rotation_delta_deg": min(r["j5_perturbation"]["gripper_rotation_delta_deg"] for r in rows),
            "max_gripper_rotation_delta_deg": max(r["j5_perturbation"]["gripper_rotation_delta_deg"] for r in rows),
        },
        "fk_position_ranges_m": {
            "wrist_link": np.round(np.ptp(wrist_points, axis=0), 6).tolist(),
            "gripper_link": np.round(np.ptp(gripper_points, axis=0), 6).tolist(),
        },
        "joint_limit_checks": limit_checks,
        "per_sample": rows,
        "not_validated": [
            "Computed FK is not physical accuracy verification.",
            "J2/J3 sign and zero-point calibration require physical/vision reference measurements.",
            "Unknown marker-to-link transforms prevent an ArUco reprojection check.",
            "OBS Side camera intrinsics and Base-to-Camera extrinsics remain unknown.",
        ],
    }


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", type=Path,
                   default=root / "data/urdf/so101/so101_new_calib.urdf")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--perturb-j5-deg", type=float, default=10.)
    args = p.parse_args()
    report = audit(args.urdf, args.manifest, perturb_j5_deg=args.perturb_j5_deg)
    print(f"FK audit: samples={report['samples']}, "
          f"J5 wrist rotation delta max={report['j5_check']['max_wrist_rotation_delta_deg']} deg, "
          f"gripper rotation delta range="
          f"{report['j5_check']['min_gripper_rotation_delta_deg']}.."
          f"{report['j5_check']['max_gripper_rotation_delta_deg']} deg")
    for item in report["joint_limit_checks"]:
        print(f"{item['joint']}: URDF limits={item['urdf_limits_deg']}, "
              f"data range={item['observed_range_deg']}, "
              f"out_of_limit={len(item['violations'])}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
