#!/usr/bin/env python3
"""Export a NEW offline Cartesian action sidecar from an existing LeRobot v3 Dataset.

For each observation.state[t], compute FK(state[t]) and FK(action[t]); export
the action's *relative* TCP translation/rotation (plus original gripper goal).
This is a proposed alternative training target, NOT data augmentation and NOT
a trainable LeRobot ACT Dataset yet. Source Dataset is never written to.

Requires Placo directly (no LeRobot installation),
pyarrow and numpy. Never imports robot hardware or sends motor commands.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll")


def rotation_log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm as a rotation vector, with stable small/near-pi handling."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError("Invalid rotation matrix")
    cos_theta = float(np.clip((np.trace(R) - 1) / 2, -1, 1))
    theta = math.acos(cos_theta)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-7:
        return vee / 2
    if math.pi - theta < 1e-5:
        diag = np.maximum((np.diag(R) + 1) / 2, 0)
        axis = np.sqrt(diag)
        i = int(np.argmax(axis))
        if axis[i] < 1e-8:
            raise ValueError("Ambiguous near-pi rotation")
        for j in range(3):
            if j != i:
                axis[j] = (R[i, j] + R[j, i]) / (4 * axis[i])
        axis /= np.linalg.norm(axis)
        return axis * theta
    return (theta / (2 * math.sin(theta))) * vee


def target_from_fk(state: np.ndarray, action: np.ndarray, fk) -> np.ndarray:
    """Translation in base frame (m), relative orientation in base frame (rad),
    gripper target in normalized 0..100. This is NOT a time derivative.
    """
    state = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    if state.shape != (6,) or action.shape != (6,) or not (
        np.isfinite(state).all() and np.isfinite(action).all()
    ):
        raise ValueError("Expected finite six-joint state and action")
    ts = np.asarray(fk(state[:5]), dtype=float).copy()
    ta = np.asarray(fk(action[:5]), dtype=float).copy()
    if ts.shape != (4, 4) or ta.shape != (4, 4):
        raise ValueError("FK must return 4x4 poses")
    if not np.isfinite(ts).all() or not np.isfinite(ta).all():
        raise ValueError("Non-finite FK poses")
    translation = ta[:3, 3] - ts[:3, 3]
    relative_rotation = ta[:3, :3] @ ts[:3, :3].T
    result = np.r_[translation, rotation_log(relative_rotation), action[5]]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite Cartesian target")
    return result


def read_rows(root: Path):
    import pyarrow.parquet as pq
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError("No episode data Parquet files found")
    for path in paths:
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "timestamp",
                           "observation.state", "action"]
        )
        yield from table.to_pylist()


def export(root: Path, output: Path, urdf: Path, frame: str, expected_episodes: int | None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import placo

    if not (root / "meta/info.json").is_file():
        raise ValueError("Not a LeRobot Dataset root")
    if not urdf.is_file():
        raise ValueError("URDF not found")
    if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError("Output must be outside original Dataset")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    info = json.loads((root / "meta/info.json").read_text())
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        if features.get(key, {}).get("shape") != [6]:
            raise ValueError(f"Unsupported feature shape: {key}")
    if int(info.get("fps", 0)) != 20:
        raise ValueError("Expected source Dataset at 20fps")
    if expected_episodes is not None and info.get("total_episodes") != expected_episodes:
        raise ValueError("Unexpected episode count")

    robot = placo.RobotWrapper(str(urdf))
    available = set(robot.joint_names())
    missing = set(JOINT_NAMES) - available
    if missing:
        raise ValueError(f"URDF joints missing: {sorted(missing)}; available: {sorted(available)}")

    def forward_kinematics(joint_deg):
        joints = np.asarray(joint_deg, dtype=float)
        if joints.shape != (5,):
            raise ValueError("Expected five arm joints")
        for name, degrees in zip(JOINT_NAMES, joints, strict=True):
            robot.set_joint(name, float(np.deg2rad(degrees)))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame(frame), dtype=float)

    forward_kinematics(np.zeros(5))
    converted = []
    counters = {}
    max_pos = 0.
    max_rot = 0.
    last = {}
    for row in read_rows(root):
        ep, ix = int(row["episode_index"]), int(row["frame_index"])
        if ix != counters.get(ep, 0):
            raise ValueError(f"Nonsequential frame in episode {ep}: {ix}")
        if ep in last and float(row["timestamp"]) <= last[ep]:
            raise ValueError(f"Nonincreasing timestamp in episode {ep}")
        last[ep] = float(row["timestamp"])
        vec = target_from_fk(row["observation.state"], row["action"],
                             forward_kinematics)
        max_pos = max(max_pos, float(np.linalg.norm(vec[:3])))
        max_rot = max(max_rot, float(np.linalg.norm(vec[3:6])))
        converted.append({"episode_index": ep, "frame_index": ix,
                          "timestamp": float(row["timestamp"]),
                          "cartesian_action": vec.tolist()})
        counters[ep] = ix + 1
    if len(converted) != info.get("total_frames"):
        raise ValueError("Total frames disagree with Dataset metadata")
    if len(counters) != info.get("total_episodes"):
        raise ValueError("Total episodes disagree with Dataset metadata")
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(converted), output / "cartesian_actions.parquet")
    metadata = {
        "source_root": str(root.resolve()),
        "source_episodes": len(counters),
        "source_frames": len(converted),
        "source_fps": 20,
        "urdf": str(urdf.resolve()),
        "frame": frame,
        "joint_order": [*JOINT_NAMES, "gripper"],
        "target_fields": ["delta_tcp_x_m_base", "delta_tcp_y_m_base",
                          "delta_tcp_z_m_base", "delta_rotvec_x_rad_base",
                          "delta_rotvec_y_rad_base", "delta_rotvec_z_rad_base",
                          "gripper_goal_normalized_0_100"],
        "semantics": "FK(action[t])-FK(state[t]), not frame-to-frame displacement",
        "max_translation_m": max_pos, "max_rotation_rad": max_rot,
        "trainable_leRobot_dataset": False,
        "notes": [
            "Derived sidecar only; source images and original dataset are not modified.",
            "The source action is an absolute joint goal; TCP delta is not a measured next-pose delta.",
            "Verify URDF zero, joint naming, TCP frame and calibration before training.",
            "Conversion to a LeRobot v3 dataset, 7D policy output, normalizers, and IK postprocessor is not implemented.",
            "A 5-DOF arm cannot generally track arbitrary 6D pose commands.",
            "No IK, collision checking, simulated images, extra episodes or real robot motion is generated.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--frame", default="gripper_frame_link")
    ap.add_argument("--expected-episodes", type=int, default=53)
    args = ap.parse_args()
    result = export(args.dataset, args.output, args.urdf, args.frame, args.expected_episodes)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
