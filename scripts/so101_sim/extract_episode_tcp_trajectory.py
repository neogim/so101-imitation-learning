#!/usr/bin/env python3
"""Extract recorded SO-101 gripper-link FK TCP proxy and original actions.

Offline only. Recorded observations are NOT automatically grasp-success labels.
The URDF gripper_link *origin* is a TCP proxy, not a calibrated fingertip TCP.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import real_to_sim as r2s


def default_joint_config():
    return {"joints": {
        "sign": {name: 1 for name in r2s.JOINTS},
        "offset_deg": {name: 0. for name in r2s.JOINTS},
    }}


def extract_tcp_rows(model, data, rows, config=None, body_name="gripper_link"):
    """Use state for FK; preserve separately logged action and source timestamp."""
    import mujoco

    config = default_joint_config() if config is None else config
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing TCP proxy body: {body_name}")
    result = []
    last_time = float("-inf")
    last_frame = -1
    for row in rows:
        time = float(row["timestamp"])
        frame = int(row["frame_index"])
        if not np.isfinite(time) or time <= last_time or frame <= last_frame:
            raise ValueError("Expected strictly increasing timestamps and frames")
        state = np.asarray(row["state"], dtype=float)
        action = np.asarray(row["action"], dtype=float)
        if state.shape != (6,) or action.shape != (6,):
            raise ValueError("Expected six logged state and action values")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Nonfinite logged joint values")
        r2s.set_pose(model, data, state.tolist(), config)
        rot = data.xmat[body_id].reshape(3, 3).copy()
        # Convert via MuJoCo to [w,x,y,z]; do not discard orientation.
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.reshape(9))
        result.append({
            "frame_index": frame,
            "timestamp_s": time,
            "tcp_proxy_body": body_name,
            "tcp_proxy_xyz_base_m": data.xpos[body_id].astype(float).tolist(),
            "tcp_proxy_quat_wxyz": quat.tolist(),
            "state_joint6_raw": state.tolist(),
            "logged_action_joint6_raw": action.tolist(),
        })
        last_time, last_frame = time, frame
    if not result:
        raise ValueError("No episode frames")
    return result


def export_rows(rows, output, *, episode, dataset, urdf, config, body_name):
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    outpath = output / "tcp_trajectory.json"
    record = {
        "schema": "so101_episode_tcp_proxy_fk_v1",
        "status": "RECORDED_STATE_FK_ONLY_NOT_A_VALIDATED_GRASP",
        "dataset": str(dataset),
        "episode": int(episode),
        "urdf": str(urdf),
        "tcp_proxy_body": body_name,
        "gripper_tcp_calibrated": False,
        "joint_mapping_calibrated": False,
        "joint_mapping": config["joints"],
        "units": {"xyz": "m", "joint5_state_action": "logged_deg",
                  "gripper_state_action": "logged_normalized_or_raw", "time": "s"},
        "rows": rows,
    }
    outpath.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    with (output / "tcp_trajectory.csv").open("x", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_index", "timestamp_s", "x_m", "y_m", "z_m",
                         "qw", "qx", "qy", "qz", *[f"state_{j}" for j in r2s.STATE_NAMES],
                         *[f"action_{j}" for j in r2s.STATE_NAMES]])
        for row in rows:
            writer.writerow([
                row["frame_index"], row["timestamp_s"],
                *row["tcp_proxy_xyz_base_m"], *row["tcp_proxy_quat_wxyz"],
                *row["state_joint6_raw"], *row["logged_action_joint6_raw"],
            ])
    return outpath


def main():
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--urdf", type=Path, default=r2s.DEFAULT_URDF)
    parser.add_argument("--body-name", default="gripper_link")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve(strict=True)
    if args.output.resolve() == dataset or dataset in args.output.resolve().parents:
        raise ValueError("Output must not be inside the original dataset")
    ep = r2s.read_episode(dataset, args.episode, args.stride, args.max_frames)
    spec = mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    spec.compiler.discardvisual = False
    model = spec.compile()
    data = mujoco.MjData(model)
    conf = default_joint_config()
    rows = extract_tcp_rows(model, data, ep["rows"], conf, args.body_name)
    target = export_rows(rows, args.output, episode=args.episode, dataset=dataset,
                         urdf=args.urdf, config=conf, body_name=args.body_name)
    print(f"Saved {len(rows)} recorded FK TCP-proxy rows: {target}")
    print("Not calibrated fingertip TCP, not grasp/pick success, no new action synthesis.")


if __name__ == "__main__":
    main()
