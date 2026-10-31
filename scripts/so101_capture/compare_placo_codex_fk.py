#!/usr/bin/env python3
"""Compare Placo FK at Codex-selected grasp frames to pre-existing FK CSV (read-only)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import placo
from analyze_fk_grasp_csv import load_rows
from export_cartesian_action_sidecar import JOINT_NAMES


def compare(dataset: Path, csv_file: Path, urdf: Path, frame: str):
    rows = load_rows(csv_file)
    wanted = {(int(r["ep"]), int(r["grasp"])): r for r in rows}
    states = {}
    for file in sorted((dataset / "data").rglob("*.parquet")):
        table = pq.read_table(file, columns=["episode_index", "frame_index", "observation.state"])
        for rec in table.to_pylist():
            key = (int(rec["episode_index"]), int(rec["frame_index"]))
            if key in wanted:
                if key in states:
                    raise ValueError(f"Duplicate selected frame: {key}")
                states[key] = np.asarray(rec["observation.state"], dtype=float)
    missing = wanted.keys() - states.keys()
    if missing:
        raise ValueError(f"Missing grasp frames: {sorted(missing)[:10]}")

    robot = placo.RobotWrapper(str(urdf))
    diffs = []
    results = []
    for key, reference in sorted(wanted.items()):
        joint = states[key]
        if joint.shape != (6,) or not np.isfinite(joint).all():
            raise ValueError(f"Invalid state: {key}")
        for name, deg in zip(JOINT_NAMES, joint[:5]):
            robot.set_joint(name, float(np.deg2rad(deg)))
        robot.update_kinematics()
        pos = np.asarray(robot.get_T_world_frame(frame), dtype=float)[:3, 3]
        old = np.array([reference[a] for a in ("x", "y", "z")])
        diff = (pos - old) * 1000
        diffs.append(float(np.linalg.norm(diff)))
        results.append({"ep": key[0], "grasp": key[1],
                        "codex_xyz_mm": (old * 1000).round(3).tolist(),
                        "placo_xyz_mm": (pos * 1000).round(3).tolist(),
                        "error_xyz_mm": diff.round(3).tolist(),
                        "error_norm_mm": round(float(np.linalg.norm(diff)), 3)})
    values = np.asarray(diffs)
    return {"frame": frame, "episodes": len(results),
            "error_mean_mm": round(float(values.mean()), 3),
            "error_p95_mm": round(float(np.percentile(values, 95)), 3),
            "error_max_mm": round(float(values.max()), 3),
            "comparisons": results,
            "warning": "Agreement between implementations does not validate physical TCP calibration or collision safety."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--codex-csv", required=True, type=Path)
    p.add_argument("--urdf", required=True, type=Path)
    p.add_argument("--frame", default="gripper_frame_link")
    p.add_argument("--json-out", type=Path)
    a = p.parse_args()
    result = compare(a.dataset, a.codex_csv, a.urdf, a.frame)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(json.dumps({k:v for k,v in result.items() if k != "comparisons"}, ensure_ascii=False, indent=2))
    if a.json_out:
        with a.json_out.open("x", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
