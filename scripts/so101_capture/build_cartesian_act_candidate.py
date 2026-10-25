#!/usr/bin/env python3
"""Build an isolated *candidate* LeRobot v3 Dataset with 7D Cartesian action.

Copies videos and metadata (never symlinks/hardlinks). Replaces action in
Parquet files, updates feature schema, and removes stale action statistics.
Leaves a NOT_TRAINING_READY marker until an independent LeRobot readback and
normalization audit is completed. Never interacts with motors.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TARGET_NAMES = [
    "delta_tcp_x_m_base", "delta_tcp_y_m_base", "delta_tcp_z_m_base",
    "delta_rotvec_x_rad_base", "delta_rotvec_y_rad_base",
    "delta_rotvec_z_rad_base", "gripper_goal_normalized_0_100",
]

def make_index(table):
    keys = zip(table["episode_index"].to_pylist(), table["frame_index"].to_pylist())
    mapping = {}
    for i, (episode, frame) in enumerate(keys):
        key = (int(episode), int(frame))
        if key in mapping:
            raise ValueError(f"Duplicate key: {key}")
        mapping[key] = i
    return mapping

def convert(source: Path, sidecar: Path, output: Path):
    source, sidecar, output = source.resolve(), sidecar.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if output == source or source in output.parents or sidecar in output.parents:
        raise ValueError("Output cannot be within source or sidecar")
    info_path = source / "meta/info.json"
    manifest_path = sidecar / "manifest.json"
    if not info_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Dataset metadata / Cartesian manifest missing")
    info = json.loads(info_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (info.get("codebase_version") != "v3.0" or int(info.get("fps", 0)) != 20
        or info.get("total_episodes") != 53 or info.get("total_frames") != 30846
        or manifest.get("source_root") != str(source)
        or manifest.get("source_frames") != info["total_frames"]
        or manifest.get("target_fields") != TARGET_NAMES):
        raise ValueError("Source/sidecar metadata or provenance mismatch")
    features = info["features"]
    if (features["action"].get("shape") != [6] or
        features["observation.state"].get("shape") != [6] or
        not {"observation.images.side", "observation.images.wrist"} <= set(features)):
        raise ValueError("Unexpected source Dataset schema")

    cart = pq.read_table(sidecar / "cartesian_actions.parquet")
    if cart.num_rows != info["total_frames"]:
        raise ValueError("Sidecar frame count differs")
    index = make_index(cart)
    vectors = cart["cartesian_action"].to_pylist()
    for i, v in enumerate(vectors):
        a = np.asarray(v, dtype=np.float64)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError(f"Bad 7D action at sidecar row {i}")
    files = sorted((source / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("No source Parquet")
    used = set()
    tables = {}
    for file in files:
        table = pq.read_table(file)
        if "action" not in table.column_names:
            raise ValueError(f"action missing: {file}")
        positions = []
        for row in zip(table["episode_index"].to_pylist(), table["frame_index"].to_pylist()):
            key = (int(row[0]), int(row[1]))
            if key not in index or key in used:
                raise ValueError(f"Missing or duplicate source frame: {key}")
            used.add(key)
            positions.append(index[key])
        # Validate episode/frame timestamps, not just matching counts.
        for row_time, offset in zip(table["timestamp"].to_pylist(), positions):
            delta = abs(float(row_time) - float(cart["timestamp"][offset].as_py()))
            if delta > 0.002:
                raise ValueError(f"Timestamp mismatch: {file}, {delta:.6f}s")
        replacement = pa.array([vectors[n] for n in positions], type=pa.list_(pa.float64(), 7))
        tables[file.relative_to(source)] = table.set_column(
            table.column_names.index("action"), "action", replacement)
    if len(used) != len(index):
        raise ValueError("Sidecar includes frames not present in source")

    try:
        shutil.copytree(source, output, symlinks=False)
        (output / "NOT_TRAINING_READY").write_text(
            "Candidate Dataset. Check 7D readback, normalization statistics and IK decoding before training.\n"
        )
        for relative, table in tables.items():
            pq.write_table(table, output / relative)
        updated = copy.deepcopy(info)
        updated["features"]["action"] = {
            "dtype": "float64", "shape": [7], "names": TARGET_NAMES,
        }
        (output / "meta/info.json").write_text(json.dumps(updated, indent=2) + "\n")
        # Inherited statistics are INVALID after changing action dimensionality.
        # Remove potential stale global and episode action stats rather than silently use them.
        for stale in (output / "meta/stats.json", output / "meta/episodes_stats.jsonl"):
            if stale.is_file():
                stale.rename(stale.with_suffix(stale.suffix + ".STALE_DO_NOT_USE"))
        for stale in (output / "meta/episodes/stats.jsonl",):
            if stale.is_file():
                stale.rename(stale.with_suffix(stale.suffix + ".STALE_DO_NOT_USE"))
        report = {
            "status": "candidate_not_training_ready",
            "episodes": info["total_episodes"], "frames": len(used),
            "action_shape": [7], "sidecar_source": str(sidecar),
            "videos": "independent copies of original videos",
            "warning": "Statistics, metadata schema compatibility and LeRobot readback must be validated before training.",
        }
        (output / "cartesian_conversion_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception:
        if output.exists():
            (output / "CONVERSION_FAILED").write_text("Conversion did not complete; do not train.\n")
        raise
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(convert(a.dataset, a.sidecar, a.output), indent=2))

if __name__ == "__main__":
    main()
