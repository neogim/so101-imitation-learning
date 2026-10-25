#!/usr/bin/env python3
"""Read-only audit of Codex SO-101 FK grasp-pose CSV.

No robot imports, IK commands, dataset writes, or control calls.
Coordinates are in meters; orientation fields are rotation-vector components
in radians as provided by the producer (not interpreted here).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

FIELDS = ("ep", "open", "close_start", "grasp", "x", "y", "z", "rx", "ry", "rz", "sigma_min")


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
        data = []
        for line, raw in enumerate(reader, 2):
            try:
                row = {name: float(raw[name]) for name in FIELDS if name != "ep"}
                row["ep"] = int(raw["ep"])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid numeric row at line {line}") from exc
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"Non-finite value at line {line}")
            if not (0 <= row["open"] <= row["close_start"] <= row["grasp"]):
                raise ValueError(f"Invalid phase order at line {line}")
            data.append(row)
    if not data:
        raise ValueError("Empty grasp-pose CSV")
    if len({row["ep"] for row in data}) != len(data):
        raise ValueError("Duplicate episode IDs")
    return data


def percentile(values: list[float], percent: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def summarize(rows: list[dict]) -> dict:
    center = {axis: statistics.mean(row[axis] for row in rows) for axis in ("x", "y", "z")}
    distances = [
        math.sqrt(sum((row[axis] - center[axis]) ** 2 for axis in center)) * 1000
        for row in rows
    ]
    medoid = min(
        rows,
        key=lambda r: sum(
            math.sqrt(sum((r[axis] - s[axis]) ** 2 for axis in center))
            for s in rows
        ),
    )
    return {
        "episodes": len(rows),
        "tcp_frame": "gripper_frame_link (as reported by analysis source; verify URDF)",
        "center_mm": {axis: round(value * 1000, 3) for axis, value in center.items()},
        "axis_stddev_mm": {
            axis: round(statistics.pstdev([row[axis] for row in rows]) * 1000, 3)
            for axis in center
        },
        "distance_to_mean_mm": {
            "p50": round(percentile(distances, 50), 3),
            "p90": round(percentile(distances, 90), 3),
            "p95": round(percentile(distances, 95), 3),
            "max": round(max(distances), 3),
            "rms": round(math.sqrt(statistics.mean(d * d for d in distances)), 3),
        },
        "position_medoid_episode": medoid["ep"],
        "singularity_proxy_sigma_min": {
            "min": round(min(row["sigma_min"] for row in rows), 6),
            "max": round(max(row["sigma_min"] for row in rows), 6),
        },
        "warnings": [
            "grasp is a gripper-action heuristic, not a verified physical contact time",
            "sigma_min alone cannot certify collision safety or full-pose nonsingularity",
            "CSV cannot validate URDF/calibration or camera-to-robot extrinsics",
            "FK-derived position labels alone do not create new image-conditioned ACT demonstrations",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Existing FK analysis CSV (read-only)")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional NEW JSON file only")
    args = parser.parse_args()
    result = summarize(load_rows(args.csv))
    report = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out is not None:
        if args.json_out.resolve() == args.csv.resolve():
            parser.error("--json-out must not overwrite input CSV")
        with args.json_out.open("x", encoding="utf-8") as out:
            out.write(report)
    print(report, end="")


if __name__ == "__main__":
    main()
