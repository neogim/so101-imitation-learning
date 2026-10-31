#!/usr/bin/env python3
"""Fit q_URDF = sign * q_LeRobot + offset from independent correspondences.

CSV columns: joint, lerobot_deg, urdf_deg, observation_id, measurement_source.
This offline tool never imports LeRobot robot classes or accesses hardware.  A
numerically good fit is deliberately reported as a candidate, never as proof of
physical validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
REQUIRED = ("joint", "lerobot_deg", "urdf_deg", "observation_id", "measurement_source")
NONINDEPENDENT_MARKERS = ("synthetic", "example", "dataset", "urdf limit", "inferred")


def fit_hypothesis(rows: list[dict], sign: int) -> dict:
    offsets = [row["urdf_deg"] - sign * row["lerobot_deg"] for row in rows]
    offset = sum(offsets) / len(offsets)
    residuals = [row["urdf_deg"] - (sign * row["lerobot_deg"] + offset) for row in rows]
    loo = []
    if len(rows) > 1:
        for index, row in enumerate(rows):
            other = offsets[:index] + offsets[index + 1 :]
            other_offset = sum(other) / len(other)
            loo.append(row["urdf_deg"] - (sign * row["lerobot_deg"] + other_offset))
    return {
        "sign": sign,
        "offset_deg": offset,
        "rmse_deg": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "mean_absolute_error_deg": sum(abs(value) for value in residuals) / len(residuals),
        "max_absolute_error_deg": max(abs(value) for value in residuals),
        "leave_one_out_rmse_deg": (
            math.sqrt(sum(value * value for value in loo) / len(loo)) if loo else None
        ),
    }


def independent_source(value: str) -> bool:
    lowered = value.strip().lower()
    return bool(lowered) and not any(marker in lowered for marker in NONINDEPENDENT_MARKERS)


def analyze_joint(rows: list[dict]) -> dict:
    distinct_angles = len({round(row["lerobot_deg"], 9) for row in rows})
    distinct_observations = len({row["observation_id"] for row in rows if row["observation_id"]})
    independent_rows = sum(independent_source(row["measurement_source"]) for row in rows)
    sufficient_diversity = distinct_angles >= 2
    sufficient_independent_references = distinct_observations >= 2 and independent_rows >= 2
    fits = [fit_hypothesis(rows, sign) for sign in (1, -1)] if rows else []
    preferred = min(fits, key=lambda item: item["rmse_deg"])["sign"] if fits else None
    return {
        "samples": len(rows),
        "distinct_lerobot_angles": distinct_angles,
        "distinct_observation_ids": distinct_observations,
        "independent_reference_rows": independent_rows,
        "sufficient_angle_diversity": sufficient_diversity,
        "sufficient_independent_references": sufficient_independent_references,
        "hypotheses": fits,
        "numerically_preferred_sign": preferred if sufficient_diversity else None,
        "mapping_status": "candidate_only" if sufficient_diversity and sufficient_independent_references else "insufficient_evidence",
        "physically_validated": False,
        "warning": "A fit is not physical validation; provenance and geometry must be reviewed independently.",
    }


def read_evidence(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            if row["joint"] not in JOINTS:
                raise ValueError(f"Line {line}: unknown joint {row['joint']!r}")
            try:
                parsed = {**row, "lerobot_deg": float(row["lerobot_deg"]), "urdf_deg": float(row["urdf_deg"])}
            except ValueError as error:
                raise ValueError(f"Line {line}: non-numeric angle") from error
            if not math.isfinite(parsed["lerobot_deg"]) or not math.isfinite(parsed["urdf_deg"]):
                raise ValueError(f"Line {line}: non-finite angle")
            grouped[row["joint"]].append(parsed)
    return grouped


def analyze(path: Path) -> dict:
    grouped = read_evidence(path)
    return {
        "model": "q_URDF_deg = sign * q_LeRobot_deg + offset_deg",
        "results": {joint: analyze_joint(grouped.get(joint, [])) for joint in JOINTS},
        "global_warning": "No result is automatically labeled physically validated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    args.output.write_text(json.dumps(analyze(args.evidence), indent=2) + "\n")


if __name__ == "__main__":
    main()
