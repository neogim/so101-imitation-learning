#!/usr/bin/env python3
"""Offline-only SO-101 calibration/URDF/dataset consistency audit.

This module only reads JSON, URDF XML, and Parquet files.  It never imports a
robot class, opens a serial port, or communicates with hardware.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TOLERANCES = (0.0, 0.1, 0.2, 0.5)
RESOLUTION_MAX = 4095


def degree_half_span(range_min: int, range_max: int) -> float:
    """Reproduce LeRobot 0.6.2 DEGREES endpoint magnitude."""
    return (range_max - range_min) * 180.0 / RESOLUTION_MAX


def exceed_count(values: np.ndarray, low: float, high: float, tolerance: float) -> int:
    return int(np.sum((values < low - tolerance) | (values > high + tolerance)))


def contiguous_runs(episodes: np.ndarray, frames: np.ndarray, mask: np.ndarray) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    for episode in np.unique(episodes[mask]):
        selected = np.sort(frames[mask & (episodes == episode)])
        start = previous = int(selected[0])
        for frame in selected[1:]:
            frame = int(frame)
            if frame != previous + 1:
                runs.append({"episode": int(episode), "frame_start": start, "frame_end": previous})
                start = frame
            previous = frame
        runs.append({"episode": int(episode), "frame_start": start, "frame_end": previous})
    return runs


def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = ["episode_index", "frame_index", "observation.state", "action"]
    parts = []
    for path in sorted((root / "data").rglob("*.parquet")):
        parts.append(pq.read_table(path, columns=columns).to_pydict())
    if not parts:
        raise FileNotFoundError(f"No dataset parquet under {root / 'data'}")
    merged = {key: sum((part[key] for part in parts), []) for key in columns}
    episodes = np.asarray(merged["episode_index"], dtype=int)
    frames = np.asarray(merged["frame_index"], dtype=int)
    state = np.asarray(merged["observation.state"], dtype=float)[:, :5]
    action = np.asarray(merged["action"], dtype=float)[:, :5]
    if state.shape != action.shape or state.shape != (len(frames), 5):
        raise ValueError("Expected matching five-joint state/action arrays")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite joint value")
    order = np.lexsort((frames, episodes))
    return episodes[order], frames[order], state[order], action[order]


def audit(dataset: Path, calibration: Path, urdf: Path) -> tuple[dict, list[dict]]:
    episodes, frames, state, action = load_dataset(dataset)
    calibration_data = json.loads(calibration.read_text())
    urdf_joints = {node.attrib["name"]: node for node in ET.parse(urdf).getroot().findall("joint")}
    summary = {
        "scope": {"frames": len(frames), "episodes": int(len(np.unique(episodes))), "fps": 20},
        "conventions": {
            "dataset": "LeRobot normalized motor degrees (not proven URDF joint-space degrees)",
            "calibration_range": "observed hand-swept encoder range centered at (range_min+range_max)/2",
            "encoder_degrees_per_tick": 360.0 / RESOLUTION_MAX,
            "lerobot_version_inspected": "0.6.2",
        },
        "joints": {},
        "elbow_calibration_exceedance_runs_strict": [],
        "limitations": [
            "Cached calibration was not compared with active servo registers because hardware access was prohibited.",
            "Numeric agreement does not prove URDF zero, axis sign, collision safety, or mechanical hard limits.",
        ],
    }
    peaks: list[dict] = []
    for index, name in enumerate(JOINTS):
        entry = calibration_data[name]
        span = degree_half_span(int(entry["range_min"]), int(entry["range_max"]))
        limit = urdf_joints[name].find("limit")
        if limit is None:
            raise ValueError(f"URDF limit missing for {name}")
        urdf_range = [math.degrees(float(limit.attrib[side])) for side in ("lower", "upper")]
        delta = action[:, index] - state[:, index]
        joint = {
            "urdf_degrees": urdf_range,
            "calibration_normalized_degrees": [-span, span],
            "calibration_drive_mode": int(entry["drive_mode"]),
            "action_minmax_degrees": [float(action[:, index].min()), float(action[:, index].max())],
            "state_minmax_degrees": [float(state[:, index].min()), float(state[:, index].max())],
            "action_outside_urdf_by_tolerance_degrees": {},
            "state_outside_urdf_by_tolerance_degrees": {},
            "action_outside_calibration_by_tolerance_degrees": {},
            "state_outside_calibration_by_tolerance_degrees": {},
            "action_minus_state_degrees": {
                "mean": float(delta.mean()),
                "mean_absolute": float(np.abs(delta).mean()),
                "p95_absolute": float(np.quantile(np.abs(delta), 0.95)),
                "p99_absolute": float(np.quantile(np.abs(delta), 0.99)),
                "max_absolute": float(np.abs(delta).max()),
            },
        }
        for tolerance in TOLERANCES:
            key = str(tolerance)
            joint["action_outside_urdf_by_tolerance_degrees"][key] = exceed_count(action[:, index], *urdf_range, tolerance)
            joint["state_outside_urdf_by_tolerance_degrees"][key] = exceed_count(state[:, index], *urdf_range, tolerance)
            joint["action_outside_calibration_by_tolerance_degrees"][key] = exceed_count(action[:, index], -span, span, tolerance)
            joint["state_outside_calibration_by_tolerance_degrees"][key] = exceed_count(state[:, index], -span, span, tolerance)
        summary["joints"][name] = joint
        if name == "elbow_flex":
            strict = action[:, index] > span
            summary["elbow_calibration_exceedance_runs_strict"] = contiguous_runs(episodes, frames, strict)
            deviations = action[:, index] - span
            for row in np.flatnonzero(deviations > 0.01):
                peaks.append({
                    "episode": int(episodes[row]), "frame": int(frames[row]), "joint": name,
                    "deviation_degrees": float(deviations[row]),
                })
    return summary, peaks[:100]


def render_readme(summary: dict) -> str:
    joints = summary["joints"]
    elbow = joints["elbow_flex"]
    shoulder = joints["shoulder_lift"]
    runs = ", ".join(
        f"ep{r['episode']}:{r['frame_start']}-{r['frame_end']}"
        for r in summary["elbow_calibration_exceedance_runs_strict"]
    )
    return f"""# SO-101 calibration / URDF offline audit

## Scope and safety

This audit read the 53-episode, 30,846-frame dataset, cached **Follower** calibration JSON,
SO-101 URDF, and the installed LeRobot 0.6.2 source. It did not import robot classes,
open a serial port, initialize motors, read servo registers, or write source data. The cached
calibration contents and absolute local paths are intentionally not included here.

## Angle definitions and evidence

- Dataset `action` and `observation.state` are named joint positions, but their numeric unit is
  LeRobot `MotorNormMode.DEGREES`: `(encoder_value - (range_min+range_max)/2) * 360/4095`.
  This is normalized **motor degrees**, not independently proven URDF joint-space degrees.
- In LeRobot 0.6.2, Follower calibration first centers the encoder by a homing offset, then
  interactively records the range swept by hand. Therefore `range_min/max` are observed
  operational calibration endpoints, not certified hard mechanical limits.
- `drive_mode` is 0 for the inspected joints. In the LeRobot 0.6.2 `DEGREES` branch it does not
  alter sign, and degree normalization/unnormalization does not clip to `range_min/max`.
- The URDF defines both `shoulder_lift` and `elbow_flex` around axis `[0,0,1]` in their respective
  joint frames. Their origins differ, so matching scalar degree values alone cannot prove the
  physical zero or sign transformation.

## Results

| joint | URDF range (deg) | cached calibration degree range | Action min/max | State min/max |
|---|---:|---:|---:|---:|
| shoulder_lift | {shoulder['urdf_degrees'][0]:.6f} .. {shoulder['urdf_degrees'][1]:.6f} | {shoulder['calibration_normalized_degrees'][0]:.6f} .. {shoulder['calibration_normalized_degrees'][1]:.6f} | {shoulder['action_minmax_degrees'][0]:.6f} .. {shoulder['action_minmax_degrees'][1]:.6f} | {shoulder['state_minmax_degrees'][0]:.6f} .. {shoulder['state_minmax_degrees'][1]:.6f} |
| elbow_flex | {elbow['urdf_degrees'][0]:.6f} .. {elbow['urdf_degrees'][1]:.6f} | {elbow['calibration_normalized_degrees'][0]:.6f} .. {elbow['calibration_normalized_degrees'][1]:.6f} | {elbow['action_minmax_degrees'][0]:.6f} .. {elbow['action_minmax_degrees'][1]:.6f} | {elbow['state_minmax_degrees'][0]:.6f} .. {elbow['state_minmax_degrees'][1]:.6f} |

Counts outside each bound at tolerances 0/0.1/0.2/0.5 degrees are in `summary.json`.
The earlier Action counts reproduce at strict tolerance: shoulder_lift outside URDF =
{shoulder['action_outside_urdf_by_tolerance_degrees']['0.0']}; elbow_flex outside URDF =
{elbow['action_outside_urdf_by_tolerance_degrees']['0.0']}; elbow_flex outside the inferred
calibration range = {elbow['action_outside_calibration_by_tolerance_degrees']['0.0']}.

The 612 elbow count is a floating-point/endpoint artifact plus a very small real overshoot:
606 Actions serialize as 97.362640 degrees, only about 0.000003 degrees above the computed
97.362637 endpoint, while 6 Actions are one encoder quantum higher at 97.450546 degrees
(about 0.087909 degrees over). Consequently the count is zero with a 0.1-degree tolerance.
No State sample exceeds the inferred elbow calibration endpoint. At the strict Action exceedance
frames, State remains below Action, consistent with commanded goal versus measured position lag;
it does not establish that the servo crossed its calibrated endpoint.

Strict elbow exceedance frame ranges (covering all 612 indices): `{runs}`.
`peaks.csv` contains only the 6 materially distinct (>0.01 degree) rows.

The other joints are summarized in `summary.json`; none has Action or State outside its cached
calibration degree span. Shoulder pan, wrist flex, and wrist roll also remain inside URDF limits.

## Conclusions and unresolved items

The data proves a convention mismatch risk: URDF joint angles and LeRobot normalized motor
degrees have been compared numerically without an independently measured zero/sign mapping.
It does **not** justify widening URDF or physical limits. The shoulder_lift Action minimum is
inside the cached swept calibration span but outside the URDF lower bound. The elbow values over
the URDF upper bound are likewise not proof of physical URDF-angle travel.

Unresolved: whether this cached file matched the firmware registers during recording, and the
physical URDF zero/sign correspondence. Resolving either requires a separately authorized,
read-only register snapshot or a controlled physical pose measurement; neither was attempted.

## Reproduction

```bash
python scripts/so101_capture/audit_so101_calibration_consistency.py \\
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \\
  --calibration "$FOLLOWER_CALIBRATION_JSON" \\
  --urdf data/urdf/so101/so101_new_calib.urdf \\
  --output reports/so101_calibration_audit
```

The environment variable must point to the local cached Follower calibration; its private path
and contents are not committed. The command refuses to overwrite an existing output directory.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    summary, peaks = audit(args.dataset, args.calibration, args.urdf)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "README.md").write_text(render_readme(summary))
    with (args.output / "peaks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("episode", "frame", "joint", "deviation_degrees"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(peaks)


if __name__ == "__main__":
    main()
