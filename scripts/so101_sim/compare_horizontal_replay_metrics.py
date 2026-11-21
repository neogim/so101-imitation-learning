#!/usr/bin/env python3
"""Compare two horizontal MuJoCo replay traces without rerunning physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_run(path: Path) -> tuple[dict, pd.DataFrame]:
    metadata = json.loads((path / "replay_comparison.json").read_text())
    trace = pd.read_csv(path / "joint_contact_trace.csv")
    return metadata, trace


def metrics(metadata: dict, trace: pd.DataFrame) -> dict:
    physics = metadata["physics"]
    return {
        "maximum_fixed_jaw_rise_mm": float(trace.fixed_jaw_centroid_delta_z_mm.max()),
        "maximum_arm_tracking_error_deg": float(physics["maximum_arm_tracking_error_deg"]),
        "maximum_muhi_xy_displacement_mm": float(physics["max_xy_displacement_mm"]),
        "final_muhi_xy_displacement_mm": float(physics["final_xy_displacement_mm"]),
        "maximum_normal_contact_force_n": float(physics["max_normal_contact_force_n"]),
        "bilateral_hold_fraction": float(physics["bilateral_hold_fraction"]),
        "deep_penetration_over_3mm": bool(physics["deep_penetration_over_3mm"]),
        "source_geometry_valid": metadata.get("source_geometry_valid"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--before-label", default="Baseline")
    parser.add_argument("--after-label", default="Recorded-hold candidate")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    before_meta, before = load_run(args.before)
    after_meta, after = load_run(args.after)
    report = {
        "before": metrics(before_meta, before),
        "after": metrics(after_meta, after),
        "interpretation_guard": "A lower jaw rise is not grasp success; source geometry validity, penetration, object motion, bilateral contact, and force must also pass.",
    }
    args.out.mkdir(parents=True)
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for label, data in ((args.before_label, before), (args.after_label, after)):
        axes[0].plot(data.time_s, data.fixed_jaw_centroid_delta_z_mm, label=label)
        axes[1].plot(data.time_s, data.bottle_xy_displacement_mm, label=label)
        axes[2].plot(data.time_s, data.max_arm_error_deg, label=label)
    axes[0].set_ylabel("Fixed-jaw Z change (mm)")
    axes[1].set_ylabel("Muhi XY displacement (mm)")
    axes[2].set_ylabel("Max J1–J5 error (deg)")
    axes[2].set_xlabel("Simulation time (s)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("Horizontal MuJoCo replay: baseline vs measured-trace-informed candidate")
    fig.savefig(args.out / "before_after.png", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
