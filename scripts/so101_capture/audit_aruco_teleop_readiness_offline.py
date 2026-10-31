#!/usr/bin/env python3
"""Offline gate for J2/J3 ArUco teleop: analyze current USB samples and calibration.

No camera, serial, motion or training dependencies. Never authorizes motor movement.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

JOINTS = {"j2": "shoulder_lift", "j3": "elbow_flex"}
URDF = {"j2": (-100., 100.), "j3": (-96.8298673771, 96.8298673771)}

def summarize(values):
    if not values:
        return None
    values = sorted(values)
    return {"count": len(values), "min": round(values[0], 3),
            "max": round(values[-1], 3),
            "mean": round(statistics.mean(values), 3),
            "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--follower-calibration", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("Output exists; choose a new path")
    with a.samples.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        p.error("No samples")
    required = ["j2_follower_deg", "j3_follower_deg", "side_image_age_ms",
                "j2_image_relative_deg", "j3_image_relative_deg"]
    for key in required:
        if key not in rows[0]:
            p.error(f"Missing column {key}")
    def column(name):
        values = []
        for row in rows:
            if row[name] not in ("", None):
                number = float(row[name])
                if not math.isfinite(number):
                    p.error(f"Nonfinite {name}")
                values.append(number)
        return values
    camera_age = summarize(column("side_image_age_ms"))
    camera_angles = {j: summarize(column(f"{j}_image_relative_deg")) for j in JOINTS}
    calibrated = None
    if a.follower_calibration is not None:
        calibrated = json.loads(a.follower_calibration.read_text(encoding="utf-8"))
    joints = {}
    for j, name in JOINTS.items():
        vals = column(f"{j}_follower_deg")
        stats = summarize(vals)
        urdf_lo, urdf_hi = URDF[j]
        current = stats["mean"] if stats else None
        margin = None if current is None else {
            "below_urdf_lower_deg": round(current-urdf_lo, 3),
            "below_urdf_upper_deg": round(urdf_hi-current, 3)
        }
        cal = None
        if calibrated is not None:
            c = calibrated[name]
            cal = {"raw_min": c["range_min"], "raw_max": c["range_max"],
                   "homing_offset": c.get("homing_offset"), "drive_mode": c.get("drive_mode")}
        joints[j] = {"motor": name, "servo_angle": stats,
                     "urdf_reference_limits_deg": [urdf_lo, urdf_hi],
                     "urdf_margin_if_same_reference": margin,
                     "motor_calibration": cal,
                     "mapping": "UNKNOWN: motor degrees cannot be directly compared to URDF joint coordinates"}
    unstable = any(x and x["std"] > .5 for x in (v["servo_angle"] for v in joints.values()))
    report = {
        "mode": "OFFLINE_REVIEW_ONLY", "samples": len(rows),
        "camera_host_age_ms": camera_age,
        "image_relative_angles": camera_angles,
        "joints": joints,
        "stationary_follower_check": "CHECK_MOTION_OR_NOISE" if unstable else "STABLE_IN_THIS_CAPTURE",
        "motion_authorized": False,
        "blockers": [
            "Do not infer safe motion direction from URDF limits until motor/URDF sign and zero offset mapping is validated.",
            "J2 is currently near/beyond the nominal -100 degree URDF bound in raw motor coordinates; this is not a verified physical limit.",
            "Live teleop integration has not been tested with the hardware; do not treat this report as motion clearance.",
            "Confirm collision-free space, calibration bounds, leader-follower alignment and supported arm before motor torque changes."
        ]
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Output:", a.out)

if __name__ == "__main__":
    main()
