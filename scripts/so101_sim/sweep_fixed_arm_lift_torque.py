#!/usr/bin/env python3
"""One-command torque sweep for SO-101 fixed-arm 5mm Muhi lift.

Simulates each J6 torque with identical physics and geometry; skips all
rendering for comparison. Optionally re-runs just the best condition
with the Git-history-restored five-view renderer for visual review.
This does NOT command the real robot.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).with_name("physical_grasp_fixed_arm_bench.py")


def one_run(root: Path, torque: float, args, *, video: bool) -> dict:
    label = f"{torque:.3f}".replace(".", "p")
    out = root / (f"visual_{label}" if video else f"torque_{label}")
    command = [
        sys.executable, str(BENCH),
        "--duration-s", str(args.duration_s),
        "--j5-deg", str(args.j5_deg),
        "--j6-speed-deg-s", str(args.j6_speed_deg_s),
        "--j6-max-torque-nm", str(torque),
        "--lift-mm", "5",
        "--lift-ramp-s", "1.0",
        "--bilateral-hold-steps", "150",
        "--render-every", "25",
        "--fps", "20",
        "--out", str(out)
    ]
    if not video:
        command.append("--no-video")
    print(f"\n=== {'VIDEO' if video else 'CSV'} {torque:.3f} Nm ===",flush=True)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout[-2200:],flush=True)
    if result.returncode:
        print(result.stderr[-2500:],file=sys.stderr,flush=True)
    row = {"torque_nm": torque, "video": video,
           "returncode": result.returncode, "output": str(out)}
    report = out / "summary.json"
    if report.is_file():
        s = json.loads(report.read_text(encoding="utf-8"))
        for key in ("status", "lift_success", "max_arm_command_lift_mm",
                    "max_object_rise_mm", "final_object_rise_mm",
                    "final_object_table_gap_approx_mm",
                    "peak_fixed_force_n", "peak_moving_force_n",
                    "bilateral_frames", "physics_steps", "video_status"):
            row[key] = s.get(key)
    else:
        row["status"] = "PROCESS_FAILED"
    if result.returncode and not report.exists():
        row["error_tail"] = result.stderr[-800:]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--torques", type=float, nargs="+",
                        default=[.12, .14, .16, .18, .20])
    parser.add_argument("--duration-s", type=float, default=4.5)
    parser.add_argument("--j5-deg", type=float, default=-90)
    parser.add_argument("--j6-speed-deg-s", type=float, default=35)
    parser.add_argument("--render-best", action="store_true",
                        help="After CSV sweep, generate one detailed 5-view MP4 at selected setting")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory exists; use a fresh timestamp")
    if not args.torques or len(args.torques)>12:
        parser.error("Specify between 1 and 12 torques")
    if len(set(args.torques)) != len(args.torques):
        parser.error("Torque values must be unique")
    if any(not math.isfinite(t) or not(.01 <= t <= .24) for t in args.torques):
        parser.error("All torques must be finite and 0.01..0.24 Nm")
    args.out.mkdir(parents=True, exist_ok=False)
    results = [one_run(args.out, t, args, video=False) for t in args.torques]
    valid = [row for row in results
             if row["returncode"] == 0 and
             row.get("status") not in ("SETUP_NOT_FOUND","INITIAL_GEOMETRY_NOT_CLEAR",
                                     "NUMERICAL_DIVERGENCE","PROCESS_FAILED")]
    # Prioritize actual successful physical lift, then observed positive Muhi
    # rise; do not choose the largest commanded torque by default.
    ranked = sorted(valid, key=lambda row: (
        int(bool(row.get("lift_success"))),
        float(row.get("max_object_rise_mm") or 0.),
        -row["torque_nm"]), reverse=True)
    best = ranked[0] if ranked else None
    if best and args.render_best:
        results.append(one_run(args.out,best["torque_nm"],args,video=True))
    with (args.out / "torque_comparison.csv").open("x",newline="",encoding="utf-8") as f:
        keys = list(dict.fromkeys(k for r in results for k in r))
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    status = {
        "source": "headless MuJoCo dynamics; identical fixed-arm J6-first 5mm task",
        "torques_nm": args.torques,
        "duration_s": args.duration_s,
        "selected_torque_nm": best["torque_nm"] if best else None,
        "selected_has_verified_lift": bool(best and best.get("lift_success")),
        "comparison": results,
        "warning": [
            "This is not a test of actual robot motor ratings or safety.",
            "Max object rise is NOT the same as sustained lift; use lift_success and table normal force.",
            "When no candidate lifts, the selected run is exploratory and is not a grasp success.",
            "Physical wrist 3 + side 2 layout is inherited from earlier physical_grasp_table_bench rendering path.",
            "Each candidate re-compiles model and repeats geometry search; no GPU required."
        ]
    }
    (args.out / "sweep_summary.json").write_text(json.dumps(status,indent=2,ensure_ascii=False)+"\n")
    print("\n=== Sweep result ===")
    for row in results:
        print(f"{row['torque_nm']:.3f}Nm  {row.get('status')}  "
              f"rise={row.get('max_object_rise_mm')}mm  "
              f"final={row.get('final_object_rise_mm')}mm")
    print("Best candidate (NOT automatically success):",status["selected_torque_nm"])
    print("Results:",args.out)


if __name__=="__main__":
    main()
