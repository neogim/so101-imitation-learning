#!/usr/bin/env python3
"""Summarize and gate offline finite-candidate DP quality without printing huge JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def summarize(data: dict, mappings: list[str]) -> tuple[list[dict], bool]:
    expected = data["frames"][1] - data["frames"][0] + 1
    rows = []
    all_pass = True
    for mapping in mappings:
        if mapping not in data["results"]:
            raise ValueError(f"Mapping not found: {mapping}")
        dirs = data["results"][mapping]
        if set(dirs) != {"x+", "x-", "y+", "y-", "z+", "z-"}:
            raise ValueError(f"Missing directions in {mapping}")
        for direction in ("x+", "x-", "y+", "y-", "z+", "z-"):
            d = dirs[direction]
            m = d["methods"]["dp_second_order"]
            covered = m["coverage"]
            passed = bool(m["quality_pass"] and covered == expected and not d["missing_frames"])
            all_pass &= passed
            metrics = m["segment_metrics"]
            def peak(key):
                vals=[x[key]["value"] for x in metrics if x.get(key) is not None]
                return max(vals) if vals else None
            rows.append({"mapping":mapping,"direction":direction,"coverage":covered,"expected":expected,
                         "pass":passed,"max_deviation_deg":peak("deviation_peak"),
                         "max_jump_deg":peak("jump_peak"),"max_velocity_deg_s":peak("velocity_peak"),
                         "max_acceleration_deg_s2":peak("acceleration_peak"),
                         "max_residual_mm":max((x["residual_max_mm"] for x in metrics),default=None),
                         "max_orientation_deg":max((x["orientation_max_deg"] for x in metrics),default=None)})
    return rows, all_pass

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",required=True,type=Path)
    p.add_argument("--mappings",nargs="+",default=["shoulder+_elbow+","shoulder-_elbow-"])
    p.add_argument("--output",type=Path,help="Optional new compact JSON path; never contains raw candidates")
    a=p.parse_args()
    rows,go=summarize(json.loads(a.summary.read_text()),a.mappings)
    for mapping in a.mappings:
        selected=[r for r in rows if r["mapping"]==mapping]
        print(f"{mapping}: {sum(r['pass'] for r in selected)}/6 PASS")
        for r in selected:
            print(f" {r['direction']:2} {r['coverage']}/{r['expected']} {'PASS' if r['pass'] else 'FAIL'}"
                  f" | jump {r['max_jump_deg']} | speed {r['max_velocity_deg_s']}"
                  f" | accel {r['max_acceleration_deg_s2']}")
    print("FULL_NUMERICAL_GATE:", "PASS" if go else "STOP")
    if a.output:
        if a.output.exists():p.error("output already exists")
        a.output.write_text(json.dumps({"classification":"CONDITIONAL_NUMERICAL_SENSITIVITY",
            "all_directions_pass":go,"rows":rows,"authorization":"offline_numerics_only"},indent=2)+"\n")

if __name__=="__main__":
    main()
