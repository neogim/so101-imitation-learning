#!/usr/bin/env python3
"""One-pass offline review: IK candidate coverage, J5/J2 branch spread, DP2 discontinuities, and pass gates.

Reads existing independent-seed per_seed.json and DP selected_paths/summary.json.
Does not perform FK/IK or hardware operations.
"""
import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
DIRECTIONS=("x+","x-","y+","y-","z+","z-")
LIMITS={"jump":5.0,"speed":100.0,"acceleration":2000.0}
def analyze(candidates, paths, summary, mapping, start, end):
    by_seed=defaultdict(list)
    for x in candidates:
        if x["mapping"]==mapping and start<=int(x["frame"])<=end and x["direction"] in DIRECTIONS and x["feasible_witness"] and float(x["position_residual_mm"])<=2:
            by_seed[(x["direction"],int(x["frame"]))].append(x)
    by_path={}
    for x in paths:
        if x["mapping"]==mapping and x["method"]=="dp_second_order" and start<=int(x["frame"])<=end:
            key=(x["direction"],int(x["frame"]))
            if key in by_path:raise ValueError("duplicate DP selected path "+str(key))
            by_path[key]=x["candidate"]
    output={"mapping":mapping,"frames":[start,end],"directions":{}}
    for direction in DIRECTIONS:
        records=[]
        for frame in range(start,end+1):
            c=by_path.get((direction,frame))
            if (direction,frame) not in by_path:raise ValueError(f"Missing DP trace {direction} {frame}")
            seeds=by_seed[(direction,frame)]
            if c is None:
                records.append({"frame":frame,"missing":True,"seed_count":len(seeds)});continue
            motor=np.array(c["motor"],dtype=float)
            j2=np.array([x["candidate_motor_deg"][1] for x in seeds],dtype=float)
            j5=np.array([x["candidate_motor_deg"][4] for x in seeds],dtype=float)
            records.append({"frame":frame,"missing":False,"motor":motor.tolist(),"seed":c["seed"],
                "bounds":c["active_bound_joints"],"seed_count":len(seeds),
                "j2_range_deg":float(np.ptp(j2)) if len(j2) else None,
                "j5_range_deg":float(np.ptp(j5)) if len(j5) else None,
                "orientation_deg":float(c["orientation_change_deg"]),
                "residual_mm":float(c["position_residual_mm"])})
        events=[]
        for a,b in zip(records,records[1:]):
            if a.get("missing") or b.get("missing"):continue
            delta=np.array(b["motor"])-np.array(a["motor"])
            for j, joint in enumerate(JOINTS):
                if abs(delta[j])>LIMITS["jump"]:
                    events.append({"from":a["frame"],"to":b["frame"],"joint":joint,"jump_deg":float(abs(delta[j])),
                        "old_seed":a["seed"],"new_seed":b["seed"],"old_bounds":a["bounds"],"new_bounds":b["bounds"],
                        "old_j5_spread":a["j5_range_deg"],"new_j5_spread":b["j5_range_deg"],
                        "old_j2_spread":a["j2_range_deg"],"new_j2_spread":b["j2_range_deg"]})
        m=summary["results"][mapping][direction]["methods"]["dp_second_order"]
        metrics=m["segment_metrics"]
        def mx(k):
            vals=[x[k]["value"] for x in metrics if x.get(k)]
            return max(vals) if vals else None
        output["directions"][direction]={
            "coverage":m["coverage"],"expected":end-start+1,"quality_pass":m["quality_pass"],
            "max_jump_deg":mx("jump_peak"),"max_speed_deg_s":mx("velocity_peak"),
            "max_accel_deg_s2":mx("acceleration_peak"),
            "max_residual_mm":max((v["residual_max_mm"] for v in metrics),default=None),
            "max_deviation_deg":mx("deviation_peak"),
            "max_orientation_deg":max((v["orientation_max_deg"] for v in metrics),default=None),
            "max_j5_candidate_spread_deg":max((v["j5_range_deg"] for v in records if v.get("j5_range_deg") is not None),default=None),
            "max_j2_candidate_spread_deg":max((v["j2_range_deg"] for v in records if v.get("j2_range_deg") is not None),default=None),
            "violating_jumps":events[:20],"total_jumps_over_5deg":len(events),
            "missing_candidate_frames":[r["frame"] for r in records if r.get("missing")],
        }
    output["pass_directions"]=sum(x["quality_pass"] for x in output["directions"].values())
    output["complete_pass"]=output["pass_directions"]==6
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("candidates","paths","summary","output"):p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--start",type=int,default=0);p.add_argument("--end",type=int,default=119)
    p.add_argument("--mappings",nargs="+",default=["shoulder+_elbow+","shoulder-_elbow-"])
    a=p.parse_args()
    if a.output.exists():p.error("output exists, use a new path")
    data=[json.loads(pth.read_text()) for pth in (a.candidates,a.paths,a.summary)]
    results=[analyze(*data,m,a.start,a.end) for m in a.mappings]
    report={"classification":"CONDITIONAL_NUMERICAL_SENSITIVITY",
        "source":"local offline candidate and trajectory JSON","mapping_results":results,
        "full_all_mapping_gate":all(x["complete_pass"] for x in results),
        "notes":["Not a physical calibration or collision/torque check.",
                 "J2/J5 spread is a finite-seed candidate distribution, not proven IK null-space dimension.",
                 "Never expand automatically on the basis of this tool."]}
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    for result in results:
        print(result["mapping"],f"{result['pass_directions']}/6 PASS")
        for direction,v in result["directions"].items():
            print(f" {direction}: {'PASS' if v['quality_pass'] else 'FAIL'}; jumps>5°={v['total_jumps_over_5deg']}; "
                  f"J5 spread max={v['max_j5_candidate_spread_deg']:.3f}°; "
                  f"jump={v['max_jump_deg']}; accel={v['max_accel_deg_s2']}")
            for e in v["violating_jumps"][:3]:
                print(f"    f{e['from']}→f{e['to']} {e['joint']} {e['jump_deg']:.3f}°, "
                      f"J5 spread {e['old_j5_spread']:.2f}→{e['new_j5_spread']:.2f}°")
    print("ALL-MAPPING GATE:","PASS" if report["full_all_mapping_gate"] else "STOP")
if __name__=="__main__":main()
