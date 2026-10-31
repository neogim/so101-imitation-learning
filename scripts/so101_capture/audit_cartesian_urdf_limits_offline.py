#!/usr/bin/env python3
"""Offline source episode URDF bounds audit with joint/frame detail and offset search.
No motors, no URDF changes, no claim of physical calibration.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from batch_cartesian_fk_ik_hypotheses_offline import HYPOTHESES, load_episode, mapping, to_urdf_rad
from export_cartesian_action_sidecar import JOINT_NAMES
from generate_offset_ik_offline import joint_limits

def audit(rows, lower, upper, signs, offsets):
    violations = []
    joint_counts = {j: {"state":0,"action":0,"either":0} for j in JOINT_NAMES}
    for row in rows:
        st=np.asarray(row["observation.state"][:5],float)
        ac=np.asarray(row["action"][:5],float)
        qst=to_urdf_rad(st,signs,offsets)
        qac=to_urdf_rad(ac,signs,offsets)
        bad_st=(qst<lower-1e-9)|(qst>upper+1e-9)
        bad_ac=(qac<lower-1e-9)|(qac>upper+1e-9)
        frame=int(row["frame_index"])
        for i,j in enumerate(JOINT_NAMES):
            joint_counts[j]["state"]+=int(bad_st[i])
            joint_counts[j]["action"]+=int(bad_ac[i])
            joint_counts[j]["either"]+=int(bad_st[i] or bad_ac[i])
            if bad_st[i] or bad_ac[i]:
                violations.append(dict(frame=frame,joint=j,
                    state_motor_deg=float(st[i]),action_motor_deg=float(ac[i]),
                    state_urdf_deg=float(np.rad2deg(qst[i])),
                    action_urdf_deg=float(np.rad2deg(qac[i])),
                    urdf_lower_deg=float(np.rad2deg(lower[i])),
                    urdf_upper_deg=float(np.rad2deg(upper[i])),
                    state_violation=bool(bad_st[i]),action_violation=bool(bad_ac[i])))
    any_frames=sorted(set(x["frame"] for x in violations))
    return joint_counts,any_frames,violations

def run(args):
    root=args.dataset.resolve()
    output=args.output.resolve()
    if output.exists() or root==output or root in output.parents:
        raise ValueError("New output outside source dataset required")
    if args.offset_step_deg<=0 or args.max_abs_offset_deg<0:
        raise ValueError("Bad offset range")
    rows=load_episode(root,args.episode)
    lo,hi=joint_limits(args.urdf)
    summary={}
    details=[]
    sweep=[]
    options=np.arange(-args.max_abs_offset_deg,args.max_abs_offset_deg+args.offset_step_deg*.25,args.offset_step_deg)
    # Note: search minimizes LIMIT VIOLATIONS ONLY. Cannot prove offset physically.
    for label in HYPOTHESES:
        signs,offset=mapping(label)
        counts,frames,viol=audit(rows,lo,hi,signs,offset)
        summary[label]={"total_frames":len(rows),"violating_frames":len(frames),
                        "joint_counts":counts,"violating_frame_indices":frames}
        details.extend(dict(hypothesis=label,**x) for x in viol)
        best=None
        for j2 in options:
            for j3 in options:
                off=np.array([0.,j2,j3,0.,0.])
                # Optimize J2/J3 violations without repeated full per-frame dict generation.
                motor_st=np.array([x["observation.state"][:5] for x in rows],dtype=float)
                motor_ac=np.array([x["action"][:5] for x in rows],dtype=float)
                qst=np.deg2rad(motor_st*signs+off)
                qac=np.deg2rad(motor_ac*signs+off)
                bad=np.any((qst<lo)|(qst>hi)|(qac<lo)|(qac>hi),axis=1)
                candidate=(int(bad.sum()),float(abs(j2)+abs(j3)),float(j2),float(j3))
                if best is None or candidate<best:best=candidate
        sweep.append({"hypothesis":label,"minimum_violating_frames":best[0],
                      "selected_j2_offset_deg":best[2],"selected_j3_offset_deg":best[3],
                      "tie_break":"smallest sum of absolute offsets",
                      "not_physical_calibration":True})
    output.mkdir(parents=True)
    result={"classification":"OFFLINE_URDF_BOUNDS_AUDIT",
            "episode":args.episode,"source_frames":len(rows),
            "urdf_joint_bounds_deg":{name:[float(np.rad2deg(a)),float(np.rad2deg(b))]
                                      for name,a,b in zip(JOINT_NAMES,lo,hi)},
            "hypotheses":summary,"offset_sweep":sweep,
            "warnings":["Changing a URDF limit to cover data is not evidence of physical reachability",
                        "Offset sweep is a limit-fit heuristic, not measured physical joint zero",
                        "No collision or hardware clearance validation"]}
    (output/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    with (output/"violations.csv").open("w",newline="") as f:
        if details:
            w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    print(json.dumps({"hypotheses":{k:{"violating_frames":v["violating_frames"],
                                     "joint_counts":v["joint_counts"]} for k,v in summary.items()},
                      "offset_sweep":sweep},indent=2))
    print("Saved",output)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--max-abs-offset-deg",type=float,default=6.)
    p.add_argument("--offset-step-deg",type=float,default=1.)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())

if __name__=="__main__":main()
