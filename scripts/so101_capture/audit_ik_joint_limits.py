#!/usr/bin/env python3
"""Audit a previous IK replay against recorded joint goals and URDF limits.

Read-only. It flags differences between sampled consecutive IK solutions;
sampled jumps are NOT per-frame velocity. Does not certify collision safety.
"""
from __future__ import annotations
import argparse,csv,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

NAMES=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
def audit(replay:Path,urdf:Path,output:Path):
    if output.exists():raise FileExistsError(output)
    rows=json.loads(replay.read_text())
    if len(rows)<2:raise ValueError("Need at least two sampled frames")
    ids=np.array([int(r["frame"]) for r in rows])
    if not np.all(np.diff(ids)>0):raise ValueError("Frames not increasing")
    q=np.array([r["ik_arm_deg"] for r in rows],dtype=float)
    goal=np.array([r["recorded_goal_arm_deg"] for r in rows],dtype=float)
    obs=np.array([r["observed_arm_deg"] for r in rows],dtype=float)
    if q.shape!=(len(rows),5) or not np.isfinite(q).all():raise ValueError("Invalid IK vectors")
    tree=ET.parse(urdf).getroot()
    joints={x.attrib["name"]:x for x in tree.findall("joint")}
    limits=[]
    for name in NAMES:
        joint=joints[name]
        lim=joint.find("limit")
        if lim is None or "lower" not in lim.attrib or "upper" not in lim.attrib:
            raise ValueError(f"Missing finite limits for {name}")
        limits.append((float(lim.attrib["lower"])*180/np.pi,float(lim.attrib["upper"])*180/np.pi))
    lower=np.array([x[0] for x in limits]);upper=np.array([x[1] for x in limits])
    bad=(q<lower)|(q>upper)
    raw_bad=(goal<lower)|(goal>upper)
    delta=np.diff(q,axis=0)
    steps=np.diff(ids)
    per_frame=delta/steps[:,None]
    # A 20-fps sampled interval average, not the maximum instantaneous motion.
    lines=[]
    for j,name in enumerate(NAMES):
        lines.append({"joint":name,"urdf_lower_deg":lower[j],"urdf_upper_deg":upper[j],
            "ik_limit_violations":int(bad[:,j].sum()),
            "recorded_goal_limit_violations":int(raw_bad[:,j].sum()),
            "mean_abs_ik_minus_goal_deg":float(np.abs(q[:,j]-goal[:,j]).mean()),
            "max_abs_ik_minus_goal_deg":float(np.abs(q[:,j]-goal[:,j]).max()),
            "max_sampled_ik_change_deg":float(np.abs(delta[:,j]).max()),
            "max_interval_avg_change_deg_per_frame":float(np.abs(per_frame[:,j]).max())})
    output.mkdir(parents=True)
    with (output/"joints.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(lines[0]));w.writeheader();w.writerows(lines)
    report={"frames":len(rows),"sample_stride_range":[int(steps.min()),int(steps.max())],
        "all_ik_violating_frames":int(np.any(bad,axis=1).sum()),
        "recorded_goal_violating_frames":int(np.any(raw_bad,axis=1).sum()),
        "max_joint_step_between_sampled_solutions_deg":float(np.abs(delta).max()),
        "max_interval_average_change_deg_per_frame":float(np.abs(per_frame).max()),
        "joints":lines,
        "limitations":["URDF joint bounds may not match physical calibration (especially J2/J3)",
                       "Sampling every 20 frames hides within-interval jumps and cannot establish safe speed",
                       "IK was independently seeded from observation.state at each point; not continuous trajectory IK",
                       "No collision, torque, or hardware safety verification"]}
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="joints"},indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay",required=True,type=Path)
    p.add_argument("--urdf",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    audit(a.replay,a.urdf,a.output)
if __name__=="__main__":main()
