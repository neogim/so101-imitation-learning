#!/usr/bin/env python3
"""Read-only URDF limit/offset hypothesis audit for recorded LeRobot states.
A limit-compatible mapping is NOT a physical calibration.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from generate_offset_ik_offline import joint_limits

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with a.predictions.open(newline="") as f:rows=list(csv.DictReader(f))
    q=np.array([[float(r[f"state_{i}"]) for i in range(5)] for r in rows])
    if q.shape[0]<1 or not np.isfinite(q).all():raise ValueError("No valid states")
    lo,hi=map(np.rad2deg,joint_limits(a.urdf))
    report={"frames":len(rows),"calibration":"UNVERIFIED","joint_audit":[]}
    for j in range(5):
        raw=q[:,j]
        out=int(np.sum((raw<lo[j])|(raw>hi[j])))
        variants={}
        for s in (+1,-1):
            x=s*raw
            # A constant offset o permits all recorded states within URDF iff
            # lo <= x+o <= hi -> lo-min(x) <=o<=hi-max(x).
            off_lo=float(lo[j]-np.min(x));off_hi=float(hi[j]-np.max(x))
            variants["sign+" if s==1 else "sign-"]={"feasible_constant_offset_interval_deg":[off_lo,off_hi],
                "compatible":bool(off_lo<=off_hi),"zero_offset_compatible":bool(off_lo<=0<=off_hi)}
        report["joint_audit"].append({"joint":j+1,"urdf_range_deg":[float(lo[j]),float(hi[j])],
            "recorded_range_deg":[float(raw.min()),float(raw.max())],
            "recorded_outside_count_if_signplus_offsetzero":out,"mapping_hypotheses":variants})
    report["warnings"]=["Static URDF interval compliance cannot establish actual joint sign, home zero, calibration or correct FK.","Offsets are feasibility intervals, not recommended configuration values.","Do not change motor calibration, URDF limits or command hardware based on this report."]
    a.output.mkdir(parents=True)
    (a.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
