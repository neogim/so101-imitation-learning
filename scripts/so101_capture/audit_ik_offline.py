#!/usr/bin/env python3
"""Offline URDF limits and IK trajectory audit; no motors or hardware."""
import argparse,csv,json,xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from export_cartesian_action_sidecar import JOINT_NAMES

def write_csv(path, rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("dataset","urdf","ik-results","output"):
        p.add_argument("--"+arg,required=True,type=Path)
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--stride",type=int,default=20)
    p.add_argument("--max-frames",type=int,default=30)
    a=p.parse_args()
    if a.stride<1 or a.max_frames<1 or a.output.exists() or a.dataset.resolve() in a.output.resolve().parents:
        p.error("Invalid sampling or existing/unsafe output")
    joints={j.attrib["name"]:j for j in ET.parse(a.urdf).getroot().findall("joint")}
    limits=np.array([[float(joints[n].find("limit").attrib[k]) for k in ("lower","upper")] for n in JOINT_NAMES])
    limits=np.rad2deg(limits)
    source=[]
    for f in sorted((a.dataset/"data").rglob("*.parquet")):
        table=pq.read_table(f,columns=["episode_index","frame_index","observation.state","action"])
        source.extend(r for r in table.to_pylist() if int(r["episode_index"])==a.episode and int(r["frame_index"])%a.stride==0)
    source=sorted(source,key=lambda r:int(r["frame_index"]))[:a.max_frames]
    if not source: raise ValueError("No matching episode data")
    original={int(r["frame_index"]):np.asarray(r["action"][:5],float) for r in source}
    limit_rows=[]
    for i,n in enumerate(JOINT_NAMES):
        vals=np.array([r[k][i] for r in source for k in ("observation.state","action")],float)
        limit_rows.append(dict(joint=n,urdf_min_deg=limits[i,0],urdf_max_deg=limits[i,1],recorded_min_deg=vals.min(),recorded_max_deg=vals.max(),outside_count=int(np.sum((vals<limits[i,0])|(vals>limits[i,1]))),sample_count=len(vals)))
    grouped=defaultdict(list)
    for r in json.loads((a.ik_results/"ik_waypoints.json").read_text()):
        grouped[r["variant"]].append(r)
    detail=[];summary=[]
    for variant,records in sorted(grouped.items()):
        prev_q=prev_orig=None; rows=[]
        for r in sorted(records,key=lambda r:int(r["frame"])):
            frame=int(r["frame"])
            if frame not in original: raise ValueError(f"Missing source frame: {frame}")
            q=np.asarray(r["ik_arm_deg"],float);orig=original[frame]
            step=None if prev_q is None else float(np.max(np.abs(q-prev_q)))
            origstep=None if prev_orig is None else float(np.max(np.abs(orig-prev_orig)))
            entry=dict(variant=variant,frame=frame,optimizer_success=bool(r["optimizer_success"]),position_error_mm=float(r["position_error_mm"]),passes_numerical_screen=bool(r["passes_numerical_screen"]),ik_step_deg=step,original_step_deg=origstep,ik_extra_step_deg=None if step is None else step-origstep,max_ik_vs_recorded_deg=float(np.max(np.abs(q-orig))),ik_outside_urdf_limits=bool(np.any((q<limits[:,0])|(q>limits[:,1]))))
            rows.append(entry);detail.append(entry);prev_q=q;prev_orig=orig
        summary.append(dict(variant=variant,samples=len(rows),passes=sum(r["passes_numerical_screen"] for r in rows),optimizer_failures=sum(not r["optimizer_success"] for r in rows),position_error_over_2mm=sum(r["position_error_mm"]>2 for r in rows),ik_step_over_25deg=sum(r["ik_step_deg"] is not None and r["ik_step_deg"]>25 for r in rows),original_step_over_25deg=sum(r["original_step_deg"] is not None and r["original_step_deg"]>25 for r in rows),max_ik_vs_recorded_deg=max(r["max_ik_vs_recorded_deg"] for r in rows)))
    a.output.mkdir(parents=True)
    write_csv(a.output/"joint_limit_audit.csv",limit_rows)
    write_csv(a.output/"ik_diagnostics.csv",detail)
    write_csv(a.output/"variant_summary.csv",summary)
    result=dict(source_episode=a.episode,sampled_frames=len(source),joint_limits=limit_rows,variants=summary,warnings=["Offline numerical diagnostic only; no robot control","Recorded angles assumed degrees: URDF zero/sign and hardware limits unverified","Sampled steps are not cycle-by-cycle motor velocity","No contact, collision or physical FK calibration verified"])
    (a.output/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=="__main__":main()
