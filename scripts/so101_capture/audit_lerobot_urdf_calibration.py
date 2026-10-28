#!/usr/bin/env python3
"""Read-only SO-101 LeRobot/URDF calibration audit.

This audit compares recorded state/action against URDF limits and, if given,
lists calibration JSON entries. It CANNOT establish physical angle equivalence
without a measured mapping. Never sends motor commands.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from generate_offset_ik_offline import joint_limits
from export_cartesian_action_sidecar import JOINT_NAMES

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("dataset","urdf","output"):p.add_argument("--"+name,type=Path,required=True)
 p.add_argument("--episode",type=int,default=0)
 p.add_argument("--calibration",type=Path,default=None,help="Optional LeRobot calibration JSON; read-only")
 a=p.parse_args()
 if a.output.exists() or a.dataset.resolve() in a.output.resolve().parents:p.error("Output must be new and outside dataset")
 low,high=joint_limits(a.urdf);lo=np.rad2deg(low);hi=np.rad2deg(high)
 state=[];action=[];frames=[]
 for file in sorted((a.dataset/"data").rglob("*.parquet")):
  schema=pq.ParquetFile(file).schema_arrow.names
  if not {"episode_index","frame_index","observation.state","action"}.issubset(schema):
   raise ValueError(f"Required dataset columns missing: {file}")
  for batch in pq.ParquetFile(file).iter_batches(columns=["episode_index","frame_index","observation.state","action"],batch_size=8192):
   for r in batch.to_pylist():
    if int(r["episode_index"])==a.episode:
     frames.append(int(r["frame_index"]));state.append(r["observation.state"][:5]);action.append(r["action"][:5])
 if not frames:raise ValueError("Episode not found")
 ix=np.argsort(frames);frames=np.asarray(frames)[ix];state=np.asarray(state,float)[ix];action=np.asarray(action,float)[ix]
 if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError("Nonfinite joint data")
 cal=None
 if a.calibration is not None:
  cal=json.loads(a.calibration.read_text())
 summaries=[];events=[]
 for j,name in enumerate(JOINT_NAMES):
  for channel,data in (("observation.state",state),("action",action)):
   x=data[:,j];lower=x<lo[j];upper=x>hi[j]
   summaries.append(dict(joint=name,channel=channel,urdf_lower_deg=float(lo[j]),urdf_upper_deg=float(hi[j]),
    min_recorded_deg=float(x.min()),max_recorded_deg=float(x.max()),below_limit_frames=int(lower.sum()),
    above_limit_frames=int(upper.sum()),first_below_frame=int(frames[np.flatnonzero(lower)[0]]) if lower.any() else "",
    first_above_frame=int(frames[np.flatnonzero(upper)[0]]) if upper.any() else "",
    max_below_excess_deg=float(max(0,np.max(lo[j]-x))),max_above_excess_deg=float(max(0,np.max(x-hi[j])))))
   for k in np.flatnonzero(lower|upper):
    events.append(dict(frame=int(frames[k]),joint=name,channel=channel,value_deg=float(x[k]),urdf_lower_deg=float(lo[j]),urdf_upper_deg=float(hi[j]),excess_deg=float(max(lo[j]-x[k],x[k]-hi[j]))))
 a.output.mkdir(parents=True)
 for name,rows in (("joint_limit_summary.csv",summaries),("out_of_limit_frames.csv",events)):
  with (a.output/name).open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["frame","joint","channel"]);w.writeheader();w.writerows(rows)
 if cal is not None:(a.output/"calibration_inspected.json").write_text(json.dumps(cal,indent=2,ensure_ascii=False)+"\n")
 result=dict(episode=a.episode,frames=len(frames),joint_summary=summaries,source_calibration=str(a.calibration) if a.calibration else None,
  calibration_structure=(list(cal.keys()) if isinstance(cal,dict) else type(cal).__name__) if cal is not None else None,
  conclusions=["Angle overlap/limits alone do NOT prove that the LeRobot calibrated degrees equal URDF joint-space degrees.",
  "Physical sign/zero/link geometry must be validated independently. Servo calibration offsets and normalized positions are distinct from URDF model joints.",
  "Do NOT widen physical or URDF limits just to make this numerical test pass.","No robot calls, movement, or writes to original inputs."])
 (a.output/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
