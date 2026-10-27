#!/usr/bin/env python3
"""Offline multi-seed bounded SO-101 IK: position, orientation and continuity.

No motor commands. Goals are FK(recorded actions), so this only validates
numerical reconstruction, not measured robot calibration or collision safety.
"""
from __future__ import annotations
import argparse,csv,json,tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pyarrow.parquet as pq
import placo
from scipy.optimize import least_squares
from export_cartesian_action_sidecar import JOINT_NAMES,rotation_log

def joint_bounds(path):
 root=ET.parse(path).getroot()
 joints={j.attrib.get("name"):j for j in root.findall("joint")}
 limits=[]
 for n in JOINT_NAMES:
  lim=joints[n].find("limit")
  limits.append((float(lim.attrib["lower"]),float(lim.attrib["upper"])))
 a=np.array(limits)
 assert np.all(a[:,0]<a[:,1])
 return a[:,0],a[:,1]

def load_frames(dataset,split_dir,ep,stride,count):
 info=json.loads((dataset/"meta/info.json").read_text())
 if info["features"]["action"]["shape"]!=[6]:raise ValueError("6D joint data required")
 split=json.loads((split_dir/"episode_split.json").read_text())
 if ep not in split["train_episode_indices"]:raise ValueError("Use train episode only")
 rows=[]
 for file in sorted((dataset/"data").rglob("*.parquet")):
  table=pq.read_table(file,columns=["episode_index","frame_index","observation.state","action"])
  rows.extend(r for r in table.to_pylist() if r["episode_index"]==ep and r["frame_index"]%stride==0)
 rows.sort(key=lambda r:r["frame_index"])
 return rows[:count]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ("dataset","split-dataset","original","experiment","output"):
  p.add_argument("--"+x,required=True,type=Path)
 p.add_argument("--episode",type=int,default=0)
 p.add_argument("--stride",type=int,default=20)
 p.add_argument("--max-frames",type=int,default=30)
 p.add_argument("--orientation-weight",type=float,default=0.2,help="Residual mm per radian after multiplying radians by 1000. 0.2 means 200mm/rad.")
 p.add_argument("--continuity-weight",type=float,default=0.2,help="Joint change penalty, mm/radian")
 p.add_argument("--max-nfev",type=int,default=150)
 a=p.parse_args()
 if a.output.exists():p.error("Output directory exists")
 if a.stride<1 or a.max_frames<1 or a.max_nfev<1:p.error("Positive stride/frame/iterations required")
 if a.orientation_weight<0 or a.continuity_weight<0:p.error("Weights must be nonnegative")
 rows=load_frames(a.dataset,a.split_dataset,a.episode,a.stride,a.max_frames)
 if not rows:raise ValueError("No frames")
 original=a.original.resolve();exp=a.experiment.resolve()
 if original==exp:raise ValueError("URDF files should differ")
 # Mesh paths in the URDF are relative: load experimental copy beside original.
 with tempfile.NamedTemporaryFile(dir=original.parent,suffix=".urdf",prefix=".ik_multiseed_",delete=False) as f:
  temp=Path(f.name)
  f.write(exp.read_bytes())
 try:
  robots={"original":placo.RobotWrapper(str(original)),"experiment":placo.RobotWrapper(str(temp))}
  lims={"original":joint_bounds(original),"experiment":joint_bounds(exp)}
  def fk(robot,q):
   for name,v in zip(JOINT_NAMES,q):robot.set_joint(name,float(v))
   robot.update_kinematics()
   return np.asarray(robot.get_T_world_frame("gripper_frame_link"),dtype=float).copy()
  out=[]
  for variant,robot in robots.items():
   low,high=lims[variant]
   prev=None
   for row in rows:
    state=np.deg2rad(np.asarray(row["observation.state"][:5],dtype=float))
    action=np.deg2rad(np.asarray(row["action"][:5],dtype=float))
    if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError("Invalid joints")
    goal=fk(robot,action)
    anchor=state if prev is None else prev
    # The recorded action is an oracle seed for diagnostics; do not use it as
    # an available seed when evaluating a real learned policy at runtime.
    seeds=[("previous",anchor),("observation",state),("recorded_oracle",action)]
    fits=[]
    for seed_name,seed in seeds:
     x0=np.clip(seed,low+1e-8,high-1e-8)
     def residual(q):
      pose=fk(robot,q)
      return np.r_[
       (pose[:3,3]-goal[:3,3])*1000,
       rotation_log(pose[:3,:3]@goal[:3,:3].T)*1000*a.orientation_weight,
       (q-anchor)*a.continuity_weight]
     fit=least_squares(residual,x0,bounds=(low,high),max_nfev=a.max_nfev)
     pose=fk(robot,fit.x)
     pos=float(np.linalg.norm(pose[:3,3]-goal[:3,3])*1000)
     ori=float(np.rad2deg(np.linalg.norm(rotation_log(pose[:3,:3]@goal[:3,:3].T))))
     move=float(np.max(np.abs(np.rad2deg(fit.x-anchor))))
     score=float(np.sum(residual(fit.x)**2))
     fits.append((score,seed_name,fit,fit.x,pos,ori,move))
    score,seed_name,fit,q,pos,ori,move=min(fits,key=lambda t:t[0])
    out.append({"variant":variant,"episode":a.episode,"frame":int(row["frame_index"]),
     "selected_seed":seed_name,"optimizer_success":bool(fit.success),
     "position_error_mm":pos,"orientation_error_deg":ori,
     "max_delta_from_previous_sample_deg":None if prev is None else float(np.max(np.abs(np.rad2deg(q-prev)))),
     "max_abs_ik_vs_recorded_deg":float(np.max(np.abs(np.rad2deg(q-action)))),
     "recorded_outside_urdf":bool(np.any(action<low)|np.any(action>high)),
     "ik_in_bounds":bool(np.all(q>=low-1e-8)&np.all(q<=high+1e-8)),
     "ik_arm_deg":np.rad2deg(q).tolist(),
     "all_seeds":[{"name":n,"score":float(sc),"optimizer_success":bool(ft.success),
                   "position_error_mm":float(pe),"orientation_error_deg":float(oe),
                   "max_delta_from_anchor_deg":float(mo)}
                  for sc,n,ft,_,pe,oe,mo in fits]})
    prev=q.copy()
  a.output.mkdir(parents=True)
  (a.output/"multiseed_ik_detailed.json").write_text(json.dumps(out,indent=2)+"\n")
  fields=[k for k in out[0] if k not in ("all_seeds","ik_arm_deg")]
  with (a.output/"multiseed_ik_comparison.csv").open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:v for k,v in r.items() if k in fields} for r in out])
  summary={}
  for v in robots:
   subset=[r for r in out if r["variant"]==v]
   summary[v]={"frames":len(subset),
    "optimizer_successes":sum(r["optimizer_success"] for r in subset),
    "recorded_outside_urdf":sum(r["recorded_outside_urdf"] for r in subset),
    "ik_bound_violations":sum(not r["ik_in_bounds"] for r in subset),
    "position_mean_mm":float(np.mean([r["position_error_mm"] for r in subset])),
    "position_max_mm":float(max(r["position_error_mm"] for r in subset)),
    "orientation_mean_deg":float(np.mean([r["orientation_error_deg"] for r in subset])),
    "orientation_max_deg":float(max(r["orientation_error_deg"] for r in subset)),
    "max_sampled_jump_deg":float(max((r["max_delta_from_previous_sample_deg"] or 0) for r in subset)),
    "seed_counts":{n:sum(r["selected_seed"]==n for r in subset) for n in ("previous","observation","recorded_oracle")}}
  summary["parameters"]={"orientation_weight":a.orientation_weight,"continuity_weight":a.continuity_weight,"stride":a.stride}
  summary["limitations"]=["Recorded action is included as an ORACLE seed; optimistic offline upper bound, not a deployable IK policy",
   "Targets use FK of recorded action; no independent measured ground truth",
   "Joint bounds are URDF-only and do not prove physical safety",
   "No collisions, velocities, acceleration or actual 20fps trajectory safety checks"]
  (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
  print(json.dumps(summary,indent=2))
 finally:
  temp.unlink(missing_ok=True)
if __name__=="__main__":main()
