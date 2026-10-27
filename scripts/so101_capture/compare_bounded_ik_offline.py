#!/usr/bin/env python3
"""Read-only bounded IK comparison: original vs experiment limits, same FK targets.

Episode 0 TRAIN only; no motor commands. Targets are FK(recorded action),
so this checks numerical self-consistency, not true physical accuracy.
"""
from __future__ import annotations
import argparse,csv,json,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import placo
from scipy.optimize import least_squares
from export_cartesian_action_sidecar import JOINT_NAMES,rotation_log

def bounds(path):
 root=ET.parse(path).getroot();joints={j.get("name"):j for j in root.findall("joint")}
 a=[]
 for n in JOINT_NAMES:
  lim=joints[n].find("limit")
  if lim is None:raise ValueError(f"No limit for {n}")
  a.append([float(lim.get("lower")),float(lim.get("upper"))])
 out=np.asarray(a,dtype=float)
 if not np.all(np.isfinite(out)) or not np.all(out[:,0]<out[:,1]):raise ValueError("Invalid bounds")
 return out[:,0],out[:,1]
def frames(root,episode,stride,max_frames):
 matched=[]
 for f in sorted((root/"data").rglob("*.parquet")):
  t=pq.read_table(f,columns=["episode_index","frame_index","observation.state","action"])
  matched.extend(r for r in t.to_pylist() if int(r["episode_index"])==episode and int(r["frame_index"])%stride==0)
 matched.sort(key=lambda r:int(r["frame_index"]))
 return matched[:max_frames] if max_frames else matched
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("dataset","split-dataset","original","experiment","output"):
  p.add_argument("--"+name,required=True,type=Path)
 p.add_argument("--episode",type=int,default=0)
 p.add_argument("--stride",type=int,default=20)
 p.add_argument("--max-frames",type=int,default=30)
 p.add_argument("--max-nfev",type=int,default=120)
 args=p.parse_args()
 if args.output.exists():p.error("Output exists")
 if args.stride<1 or args.max_frames<1 or args.max_nfev<1:p.error("Invalid numerical args")
 split=json.loads((args.split_dataset/"episode_split.json").read_text())
 train=set(map(int,split["train_episode_indices"]))
 val=set(map(int,split["validation_episode_indices"]))
 if len(train)!=45 or len(val)!=8 or train&val or train|val!=set(range(53)) or args.episode not in train:
  raise ValueError("Invalid split or heldout episode")
 info=json.loads((args.dataset/"meta/info.json").read_text())
 if info["features"]["action"]["shape"]!=[6]:raise ValueError("Expected original 6D data")
 rows=frames(args.dataset,args.episode,args.stride,args.max_frames)
 if not rows:raise ValueError("No frames")
 # Source and experimental URDF both load from source URDF's mesh directory.
 original=args.original.resolve();experimental=args.experiment.resolve()
 if original==experimental:raise ValueError("Need distinct URDFs")
 if not all(q.exists() for q in (original,experimental)):raise FileNotFoundError("Missing URDF")
 with tempfile.NamedTemporaryFile(mode="wb",suffix=".urdf",prefix=".ik_shadow_",dir=original.parent,delete=False) as fh:
  temp=Path(fh.name);fh.write(experimental.read_bytes())
 try:
  robots={"original":placo.RobotWrapper(str(original)),"experiment":placo.RobotWrapper(str(temp))}
  limits={"original":bounds(original),"experiment":bounds(experimental)}
  def fk(robot,q):
   for n,x in zip(JOINT_NAMES,q):robot.set_joint(n,float(x))
   robot.update_kinematics()
   return np.asarray(robot.get_T_world_frame("gripper_frame_link"),dtype=float).copy()
  outputs=[]
  for variant,robot in robots.items():
   lower,upper=limits[variant]
   previous=None
   for row in rows:
    frame=int(row["frame_index"])
    observed=np.deg2rad(np.asarray(row["observation.state"][:5],dtype=float))
    gt=np.deg2rad(np.asarray(row["action"][:5],dtype=float))
    if not np.isfinite(observed).all() or not np.isfinite(gt).all():raise ValueError("Bad state/action")
    target=fk(robot,gt)
    # Clip seed strictly to feasible interval; scipy rejects out-of-bounds x0.
    seed=np.clip(observed if previous is None else previous,lower+1e-9,upper-1e-9)
    def residual(q):
     pose=fk(robot,q)
     return (pose[:3,3]-target[:3,3])*1000.0
    fit=least_squares(residual,seed,bounds=(lower,upper),max_nfev=args.max_nfev)
    q=fit.x
    pose=fk(robot,q)
    pmm=float(np.linalg.norm((pose[:3,3]-target[:3,3])*1000))
    om=float(np.rad2deg(np.linalg.norm(rotation_log(pose[:3,:3]@target[:3,:3].T))))
    delta=(np.rad2deg(q-previous) if previous is not None else np.full(5,np.nan))
    outputs.append({"variant":variant,"episode":args.episode,"frame":frame,
      "optimizer_success":bool(fit.success),"position_error_mm":pmm,"orientation_error_deg":om,
      "max_consecutive_sampled_delta_deg":float(np.max(np.abs(delta))) if previous is not None else None,
      "max_abs_ik_vs_recorded_deg":float(np.max(np.abs(np.rad2deg(q-gt)))),
      "recorded_outside_model_limits":bool(np.any(gt<lower)|np.any(gt>upper)),
      "ik_inside_model_limits":bool(np.all(q>=lower-1e-9)&np.all(q<=upper+1e-9)),
      "ik_arm_deg":np.rad2deg(q).tolist()})
    previous=q.copy()
  args.output.mkdir(parents=True)
  with (args.output/"bounded_ik_comparison.csv").open("w",newline="") as f:
   fieldnames=[k for k in outputs[0] if k!="ik_arm_deg"]
   w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows([{k:r[k] for k in fieldnames} for r in outputs])
  (args.output/"bounded_ik_detailed.json").write_text(json.dumps(outputs,indent=2)+"\n")
  summary={}
  for v in robots:
   a=[r for r in outputs if r["variant"]==v]
   summary[v]={"frames":len(a),"optimizer_successes":sum(x["optimizer_success"] for x in a),
    "recorded_outside_model_limits":sum(x["recorded_outside_model_limits"] for x in a),
    "ik_limit_violations":sum(not x["ik_inside_model_limits"] for x in a),
    "position_error_mean_mm":float(np.mean([x["position_error_mm"] for x in a])),
    "position_error_max_mm":float(max(x["position_error_mm"] for x in a)),
    "orientation_error_mean_deg":float(np.mean([x["orientation_error_deg"] for x in a])),
    "max_delta_between_sampled_ik_deg":float(max((x["max_consecutive_sampled_delta_deg"] or 0) for x in a))}
  summary["limitations"]=[
   "Each target is FK(recorded action) in same URDF; this is self-consistency only",
   "Position-only IK: orientation is a diagnostic, not an optimization constraint",
   "Warm-start from previous sampled IK; sampled stride is not an actual 20-fps trajectory",
   "URDF limits do not establish real servo limits; joint zero and sign unverified",
   "No collision checking, dynamic constraints, contact, or robot commands"]
  (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
  print(json.dumps(summary,ensure_ascii=False,indent=2))
 finally:
  temp.unlink(missing_ok=True)
if __name__=="__main__":main()
