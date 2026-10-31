#!/usr/bin/env python3
"""Compare Placo FK from original and limit-only experimental URDF (NO motors).

Samples recorded 6D actions from an existing full-frame audit CSV.
Same joint angles should produce identical TCP transforms when only <limit>
changes. This tests numerical equivalence, NOT physical zero, sign, TCP calibration,
collision-free IK, or safe control.
"""
from __future__ import annotations
import argparse,csv,json,tempfile
from pathlib import Path
import numpy as np
import placo
from export_cartesian_action_sidecar import JOINT_NAMES,rotation_log

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--original",type=Path,required=True)
 p.add_argument("--experiment",type=Path,required=True)
 p.add_argument("--audit-csv",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--frame",default="gripper_frame_link")
 p.add_argument("--stride",type=int,default=20)
 a=p.parse_args()
 if a.output.exists():p.error("Output already exists")
 if a.stride<1:p.error("Stride must be positive")
 # Resolve mesh paths relative to the original URDF's directory.
 # Temporary experimental copy is removed afterwards; original is untouched.
 a.original=a.original.resolve()
 a.experiment=a.experiment.resolve()
 if a.original==a.experiment:raise ValueError("URDF paths must be different")
 with tempfile.NamedTemporaryFile(mode="wb",suffix=".urdf",prefix=".fk_compare_",dir=a.original.parent,delete=False) as fp:
  temporary=Path(fp.name)
  fp.write(a.experiment.read_bytes())
 try:
  robots=[placo.RobotWrapper(str(v)) for v in (a.original,temporary)]
  compare_fk(robots,a)
 finally:
  temporary.unlink(missing_ok=True)

def compare_fk(robots,a):
 for robot in robots:
  if not set(JOINT_NAMES).issubset(robot.joint_names()):raise ValueError("URDF missing arm joint")
 # Read only fixed train episode 0 for a quick deterministic FK equivalence test.
 with a.audit_csv.open(newline="") as f:
  rows=[r for r in csv.DictReader(f) if int(r["episode"])==0 and int(r["frame"])%a.stride==0]
 if not rows:raise ValueError("No episode 0 frames found")
 results=[]
 for row in rows:
  q=np.array([float(row[f"{n}_goal"]) for n in JOINT_NAMES])
  if not np.isfinite(q).all():raise ValueError("Nonfinite joint value")
  ts=[]
  for robot in robots:
   for name,deg in zip(JOINT_NAMES,q):
    robot.set_joint(name,float(np.deg2rad(deg)))
   robot.update_kinematics()
   ts.append(np.asarray(robot.get_T_world_frame(a.frame),dtype=float).copy())
  dpos=float(np.linalg.norm(ts[0][:3,3]-ts[1][:3,3])*1000)
  drot=float(np.rad2deg(np.linalg.norm(rotation_log(ts[0][:3,:3]@ts[1][:3,:3].T))))
  results.append({"episode":0,"frame":int(row["frame"]),"delta_position_mm":dpos,"delta_orientation_deg":drot})
 a.output.mkdir(parents=True)
 with (a.output/"fk_comparison.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 out={"frames":len(results),"sample_stride":a.stride,"original_urdf":str(a.original),
      "experimental_urdf":str(a.experiment),"frame":a.frame,
      "max_fk_position_difference_mm":max(r["delta_position_mm"] for r in results),
      "max_fk_orientation_difference_deg":max(r["delta_orientation_deg"] for r in results),
      "fk_equivalent_within_tolerance":all(r["delta_position_mm"]<1e-4 and r["delta_orientation_deg"]<1e-5 for r in results),
      "warning":"Only FK equivalence of both URDF files. Does NOT verify physical calibration or safe IK."}
 (a.output/"summary.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
