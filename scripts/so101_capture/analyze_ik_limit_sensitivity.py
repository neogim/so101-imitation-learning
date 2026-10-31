#!/usr/bin/env python3
"""Offline sensitivity study near an IK velocity spike; NEVER command the robot.

Compares URDF solver bounds with HYPOTHETICAL bounds expanded by 3/5 degrees.
Expanded limits are diagnostics only, not permission to alter hardware limits.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import placo
from scipy.optimize import least_squares
from generate_offset_ik_offline import joint_limits
from export_cartesian_action_sidecar import JOINT_NAMES

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ("input","urdf","output"):p.add_argument("--"+k,required=True,type=Path)
 p.add_argument("--start-frame",type=int,default=565)
 p.add_argument("--end-frame",type=int,default=580)
 p.add_argument("--offset-mm",type=float,default=3.)
 a=p.parse_args()
 if a.output.exists():p.error("Output already exists")
 items=json.loads(a.input.read_text())
 rows={(r["variant"],r["method"],r["frame"]):r for r in items}
 variants=["x+","y+","z-"]
 axes={"x+":np.array([1,0,0]),"y+":np.array([0,1,0]),"z-":np.array([0,0,-1])}
 low,high=joint_limits(a.urdf)
 robot=placo.RobotWrapper(str(a.urdf.resolve()))
 def fk(q):
  for n,v in zip(JOINT_NAMES,q):robot.set_joint(n,float(v))
  robot.update_kinematics()
  return np.asarray(robot.get_T_world_frame("gripper_frame_link"),float)[:3,3].copy()
 out=[];summary=[]
 for variant in variants:
  selected=[rows[(variant,"regularized",i)] for i in range(a.start_frame,a.end_frame+1)]
  ts=np.array([r["time_s"] for r in selected])
  if np.any(np.diff(ts)<=0):raise ValueError("Invalid timestamps")
  rec=np.deg2rad(np.array([r["original_arm_deg"] for r in selected]))
  target=np.array([fk(q) for q in rec])+axes[variant]*a.offset_mm/1000.
  for expand_deg in [0.,3.,5.]:
   l=low-np.deg2rad(expand_deg);h=high+np.deg2rad(expand_deg)
   for rw,pw in [(0.05,0.01),(0.15,0.03),(0.5,0.1),(0.,0.)]:
    prev=None;qs=[];errs=[];status=[]
    for i in range(len(rec)):
     seed=np.clip(rec[i] if rw else (prev if prev is not None else rec[i]),l+1e-9,h-1e-9)
     def residual(q):
      position=(fk(q)-target[i])*1000.
      if rw==0:return position
      terms=[position,(np.rad2deg(q-rec[i]))*rw]
      if prev is not None:terms.append(np.rad2deg(q-prev)*pw)
      return np.concatenate(terms)
     opt=least_squares(residual,seed,bounds=(l,h),max_nfev=150)
     prev=opt.x.copy();qs.append(np.rad2deg(opt.x));errs.append(np.linalg.norm((fk(opt.x)-target[i])*1000));status.append(opt.success)
    qs=np.array(qs);speeds=np.abs(np.diff(qs,axis=0)/np.diff(ts)[:,None])
    accelerations=np.abs(np.diff(np.diff(qs,axis=0)/np.diff(ts)[:,None],axis=0)/((np.diff(ts)[:-1]+np.diff(ts)[1:])/2)[:,None])
    for k,frame in enumerate(range(a.start_frame,a.end_frame+1)):
     out.append(dict(variant=variant,expanded_bounds_deg=expand_deg,recorded_weight=rw,previous_weight=pw,frame=frame,time_s=ts[k],recorded_shoulder_lift_deg=np.rad2deg(rec[k,1]),candidate_shoulder_lift_deg=qs[k,1],position_error_mm=errs[k],optimizer_success=status[k],max_velocity_deg_s="" if k==0 else float(speeds[k-1].max()),max_acceleration_deg_s2="" if k<2 else float(accelerations[k-2].max())))
    summary.append(dict(variant=variant,expanded_bounds_deg=expand_deg,recorded_weight=rw,previous_weight=pw,max_velocity_deg_s=float(speeds.max()),velocity_peak_frame=int(a.start_frame+np.unravel_index(np.argmax(speeds),speeds.shape)[0]+1),velocity_peak_joint=JOINT_NAMES[np.unravel_index(np.argmax(speeds),speeds.shape)[1]],max_acceleration_deg_s2=float(accelerations.max()),max_error_mm=float(max(errs)),all_optimizer_success=all(status)))
 a.output.mkdir(parents=True)
 for name,data in [("per_frame_sensitivity.csv",out),("sensitivity_summary.csv",summary)]:
  with (a.output/name).open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (a.output/"sensitivity_summary.json").write_text(json.dumps(dict(analysis_frames=[a.start_frame,a.end_frame],variants=variants,results=summary,caveats=["Expanded URDF limits are hypothetical numerical experiments only","No physical joint limit, motor safety, collision, orientation, or TCP calibration validated","Short-window warm start differs from whole-episode run; reference setting may not reproduce identical peak","No motor commands were sent"]),indent=2)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
