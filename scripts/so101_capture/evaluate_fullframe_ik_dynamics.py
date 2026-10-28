#!/usr/bin/env python3
"""Full-frame offline IK comparison and finite-difference velocity/acceleration.

Never sends robot commands. Uses timestamps from dataset when available;
otherwise requires --fps. Rates are finite differences of recorded action
targets, not motor telemetry. No collision or physical calibration checks.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import placo
from scipy.optimize import least_squares
from generate_offset_ik_offline import joint_limits
from export_cartesian_action_sidecar import JOINT_NAMES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","split-dataset","urdf","output"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--max-frames",type=int,default=0,help="0 means all episode frames")
    p.add_argument("--fps",type=float,default=None,help="Only used if timestamps unavailable")
    p.add_argument("--offset-mm",type=float,default=3.)
    p.add_argument("--recorded-weight",type=float,default=.05)
    p.add_argument("--previous-weight",type=float,default=.01)
    p.add_argument("--max-nfev",type=int,default=100)
    p.add_argument("--max-position-error-mm",type=float,default=2.)
    p.add_argument("--max-deviation-deg",type=float,default=15.)
    p.add_argument("--max-velocity-deg-s",type=float,default=None,help="Optional diagnostic threshold; not certified safety limit")
    p.add_argument("--max-acceleration-deg-s2",type=float,default=None,help="Optional diagnostic threshold; not certified safety limit")
    a=p.parse_args()
    if a.output.exists() or a.dataset.resolve() in a.output.resolve().parents:
        p.error("Output must not exist and must be outside dataset")
    if a.max_frames<0 or a.max_nfev<1 or a.offset_mm<=0 or a.recorded_weight<0 or a.previous_weight<0 or (a.fps is not None and a.fps<=0):
        p.error("Invalid parameters")
    info=json.loads((a.dataset/"meta/info.json").read_text())
    if info["features"]["action"]["shape"]!=[6]:raise ValueError("Require 6D action dataset")
    split=json.loads((a.split_dataset/"episode_split.json").read_text())
    if a.episode not in split["train_episode_indices"] or set(split["train_episode_indices"]) & set(split["validation_episode_indices"]):
        raise ValueError("Only a train episode from disjoint split is allowed")
    frames=[]
    for file in sorted((a.dataset/"data").rglob("*.parquet")):
        pf=pq.ParquetFile(file)
        cols=["episode_index","frame_index","action"]
        has_ts="timestamp" in pf.schema_arrow.names
        if has_ts:cols.append("timestamp")
        for batch in pf.iter_batches(batch_size=8192,columns=cols):
            frames.extend(r for r in batch.to_pylist() if int(r["episode_index"])==a.episode)
    frames.sort(key=lambda r:int(r["frame_index"]))
    if a.max_frames:frames=frames[:a.max_frames]
    if len(frames)<3:raise ValueError("At least three frames required")
    indices=np.array([r["frame_index"] for r in frames],dtype=int)
    if np.any(np.diff(indices)!=1):raise ValueError("Nonconsecutive frames; cannot claim one-frame analysis")
    if all("timestamp" in r and r["timestamp"] is not None for r in frames):
        ts=np.array([r["timestamp"] for r in frames],dtype=float)
        time_source="dataset_timestamp"
        if not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):raise ValueError("Nonmonotonic timestamps")
    else:
        fps=a.fps if a.fps is not None else info.get("fps")
        if not fps or float(fps)<=0:raise ValueError("No timestamps or valid fps; specify --fps explicitly")
        ts=(indices-indices[0])/float(fps)
        time_source="fps_fallback"
    lower,upper=joint_limits(a.urdf)
    robot=placo.RobotWrapper(str(a.urdf.resolve()))
    def fk(q):
        for name,v in zip(JOINT_NAMES,q):robot.set_joint(name,float(v))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"),float)[:3,3].copy()
    original=np.array([r["action"][:5] for r in frames],float)
    if not np.isfinite(original).all():raise ValueError("Nonfinite action")
    def derivatives(q):
        v=np.full_like(q,np.nan);acc=np.full_like(q,np.nan)
        dt=np.diff(ts)
        v[1:]=np.diff(q,axis=0)/dt[:,None]
        # Compare consecutive interval velocities, using midpoint interval dt.
        acc[2:]=np.diff(v[1:],axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
        return v,acc
    orig_v,orig_a=derivatives(original)
    arm_positions=np.empty((len(frames),3))
    for i in range(len(frames)):arm_positions[i]=fk(np.deg2rad(original[i]))
    directions=[("x+",[1,0,0]),("x-",[-1,0,0]),("y+",[0,1,0]),("y-",[0,-1,0]),("z+",[0,0,1]),("z-",[0,0,-1])]
    output=[]
    variant_summary=[]
    for label,axis in directions:
        target=arm_positions+np.array(axis)*a.offset_mm/1000
        angles={}
        errors={}
        successes={}
        for method in ("baseline","regularized"):
            qout=[];errs=[];ok=[];prev=None
            for i in range(len(frames)):
                recorded=np.deg2rad(original[i])
                seed=np.clip(recorded if method=="regularized" or prev is None else prev,lower+1e-9,upper-1e-9)
                def residual(q):
                    pos=(fk(q)-target[i])*1000.
                    if method=="baseline":return pos
                    parts=[pos,(np.rad2deg(q-recorded)*a.recorded_weight)]
                    if prev is not None:parts.append(np.rad2deg(q-prev)*a.previous_weight)
                    return np.concatenate(parts)
                fit=least_squares(residual,seed,bounds=(lower,upper),max_nfev=a.max_nfev)
                qout.append(np.rad2deg(fit.x))
                errs.append(float(np.linalg.norm((fk(fit.x)-target[i])*1000)))
                ok.append(bool(fit.success))
                prev=fit.x.copy()
            angles[method]=np.asarray(qout)
            errors[method]=np.asarray(errs)
            successes[method]=np.asarray(ok)
        rows_by_method={}
        for method in ("baseline","regularized"):
            q=angles[method];v,acc=derivatives(q)
            dev=np.max(np.abs(q-original),axis=1)
            outside=np.any((q<np.rad2deg(lower))|(q>np.rad2deg(upper)),axis=1)
            orig_outside=np.any((original<np.rad2deg(lower))|(original>np.rad2deg(upper)),axis=1)
            dv=np.nanmax(np.abs(v),axis=1)
            da=np.nanmax(np.abs(acc),axis=1)
            ov=np.nanmax(np.abs(orig_v),axis=1)
            oa=np.nanmax(np.abs(orig_a),axis=1)
            good=successes[method] & (errors[method]<=a.max_position_error_mm) & (dev<=a.max_deviation_deg)
            rec=[]
            for i,frame in enumerate(indices):
                rec.append(dict(variant=label,method=method,frame=int(frame),time_s=float(ts[i]),
                    position_error_mm=float(errors[method][i]),optimizer_success=bool(successes[method][i]),
                    max_deviation_from_recorded_deg=float(dev[i]),numeric_screen_pass=bool(good[i]),
                    urdf_limit_recorded_outside=bool(orig_outside[i]),urdf_limit_candidate_outside=bool(outside[i]),
                    original_max_velocity_deg_s=None if i<1 else float(ov[i]),
                    candidate_max_velocity_deg_s=None if i<1 else float(dv[i]),
                    original_max_acceleration_deg_s2=None if i<2 else float(oa[i]),
                    candidate_max_acceleration_deg_s2=None if i<2 else float(da[i]),
                    original_arm_deg=original[i].tolist(),ik_arm_deg=q[i].tolist()))
            output.extend(rec)
            rows_by_method[method]=dict(passes=int(sum(good)),max_deviation_deg=float(dev.max()),
                max_position_error_mm=float(errors[method].max()),max_velocity_deg_s=float(np.nanmax(dv[1:])),
                max_acceleration_deg_s2=float(np.nanmax(da[2:])),
                outside_urdf_frames=int(sum(outside)),
                velocity_over_threshold=None if a.max_velocity_deg_s is None else int(sum(dv[1:]>a.max_velocity_deg_s)),
                acceleration_over_threshold=None if a.max_acceleration_deg_s2 is None else int(sum(da[2:]>a.max_acceleration_deg_s2)))
        variant_summary.append(dict(variant=label,frames=len(frames),baseline=rows_by_method["baseline"],regularized=rows_by_method["regularized"]))
    a.output.mkdir(parents=True)
    with (a.output/"per_frame.csv").open("w",newline="") as f:
        cols=[k for k in output[0] if k not in ("original_arm_deg","ik_arm_deg")]
        writer=csv.DictWriter(f,cols,extrasaction="ignore");writer.writeheader();writer.writerows(output)
    (a.output/"per_frame_detailed.json").write_text(json.dumps(output,indent=2)+"\n")
    summary=dict(episode=a.episode,frames=len(frames),frame_start=int(indices[0]),frame_end=int(indices[-1]),
        time_source=time_source,median_frame_dt_s=float(np.median(np.diff(ts))),
        original_max_velocity_deg_s=float(np.nanmax(np.abs(orig_v[1:]))),
        original_max_acceleration_deg_s2=float(np.nanmax(np.abs(orig_a[2:]))),
        original_urdf_limit_outside_frames=int(np.sum(np.any((original<np.rad2deg(lower))|(original>np.rad2deg(upper)),axis=1))),
        weights=dict(recorded=a.recorded_weight,previous=a.previous_weight),variants=variant_summary,
        warning=["OFFLINE ONLY; do not replay as motor commands","These are action-target derivatives, not measured motor joint velocity or acceleration","Original and generated actions are compared using unverified URDF frame/sign/zero calibration","No collision, contact, self-collision, tool calibration, pose orientation, torque or swept-path screening","No paired synthetic images or ACT training examples created","Do not interpret arbitrary velocity thresholds as verified SO-101 specifications"])
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
