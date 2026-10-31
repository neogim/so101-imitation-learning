#!/usr/bin/env python3
"""Offline inverse-kinematics replay of recorded SO-101 joint goals.

Reconstruct FK(action) target poses and solve five arm joints from observed
state, WITHOUT motor commands, simulation images, or synthetic ACT data.
Position priority + optionally orientation. Diagnostic only: no collisions
or calibrated robot limits are enforced.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import placo
from scipy.optimize import least_squares
from export_cartesian_action_sidecar import JOINT_NAMES,rotation_log

def rows_for(root,episode,stride,max_frames):
    matches=[]
    for file in sorted((root/"data").rglob("*.parquet")):
        tab=pq.read_table(file,columns=["episode_index","frame_index","observation.state","action"])
        for r in tab.to_pylist():
            if int(r["episode_index"])==episode and int(r["frame_index"])%stride==0:
                matches.append(r)
    matches.sort(key=lambda r:int(r["frame_index"]))
    if max_frames:matches=matches[:max_frames]
    if not matches:raise ValueError("No matching frames")
    return matches

def compute(dataset,split_dataset,urdf,output,episode,stride,max_frames,orientation_weight,regularization):
    for p in (dataset,split_dataset,urdf): 
        if not p.exists():raise FileNotFoundError(p)
    dataset,output=dataset.resolve(),output.resolve()
    if output.exists():raise FileExistsError(output)
    if dataset==output or dataset in output.parents:raise ValueError("Output inside Dataset")
    if orientation_weight<0 or regularization<0:raise ValueError("weights must be nonnegative")
    info=json.loads((dataset/"meta/info.json").read_text())
    if info["features"]["action"]["shape"]!=[6]:
        raise ValueError("MUST use original 6D joint-action Dataset, not Cartesian 7D candidate")
    split=json.loads((split_dataset/"episode_split.json").read_text())
    if episode not in split["train_episode_indices"]:
        raise ValueError("Select TRAIN episode; keep validation data untouched")
    robot=placo.RobotWrapper(str(urdf))
    if not set(JOINT_NAMES).issubset(set(robot.joint_names())):
        raise ValueError("Missing joint name in URDF")
    def fk(q):
        for n,val in zip(JOINT_NAMES,q):
            robot.set_joint(n,float(val))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"),dtype=float).copy()
    frames=rows_for(dataset,episode,stride,max_frames)
    result=[]
    for row in frames:
        state=np.deg2rad(np.asarray(row["observation.state"],dtype=float)[:5])
        target_joint=np.deg2rad(np.asarray(row["action"],dtype=float)[:5])
        goal=fk(target_joint)
        if not np.isfinite(state).all() or not np.isfinite(target_joint).all():
            raise ValueError("Nonfinite input")
        def residual(q):
            pose=fk(q)
            pos=(pose[:3,3]-goal[:3,3])*1000.0 # mm
            orientation=rotation_log(pose[:3,:3] @ goal[:3,:3].T)*1000*orientation_weight
            prior=(q-state)*regularization
            return np.r_[pos,orientation,prior]
        # State-seeded IK, not target-action-seeded. This is NOT guaranteed to
        # respect mechanical joint limits or prevent collision.
        fit=least_squares(residual,state,max_nfev=100,xtol=1e-9,gtol=1e-9,ftol=1e-9)
        pose=fk(fit.x)
        pmm=float(np.linalg.norm(pose[:3,3]-goal[:3,3])*1000)
        odeg=float(np.rad2deg(np.linalg.norm(rotation_log(pose[:3,:3] @ goal[:3,:3].T))))
        errq=float(np.linalg.norm(np.rad2deg(fit.x-target_joint)))
        jump=float(np.max(np.abs(np.rad2deg(fit.x-state))))
        result.append({"episode":episode,"frame":int(row["frame_index"]),
           "success_optimizer":bool(fit.success),"position_error_mm":pmm,
           "orientation_error_deg":odeg,"joint_distance_to_recorded_deg":errq,
           "max_delta_from_observed_state_deg":jump,"nfev":int(fit.nfev),
           "observed_arm_deg":np.rad2deg(state).round(4).tolist(),
           "recorded_goal_arm_deg":np.rad2deg(target_joint).round(4).tolist(),
           "ik_arm_deg":np.rad2deg(fit.x).round(4).tolist()})
    output.mkdir(parents=True)
    with (output/"ik_replay.csv").open("w",newline="") as f:
        cols=[k for k in result[0] if not k.endswith("_deg") or k in ("position_error_mm","orientation_error_deg","joint_distance_to_recorded_deg","max_delta_from_observed_state_deg")]
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader();w.writerows(result)
    (output/"ik_replay_detailed.json").write_text(json.dumps(result,indent=2)+"\n")
    errs=np.array([x["position_error_mm"] for x in result])
    summary={"episode":episode,"frames_evaluated":len(result),"max_nfev":100,
        "position_mae_mm":float(errs.mean()),"position_p95_mm":float(np.percentile(errs,95)),
        "position_max_mm":float(errs.max()),
        "orientation_mean_deg":float(np.mean([r["orientation_error_deg"] for r in result])),
        "optimizer_successes":sum(r["success_optimizer"] for r in result),
        "within_5mm":sum(r["position_error_mm"]<=5 for r in result),
        "orientation_weight":orientation_weight,"regularization":regularization,
        "warnings":["Goal created from FK(recorded action); not synthetic data augmentation or proof of general IK accuracy",
                    "No collision/physical limits enforced; never drive motors from these results",
                    "URDF calibration/FK absolute offset (~10mm) unresolved",
                    "SO-101 5 DOF cannot generally satisfy arbitrary 6D pose"]}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--split-dataset",required=True,type=Path,help="Cartesian candidate with episode_split.json")
    p.add_argument("--urdf",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--stride",type=int,default=20)
    p.add_argument("--max-frames",type=int,default=30)
    p.add_argument("--orientation-weight",type=float,default=0.0,help="mm per milliradian, default position only")
    p.add_argument("--regularization",type=float,default=0.0,help="Penalty for deviation from observed pose")
    a=p.parse_args()
    if a.stride<1 or a.max_frames<1:raise ValueError("Positive stride/max-frames required")
    compute(a.dataset,a.split_dataset,a.urdf,a.output,a.episode,a.stride,a.max_frames,a.orientation_weight,a.regularization)
if __name__=="__main__":main()
