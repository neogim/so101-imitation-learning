#!/usr/bin/env python3
"""Offline comparison of baseline position-only IK with joint-regularized IK.

No robot calls. Results are counterfactual waypoint candidates, NOT synchronized
ACT training data or cleared trajectories for hardware execution.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import placo
from scipy.optimize import least_squares

from generate_offset_ik_offline import joint_limits
from export_cartesian_action_sidecar import JOINT_NAMES


def run(a):
    if a.output.exists() or a.output.resolve() == a.baseline.resolve() or a.baseline.resolve() in a.output.resolve().parents:
        raise ValueError("Output must be a new folder outside baseline")
    if a.position_weight <= 0 or a.recorded_weight < 0 or a.previous_weight < 0 or a.max_nfev <= 0:
        raise ValueError("Invalid weights or maximum evaluations")
    if a.max_position_error_mm <= 0 or a.max_joint_step_deg <= 0 or a.max_deviation_deg <= 0:
        raise ValueError("Screening thresholds must be positive")
    info = json.loads((a.dataset/"meta/info.json").read_text())
    if info["features"]["action"]["shape"] != [6]:
        raise ValueError("Expected 6D source actions")
    split = json.loads((a.split_dataset/"episode_split.json").read_text())
    train = set(map(int,split["train_episode_indices"]))
    val = set(map(int,split["validation_episode_indices"]))
    baseline = json.loads((a.baseline/"ik_waypoints.json").read_text())
    if train & val or not baseline or any(r["episode"] not in train for r in baseline):
        raise ValueError("Baseline includes held-out or invalid episodes")
    lower,upper = joint_limits(a.urdf)
    robot = placo.RobotWrapper(str(a.urdf.resolve()))
    if not set(JOINT_NAMES).issubset(set(robot.joint_names())):
        raise ValueError("URDF joint mismatch")

    def fk(q):
        for name,x in zip(JOINT_NAMES,q):
            robot.set_joint(name,float(x))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"),float)[:3,3].copy()

    groups=defaultdict(list)
    for r in baseline:groups[(r["episode"],r["variant"])].append(r)
    details=[]
    summary=[]
    for (episode,variant),records in sorted(groups.items()):
        oldprev=newprev=None
        for r in sorted(records,key=lambda x:x["frame"]):
            recorded=np.deg2rad(np.asarray(r["original_arm_deg"],float))
            target=np.asarray(r["target_xyz_m"],float)
            old=np.deg2rad(np.asarray(r["ik_arm_deg"],float))
            if not all(np.isfinite(x).all() for x in (recorded,target,old)):
                raise ValueError("Nonfinite baseline values")
            # Seed from each frame's recorded action to discourage branch jumping.
            seed=np.clip(recorded,lower+1e-9,upper-1e-9)
            def residual(q):
                parts=[(fk(q)-target)*1000*a.position_weight,
                       np.rad2deg(q-recorded)*a.recorded_weight]
                if newprev is not None:
                    parts.append(np.rad2deg(q-newprev)*a.previous_weight)
                return np.concatenate(parts)
            fit=least_squares(residual,seed,bounds=(lower,upper),max_nfev=a.max_nfev)
            q=fit.x
            newerr=float(np.linalg.norm((fk(q)-target)*1000))
            older=float(r["position_error_mm"])
            oldstep=None if oldprev is None else float(np.max(np.abs(np.rad2deg(old-oldprev))))
            newstep=None if newprev is None else float(np.max(np.abs(np.rad2deg(q-newprev))))
            olddev=float(np.max(np.abs(np.rad2deg(old-recorded))))
            newdev=float(np.max(np.abs(np.rad2deg(q-recorded))))
            oldpass=bool(r["passes_numerical_screen"])
            newpass=bool(fit.success and newerr<=a.max_position_error_mm
                         and (newstep is None or newstep<=a.max_joint_step_deg)
                         and newdev<=a.max_deviation_deg)
            details.append(dict(episode=episode,variant=variant,frame=int(r["frame"]),
                original_optimizer_success=bool(r["optimizer_success"]),regularized_optimizer_success=bool(fit.success),
                baseline_position_error_mm=older,regularized_position_error_mm=newerr,
                baseline_joint_step_deg=oldstep,regularized_joint_step_deg=newstep,
                baseline_max_deviation_deg=olddev,regularized_max_deviation_deg=newdev,
                baseline_original_screen=oldpass,regularized_strict_screen=newpass,
                baseline_strict_screen=bool(oldpass and olddev<=a.max_deviation_deg),
                regularized_arm_deg=np.rad2deg(q).tolist(),original_arm_deg=np.rad2deg(recorded).tolist(),
                target_xyz_m=target.tolist(),original_gripper=r["original_gripper"]))
            oldprev=old.copy();newprev=q.copy()
        rr=[x for x in details if x["episode"]==episode and x["variant"]==variant]
        summary.append(dict(episode=episode,variant=variant,samples=len(rr),
            baseline_original_passes=sum(x["baseline_original_screen"] for x in rr),
            baseline_strict_passes=sum(x["baseline_strict_screen"] for x in rr),
            regularized_strict_passes=sum(x["regularized_strict_screen"] for x in rr),
            baseline_over_25deg=sum(x["baseline_joint_step_deg"] is not None and x["baseline_joint_step_deg"]>a.max_joint_step_deg for x in rr),
            regularized_over_25deg=sum(x["regularized_joint_step_deg"] is not None and x["regularized_joint_step_deg"]>a.max_joint_step_deg for x in rr),
            baseline_max_deviation_deg=max(x["baseline_max_deviation_deg"] for x in rr),
            regularized_max_deviation_deg=max(x["regularized_max_deviation_deg"] for x in rr),
            baseline_mean_position_error_mm=float(np.mean([x["baseline_position_error_mm"] for x in rr])),
            regularized_mean_position_error_mm=float(np.mean([x["regularized_position_error_mm"] for x in rr]))))
    a.output.mkdir(parents=True)
    with (a.output/"comparison.csv").open("w",newline="") as f:
        cols=[k for k in details[0] if k not in ("regularized_arm_deg","original_arm_deg","target_xyz_m")]
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader();w.writerows(details)
    with (a.output/"variant_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (a.output/"comparison_detailed.json").write_text(json.dumps(details,indent=2)+"\n")
    report=dict(weights=dict(position_weight=a.position_weight,recorded_weight=a.recorded_weight,previous_weight=a.previous_weight),
                thresholds=dict(max_position_error_mm=a.max_position_error_mm,max_joint_step_deg=a.max_joint_step_deg,max_deviation_deg=a.max_deviation_deg),
                variants=summary,totals=dict(samples=len(details),
                baseline_original_passes=sum(x["baseline_original_screen"] for x in details),
                baseline_strict_passes=sum(x["baseline_strict_screen"] for x in details),
                regularized_strict_passes=sum(x["regularized_strict_screen"] for x in details)),
                caveats=["Offline only; not hardware-safe or synchronized ACT data",
                         "The strict screen adds an absolute deviation criterion to both methods; original screen used no such criterion",
                         "URDF limits may conflict with calibrated recorded positions",
                         "Position-only IK, no collision checking or verified tool calibration",
                         "Sparse sampled steps do not establish motion velocity or full path continuity",
                         "Regularization trades positional accuracy against closeness to recorded joints"])
    (a.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("dataset","split-dataset","urdf","baseline","output"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--position-weight",type=float,default=1.0,help="Residual multiplier for mm TCP error")
    p.add_argument("--recorded-weight",type=float,default=0.05,help="Residual multiplier for degree difference to original joint action")
    p.add_argument("--previous-weight",type=float,default=0.01,help="Residual multiplier for degree difference to preceding IK solution")
    p.add_argument("--max-position-error-mm",type=float,default=2.0)
    p.add_argument("--max-joint-step-deg",type=float,default=25.0)
    p.add_argument("--max-deviation-deg",type=float,default=15.0)
    p.add_argument("--max-nfev",type=int,default=150)
    run(p.parse_args())
