#!/usr/bin/env python3
"""Read-only representative-frame FK pose comparison for SO-101 sign hypotheses.

Uses recorded observation.state, NOT action, because camera frames show observed
robot positions. Produces FK link-joint origins and multi-view plots; it does
not claim camera reprojection or determine the correct sign automatically.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import placo
from export_cartesian_action_sidecar import JOINT_NAMES
from evaluate_mapping_hypotheses_offline import HYPOTHESES

LINKS=("base_link","shoulder_link","upper_arm_link","lower_arm_link",
       "wrist_link","gripper_link","gripper_frame_link")

def read_observations(dataset, episode, indices):
    found={}
    for file in sorted((dataset/"data").rglob("*.parquet")):
        parquet=pq.ParquetFile(file)
        for batch in parquet.iter_batches(columns=["episode_index","frame_index",
                                                    "timestamp","observation.state","action"]):
            for r in batch.to_pylist():
                ix=int(r["frame_index"])
                if int(r["episode_index"])==episode and ix in indices:
                    if ix in found:raise ValueError(f"Duplicate frame {ix}")
                    found[ix]=r
    missing=sorted(indices-found.keys())
    if missing:raise ValueError(f"Missing frames: {missing}")
    return found

def fk_points(robot,motor,signs):
    q=np.asarray(motor,dtype=float)
    if q.shape[0]<5 or not np.isfinite(q).all():raise ValueError("Bad state")
    for name,deg in zip(JOINT_NAMES,q[:5]*signs,strict=True):
        robot.set_joint(name,float(np.deg2rad(deg)))
    robot.update_kinematics()
    return {name:np.asarray(robot.get_T_world_frame(name),dtype=float)[:3,3].tolist()
            for name in LINKS}

def run(args):
    if args.output.exists() and not args.output.is_dir():raise FileExistsError(f"Output is a file: {args.output}")
    if args.output.resolve()==args.dataset.resolve() or args.dataset.resolve() in args.output.resolve().parents:
        raise ValueError("Output must be outside source dataset")
    frames=sorted(set(args.frames))
    if not frames or min(frames)<0:raise ValueError("Invalid frame choices")
    obs=read_observations(args.dataset,args.episode,set(frames))
    robot=placo.RobotWrapper(str(args.urdf.resolve()))
    results=[]
    for ix in frames:
        r=obs[ix]
        state=np.asarray(r["observation.state"],dtype=float)
        action=np.asarray(r["action"],dtype=float)
        entry={"episode":args.episode,"frame":ix,"timestamp":float(r["timestamp"]),
               "observation_state_motor_deg":state[:5].tolist(),
               "recorded_action_motor_deg":action[:5].tolist(),
               "fk_hypotheses":{}}
        for label,signs in HYPOTHESES.items():
            entry["fk_hypotheses"][label]=fk_points(robot,state,signs)
        results.append(entry)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("Plotting requires matplotlib in the active so101-ik environment: python -m pip install matplotlib") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and any((args.output / f"frame_{ix:04d}_fk.png").exists() for ix in frames):
        raise FileExistsError("Existing plot files found; use a new --output directory")
    (args.output/"fk_comparison.json").write_text(json.dumps({
        "classification":"HYPOTHESIS_COMPARISON_NOT_PHYSICAL_VALIDATION",
        "episode":args.episode,"source":"observation.state (not action)",
        "coordinate_frame":"URDF world, not image pixel coordinates",
        "limitations":["No camera intrinsics/extrinsics or image overlay",
                       "Joint signs assumed; offsets all zero",
                       "URDF link frame origins are not the exact motor pivot centers",
                       "Cannot infer physical mapping from these model plots alone"],
        "rows":results},indent=2,ensure_ascii=False)+"\n")
    for entry in results:
        fig,axes=plt.subplots(4,2,figsize=(12,17))
        for row,(label,points) in enumerate(entry["fk_hypotheses"].items()):
            xyz=np.array([points[link] for link in LINKS],dtype=float)
            for col,axis_idx in enumerate((0,1)):
                a=axes[row,col]
                a.plot(xyz[:,axis_idx],xyz[:,2],"-o")
                for joint,coord in zip(("base","J1","J2","J3","J4","J5","TCP"),xyz,strict=True):
                    a.annotate(joint,(coord[axis_idx],coord[2]),fontsize=8)
                a.set_title(f"{label} | {'XZ' if col==0 else 'YZ'}")
                a.set_xlabel(("x" if col==0 else "y")+" URDF world [m]")
                a.set_ylabel("z URDF world [m]")
                a.axis("equal");a.grid(True)
        fig.suptitle(f"Episode {args.episode} frame {entry['frame']} | observation.state | signs only, offset=0")
        fig.tight_layout()
        fig.savefig(args.output/f"frame_{entry['frame']:04d}_fk.png",dpi=140)
        plt.close(fig)
    print("Episode",args.episode,"frames",frames,"hypotheses",list(HYPOTHESES))
    print("Saved:",args.output)
    print("NOT camera-aligned; physical signs and zero offsets remain unverified.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--urdf",type=Path,required=True)
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--frames",type=int,nargs="+",default=[0,60,120,240,360,480,647])
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
if __name__=="__main__":main()
