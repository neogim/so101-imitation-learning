#!/usr/bin/env python3
"""Evaluate J2/J3 coupling, gripper-table clearance, and saved scene geometry.

This diagnostic does NOT run a policy, simulate dynamics, change URDF/table
poses, or claim obstacle-free paths. It samples static SO-101 arm configurations
while holding J1,J4,J5 at the saved Episode-4 position, and reports both fingers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import demo_real2sim_muhi_cup_reach as scene
from table_collision_guard import TableCollisionGuard
import wrist_camera_sim as wrist


def sampled_angles(center, low, high, span_deg, count):
    if count < 3 or span_deg <= 0 or not math.isfinite(span_deg):
        raise ValueError("count >=3 and positive span_deg required")
    x=np.linspace(center-np.deg2rad(span_deg),center+np.deg2rad(span_deg),count)
    return np.unique(np.clip(x,low,high))


def scan_coupled_elbow(guard, start, low, high, tcp_id, *,
                       span_deg=30., count=25, clearance_mm=3.):
    """Grid of (J2,J3) pair candidates. Does NOT assert path feasibility."""
    q=np.asarray(start,float)
    j2=sampled_angles(q[1],low[1],high[1],span_deg,count)
    j3=sampled_angles(q[2],low[2],high[2],span_deg,count)
    rows=[]
    for q2 in j2:
        for q3 in j3:
            candidate=q.copy()
            candidate[1]=q2
            candidate[2]=q3
            result=guard.inspect(candidate)
            # Independent forward kinematics; never silently move the saved state.
            previous=guard.data.qpos[guard.qids].copy()
            try:
                guard.data.qpos[guard.qids]=candidate
                mujoco.mj_forward(guard.model,guard.data)
                tcp=np.asarray(guard.data.xpos[tcp_id],float).copy()
            finally:
                guard.data.qpos[guard.qids]=previous
                mujoco.mj_forward(guard.model,guard.data)
            rows.append({
                "j2_deg":float(np.degrees(q2)),
                "j3_deg":float(np.degrees(q3)),
                "j2_delta_deg":float(np.degrees(q2-q[1])),
                "j3_delta_deg":float(np.degrees(q3-q[2])),
                "min_table_clearance_mm":float(result["minimum_distance_mm"]),
                "safe_static":bool(result["minimum_distance_mm"]>=clearance_mm),
                "tcp_z_m":float(tcp[2]),
                "nearest_pair":str(result["nearest_pair"]),
            })
    return rows


def summarize_scan(rows, *, required_clearance_mm=3.):
    possible=[r for r in rows if r["safe_static"]]
    ranked=sorted(possible,key=lambda r:(
        abs(r["j2_delta_deg"])+abs(r["j3_delta_deg"]),
        -r["min_table_clearance_mm"]))
    # For every proposed J2 angle, report safe J3 values if any. This is
    # intentionally a joint-coupled lookup instead of a fixed elbow rule.
    per_j2=[]
    for j2 in sorted({r["j2_deg"] for r in rows}):
        viable=[r for r in possible if r["j2_deg"]==j2]
        best=min(viable,key=lambda r:abs(r["j3_delta_deg"])) if viable else None
        per_j2.append({
            "j2_deg":j2,
            "safe_j3_count":len(viable),
            "nearest_safe_j3_deg":best["j3_deg"] if best else None,
            "nearest_safe_j3_delta_deg":best["j3_delta_deg"] if best else None,
        })
    return {
        "safe_static_pairs":len(possible),
        "sampled_pairs":len(rows),
        "nearest_safe_static_candidates":ranked[:10],
        "j2_to_safe_j3_lookup":per_j2,
        "important":"Static configurations only. No swept-path / self-collision / physical-torque guarantee.",
        "required_clearance_mm":required_clearance_mm,
    }


def plot_scan(rows, output, *, clearance_mm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=np.array(sorted({r["j2_deg"] for r in rows}))
    y=np.array(sorted({r["j3_deg"] for r in rows}))
    scores={(r["j2_deg"],r["j3_deg"]):r["min_table_clearance_mm"] for r in rows}
    z=np.array([[scores[(a,b)] for a in x] for b in y],dtype=float)
    fig,ax=plt.subplots(figsize=(9,7))
    img=ax.imshow(z,origin="lower",aspect="auto",extent=(x[0],x[-1],y[0],y[-1]),
                  cmap="coolwarm",vmin=-15,vmax=30)
    ax.contour(x,y,z,levels=[clearance_mm],colors=["black"],linewidths=1.7)
    ax.set(xlabel="J2 shoulder angle (deg)",ylabel="J3 elbow angle (deg)",
           title="SO-101 table clearance (mm) vs coupled J2/J3 pose")
    ax.grid(alpha=.15)
    cbar=fig.colorbar(img,ax=ax)
    cbar.set_label("Minimum moving-link / table signed distance (mm)")
    fig.tight_layout()
    fig.savefig(output,dpi=155)
    plt.close(fig)


def run(args):
    args.physical_objects=False
    args.wrist_camera=True
    (model,data,qids,dids,low,high,tcp,side,goals,
     *_remainder)=scene.create_scene(args)
    source=np.asarray(data.qpos[qids],float).copy()
    start=np.clip(source,low,high)
    guard=TableCollisionGuard(model,data,qids,clearance_mm=args.clearance_mm,
                             include_cup=args.include_cup)
    source_check=guard.inspect(source)
    clipped_check=guard.inspect(start)
    rows=scan_coupled_elbow(guard,start,low,high,tcp,
                            span_deg=args.span_deg,count=args.samples,
                            clearance_mm=args.clearance_mm)
    summary={
        "status":"J2_J3_STATIC_TABLE_CLEARANCE_SCAN",
        "source_start_joint_deg":np.degrees(source).tolist(),
        "clipped_start_joint_deg":np.degrees(start).tolist(),
        "source_initial_clearance":source_check,
        "clipped_initial_clearance":clipped_check,
        "saved_episode":args.episode,
        "source_urdf":str(args.urdf),
        "table_not_modified":True,
        "sac_checkpoint_not_used":True,
        "physics_enabled":False,
        "span_deg":args.span_deg,
        "analysis":summarize_scan(rows,required_clearance_mm=args.clearance_mm),
        "notes":[
            "J1, J4 and J5 are fixed; J2/J3 alone may not find a safe solution.",
            "The original home pose violates J2/J3 URDF joint limits. Scan uses clipped starting pose.",
            "Geometric meshes can be conservative convex approximations.",
            "A safe sampled pose is NOT proof that a collision-free path exists from the penetrating start.",
            "Do not adjust the physical table, visual calibration, or collision masks from this scan alone.",
        ]
    }
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    with (args.out/"j2_j3_clearance.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    plot_scan(rows,args.out/"j2_j3_clearance_map.png",clearance_mm=args.clearance_mm)
    print(json.dumps({
        "clipped_initial_clearance":clipped_check,
        "safe_static_pairs":summary["analysis"]["safe_static_pairs"],
        "sampled_pairs":summary["analysis"]["sampled_pairs"],
        "results_dir":str(args.out),
    },indent=2,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--span-deg",type=float,default=30.)
    p.add_argument("--samples",type=int,default=25)
    p.add_argument("--clearance-mm",type=float,default=3.)
    p.add_argument("--include-cup",action="store_true")
    p.add_argument("--dataset-root",type=Path,default=scene.DATASET)
    p.add_argument("--episode",type=int,default=4)
    p.add_argument("--object-layout-json",type=Path,default=scene.OBJECTS)
    p.add_argument("--cup-pose-json",type=Path,default=scene.CUP_POSE)
    p.add_argument("--extrinsics",type=Path,default=scene.EXTRINSICS)
    p.add_argument("--intrinsics",type=Path,default=scene.INTRINSICS)
    p.add_argument("--urdf",type=Path,default=scene.r2s.DEFAULT_URDF)
    p.add_argument("--robot-joint-offsets-json",type=Path)
    p.add_argument("--muhi-clearance-mm",type=float,default=75)
    p.add_argument("--cup-clearance-mm",type=float,default=45)
    p.add_argument("--wrist-mount-pos",type=float,nargs=3,default=wrist.DEFAULT_MOUNT_POS)
    p.add_argument("--wrist-mount-rpy-deg",type=float,nargs=3,default=wrist.DEFAULT_MOUNT_RPY_DEG)
    p.add_argument("--wrist-fovy",type=float,default=wrist.DEFAULT_FOVY_DEG)
    p.add_argument("--wrist-look-at",type=float,nargs=3,default=wrist.DEFAULT_LOOK_AT)
    args=p.parse_args()
    if (args.samples<3 or args.samples>101 or not math.isfinite(args.span_deg)
        or args.span_deg<=0 or args.span_deg>90
        or not math.isfinite(args.clearance_mm) or not 0<=args.clearance_mm<=50):
        p.error("Invalid scan parameters")
    run(args)


if __name__=="__main__":
    main()
