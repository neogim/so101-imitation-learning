#!/usr/bin/env python3
"""Read-only URDF positional Jacobian SVD diagnostic for selected SO-101 DP paths.

Numerically differentiate FK: 3x5 translational Jacobian, meters/radian.
Requires placo but never touches real hardware, motors, datasets or URDF files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import placo
from evaluate_mapping_hypotheses_offline import HYPOTHESES, motor_to_urdf_degrees
from export_cartesian_action_sidecar import JOINT_NAMES

def finite_difference_jacobian(fk_position, q, h=1e-5):
    q=np.asarray(q,dtype=float)
    if q.shape != (5,) or not np.isfinite(q).all() or h <= 0:
        raise ValueError("Invalid angles or derivative step")
    columns=[]
    for k in range(5):
        delta=np.zeros(5);delta[k]=h
        plus=np.asarray(fk_position(q+delta),dtype=float)
        minus=np.asarray(fk_position(q-delta),dtype=float)
        if plus.shape!=(3,) or minus.shape!=(3,):
            raise ValueError("FK position shape must be (3,)")
        columns.append((plus-minus)/(2*h))
    return np.column_stack(columns)

def svd_metrics(j):
    s=np.linalg.svd(j,compute_uv=False)
    return {"singular_values_m_per_rad":s.tolist(),
            "sigma_min_m_per_rad":float(s[-1]),
            "condition_number":float(s[0]/s[-1]) if s[-1] > 1e-10 else None,
            "near_singular_numeric":bool(s[-1] < 1e-5)}

def run(args):
    if args.output.exists():raise ValueError("Output exists")
    if args.start > args.end:raise ValueError("Invalid frame window")
    rows=json.loads(args.paths.read_text())
    selected=sorted([x for x in rows if x["mapping"]==args.mapping and
                     x["method"]==args.method and x["direction"]==args.direction and args.start <= int(x["frame"]) <= args.end],
                    key=lambda x:int(x["frame"]))
    if not selected:raise ValueError("No selected path found")
    if [x["frame"] for x in selected] != list(range(args.start,args.end+1)):
        raise ValueError("Missing or duplicate frames in requested window")
    signs=HYPOTHESES[args.mapping]
    robot=placo.RobotWrapper(str(args.urdf.resolve()))
    def fk(q):
        for name, angle in zip(JOINT_NAMES,q):
            robot.set_joint(name,float(angle))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"),dtype=float)[:3,3].copy()
    result=[]
    for row in selected:
        c=row["candidate"]
        if c is None:
            result.append({"frame":row["frame"],"missing_candidate":True})
            continue
        motor=np.asarray(c["motor"],dtype=float)
        q=np.deg2rad(motor_to_urdf_degrees(motor,signs))
        j=finite_difference_jacobian(fk,q)
        stat=svd_metrics(j)
        stat.update({"frame":int(row["frame"]),"seed":c["seed"],
                     "active_bound_joints":c["active_bound_joints"]})
        result.append(stat)
    finite=[x for x in result if not x.get("missing_candidate")]
    minrow=min(finite,key=lambda x:x["sigma_min_m_per_rad"]) if finite else None
    output={"classification":"CONDITIONAL_NUMERICAL_SENSITIVITY",
            "mapping":args.mapping,"direction":args.direction,"method":args.method,
            "frames":[args.start,args.end],"jacobian":"URDF TCP position 3x5 m/rad, centered finite difference",
            "fd_step_rad":1e-5,"min_sigma":minrow,
            "near_singular_frames":[x["frame"] for x in finite if x["near_singular_numeric"]],
            "missing_frames":[x["frame"] for x in result if x.get("missing_candidate")],
            "rows":result,
            "warning":"Condition number is a numeric diagnostic, not proof of physical singularity or safe hardware operation."}
    args.output.write_text(json.dumps(output,indent=2)+"\n")
    print("direction:",args.direction,"min_sigma:",minrow["sigma_min_m_per_rad"] if minrow else "N/A",
          "at frame:",minrow["frame"] if minrow else "N/A")
    print("near_singular_frames:",output["near_singular_frames"])
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths",type=Path,required=True)
    p.add_argument("--urdf",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mapping",default="shoulder+_elbow+")
    p.add_argument("--direction",choices=("x+","x-","y+","y-","z+","z-"),required=True)
    p.add_argument("--method",default="dp_second_order")
    p.add_argument("--start",type=int,default=50)
    p.add_argument("--end",type=int,default=100)
    a=p.parse_args()
    run(a)

if __name__=="__main__":
    main()
