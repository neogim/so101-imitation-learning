#!/usr/bin/env python3
"""Build experimental LeRobot v3 Cartesian-action dataset COPY, offline only.

Copies camera videos, observations, meta and ancillary files from source, then
replaces action (6D joint goal) with 7D FK(action)-FK(state) + gripper target.
The resulting dataset is EXPERIMENTAL until loaded and trained with LeRobot.
No robot access; no changes to the source; no IK on motors.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import placo

from batch_cartesian_fk_ik_hypotheses_offline import mapping,to_urdf_rad
from export_cartesian_action_sidecar import JOINT_NAMES,target_from_fk

TARGET_NAMES=[
    "delta_tcp_x_m_base","delta_tcp_y_m_base","delta_tcp_z_m_base",
    "delta_rotvec_x_rad_base","delta_rotvec_y_rad_base","delta_rotvec_z_rad_base",
    "gripper_goal_normalized_0_100",
]

def reshape_stat(old,values):
    """Keep existing stats keys where possible; update action moments correctly."""
    x=np.asarray(values,dtype=np.float64)
    stats={"mean":x.mean(0),"std":x.std(0),"min":x.min(0),
           "max":x.max(0),"count":[len(x)]}
    if not isinstance(old,dict):
        return {k:v.tolist() for k,v in stats.items()}
    result=dict(old)
    for key,vec in stats.items():
        prior=old.get(key)
        data=vec.tolist() if hasattr(vec, "tolist") else list(vec)
        # LeRobot metadata generally serializes statistics as nested lists.
        if key!="count" and isinstance(prior,list) and prior and isinstance(prior[0],list):
            data=[data]
        result[key]=data
    return result

def convert_dataset(source,output,urdf,mapping_label,j2_offset,j3_offset):
    source=source.resolve()
    output=output.resolve()
    if output.exists() or source==output or source in output.parents:
        raise ValueError("Output must be a new path outside the source dataset")
    info_path=source/"meta/info.json"
    if not info_path.is_file():raise FileNotFoundError(info_path)
    info=json.loads(info_path.read_text())
    features=info.get("features",{})
    if features.get("action",{}).get("shape")!=[6] or features.get("observation.state",{}).get("shape")!=[6]:
        raise ValueError("Require original state and action features of shape [6]")
    if int(info.get("fps",0))!=20:
        raise ValueError("Expected 20 FPS source")
    originals=sorted((source/"data").rglob("*.parquet"))
    if not originals:raise ValueError("No parquet files")
    signs,offsets=mapping(mapping_label,j2_offset,j3_offset)
    robot=placo.RobotWrapper(str(urdf.resolve()))
    if not set(JOINT_NAMES).issubset(set(robot.joint_names())):
        raise ValueError("URDF joints missing")
    def fk(motor_deg):
        q=to_urdf_rad(motor_deg,signs,offsets)
        for name,value in zip(JOINT_NAMES,q,strict=True):
            robot.set_joint(name,float(value))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"),float).copy()

    converted={}
    action_values=[]
    total=0
    for path in originals:
        original=pq.read_table(path)
        if "observation.state" not in original.column_names or "action" not in original.column_names:
            raise ValueError("Missing action/state: "+str(path))
        states=original["observation.state"].to_pylist()
        actions=original["action"].to_pylist()
        vals=[]
        for st,ac in zip(states,actions,strict=True):
            vals.append(target_from_fk(np.asarray(st,float),np.asarray(ac,float),fk).tolist())
        arr=pa.array(vals,type=pa.list_(pa.float32(),7))
        index=original.schema.get_field_index("action")
        modified=original.set_column(index,"action",arr)
        converted[path.relative_to(source)]=modified
        action_values.extend(vals)
        total+=len(vals)
    if total!=int(info.get("total_frames",-1)):
        raise ValueError(f"All-frame count mismatch: {total} != {info.get('total_frames')}")
    action_values=np.asarray(action_values)
    if not np.isfinite(action_values).all():raise ValueError("Nonfinite Cartesian actions")

    # Validate statistics BEFORE creating the destination or copying videos.
    prepared_stats = None
    source_stats = source/"meta/stats.json"
    if source_stats.exists():
        prepared_stats = json.loads(source_stats.read_text())
        if not isinstance(prepared_stats,dict):
            raise ValueError("Unexpected statistics JSON")
        prepared_stats["action"] = reshape_stat(prepared_stats.get("action"),action_values)\n
    # Copy everything, preserving video references and parquet partition paths.
    # No hardlinks: source and output are independent even if one is edited.
    shutil.copytree(source,output)
    for relative,table in converted.items():
        pq.write_table(table,output/relative)
    updated=dict(info)
    new_features=dict(features)
    action_feature=dict(features["action"])
    action_feature["shape"]=[7]
    action_feature["names"]=TARGET_NAMES
    new_features["action"]=action_feature
    updated["features"]=new_features
    (output/"meta/info.json").write_text(json.dumps(updated,ensure_ascii=False,indent=2)+"\n")
    # Never preserve stale 6D action statistics.
    stats_path=output/"meta/stats.json"
    if prepared_stats is not None:
        stats_path.write_text(json.dumps(prepared_stats,ensure_ascii=False,indent=2)+"\n")

    manifest={
        "status":"EXPERIMENTAL_CARTESIAN_ACTION_COPY_NOT_HARDWARE_APPROVED",
        "source":str(source),"output":str(output),"urdf":str(urdf.resolve()),
        "mapping":mapping_label,"signs":signs.tolist(),"offsets_deg":offsets.tolist(),
        "source_frames":total,"action_shape":[7],"action_names":TARGET_NAMES,
        "action_semantics":"FK(joint_goal[t])-FK(measured_joint_state[t]), base frame, not observed next-state",
        "source_unchanged":True,"camera_videos_copied":True,"robot_commands_sent":False,
        "known_gaps":[
            "LeRobot training dataset read and training not verified",
            "Physical J2/J3 signs and offsets unknown; do not replay on robot",
            "7D ACT output requires matching normalization, action conversion and IK postprocessor",
            "SO-101 has only 5 arm joints; arbitrary 6D Cartesian targets may be infeasible",
            "No collision checking and source camera images are not counterfactual images",
        ],
    }
    (output/"cartesian_experimental_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(output),"frames":total,"action_shape":[7],
                      "source_unchanged":True,"status":manifest["status"]},indent=2))
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    p.add_argument("--mapping",choices=["j2+_j3+","j2-_j3+","j2+_j3-","j2-_j3-"],required=True)
    p.add_argument("--j2-offset-deg",type=float,default=0.)
    p.add_argument("--j3-offset-deg",type=float,default=0.)
    a=p.parse_args()
    convert_dataset(a.dataset,a.output,a.urdf,a.mapping,a.j2_offset_deg,a.j3_offset_deg)

if __name__=="__main__":main()
