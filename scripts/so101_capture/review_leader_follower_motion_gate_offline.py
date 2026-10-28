#!/usr/bin/env python3
"""Offline-only SO-101 Leader/Follower J2/J3 teleop calibration safety gate.

Reads a preflight JSON; never imports hardware, opens ports or sends commands.
A PASS here is NOT permission to actuate the arm.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")
TARGET=("shoulder_lift","elbow_flex")
def analyze(report, threshold=2.0, margin=5.0):
    if report.get("mode")!="READ_ONLY_NO_CONFIGURE_NO_GOAL_NO_TORQUE_WRITE":
        raise ValueError("Unexpected input: not a read-only preflight")
    readings=report["readings"]
    leader=readings["leader"]; follower=readings["follower"]
    for device in (leader,follower):
        if set(device["position_degrees"])!=set(JOINTS) or set(device["torque_enable_raw"])!=set(JOINTS):
            raise ValueError("Six-axis data incomplete")
    differences={j:abs(float(leader["position_degrees"][j])-float(follower["position_degrees"][j])) for j in JOINTS}
    torque_ok=all(int(device["torque_enable_raw"][j])==0 for device in (leader,follower) for j in JOINTS)
    # These are model (URDF) limits only; motor calibration and actual safe range can differ.
    model_limits={"shoulder_lift":(-100.0,100.0),"elbow_flex":(-96.8298673771,96.8298673771)}
    near=[]
    for j in TARGET:
        angle=float(follower["position_degrees"][j])
        lo,hi=model_limits[j]
        # Evaluates both sign hypotheses; symmetric model bounds apply to both.
        if not lo+margin<=angle<=hi-margin:
            near.append({"joint":j,"follower_deg":angle,"urdf_limit_deg":[lo,hi],
                         "note":"At/outside conservative URDF margin; mapping still unknown"})
    return {"classification":"OFFLINE_REVIEW_ONLY_NO_MOTION",
            "preflight_torque_all_off":torque_ok,
            "initial_difference_deg":differences,
            "max_initial_difference_deg":max(differences.values()),
            "initial_match_within_threshold":all(d<=threshold for d in differences.values()),
            "j2_j3_urdf_margin_warnings":near,
            "motion_authorization":"NOT_AUTHORIZED",
            "reason":"Torque-off, small initial differences do not establish safe teleop start. "
                     "J2/J3 sit close to or beyond unverified URDF angular bounds. "
                     "Requires operator inspection of startup torque, calibration, joint limits and clearance."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Refusing to overwrite output")
    result=analyze(json.loads(a.preflight.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("Initial difference max:",round(result["max_initial_difference_deg"],3),"deg")
    print("Torque all off:",result["preflight_torque_all_off"])
    print("URDF margin warnings:",result["j2_j3_urdf_margin_warnings"])
    print("Motion:",result["motion_authorization"])
    print("Saved:",a.output)
if __name__=="__main__":
    main()
