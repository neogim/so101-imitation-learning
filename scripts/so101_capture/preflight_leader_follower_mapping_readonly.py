#!/usr/bin/env python3
"""SO-101 Leader/Follower read-only serial preflight; NEVER sends a motor command.

IMPORTANT: Run only when no other process owns either port. Exits and closes
both buses without disabling torque or configuring hardware. Requires LeRobot.
"""
from __future__ import annotations
import argparse, json, hashlib, math
from pathlib import Path
from datetime import datetime, timezone

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")
def run(args):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
    if args.output.exists():raise FileExistsError(args.output)
    paths=(HF_LEROBOT_CALIBRATION/"teleoperators/so_leader/my_leader_arm.json",
           HF_LEROBOT_CALIBRATION/"robots/so_follower/my_follower_arm.json")
    if not all(x.is_file() for x in paths):
        raise FileNotFoundError(f"Missing calibration: {[str(x) for x in paths if not x.is_file()]}")
    leader=SO101Leader(SO101LeaderConfig(port=args.leader_port,id="my_leader_arm",use_degrees=True))
    follower=SO101Follower(SO101FollowerConfig(port=args.follower_port,id="my_follower_arm",
                                               use_degrees=True,disable_torque_on_disconnect=False))
    opened=[]
    try:
        for device in (leader,follower):
            device.bus.connect()
            opened.append(device)
        if not leader.is_calibrated or not follower.is_calibrated:
            raise RuntimeError("Calibration mismatch; no configure / repair attempted.")
        readings={}
        for role,device in (("leader",leader),("follower",follower)):
            readings[role]={
                "position_degrees":device.bus.sync_read("Present_Position",num_retry=0),
                "torque_enable_raw":device.bus.sync_read("Torque_Enable",normalize=False,num_retry=0),
            }
        if any(set(readings[r]["position_degrees"])!=set(JOINTS) for r in readings):
            raise RuntimeError("Unexpected joint names")
        delta={name:round(float(readings["leader"]["position_degrees"][name]-
                                readings["follower"]["position_degrees"][name]),3)
               for name in JOINTS}
        if not all(math.isfinite(v) for v in delta.values()):raise ValueError("Nonfinite angles")
        obj={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
             "mode":"READ_ONLY_NO_CONFIGURE_NO_GOAL_NO_TORQUE_WRITE",
             "calibration_sha256":{"leader":hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                                   "follower":hashlib.sha256(paths[1].read_bytes()).hexdigest()},
             "readings":readings,"leader_minus_follower":delta,
             "alert": "Position difference is NOT proof that starting teleop is safe. "
                      "Existing startup code may align and enable torque."}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
        for name in JOINTS:
            print(f"{name:16s} L-F={delta[name]:+8.3f} "
                  f"L={readings['leader']['position_degrees'][name]:8.3f} "
                  f"F={readings['follower']['position_degrees'][name]:8.3f}")
        for role in ("leader","follower"):
            print(role,"torque",readings[role]["torque_enable_raw"])
        print("Saved",args.output)
        print("NO movement commands sent. Do not start teleop solely based on this report.")
    finally:
        for device in reversed(opened):
            device.bus.disconnect(disable_torque=False)

def main():
    from record_episode import FOLLOWER, LEADER
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--follower-port",default=FOLLOWER)
    p.add_argument("--leader-port",default=LEADER)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
if __name__=="__main__":main()
