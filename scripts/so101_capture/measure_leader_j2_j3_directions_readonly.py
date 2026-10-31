#!/usr/bin/env python3
"""Interactive read-only J2/J3 direction experiment using SO-101 Leader.

Only Leader serial Present_Position reads. Never configure, torque or Goal.
Take a photo or note direction after each manual pose, then hit Enter.
"""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
from datetime import datetime,timezone

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")
def capture(bus,label):
    vals=bus.sync_read("Present_Position",num_retry=0)
    if set(vals)!=set(JOINTS) or any(not math.isfinite(float(x)) for x in vals.values()):
        raise RuntimeError("Incomplete/nonfinite joint positions")
    print("\n",label, "J2:",round(vals["shoulder_lift"],3),
          "J3:",round(vals["elbow_flex"],3),flush=True)
    return {"name":label,"utc":datetime.now(timezone.utc).isoformat(),
            "angles_deg_or_gripper_units":vals}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("outputs/so101-leader-j2-j3-directions-v1.json"))
    p.add_argument("--port")
    a=p.parse_args()
    if a.output.exists():p.error("Output exists: choose v2 etc")
    from record_episode import LEADER
    from lerobot.teleoperators.so_leader import SO101Leader,SO101LeaderConfig
    leader=SO101Leader(SO101LeaderConfig(port=a.port or LEADER,id="my_leader_arm",use_degrees=True))
    records=[]
    try:
        leader.bus.connect()
        if not leader.is_calibrated:raise RuntimeError("Leader calibration mismatch")
        torque=leader.bus.sync_read("Torque_Enable",normalize=False,num_retry=0)
        if set(torque)!=set(JOINTS) or any(v!=0 for v in torque.values()):
            raise RuntimeError("Leader torque must already be OFF (no changes performed)")
        print("LEADER ONLY read-only probe. Follower NOT connected, no commands sent.")
        print("Support arm as needed. Avoid hard stops, cables and finger pinch points.")
        print("Leave Leader at a comfortable starting position; do not force joints.")
        input("Press ENTER to record BASELINE (do not move while sampling): ")
        records.append(capture(leader.bus,"baseline"))
        for axis in ("J2","J3"):
            print("\nMOVE",axis,"of LEADER slowly through a visible, comfortable angle.")
            print("Observe and remember which direction the corresponding link rotates.")
            input("When motion has STOPPED, press ENTER to record: ")
            row=capture(leader.bus,axis+"_after")
            row["relative_to_baseline"]={j:round(row["angles_deg_or_gripper_units"][j]-
                 records[0]["angles_deg_or_gripper_units"][j],3) for j in JOINTS}
            records.append(row)
            for name in ("shoulder_lift","elbow_flex","wrist_flex"):
                print("  Δ",name,round(row["relative_to_baseline"][name],3))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({"mode":"LEADER_READ_ONLY","records":records,
            "warning":"Leader pose alone does not establish Follower-URDF zero offset or sign."},
            indent=2,ensure_ascii=False)+"\n")
        print("Saved",a.output)
        print("Please report visual direction: for J2 and J3, which link moved and clockwise/counterclockwise FROM WHICH CAMERA SIDE.")
    finally:
        if leader.bus.is_connected:leader.bus.disconnect(disable_torque=False)
if __name__=="__main__":main()
