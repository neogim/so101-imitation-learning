#!/usr/bin/env python3
"""Offline SO-101 calibration/URDF readiness audit. NEVER contacts motors.

Crucial: SOFollower's body joints may use MotorNormMode.RANGE_M100_100
(use_degrees=False) OR MotorNormMode.DEGREES (use_degrees=True).
A LeRobot value must NOT be interpreted as a physical degree without
verifying the exact installed LeRobot version and config.
"""
from __future__ import annotations
import argparse,csv,json,math,xml.etree.ElementTree as ET
from pathlib import Path

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
ALL=(*JOINTS,"gripper")

def inverse_normalize(value,cal,mode):
    lo,hi=cal["range_min"],cal["range_max"]
    if mode=="range_m100_100":
        # LeRobot motor bus applies drive_mode in software.
        v=-value if cal["drive_mode"] else value
        return lo+(v+100)*(hi-lo)/200
    if mode=="range_0_100":
        v=100-value if cal["drive_mode"] else value
        return lo+v*(hi-lo)/100
    if mode=="degrees":
        # STS3215 12-bit model; DEGREES normalization is relative to
        # calibrated encoder range midpoint, not a kinematic zero.
        return (lo+hi)/2+value*4095/360
    raise ValueError(mode)

def analyze(follower,urdf,predictions,use_degrees):
    cal=json.loads(follower.read_text())
    root=ET.parse(urdf).getroot()
    urdf_joints={j.attrib.get("name"):j for j in root.findall("joint")}
    for i,name in enumerate(ALL,1):
        if name not in cal or cal[name].get("id")!=i:raise ValueError(f"Missing/wrong ID {name}")
        c=cal[name]
        if not 0<=c["range_min"]<c["range_max"]<=4095:raise ValueError(f"Invalid range {name}")
        if c["drive_mode"] not in (0,1):raise ValueError(f"Invalid drive mode {name}")
    with predictions.open(newline="") as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError("Empty predictions")
    mode="degrees" if use_degrees else "range_m100_100"
    res=[]
    for j,name in enumerate(JOINTS):
        c=cal[name]
        values=[float(r[f"state_{j}"]) for r in rows]
        if not all(math.isfinite(v) for v in values):raise ValueError("Nonfinite values")
        inferred=[inverse_normalize(v,c,mode) for v in values]
        # LeRobot may clamp at normalization. Here we flag unclamped inverse
        # and cannot infer real encoder values outside calibration.
        outside=sum(not c["range_min"]<=v<=c["range_max"] for v in inferred)
        jo=urdf_joints.get(name)
        limit=jo.find("limit") if jo is not None else None
        limits=None if limit is None else [
           float(limit.attrib["lower"])*180/math.pi,
           float(limit.attrib["upper"])*180/math.pi]
        res.append({"joint":name,"le_robot_state_min_max":[min(values),max(values)],
            "mode_assumed":mode,
            "calibrated_encoder_range":[c["range_min"],c["range_max"]],
            "inverse_normalized_encoder_min_max":[min(inferred),max(inferred)],
            "inverse_outside_calibrated_range_count":outside,
            "URDF_joint_limit_deg":limits,
            "WARNING":"Calibrated encoder counts do not directly determine URDF joint sign/zero"})
    collisions=[]
    for link in root.findall("link"):
        els=link.findall("collision")
        if els:
            collisions.append({"link":link.attrib.get("name"),"collision_geometries":len(els),
                "types":[next((x.tag for x in list(el.find("geometry") or []) if isinstance(x.tag,str)),"none") for el in els]})
    return {"classification":"OFFLINE_DIAGNOSTIC_NOT_PHYSICAL_CALIBRATION",
        "follower":str(follower),"source_frames":len(rows),"body_mode_assumption":mode,
        "body_joints":res,"collision_geometry_inventory":collisions,
        "collision_checker":"NOT_IMPLEMENTED","swept_path_checker":"NOT_IMPLEMENTED",
        "critical":["If use_degrees=False, LeRobot body actions represent calibrated [-100,100], NOT physical degrees. Existing Cartesian7 FK labels, IK, and 60% TCP comparison cannot be physically interpreted.",
                    "If use_degrees=True, values are servo-centered degrees; correspondence to URDF sign/zero still requires physical calibration.",
                    "Homing_offset acts inside Feetech position reporting; do not add it again to normalized positions.",
                    "Inferred encoder coordinates outside calibrated range are diagnostic only; LeRobot normalizing clamps bounded values.",
                    "URDF neutral-position self-collision warnings are not evidence that individual predicted waypoints are collision-free or colliding.",
                    "No motor commands. No permission to execute existing IK candidates."]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--follower",type=Path,required=True)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    p.add_argument("--body-mode",choices=("range_m100_100","degrees"),required=True,
      help="Verify SOFollowerRobotConfig.use_degrees on the actual training installation")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report=analyze(a.follower,a.urdf,a.predictions,a.body_mode=="degrees")
    a.output.mkdir(parents=True)
    (a.output/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="collision_geometry_inventory"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
