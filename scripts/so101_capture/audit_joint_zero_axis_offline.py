#!/usr/bin/env python3
"""Offline zero/axis review sheet: LeRobot calibration vs URDF; NO motor access.

The signs and physical zero alignment require real-world evidence: this tool
does NOT infer them from URDF metadata or recorded actuator targets.
"""
import argparse,csv,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
NAMES=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--calibration",type=Path,required=True)
 p.add_argument("--urdf",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error("Output already exists")
 calibration=json.loads(a.calibration.read_text())
 root=ET.parse(a.urdf).getroot()
 urdf={j.get("name"):j for j in root.findall("joint")}
 results=[]
 for n in NAMES:
  c=calibration[n]; j=urdf[n]
  lo,hi=int(c["range_min"]),int(c["range_max"])
  midpoint=(lo+hi)/2
  axis=j.find("axis")
  origin=j.find("origin")
  row={"joint":n,"calibrated_midpoint_ticks":midpoint,
       "range_min_ticks":lo,"range_max_ticks":hi,
       "midpoint_degrees_definition":0,
       "homing_offset_ticks":int(c["homing_offset"]),
       "drive_mode":int(c["drive_mode"]),
       "urdf_axis_xyz":axis.get("xyz","1 0 0") if axis is not None else "1 0 0",
       "urdf_origin_xyz":origin.get("xyz","0 0 0") if origin is not None else "0 0 0",
       "urdf_origin_rpy":origin.get("rpy","0 0 0") if origin is not None else "0 0 0",
       "zero_alignment":"UNVERIFIED",
       "positive_direction_alignment":"UNVERIFIED",
       "evidence":"NOT_PROVIDED"}
  results.append(row)
 a.output.mkdir(parents=True)
 with (a.output/"joint_zero_axis_review.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 report={"joint_count":len(results),"joint_review":results,
  "interpretation":"LeRobot DEGREES zero is calibration range midpoint; URDF zero is kinematic reference. Equality cannot be inferred from both being named zero.",
  "evidence_required":["At least two independently measured physical configurations of the robot showing joint readings AND visually measured link orientation, or validated manufacturer URDF+calibration mapping",
                       "Positive direction sign for each joint using existing time-separated images and joint states; avoid moving motors until safety is established"],
  "warning":"No motor connection, no motion, no physical alignment or collision certification."}
 (a.output/"joint_zero_axis_review.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps({"output":str(a.output),"joints":[{"name":x["joint"],"midpoint_ticks":x["calibrated_midpoint_ticks"],"urdf_axis":x["urdf_axis_xyz"],"zero":x["zero_alignment"],"sign":x["positive_direction_alignment"]} for x in results],"warning":report["warning"]},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
