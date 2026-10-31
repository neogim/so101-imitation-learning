#!/usr/bin/env python3
"""Offline analysis of LeRobot calibrated angles vs SO101 URDF joint limits.

Does not access serial ports or modify calibration/URDF.
"""
import argparse,json,math
from pathlib import Path

TARGETS={"shoulder_lift":(-100.,100.),"elbow_flex":(-96.8298673771,96.8298673771)}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--preflight",type=Path,required=True)
 p.add_argument("--follower-calibration",type=Path,required=True)
 p.add_argument("--leader-calibration",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 args=p.parse_args()
 if args.output.exists():p.error("Refusing overwrite")
 record=json.loads(args.preflight.read_text())
 if record.get("mode")!="READ_ONLY_NO_CONFIGURE_NO_GOAL_NO_TORQUE_WRITE":p.error("not a verified preflight document")
 l=json.loads(args.leader_calibration.read_text())
 f=json.loads(args.follower_calibration.read_text())
 import hashlib
 for label,path in (("leader",args.leader_calibration),("follower",args.follower_calibration)):
  actual=hashlib.sha256(path.read_bytes()).hexdigest()
  expected=record["calibration_sha256"][label]
  if actual!=expected:p.error(f"{label} calibration hash differs from preflight; abort")
 results={}
 for j,(lo,hi) in TARGETS.items():
  current=float(record["readings"]["follower"]["position_degrees"][j])
  target=float(record["readings"]["leader"]["position_degrees"][j])
  data={}
  for name,cal in (("leader",l),("follower",f)):
   item=cal[j]
   lower,upper=int(item["range_min"]),int(item["range_max"])
   # LeRobot DEGREES mode uses midpoint and 4095, regardless of drive-mode.
   if lower>=upper:raise ValueError(f"Bad calibration endpoints: {j} {name}")
   span=(upper-lower)*360/4095
   data[name]={"range_raw":[lower,upper],"range_span_degrees":round(span,3),
               "homing_offset_raw":item.get("homing_offset"),
               "drive_mode":item.get("drive_mode")}
  results[j]={"follower_current_deg":current,"leader_current_deg":target,
              "urdf_bounds_deg":[lo,hi],
              "distance_to_urdf_lower_deg":round(current-lo,3),
              "distance_to_urdf_upper_deg":round(hi-current,3),
              "calibration":data,
              "caution":"URDF limit and motor calibration are not directly interchangeable; sign/offset unknown"}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 report={"mode":"OFFLINE_ONLY","joints":results,"motion_authorization":"NOT_AUTHORIZED"}
 args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
 for name,item in results.items():
  print(name,"Follower",item["follower_current_deg"],"URDF distances",item["distance_to_urdf_lower_deg"],item["distance_to_urdf_upper_deg"])
  print("  follower calibrated raw span",item["calibration"]["follower"]["range_raw"],"angular span",item["calibration"]["follower"]["range_span_degrees"])
 print("Saved",args.output,"(offline; no motor commands)")
if __name__=="__main__":main()
