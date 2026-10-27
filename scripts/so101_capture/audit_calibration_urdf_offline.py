#!/usr/bin/env python3
"""Offline calibration-vs-URDF angle audit. NO robot connection or commands.

Checks stored STS3215 ranges under LeRobot DEGREES assumption against URDF
bounds and previously audited joint-action CSV. Does NOT establish physical
zero position, axis sign, collision safety or actual firmware calibration.
"""
from __future__ import annotations
import argparse,csv,json,inspect,importlib.metadata,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--calibration",type=Path,required=True)
 p.add_argument("--urdf",type=Path,required=True)
 p.add_argument("--audit-csv",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists; will not overwrite")
 for f in (a.calibration,a.urdf,a.audit_csv):
  if not f.is_file():raise FileNotFoundError(f)
 cal=json.loads(a.calibration.read_text())
 nodes={n.attrib.get("name"):n for n in ET.parse(a.urdf).getroot().findall("joint")}
 with a.audit_csv.open(newline="") as f:
  reader=csv.DictReader(f)
  cols=set(reader.fieldnames or [])
  need={f"{n}_goal" for n in JOINTS}
  if not need.issubset(cols):raise ValueError(f"Missing columns: {need-cols}")
  values={n:[] for n in JOINTS}
  for r in reader:
   for n in JOINTS:values[n].append(float(r[f"{n}_goal"]))
 report={"assumption":"LeRobot MotorNormMode.DEGREES, STS3215 4096 resolution (max_res=4095); dataset recorded in degrees",
         "motors":{},"action_frames":len(values[JOINTS[0]]),
         "not_proven":["physical joint axis sign","URDF zero alignment","firmware active calibration","actual motor mechanical limits","robot safety"]}
 for n in JOINTS:
  obj=cal[n]; low_tick=int(obj["range_min"]);hi_tick=int(obj["range_max"])
  if low_tick>=hi_tick:raise ValueError(f"Invalid range: {n}")
  degree_span=(hi_tick-low_tick)*360/4095
  m=nodes[n].find("limit")
  if m is None:raise ValueError(f"Missing URDF limit {n}")
  urdf=[float(m.attrib[x])*180/np.pi for x in ("lower","upper")]
  data=np.asarray(values[n])
  if not np.isfinite(data).all():raise ValueError(f"Nonfinite {n}")
  cap=[-degree_span/2,degree_span/2]
  r={"homing_offset":int(obj["homing_offset"]),"drive_mode":int(obj["drive_mode"]),
     "range_ticks":[low_tick,hi_tick],"lerobot_degrees_range":cap,"urdf_degrees_range":urdf,
     "dataset_degrees_minmax":[float(data.min()),float(data.max())],
     "dataset_outside_urdf_frames":int(np.sum((data<urdf[0])|(data>urdf[1]))),
     "dataset_outside_calibrated_range_frames":int(np.sum((data<cap[0]-1e-6)|(data>cap[1]+1e-6)))}
  report["motors"][n]=r
 try:
  report["installed_lerobot_version"]=importlib.metadata.version("lerobot")
 except importlib.metadata.PackageNotFoundError:
  report["installed_lerobot_version"]="unknown"
 a.output.mkdir(parents=True)
 (a.output/"calibration_urdf_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
