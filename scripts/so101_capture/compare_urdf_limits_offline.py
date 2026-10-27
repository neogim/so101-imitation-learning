#!/usr/bin/env python3
"""Create a separate, limit-only experimental SO-101 URDF and compare recorded actions.

Read-only for input files and motors. Experimental limits are NOT hardware-safe
limits. No FK/IK solver is run here and no collision certification is implied.
"""
from __future__ import annotations
import argparse,csv,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--audit-csv",type=Path,required=True)
 p.add_argument("--urdf",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--margin-deg",type=float,default=0.5)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists")
 if not np.isfinite(a.margin_deg) or a.margin_deg<0 or a.margin_deg>2:p.error("margin must be 0..2 degrees")
 rows=list(csv.DictReader(a.audit_csv.open(newline="")))
 if len(rows)!=30846:raise ValueError("Expected all 30846 frames")
 q=np.array([[float(r[f"{n}_goal"]) for n in JOINTS] for r in rows])
 if not np.isfinite(q).all():raise ValueError("Non-finite action")
 tree=ET.parse(a.urdf);root=tree.getroot()
 joints={j.get("name"):j for j in root.findall("joint")}
 originals={};modified={}
 for n in JOINTS:
  lim=joints[n].find("limit")
  if lim is None:raise ValueError(f"Missing joint limits {n}")
  originals[n]=[float(lim.get("lower")),float(lim.get("upper"))]
  modified[n]=originals[n].copy()
 for i,n in ((1,"shoulder_lift"),(2,"elbow_flex")):
  current=modified[n]
  lo=min(current[0],float(np.deg2rad(q[:,i].min()-a.margin_deg)))
  hi=max(current[1],float(np.deg2rad(q[:,i].max()+a.margin_deg)))
  modified[n]=[lo,hi]
  joints[n].find("limit").set("lower",f"{lo:.12f}")
  joints[n].find("limit").set("upper",f"{hi:.12f}")
 def violation(limits):
  bounds=np.array([limits[n] for n in JOINTS])*180/np.pi
  bad=(q<bounds[:,0])|(q>bounds[:,1])
  return {"any_frames":int(bad.any(axis=1).sum()),"by_joint":{n:int(bad[:,i].sum()) for i,n in enumerate(JOINTS)}}
 a.output.mkdir(parents=True)
 out=a.output/"so101_limits_experiment_ONLY.urdf"
 tree.write(out,encoding="utf-8",xml_declaration=True)
 report={"input_urdf":str(a.urdf),"sha256":hashlib.sha256(a.urdf.read_bytes()).hexdigest(),
 "experiment_urdf":str(out),"margin_deg":a.margin_deg,
 "original_limits_deg":{k:list(np.rad2deg(v)) for k,v in originals.items()},
 "experimental_limits_deg":{k:list(np.rad2deg(v)) for k,v in modified.items()},
 "recorded_action_range_deg":{n:[float(q[:,i].min()),float(q[:,i].max())] for i,n in enumerate(JOINTS)},
 "baseline_violations":violation(originals),"experimental_violations":violation(modified),
 "WARNING":"DO NOT use experimental URDF for robot control. It is a limit-only offline hypothesis; zero/sign and physical clearances unverified."}
 (a.output/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
