#!/usr/bin/env python3
"""One-pass offline topology + URDF joint-direction evidence audit.

Uses previously generated Codex episode audit; no motors or robot connection.
Produces paired URDF 3D direction views and a physical verification checklist.
Does NOT run collision detection or prove physical zero/sign alignment.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--urdf",type=Path,required=True)
 p.add_argument("--codex-summary",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():p.error("Output exists")
 summary=json.loads(a.codex_summary.read_text())
 if summary["episode"]!=0 or summary["dataset_frames_read"]!=648:raise ValueError("Expected Episode 0, 648 frames")
 root=ET.parse(a.urdf).getroot()
 joints=root.findall("joint")
 parent_pairs={frozenset([j.find("parent").get("link"),j.find("child").get("link")]) for j in joints}
 mesh_links={}
 for link in root.findall("link"):
  mesh_links[link.attrib["name"]]=len(link.findall("collision"))
 if len(summary["joint_results"])!=5:raise ValueError("Need 5 joints")
 reported=[
 ("base_link","shoulder_link"),("shoulder_link","upper_arm_link"),
 ("upper_arm_link","lower_arm_link"),("lower_arm_link","wrist_link"),
 ("wrist_link","gripper_link"),("gripper_link","moving_jaw_so101_v1_link")]
 classes=[{"link_a":x,"link_b":y,"direct_joint":frozenset((x,y)) in parent_pairs,
           "status":"NEUTRAL_WARNING_ONLY_NO_CLEARANCE_TEST"} for x,y in reported]
 collision_links=[k for k,v in mesh_links.items() if v]
 nonadjacent=[(x,y) for i,x in enumerate(collision_links) for y in collision_links[i+1:]
              if frozenset((x,y)) not in parent_pairs]
 joint_names=[x["joint"] for x in summary["joint_results"]]
 # Report measured joint angle differences and URDF skeleton-predicted displacement.
 # Local, dependency-light kinematics helpers: no OpenCV/video decoding needed.
 def rpy_matrix(v):
  r,p,y=v
  cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
  return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                   [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],
                   [-sp,cp*sr,cp*cr]])
 def axis_angle(axis,angle):
  axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
  x,y,z=axis;c=math.cos(angle);s=math.sin(angle);t=1-c
  return np.array([[t*x*x+c,t*x*y-s*z,t*x*z+s*y],
                   [t*x*y+s*z,t*y*y+c,t*y*z-s*x],
                   [t*x*z-s*y,t*y*z+s*x,t*z*z+c]])
 def parse_urdf(path):
  root=ET.parse(path).getroot()
  joints={j.get("name"):j for j in root.findall("joint")}
  names=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
  out=[]
  for name in names:
   j=joints[name];origin=j.find("origin");axis=j.find("axis")
   out.append({"origin_xyz":[float(x) for x in origin.get("xyz","0 0 0").split()],
    "origin_rpy":[float(x) for x in origin.get("rpy","0 0 0").split()],
    "axis_local":[float(x) for x in axis.get("xyz","1 0 0").split()]})
  return out
 def fk_origins(specs,qdeg):
  T=np.eye(4);points=[T[:3,3].copy()];axes=[]
  for spec,deg in zip(specs,qdeg):
   O=np.eye(4);O[:3,:3]=rpy_matrix(spec["origin_rpy"]);O[:3,3]=spec["origin_xyz"]
   T=T@O
   axis=np.asarray(spec["axis_local"],float)
   axes.append(T[:3,:3]@axis);points.append(T[:3,3].copy())
   R=np.eye(4);R[:3,:3]=axis_angle(axis,math.radians(float(deg)))
   T=T@R
  points.append(T[:3,3]+T[:3,:3]@np.array([0.,-.08,0.]))
  return np.asarray(points),np.asarray(axes)
 specs=parse_urdf(a.urdf)
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 figures=[]
 for result,pair in zip(summary["joint_results"],summary["direction_pairs"]):
  if result["joint"]!=pair["joint"]:raise ValueError("Joint ordering mismatch")
  qa=np.asarray(pair["state_a_deg"],dtype=float)
  qb=np.asarray(pair["state_b_deg"],dtype=float)
  pa,_=fk_origins(specs,qa); pb,_=fk_origins(specs,qb)
  # Points from fk_origins are origins, not physical measured joint centers.
  fig=plt.figure(figsize=(9,5))
  ax=fig.add_subplot(111,projection="3d")
  ax.plot(pa[:,0],pa[:,1],pa[:,2],"o-",label=f"frame {pair['frame_a']}")
  ax.plot(pb[:,0],pb[:,1],pb[:,2],"o-",label=f"frame {pair['frame_b']}")
  ax.set_xlim(-.35,.20);ax.set_ylim(-.25,.25);ax.set_zlim(-.15,.35)
  ax.set_xlabel("base X (m)");ax.set_ylabel("base Y (m)");ax.set_zlabel("base Z (m)")
  ax.set_title(f"{result['joint']}  delta={pair['delta_target_deg']:+.2f} deg\nURDF model prediction, NOT camera registration")
  ax.legend()
  path=f"urdf_direction_{result['joint']}.png"
  figures.append((fig,path))
  result["predicted_tool_displacement_mm"]=float(np.linalg.norm(pa[-1]-pb[-1])*1000)
 a.output.mkdir(parents=True)
 for fig,path in figures:
  fig.savefig(a.output/path,dpi=130,bbox_inches="tight")
  plt.close(fig)
 report={"source_summary":str(a.codex_summary),"collision_mesh_links":mesh_links,
         "neutral_warning_pairs":classes,"unexamined_nonadjacent_mesh_pairs":nonadjacent,
         "joint_comparisons":[{"joint":x["joint"],"frame_pair":x["direction_pair_frames"],
          "delta_deg":x["direction_delta_deg"],"zero_candidate_deg":x["zero_candidate_deg"],
          "predicted_tool_displacement_mm":x["predicted_tool_displacement_mm"],
          "zero_status":"UNVERIFIED","sign_status":"UNVERIFIED"} for x in summary["joint_results"]],
         "results":{"reported_pairs_all_adjacent":all(x["direct_joint"] for x in classes),
          "nonadjacent_collision_checked":False,"physical_zero_verified":False,"physical_sign_verified":False},
         "next_action":"Independent physical reference: record fixed-base image, joint readings and angle/axis measurements with motors unpowered. Determine zero and positive direction for each joint, especially J4/J5, before commanding IK.",
         "warnings":["No collision engine was run: enumerating nonadjacent pairs is NOT a collision check.",
         "URDF trajectory overlays do not prove physical axis mapping or safety.",
         "Do NOT use experimental URDF for robot motion."]}
 (a.output/"combined_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
 lines=["# SO-101 joint mapping and collision review (offline)","",
  f"- All six neutral-position warnings are adjacent parent-child links: **{report['results']['reported_pairs_all_adjacent']}**",
  f"- Nonadjacent collision-mesh pairs requiring actual collision-engine checks: **{len(nonadjacent)}**",
  "- **No nonadjacent collision check was executed.**","",
  "|Joint|Frame pair|Delta (deg)|Zero candidate (deg)|Physical zero/sign|",
  "|---|---|---:|---:|---|"]
 for x in report["joint_comparisons"]:
  lines.append(f"|{x['joint']}|{x['frame_pair'][0]} -> {x['frame_pair'][1]}|{x['delta_deg']:+.2f}|{x['zero_candidate_deg']:+.2f}|UNVERIFIED|")
 lines.extend(["","## Files","- urdf_direction_<joint>.png: model-predicted 3D poses at recorded frames, not camera registered.",
               "- combined_audit.json: every collision pair and all diagnostic metadata.",
               "","## Safety gate","Do not write IK results to motors until physical zero, sign, collision pairs and joint dynamics are validated."])
 (a.output/"report.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"output":str(a.output),"warning_pairs_adjacent":len(classes),
       "nonadjacent_pairs_awaiting_collision_check":len(nonadjacent),
       "physical_zero_and_sign_verified":False,"plots":len(figures)},indent=2))
if __name__=="__main__":main()
