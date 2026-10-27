#!/usr/bin/env python3
"""Offline Placo mesh self-collision audit for SO-101 recorded states.

Only uses recorded observation.state and the existing URDF. No motor access.
Does not exclude adjacent pairs or change collision geometry. Computes observed
collisions at sampled configurations (not continuous swept-volume clearance).
"""
from __future__ import annotations
import argparse,collections,csv,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pyarrow.parquet as pq
import placo

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--dataset",required=True,type=Path)
 p.add_argument("--urdf",required=True,type=Path)
 p.add_argument("--output",required=True,type=Path)
 p.add_argument("--episode",type=int,default=0)
 p.add_argument("--stride",type=int,default=10)
 a=p.parse_args()
 if a.stride<1:p.error("stride must be >=1")
 if a.output.exists():p.error("output exists")
 r=ET.parse(a.urdf).getroot()
 links={x.get("name") for x in r.findall("link")}
 adjacent={frozenset((j.find("parent").get("link"),j.find("child").get("link"))) for j in r.findall("joint")}
 ordered_links=sorted(links,key=len,reverse=True)
 def link_of(body):
  body=str(body)
  for name in ordered_links:
   if body==name or body.startswith(name+"_") and body[len(name)+1:].isdigit():return name
  return None
 robot=placo.RobotWrapper(str(a.urdf.resolve()))
 if not hasattr(robot,"self_collisions"):raise RuntimeError("Placo lacks self_collisions()")
 rows=[]
 for f in sorted((a.dataset/"data").rglob("*.parquet")):
  tab=pq.read_table(f,columns=["episode_index","frame_index","observation.state"])
  rows.extend(rec for rec in tab.to_pylist() if int(rec["episode_index"])==a.episode and int(rec["frame_index"])%a.stride==0)
 rows.sort(key=lambda x:int(x["frame_index"]))
 if not rows:raise ValueError("No matching episode frames")
 records=[]
 for row in rows:
  q=np.asarray(row["observation.state"][:5],float)
  if not np.isfinite(q).all():raise ValueError("Nonfinite joints")
  for joint,deg in zip(JOINTS,q):
   robot.set_joint(joint,float(np.deg2rad(deg)))
  robot.update_kinematics()
  # False requests the full collision list instead of stopping at the first.
  contacts=robot.self_collisions(False)
  for contact in contacts:
   ba=str(contact.bodyA);bb=str(contact.bodyB)
   la=link_of(ba);lb=link_of(bb)
   category="unmapped" if la is None or lb is None else ("same_link" if la==lb else "adjacent" if frozenset((la,lb)) in adjacent else "nonadjacent")
   records.append({"frame":int(row["frame_index"]),"bodyA":ba,"bodyB":bb,
                   "linkA":la,"linkB":lb,"category":category})
 a.output.mkdir(parents=True)
 with (a.output/"collision_events.csv").open("w",newline="") as f:
  fields=["frame","bodyA","bodyB","linkA","linkB","category"]
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
 counts=collections.Counter(x["category"] for x in records)
 unique={category:sorted({tuple(sorted((x["linkA"],x["linkB"]))) for x in records if x["category"]==category and x["linkA"] and x["linkB"]}) for category in ("adjacent","nonadjacent")}
 flagged=sorted({x["frame"] for x in records if x["category"] in ("nonadjacent","unmapped")})
 summary={"episode":a.episode,"frames_sampled":len(rows),"stride":a.stride,
          "collision_events":len(records),"categories":dict(counts),
          "adjacent_link_pairs_observed":unique["adjacent"],
          "nonadjacent_link_pairs_observed":unique["nonadjacent"],
          "frames_with_nonadjacent_or_unmapped":flagged,
          "unmapped_events":sum(x["category"]=="unmapped" for x in records),
          "warnings":["Sampled configurations only, not continuous motion.",
                      "Placo mesh collisions may reflect nominal mesh overlaps; inspect visually.",
                      "No environment/table collision checking.",
                      "Zero/sign mapping to physical SO-101 remains unverified.",
                      "Do not command motors from these results."]}
 (a.output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
