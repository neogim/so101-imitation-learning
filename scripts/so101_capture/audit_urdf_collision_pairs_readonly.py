#!/usr/bin/env python3
"""Read-only inspection of URDF kinematic chain and collision mesh pairs.

Do not infer actual collisions from collision elements; Placo warnings must be
independently validated with pairwise geometry and permitted link pairs.
"""
import argparse, json, xml.etree.ElementTree as ET
from pathlib import Path

def inspect(root):
    joints=root.findall("joint")
    parent_child={}
    for j in joints:
        p,c=j.find("parent"),j.find("child")
        if p is not None and c is not None:parent_child[frozenset((p.get("link"),c.get("link")))]=j.get("name")
    links={}
    same_mesh=[]
    for link in root.findall("link"):
        def meshes(tag):
            out=[]
            for element in link.findall(tag):
                geo=element.find("geometry")
                mesh=geo.find("mesh") if geo is not None else None
                if mesh is not None:out.append(mesh.get("filename"))
            return out
        vis,coll=meshes("visual"),meshes("collision")
        links[link.get("name")]={"visual_meshes":vis,"collision_meshes":coll,
                                  "collision_count":len(link.findall("collision"))}
        same_mesh.extend({"link":link.get("name"),"mesh":m} for m in coll if m in vis)
    warnings=[
        ("base_link","shoulder_link"),("shoulder_link","upper_arm_link"),
        ("upper_arm_link","lower_arm_link"),("lower_arm_link","wrist_link"),
        ("wrist_link","gripper_link"),("gripper_link","moving_jaw_so101_v1_link")
    ]
    findings=[]
    for a,b in warnings:
        findings.append({"parent_link":a,"child_link":b,
            "directly_connected_joint":parent_child.get(frozenset((a,b))),
            "a_collision_elements":links.get(a,{}).get("collision_count"),
            "b_collision_elements":links.get(b,{}).get("collision_count"),
            "note":"Link-level pair approximates Placo's part-level warning; not a geometry intersection calculation."})
    return {"links":links,"same_visual_collision_mesh_records":same_mesh,
            "reported_neutral_warning_pairs":findings,
            "conclusion":"All six reported pairs are listed for adjacency checking; mesh intersection and actual collision safety are unverified."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Output path already exists")
    result=inspect(ET.parse(a.urdf).getroot())
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("Shared visual/collision mesh records:",len(result["same_visual_collision_mesh_records"]))
    for x in result["reported_neutral_warning_pairs"]:
        print(x["parent_link"],"<->",x["child_link"],
              "direct joint:",x["directly_connected_joint"],
              "collision elements:",x["a_collision_elements"],x["b_collision_elements"])
    print("No geometric collision or physical sign mapping was verified.")
if __name__=="__main__":main()
