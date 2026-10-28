#!/usr/bin/env python3
"""Read-only SO-101 joint mapping / URDF collision metadata audit. No hardware or IK."""
import argparse, json, math, xml.etree.ElementTree as ET
from pathlib import Path

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")
def audit(urdf,calibration=None):
    root=ET.parse(urdf).getroot()
    joints={j.attrib["name"]:j for j in root.findall("joint")}
    rows=[]
    calib=json.loads(Path(calibration).read_text()) if calibration else {}
    for idx,name in enumerate(JOINTS,1):
        node=joints.get(name)
        item={"axis":f"J{idx}","joint":name,"urdf_present":node is not None}
        if node is not None:
            item["urdf_type"]=node.get("type")
            axis=node.find("axis")
            item["axis_xyz"]=axis.get("xyz") if axis is not None else None
            limit=node.find("limit")
            if limit is not None:
                lo,hi=limit.get("lower"),limit.get("upper")
                item["urdf_limit_deg"]=[math.degrees(float(lo)),math.degrees(float(hi))] if lo is not None and hi is not None else None
            item["parent"]=node.find("parent").get("link") if node.find("parent") is not None else None
            item["child"]=node.find("child").get("link") if node.find("child") is not None else None
        if name in calib:
            c=calib[name]
            item["calibration_fields_available"]=sorted(c.keys())
            item["calibration_range_ticks"]=[c.get("range_min"),c.get("range_max")]
            # Calibration range alone never determines sign or zero offset.
        rows.append(item)
    links=root.findall("link")
    geometry={"links_total":len(links),
              "links_with_collision":sum(bool(link.findall("collision")) for link in links),
              "collision_elements":sum(len(link.findall("collision")) for link in links),
              "collision_geometry_types":{}}
    for link in links:
        for coll in link.findall("collision"):
            geo=coll.find("geometry")
            kind=(next(iter(geo)).tag if geo is not None and len(geo) else "unknown")
            geometry["collision_geometry_types"][kind]=geometry["collision_geometry_types"].get(kind,0)+1
    return {"joint_table":rows,"collision_metadata":geometry,"hypothetical_mapping_signs":{
        "shoulder+_elbow+":{"J2":1,"J3":1},"shoulder-_elbow-":{"J2":-1,"J3":-1}},
        "not_verified":["True motor-to-URDF direction/zero offset needs independent physical evidence",
                        "Neutral-position collision warnings are not independently verified here",
                        "Collision meshes, adjacent-link exclusions, continuous swept-path and torque not checked",
                        "Numerical DP success does not imply hardware-safe trajectory"]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf",type=Path,required=True)
    p.add_argument("--calibration",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Output exists")
    report=audit(a.urdf,a.calibration)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    for r in report["joint_table"]:
        print(r["axis"],r["joint"],"URDF limits (deg):",r.get("urdf_limit_deg"),
              "axis:",r.get("axis_xyz"))
    print("Collision elements:",report["collision_metadata"]["collision_elements"])
    print("True J2/J3 motor sign and offset: NOT VERIFIED; no hardware touched")
    print("JSON:",a.output)
if __name__=="__main__":main()
