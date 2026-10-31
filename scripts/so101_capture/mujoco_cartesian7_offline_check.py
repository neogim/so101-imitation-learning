#!/usr/bin/env python3
"""Read-only MuJoCo validation of Cartesian7 offline IK candidates.

Never connects to a motor or camera. Loads the URDF directly. MuJoCo contact
data alone does NOT certify safety: imported URDF collision coverage, workcell
geometry, physical calibration, and swept-path dynamics still need validation.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

NAMES=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    p.add_argument("--candidates",type=Path,help="Existing ik_candidates.json; optional")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--show",action="store_true",help="Open MuJoCo viewer (Mac: run with mjpython)")
    p.add_argument("--svg",type=Path,help="Save GUI-free 2D model/trajectory visualization as SVG")
    a=p.parse_args()
    if a.output.exists():p.error("Output already exists")
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(a.urdf.resolve()))
    data=mujoco.MjData(model)
    names={}
    for n in NAMES:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
        if jid<0:raise RuntimeError("Joint missing in MuJoCo: "+n)
        if model.jnt_type[jid]!=mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError("Unexpected joint type: "+n)
        names[n]=int(model.jnt_qposadr[jid])
    def inspect(angles):
        if len(angles)!=5 or not all(math.isfinite(float(x)) for x in angles):
            raise ValueError("Invalid joint angles")
        data.qpos[:]=model.qpos0
        for n,deg in zip(NAMES,angles):
            data.qpos[names[n]]=math.radians(float(deg))
        mujoco.mj_forward(model,data)
        contacts=[]
        for i in range(data.ncon):
            c=data.contact[i]
            g1,g2=int(c.geom1),int(c.geom2)
            b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
            contacts.append({"geom1":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g1) or f"geom#{g1}",
                             "geom2":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g2) or f"geom#{g2}",
                             "body1":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b1) or f"body#{b1}",
                             "body2":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b2) or f"body#{b2}",
                             "distance_m":float(c.dist)})
        limit=[]
        for n in NAMES:
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
            q=float(data.qpos[names[n]])
            if bool(model.jnt_limited[jid]) and not (model.jnt_range[jid,0]<=q<=model.jnt_range[jid,1]):
                limit.append(n)
        pairs={}
        for c in contacts:
            pair=" <> ".join(sorted((c["body1"],c["body2"])))
            entry=pairs.setdefault(pair,{"count":0,"min_distance_m":float("inf")})
            entry["count"]+=1
            entry["min_distance_m"]=min(entry["min_distance_m"],c["distance_m"])
        return {"joints_deg":list(angles),"outside_mujoco_joint_limits":limit,
                "contacts_detected":len(contacts),"contact_pairs":pairs,"contacts":contacts[:20],
                "body_positions_xyz":[[float(x) for x in data.xpos[i]] for i in range(model.nbody)]}
    neutral=inspect([0]*5)
    rows=[]
    if a.candidates:
        src=json.loads(a.candidates.read_text())
        if not isinstance(src,list):raise ValueError("Expected list of candidates")
        for idx,r in enumerate(src):
            if not isinstance(r,dict) or "candidate_motor_deg" not in r:
                raise ValueError(f"Missing candidate_motor_deg at {idx}")
            rec=inspect(r["candidate_motor_deg"])
            rec["source_index"]=idx
            rows.append(rec)
    result={"status":"SIMULATION_DIAGNOSTIC_NOT_HARDWARE_APPROVAL",
            "urdf":str(a.urdf),"mujoco_version":mujoco.__version__,
            "ngeom":int(model.ngeom),"nconmax":int(model.nconmax),
            "neutral_pose":neutral,
            "candidates_count":len(rows),
            "limit_violation_candidates":sum(bool(r["outside_mujoco_joint_limits"]) for r in rows),
            "contact_candidates":sum(bool(r["contacts_detected"]) for r in rows),
            "candidate_details":rows,
            "UNVERIFIED":["LeRobot-to-URDF mapping","collision geom completeness",
                          "desk, Muhi and cup absent from URDF",
                          "swept/interpolated trajectories","physical calibration"],
            "NO_GO":"Never pass these outputs directly to motor control."}
    if a.svg:
        # No OpenGL/GUI dependency. Two orthographic panels, schematic body centers only.
        from html import escape
        poses=[neutral]+rows
        panels=[]
        for axis1,axis2,label in ((0,2,"X-Z"),(1,2,"Y-Z")):
            allpts=[(q["body_positions_xyz"][b][axis1],q["body_positions_xyz"][b][axis2]) for q in poses for b in range(1,model.nbody)]
            lo1=min(x for x,y in allpts);hi1=max(x for x,y in allpts)
            lo2=min(y for x,y in allpts);hi2=max(y for x,y in allpts)
            span=max(hi1-lo1,hi2-lo2,.01)*1.25
            cx=(lo1+hi1)/2;cy=(lo2+hi2)/2
            offx=0 if axis1==0 else 540
            def xy(p):return (offx+270+450*(p[axis1]-cx)/span,285-450*(p[axis2]-cy)/span)
            segments=[]
            for qi,q in enumerate(poses):
                pts=q["body_positions_xyz"]
                for b in range(1,model.nbody):
                    parent=int(model.body_parentid[b]);
                    if parent<=0:continue
                    x1,y1=xy(pts[parent]);x2,y2=xy(pts[b]);
                    color="#F29E4C" if qi==0 else "#5A9ACB"
                    opacity="0.9" if qi==0 else "0.055"
                    segments.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2" opacity="{opacity}"/>')
            panels.append(f'<text x="{offx+25}" y="32" fill="#222" font-size="21">{escape(label)} projection</text>'+"".join(segments))
        svg='<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="570" viewBox="0 0 1080 570"><rect width="1080" height="570" fill="#fff"/>'+"".join(panels)+'<text x="25" y="550" fill="#444" font-size="15">Orange: neutral | Blue: 100 IK candidates (body-center skeleton, NOT collision geometry) | No physical calibration</text></svg>'
        a.svg.parent.mkdir(parents=True,exist_ok=True)
        a.svg.write_text(svg)
        print("Saved SVG:",a.svg)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    small={k:v for k,v in result.items() if k not in ("candidate_details","neutral_pose")}
    small["neutral_contact_pairs"]=neutral["contact_pairs"]
    small["candidate_contact_pairs_first"]=rows[0]["contact_pairs"] if rows else {}
    print(json.dumps(small,ensure_ascii=False,indent=2))
    print("Saved:",a.output)
    if a.show:
        import mujoco.viewer
        if rows:inspect(rows[0]["joints_deg"])
        else:inspect([0]*5)
        mujoco.viewer.launch(model,data)

if __name__=="__main__":
    main()
