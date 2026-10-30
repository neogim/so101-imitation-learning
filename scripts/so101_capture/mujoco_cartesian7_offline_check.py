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
            contacts.append({"geom1":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom1)),
                             "geom2":mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(c.geom2)),
                             "distance_m":float(c.dist)})
        limit=[]
        for n in NAMES:
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
            q=float(data.qpos[names[n]])
            if bool(model.jnt_limited[jid]) and not (model.jnt_range[jid,0]<=q<=model.jnt_range[jid,1]):
                limit.append(n)
        return {"joints_deg":list(angles),"outside_mujoco_joint_limits":limit,
                "contacts_detected":len(contacts),"contacts":contacts[:20]}
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
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="candidate_details"},ensure_ascii=False,indent=2))
    print("Saved:",a.output)
    if a.show:
        import mujoco.viewer
        if rows:inspect(rows[0]["joints_deg"])
        else:inspect([0]*5)
        mujoco.viewer.launch(model,data)

if __name__=="__main__":
    main()
