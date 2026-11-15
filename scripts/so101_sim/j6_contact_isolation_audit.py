#!/usr/bin/env python3
"""Isolate J6 collision constraints from PD response (MuJoCo-only diagnosis).

No Muhi/table/wrist, no edits to URDF or production grasp controller.
Compare the identical J6 reference with contacts ON versus OFF. The OFF
case is diagnostic ONLY and must never be used as proof of a valid grasp.
"""
import argparse, csv, json, math
from pathlib import Path
import mujoco
import numpy as np
import physical_grasp_j6_bench as bench
import real_to_sim as r2s

def objname(model, kind, idx):
    return mujoco.mj_id2name(model, kind, int(idx)) or str(idx)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): p.error("output exists")
    model=mujoco.MjSpec.from_file(str(a.urdf.resolve(strict=True))).compile()
    model.opt.timestep=.002
    data=mujoco.MjData(model)
    armq=[]; armd=[]
    for name in r2s.JOINTS:
        q,d,_=bench.joint_addr(model,name)
        armq.append(q); armd.append(d)
    jawq,jawd,lims=bench.joint_addr(model,"gripper")
    model.dof_armature[jawd]=max(float(model.dof_armature[jawd]),.002)
    model.dof_damping[jawd]=max(float(model.dof_damping[jawd]),.06)
    initial=np.array(model.qpos0,copy=True)
    initial[armq]=0
    initial[armq[r2s.JOINTS.index("wrist_roll")]]=math.radians(-90)
    initial[jawq]=float(lims[1])
    orig_disable=int(model.opt.disableflags)
    records=[]; contacts=[]; results=[]
    for case,off in [("contact_on",False),("contact_off_DIAGNOSTIC",True)]:
        model.opt.disableflags=orig_disable
        if off:
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mujoco.mj_resetData(model,data)
        data.qpos[:]=initial
        data.qvel[:]=0
        mujoco.mj_forward(model,data)
        case_rows=[]
        for step in range(1500):
            time=float(data.time)
            f=min(1,time/2.2)
            s=f*f*(3-2*f)
            target=float(lims[1])+s*(float(lims[0])-float(lims[1]))
            data.qpos[armq]=initial[armq]
            data.qvel[armd]=0
            mujoco.mj_forward(model,data)
            cmd=float(np.clip(.15*(target-data.qpos[jawq])-.08*data.qvel[jawd],-.015,.015))
            data.qfrc_applied[:]=0
            data.qfrc_applied[jawd]=cmd
            mujoco.mj_step(model,data)
            # Capture contact names and forces before kinematically restoring arm joints.
            if not off and step in (0,80,400,800,1100,1499):
                for j,c in enumerate(data.contact[:data.ncon]):
                    force=np.zeros(6)
                    mujoco.mj_contactForce(model,data,j,force)
                    contacts.append(dict(case=case,time_s=float(data.time),
                        geom1=objname(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),
                        geom2=objname(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),
                        body1=objname(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[c.geom1]),
                        body2=objname(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[c.geom2]),
                        penetration_mm=min(0.,float(c.dist)*1000),
                        normal_force_n=float(force[0]),efc_address=int(c.efc_address)))
            constraint=float(data.qfrc_constraint[jawd])
            data.qpos[armq]=initial[armq]
            data.qvel[armd]=0
            mujoco.mj_forward(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError(f"nonfinite state at {case} step {step}")
            row=dict(case=case,time_s=float(data.time),target_deg=math.degrees(target),
                     actual_deg=math.degrees(float(data.qpos[jawq])),torque_nm=cmd,
                     constraint_nm=constraint,contacts=int(data.ncon))
            case_rows.append(row)
        records.extend(case_rows)
        results.append(dict(case=case,final_actual_deg=case_rows[-1]["actual_deg"],
            final_error_deg=case_rows[-1]["actual_deg"]-case_rows[-1]["target_deg"],
            max_abs_constraint_nm=max(abs(r["constraint_nm"]) for r in case_rows)))
    model.opt.disableflags=orig_disable
    a.out.mkdir(parents=True)
    for name,rs in [("traces.csv",records),("contact_pairs.csv",contacts)]:
        if not rs: continue
        with (a.out/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rs[0].keys());w.writeheader();w.writerows(rs)
    report=dict(status="CONTACT_ISOLATION_DIAGNOSTIC_ONLY",cases=results,
        contact_pairs_logged=len(contacts),note="Contact OFF is diagnosis only, not a safe physical grasp or production setting.")
    (a.out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5))
        for case in ("contact_on","contact_off_DIAGNOSTIC"):
            rows=[r for r in records if r["case"]==case]
            ax.plot([r["time_s"] for r in rows],[r["actual_deg"] for r in rows],label=case)
        ax.plot([r["time_s"] for r in records[:1500]],
                [r["target_deg"] for r in records[:1500]],"--",label="J6 target")
        ax.set(xlabel="Time (s)",ylabel="J6 angle (deg)",
               title="Contact isolation diagnosis (contact OFF is NOT physical validation)")
        ax.legend();ax.grid(alpha=.3);fig.tight_layout()
        fig.savefig(a.out/"j6_contact_on_off.png",dpi=150);plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
