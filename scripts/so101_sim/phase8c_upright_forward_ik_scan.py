#!/usr/bin/env python3
"""Phase8C: move the SO-101 fixed jaw FORWARD (+world X), not the upright bottle.

Uses the recorded Phase8B pair as starting joint postures. J5 remains fixed;
DLS on J1-J4 changes the finger body's 3D origin (x shift only). Then execute
J6 closing and a physical 50 mm lift, compare measured bottle displacement.
Simulation-only; not calibrated to hardware and not proof of real-world success.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import phase6c_upright_reach_grasp as phase6c
import phase6g_fixed_workspace_grasp_depth as upright
import physical_grasp_j6_bench as grasp
import physical_workcell_sim as physical

CASES=('middle','middle_j4_nonnegative')

def dump(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=float)+'\n')

def forward_ik(model, qadr, qseed, ref_body, advance_mm):
    """Keep gripper roll fixed and move arm-side reference in world +X."""
    d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
    d.qpos[qadr[:5]]=qseed
    mujoco.mj_forward(model,d)
    start=np.array(d.xpos[ref_body],dtype=float)
    target=start+np.array([advance_mm/1000.,0.,0.])
    q=qseed.copy()
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in upright.JOINTS]
    dofs=[int(model.jnt_dofadr[j]) for j in joints[:4]]
    limits=np.asarray(model.jnt_range[joints[:4]],dtype=float)
    for iteration in range(70):
        d.qpos[qadr[:5]]=q
        mujoco.mj_forward(model,d)
        error=target-d.xpos[ref_body]
        if np.linalg.norm(error)<.00025:break
        jp=np.zeros((3,model.nv));jr=np.zeros((3,model.nv))
        mujoco.mj_jacBody(model,d,jp,jr,ref_body)
        J=jp[:,dofs]
        damping=.025
        delta=J.T@np.linalg.solve(J@J.T+damping*damping*np.eye(3),error)
        delta=np.clip(delta,-math.radians(3),math.radians(3))
        q[:4]=np.clip(q[:4]+delta,limits[:,0]+.001,limits[:,1]-.001)
    d.qpos[qadr[:5]]=q;mujoco.mj_forward(model,d)
    achieved=np.asarray(d.xpos[ref_body])-start
    return q,dict(requested_forward_mm=advance_mm,world_forward_axis='+X',
        achieved_xyz_mm=(achieved*1000).tolist(),
        error_mm=float(np.linalg.norm(target-d.xpos[ref_body])*1000),
        j5_preserved_deg=float(np.rad2deg(q[4])),
        q_deg=np.rad2deg(q).tolist())

def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    input_bottle_xyz=np.array([args.target_x,args.target_y,args.table_z+physical.MUHI_LENGTH/2],float)
    results=[]
    for name in CASES:
        input_result=args.baseline/name/'result.json'
        prior=json.loads(input_result.read_text())
        old_pose=prior.get('ik',{}).get('ik_angles_deg',{})
        if any(j not in old_pose for j in upright.JOINTS):
            raise ValueError(f'No 5-joint reference in {input_result}: {old_pose}')
        q0=np.deg2rad(np.array([old_pose[j] for j in upright.JOINTS],float))
        for advance_mm in args.forward_mm:
            folder=args.out/name/f'forward_{advance_mm:+05.1f}mm'
            folder.mkdir(parents=True,exist_ok=False)
            options=SimpleNamespace(urdf=args.urdf,out=folder,dt=args.dt,
                jaw_collision_mode='axial_cross_convex_patches',jaw_collision_slabs=12,
                jaw_collision_cross_bands=3,support_plane_z_m=args.table_z,
                arm_kp=18.,arm_kd=1.8,arm_torque=3.,j6_kp=2.,j6_kd=.08,j6_torque=.4,
                wrist_camera=args.wrist_camera,settle_s=.4,close_s=1.3,hold_s=1.,
                lift_mm=args.lift_mm,lift_s=2.4,retain_s=.9,fps=args.fps)
            record={'pose':name,'forward_mm':advance_mm,'status':'STARTED','output':str(folder)}
            try:
                model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor=phase6c.model_build(options)
                ref=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,grasp.FINGER_BODY)
                fj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
                bq=int(model.jnt_qposadr[fj]);bd=int(model.jnt_dofadr[fj])
                q,ik=forward_ik(model,qadr,q0,ref,advance_mm)
                record['ik']=ik
                # Bottle remains fixed at the original input XYZ BEFORE simulation.
                # It is fully dynamic and may move naturally DURING mj_step.
                phys=upright.physics(model,qadr,vadr,aids,lo,hi,input_bottle_xyz,
                      bq,bd,fixed,moving,bottle,floor,q,options,folder)
                record.update(status=phys['status'],lift_status=phys.get('lift_status'),
                    lift_success=phys.get('lift_success'),
                    rise_mm=phys.get('lift_final_bottle_rise_mm'),
                    xy_drift_mm=phys.get('lift_final_xy_drift_mm'),
                    bilateral_fraction=phys.get('retain_bilateral_fraction'),
                    deep_penetration=phys.get('deep_penetration_over_3mm'),
                    video=str(folder/'grasp.mp4') if (folder/'grasp.mp4').exists() else None)
            except Exception as exc:
                record.update(status='ERROR',error=repr(exc))
            dump(folder/'phase8c_case.json',record)
            results.append(record)
            dump(args.out/'phase8c_summary.json',{'baseline':str(args.baseline),
                'forward_axis':'world +X; target object remains fixed',
                'target_bottle_xyz_m':input_bottle_xyz.tolist(),
                'results':results})
            print(f"{name} forward={advance_mm:+g}mm {record['status']} "
                  f"lift={record.get('lift_status')} rise={record.get('rise_mm')}mm "
                  f"drift={record.get('xy_drift_mm')}mm",flush=True)
    print('RESULT_DIR:',args.out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True,type=Path,help='Phase8B upright directory containing both result.json cases')
    p.add_argument('--urdf',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--forward-mm',nargs='+',type=float,default=[0.,5.,10.,15.])
    p.add_argument('--target-x',type=float,default=.22)
    p.add_argument('--target-y',type=float,default=0.)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--lift-mm',type=float,default=50.)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--wrist-camera',action='store_true')
    a=p.parse_args()
    if not a.baseline.is_dir():p.error(f'Baseline not found: {a.baseline}')
    if a.dt<=0 or a.fps<=0 or any(not math.isfinite(n) or n<0 or n>30 for n in a.forward_mm):p.error('Invalid scan configuration')
    run(a)
if __name__=='__main__':main()
