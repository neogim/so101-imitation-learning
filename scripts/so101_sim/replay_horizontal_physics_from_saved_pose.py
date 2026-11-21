#!/usr/bin/env python3
"""Controlled MuJoCo mj_step replay of a saved horizontal grasp candidate.

Reuses the Phase6G upright PD actuator/contact machinery and the *current*
12x3 split-convex jaw collision, but does NOT silently pass failed IK.
The bottle stays in the scene and is a FREE body (unlike J6 air smoke).
The run is explicitly diagnostic, not accepted as a grasp or a lift.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import phase6c_upright_reach_grasp as phase6c
import phase7c_horizontal_from_upright_baseline as phase7
import physical_workcell_sim as physical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--result-json',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--target-x',type=float,default=.220)
    p.add_argument('--target-y',type=float,default=0.)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--yaw-deg',type=float,default=0.)
    p.add_argument('--object-height-clearance-mm',type=float,default=.5)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--max-start-penetration-mm',type=float,default=3.)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    entry=json.loads(args.result_json.read_text(encoding='utf-8'))
    qmap=entry['ik']['ik_angles_deg']
    q=np.deg2rad(np.array([qmap[j] for j in phase7.JOINTS],dtype=float))
    r=Rotation.from_euler('z',args.yaw_deg,degrees=True)*Rotation.from_euler(
        'y',90.,degrees=True)
    xyzw=r.as_quat()
    quat=np.array([xyzw[3],xyzw[0],xyzw[1],xyzw[2]])
    xyz=np.array([args.target_x,args.target_y,args.table_z
                  +physical.MUHI_RADIUS+args.object_height_clearance_mm/1000.])
    setup=SimpleNamespace(
        urdf=args.urdf.resolve(),out=args.out,
        jaw_collision_mode='axial_cross_convex_patches',
        jaw_collision_slabs=12,jaw_collision_cross_bands=3,
        support_plane_z_m=args.table_z,table_z=args.table_z,
        object_quat=quat.tolist(),dt=args.dt,fps=args.fps,
        settle_s=.3,close_s=1.3,hold_s=.6,
        arm_kp=18.,arm_kd=1.8,arm_torque=3.,
        j6_kp=2.,j6_kd=.08,j6_torque=.4)
    (model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor)=phase6c.model_build(setup)
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    bq=int(model.jnt_qposadr[jid]);bd=int(model.jnt_dofadr[jid])
    d=mujoco.MjData(model)
    d.qpos[qadr[:5]]=q
    d.qpos[qadr[5]]=hi[5]
    d.qpos[bq:bq+3]=xyz
    d.qpos[bq+3:bq+7]=quat
    mujoco.mj_forward(model,d)
    fixed_gap=phase7.jaw_gap(model,d,fixed,bottle) if hasattr(phase7,'jaw_gap') else None
    # Explicit fail closed before mj_step, including severe intersecting jaws.
    depths=[]
    for group in (fixed,moving):
        for g in group:
            for b in bottle:
                depth=float(mujoco.mj_geomDistance(model,d,int(g),int(b),.5,np.zeros(6)))
                depths.append(depth)
    start_pen=max(0.,-1000*min(depths))
    meta=dict(
       source_result=str(args.result_json),
       collision_mode=setup.jaw_collision_mode,
       physical_method='MuJoCo mj_step, fixed arm PD, dynamic free Muhi',
       reason='Diagnostic physics replay even if saved pose failed IK acceptance',
       no_ik_reoptimization=True,
       no_lift=True,
       source_geometry_valid=entry.get('ik',{}).get('horizontal_clearance_search',{})
            .get('chosen',{}).get('valid_for_physics'),
       initial_jaw_bottle_penetration_mm=start_pen,
       threshold_mm=args.max_start_penetration_mm,
       jaw_table_physical_contacts=False,
       note='No physical grasp success should be inferred from contact or movement alone')
    if start_pen>args.max_start_penetration_mm:
        meta['status']='SKIPPED_SEVERE_INITIAL_INTERPENETRATION'
    else:
        try:
            metrics=phase7.physics(
                model,qadr,vadr,aids,lo,hi,xyz,bq,bd,
                fixed,moving,bottle,floor,q,setup,args.out)
            meta['status']='PHYSICAL_DIAGNOSTIC_FINISHED'
            meta['physics']=metrics
        except Exception as exc:
            meta['status']='PHYSICAL_DIAGNOSTIC_ERROR'
            meta['error']=repr(exc)
    (args.out/'replay_comparison.json').write_text(
        json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:meta.get(k) for k in (
        'status','initial_jaw_bottle_penetration_mm','source_geometry_valid')},
        ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
