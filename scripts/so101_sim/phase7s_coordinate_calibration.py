#!/usr/bin/env python3
"""SO-101 Phase7S: explicit motor/URDF and tabletop calibration, never inferred from penetration alone.

No real-hardware control, mj_step, or modification to source URDF. Measured anchors
are required to promote motor-to-URDF mapping to calibrated. Real tabletop height
in URDF base frame is required independently. Otherwise the output is diagnostic.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
import phase7q_recorded_success_physics as phase7q
import phase6c_upright_reach_grasp as phase6c

JOINTS=['J1 shoulder_pan','J2 shoulder_lift','J3 elbow_flex','J4 wrist_flex','J5 wrist_roll']

def fit_anchors(path:Path):
    """File: {anchors:[{motor_deg:[5], urdf_deg:[5]},...]}, at least two separated poses.

    URDF values must be externally grounded in joint conventions, not fitted
    from the same simulation jaw/table collision! Each joint: q=s*m+offset.
    """
    data=json.loads(path.read_text())
    anchors=data.get('anchors',[])
    if len(anchors)<2:raise ValueError('Need >=2 independently measured calibration poses')
    motor=np.asarray([r['motor_deg'] for r in anchors],float)
    target=np.asarray([r['urdf_deg'] for r in anchors],float)
    if motor.shape!=(len(anchors),5) or target.shape!=motor.shape or not np.isfinite(motor).all() or not np.isfinite(target).all():
        raise ValueError('Each anchor must have finite motor_deg and urdf_deg arrays of five values')
    signs=[];offsets=[];rows=[]
    for j,name in enumerate(JOINTS):
        if np.ptp(motor[:,j])<5:raise ValueError(f'{name}: anchor motor excursion <5 degrees; add a separated reference pose')
        trials=[]
        for sign in (-1,1):
            offset=float(np.mean(target[:,j]-sign*motor[:,j]))
            err=sign*motor[:,j]+offset-target[:,j]
            trials.append((float(np.sqrt(np.mean(err*err))),sign,offset,err))
        rms,sign,offset,err=min(trials,key=lambda r:r[0])
        signs.append(int(sign));offsets.append(offset)
        rows.append(dict(joint=name,sign=int(sign),offset_deg=offset,rms_deg=rms,
                         max_abs_error_deg=float(np.max(np.abs(err))),motor_span_deg=float(np.ptp(motor[:,j]))))
    return dict(sign=signs,offset_deg=offsets,anchors=len(anchors),fit_details=rows,
                source=str(path),verified=False,
                note='Fitted to supplied independent anchors; verify anchor origin and camera/physical setup before claiming independent calibration')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--table-z-mm',type=float,default=None,help='Measured tabletop height in URDF base/world coordinates (mm)')
    p.add_argument('--anchors-json',type=Path,default=None,help='Independent paired motor-deg and URDF-deg anchor poses')
    p.add_argument('--time',type=float,default=14.8)
    p.add_argument('--j6-closed-deg',type=float,default=-10.)
    p.add_argument('--j6-open-deg',type=float,default=100.)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    result={'episode_time_s':args.time,'physics_integrated':False,'table_height_measured':args.table_z_mm is not None,
            'joint_mapping_anchors_used':args.anchors_json is not None,
            'table_z_mm':args.table_z_mm,'status':'STARTED',
            'caveats':['URDF base/world origin must be physically located; a ruler value from a different frame is not valid',
                       'J6 normalized mapping not independently calibrated',
                       'MUHI world pose and camera extrinsics unmeasured']}
    try:
        times,states,_=phase7q.read_recorded_csv(args.csv)
        mapping=fit_anchors(args.anchors_json) if args.anchors_json else json.loads(args.mapping.read_text())
        qstate=phase7q.joint_mapping(states,mapping,args.j6_closed_deg,args.j6_open_deg)
        q0=np.array([np.interp(args.time,times,qstate[:,j]) for j in range(6)])
        # Diagnostic model with candidate tabletop z=0 if unmeasured; does not quietly calibrate it.
        table_z=0. if args.table_z_mm is None else args.table_z_mm/1000.
        spec=SimpleNamespace(urdf=args.urdf,out=args.out,dt=.002,jaw_collision_mode='axial_cross_convex_patches',
                             jaw_collision_slabs=12,jaw_collision_cross_bands=3,jaw_table_contact=True,
                             support_plane_z_m=table_z,arm_kp=18.,arm_kd=1.8,arm_torque=3.,
                             j6_kp=2.,j6_kd=.08,j6_torque=.4)
        model,qadr,_,_,_,_,fixed,moving,_,floor=phase6c.model_build(spec)
        d=mujoco.MjData(model);mujoco.mj_resetData(model,d);d.qpos[qadr]=q0;mujoco.mj_forward(model,d)
        fromto=np.zeros(6)
        def worst(ids):
            hits=[]
            for gid in ids:
                dist=float(mujoco.mj_geomDistance(model,d,int(gid),int(floor),.3,fromto))
                hits.append(dict(geom_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(gid)),
                                 signed_gap_mm=dist*1000))
            return min(hits,key=lambda r:r['signed_gap_mm'])
        fixed_hit,moving_hit=worst(fixed),worst(moving)
        upper=min(fixed_hit['signed_gap_mm'],moving_hit['signed_gap_mm'])
        # A shift of the support plane changes the gap by the opposite amount.
        # At most table_z + min_gap (minus desired positive clearance) avoids penetration.
        result.update(fixed=fixed_hit,moving=moving_hit,urdf_joint_deg=np.rad2deg(q0).tolist(),
                      current_assumed_table_z_mm=table_z*1000,
                      required_table_z_upper_bound_for_nonpenetration_mm=table_z*1000+upper,
                      minimum_geometry_clearance_mm=upper,
                      mapping=mapping,
                      note='The computed nonpenetrating table upper bound is a simulation sensitivity result, NOT a physical tabletop measurement')
        if args.table_z_mm is None:
            result['status']='NEEDS_MEASURED_TABLE_Z'
        elif args.anchors_json is None:
            result['status']='NEEDS_JOINT_CALIBRATION_ANCHORS'
        elif upper < -.5:
            result['status']='CALIBRATED_INPUTS_STILL_INTERPENETRATE'
        else:
            result['status']='CALIBRATION_INPUTS_PASSED_INITIAL_TABLE_GEOMETRY_ONLY'
        if args.anchors_json is not None:
            (args.out/'fitted_mapping_provisional.json').write_text(json.dumps(mapping,indent=2))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4.8),facecolor='white')
        values=[fixed_hit['signed_gap_mm'],moving_hit['signed_gap_mm']]
        bars=ax.bar(['Fixed jaw','Moving jaw'],values,color=['#dc2626' if x<0 else '#2563eb' for x in values])
        ax.axhline(0,color='#111827',lw=2,label='Assumed/measured tabletop')
        ax.axhline(-.5,color='#d97706',ls='--',label='Penetration tolerance (-0.5 mm)')
        for bar,v in zip(bars,values):ax.annotate(f'{v:+.2f} mm',(bar.get_x()+bar.get_width()/2,v),
            xytext=(0,8 if v>=0 else -17),textcoords='offset points',ha='center',color='#111827')
        ax.set_ylabel('Signed gap relative to tabletop (mm)')
        ax.set_title('SO-101 recorded Episode015 initial geometry - table reference sensitivity')
        ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(args.out/'joint_table_initial_check.png',dpi=165)
        plt.close(fig)
    except Exception as exc:
        result.update(status='ERROR',error=repr(exc))
        raise
    finally:
        (args.out/'phase7s_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:result.get(k) for k in ('status','minimum_geometry_clearance_mm','required_table_z_upper_bound_for_nonpenetration_mm','error')}))
if __name__=='__main__':main()
