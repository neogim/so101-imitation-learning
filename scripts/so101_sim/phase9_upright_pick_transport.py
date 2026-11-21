#!/usr/bin/env python3
"""Phase 9: SO-101 upright Muhi home -> pregrasp -> grasp -> lift -> box proxy.

Simulation only. Input grasp is the measured Phase8C J4-positive +15mm candidate.
Does NOT teleport the bottle during physics; DOES NOT claim physical placement
into a calibrated real box. The box here is a user-specified visual marker only.
Both robot joint mapping and actual box frame must be calibrated for real use.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

import phase6c_upright_reach_grasp as phase6c
import phase7v_minus35_physical_grasp as p7v
import phase7q_recorded_success_physics as p7q
import phase6g_fixed_workspace_grasp_depth as upright
import physical_grasp_j6_bench as grasp
import physical_workcell_sim as physical


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float)+'\n', encoding='utf-8')


def reference(model, d, qadr, body_id, q):
    d.qpos[qadr[:5]]=q
    mujoco.mj_forward(model,d)
    return np.asarray(d.xpos[body_id]).copy()


def solve_relative_ik(model, qadr, body_id, seed, delta_xyz, *, max_iter=100):
    """Damped least-squares xyz motion, hold J5 at seed value (4-DoF)."""
    d=mujoco.MjData(model)
    mujoco.mj_resetData(model,d)
    q=np.asarray(seed,dtype=float).copy()
    initial=reference(model,d,qadr,body_id,q)
    target=initial+np.asarray(delta_xyz,float)
    jids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
          for n in upright.JOINTS[:4]]
    assert all(j>=0 for j in jids)
    dofs=[int(model.jnt_dofadr[j]) for j in jids]
    limits=np.asarray(model.jnt_range[jids],float)
    for i in range(max_iter):
        pos=reference(model,d,qadr,body_id,q)
        error=target-pos
        if np.linalg.norm(error)<0.00075:break
        jp=np.zeros((3,model.nv));jr=np.zeros((3,model.nv))
        mujoco.mj_jacBody(model,d,jp,jr,body_id)
        jac=jp[:,dofs]
        damping=0.035
        dq=jac.T@np.linalg.solve(jac@jac.T+(damping*damping)*np.eye(3),error)
        q[:4]=np.clip(q[:4]+np.clip(dq,-math.radians(3),math.radians(3)),
                      limits[:,0]+0.001,limits[:,1]-0.001)
    actual=reference(model,d,qadr,body_id,q)
    return q,dict(start_xyz_m=initial.tolist(),target_xyz_m=target.tolist(),
                  achieved_xyz_m=actual.tolist(),error_mm=float(np.linalg.norm(target-actual)*1000),
                  j5_deg=float(np.rad2deg(q[4])),iterations=i+1,
                  q_deg=np.rad2deg(q).tolist())


def selected_pose(path, pose, advance_mm):
    info=json.loads(path.read_text(encoding='utf-8'))
    cases=[r for r in info['results']
           if r['pose']==pose and abs(float(r['forward_mm'])-advance_mm)<0.01]
    if len(cases)!=1 or not cases[0].get('ik',{}).get('q_deg'):
        raise ValueError(f'Phase8C pose not found: {pose} +{advance_mm}mm')
    case=cases[0]
    return np.deg2rad(np.asarray(case['ik']['q_deg'],float)),case


def scene_box_marker(spec, box_xy, plane_z, half_x, half_y, height):
    """Synthetic visible box only; not a measured or collision-enabled scene."""
    bx,by=box_xy
    z=plane_z+height/2
    for name,pos,size in (
        ('phase9_box_left',[bx-half_x,by,z],[0.004,half_y, height/2]),
        ('phase9_box_right',[bx+half_x,by,z],[0.004,half_y,height/2]),
        ('phase9_box_front',[bx,by-half_y,z],[half_x,0.004,height/2]),
        ('phase9_box_back',[bx,by+half_y,z],[half_x,0.004,height/2])):
        spec.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,
             pos=pos,size=size,rgba=[0.20,0.57,0.85,0.42],
             contype=0,conaffinity=0)


def model_with_proxy(args,out):
    options=SimpleNamespace(urdf=args.urdf,out=out,dt=args.dt,
        jaw_collision_mode='axial_cross_convex_patches',jaw_collision_slabs=12,
        jaw_collision_cross_bands=3,support_plane_z_m=args.table_z,
        arm_kp=18.,arm_kd=1.8,arm_torque=3.,j6_kp=2.,j6_kd=.08,
        j6_torque=.4,wrist_camera=args.wrist_camera)
    # The optional proxy is passed to the common scene builder, so the base
    # physics and wrist camera stay consistent with Phase8C.
    options.transport_box_proxy=dict(xy=[args.box_x,args.box_y],
        half_size=[args.box_half_x,args.box_half_y],height=args.box_height)
    return phase6c.model_build(options)


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    qgrasp,base_case=selected_pose(args.phase8c_summary,args.pose,args.forward_mm)
    model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor=model_with_proxy(args,args.out)
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,grasp.FINGER_BODY)
    fj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    assert bid>=0 and fj>=0
    bq=int(model.jnt_qposadr[fj])
    initial_bottle=np.array([args.bottle_x,args.bottle_y,args.table_z+physical.MUHI_LENGTH/2],float)
    probe=mujoco.MjData(model);mujoco.mj_resetData(model,probe)
    finger_grasp=reference(model,probe,qadr,bid,qgrasp)
    offset_xy=initial_bottle[:2]-finger_grasp[:2]
    # Work backward from the demonstrated grasp configuration to a detached
    # pregrasp and initial (home-like) pose; do not assume an unsafe URDF-zero home.
    qpre,pre_ik=solve_relative_ik(model,qadr,bid,qgrasp,[-args.pregrasp_mm/1000,0,0.025])
    qhome,home_ik=solve_relative_ik(model,qadr,bid,qpre,[-args.home_retreat_mm/1000,0,args.home_raise_mm/1000])
    # The nominal 60mm retreat + 80mm rise is unreachable for some Phase8C poses.
    # Search smaller, reachable home-like offsets before rejecting the whole run.
    home_candidates=[(qhome,home_ik,args.home_retreat_mm,args.home_raise_mm)]
    if home_ik['error_mm']>args.ik_max_error_mm:
        for retreat,raise_mm in ((45,60),(30,40),(15,20),(0,0)):
            if retreat==0 and raise_mm==0:
                home_candidates.append((qpre,dict(pre_ik,home_fallback_to_pregrasp=True),0,0))
            else:
                qc,ic=solve_relative_ik(model,qadr,bid,qpre,[-retreat/1000,0,raise_mm/1000])
                home_candidates.append((qc,ic,retreat,raise_mm))
        reachable=[c for c in home_candidates if c[1]['error_mm']<=args.ik_max_error_mm]
        if reachable:
            qhome,home_ik,retreat,raise_mm=reachable[0]
            print(f'HOME_IK_FALLBACK: retreat={retreat}mm, raise={raise_mm}mm, error={home_ik["error_mm"]:.3f}mm',flush=True)
    lift_data=mujoco.MjData(model);mujoco.mj_resetData(model,lift_data)
    lift_data.qpos[qadr[:5]]=qgrasp
    mujoco.mj_forward(model,lift_data)
    qlift,lift_ik=p7v.lift_target(model,lift_data,qadr,bid,np.r_[qgrasp,hi[5]],args.lift_mm)
    pos_lift=reference(model,probe,qadr,bid,qlift)
    # Keep object-to-finger XY displacement constant when calculating final
    # finger target. This offset is a planning assumption, not enforced physics.
    target_finger_xy=np.array([args.box_x,args.box_y])-offset_xy
    target_finger_z=max(pos_lift[2],args.table_z+args.box_height+args.hover_clearance_mm/1000)
    qbox,box_ik=solve_relative_ik(model,qadr,bid,qlift,
        [target_finger_xy[0]-pos_lift[0],target_finger_xy[1]-pos_lift[1],
         target_finger_z-pos_lift[2]])
    plan={'schema':'phase9_upright_pick_transport_v1','source_case':base_case,
          'box_is_provisional_uncalibrated_visual_proxy':True,
          'box_collision_enabled':False,'bottle_world_xyz_initial_m':initial_bottle.tolist(),
          'box_center_xy_m':[args.box_x,args.box_y],
          'box_top_z_m':args.table_z+args.box_height,
          'target_finger_world_xy_m':target_finger_xy.tolist(),
          'grasp_finger_xyz_m':finger_grasp.tolist(),'pregrasp_ik':pre_ik,
          'home_ik':home_ik,'lift_ik':lift_ik,'box_ik':box_ik}
    save_json(args.out/'phase9_plan.json',plan)
    if args.plan_only:
        print('PLAN_ONLY; no MuJoCo mj_step; output:',args.out)
        return
    # Do not claim success if geometrically unreachable. Record full plan.
    if max(pre_ik['error_mm'],home_ik['error_mm'],box_ik['error_mm'])>args.ik_max_error_mm:
        save_json(args.out/'phase9_summary.json',dict(status='IK_UNREACHABLE',
             ik_max_error_mm=args.ik_max_error_mm,stage='planning'))
        print('IK_UNREACHABLE. See phase9_plan.json');return
    d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
    d.qpos[qadr[:5]]=qhome;d.qpos[qadr[5]]=hi[5]
    d.qpos[bq:bq+3]=initial_bottle;d.qpos[bq+3:bq+7]=[1,0,0,0]
    d.qvel[:]=0;d.ctrl[aids]=np.r_[qhome,hi[5]]
    mujoco.mj_forward(model,d)
    stages=[('home',.5,qhome,qhome,hi[5],hi[5]),
            ('pregrasp',1.5,qhome,qpre,hi[5],hi[5]),
            ('approach',1.5,qpre,qgrasp,hi[5],hi[5]),
            ('close',1.3,qgrasp,qgrasp,hi[5],lo[5]),
            ('hold',.5,qgrasp,qgrasp,lo[5],lo[5]),
            ('lift',2.4,qgrasp,qlift,lo[5],lo[5]),
            ('transport',3.0,qlift,qbox,lo[5],lo[5]),
            ('above_box',1.0,qbox,qbox,lo[5],lo[5])]
    total=sum(s[1] for s in stages)
    stride=max(1,int(round(1/(args.dt*args.fps))))
    samples=[];frames=[];elapsed=0.0
    nsteps=int(math.ceil(total/args.dt))
    stage_duration_acc=[sum(row[1] for row in stages[:i+1]) for i in range(len(stages))]
    current_stage='home';stage_entered=[];stage_mark=[];invalid=None;max_pen=0.0
    for k in range(nsteps+1):
        t=min(k*args.dt,total)
        idx=next((i for i,end in enumerate(stage_duration_acc) if t<end),len(stages)-1)
        name,duration,qstart,qend,gstart,gend=stages[idx]
        begin=0.0 if idx==0 else stage_duration_acc[idx-1]
        u=min(1.0,max(0.0,(t-begin)/max(duration,1e-9)))
        # Smooth interpolation avoids acceleration discontinuity at stage ends.
        alpha=u*u*(3-2*u)
        goal=qstart+(qend-qstart)*alpha
        j6_goal=gstart+(gend-gstart)*alpha
        d.ctrl[aids]=np.r_[goal,j6_goal]
        if k:mujoco.mj_step(model,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            invalid='NONFINITE_PHYSICS_STATE';break
        fc,mc,dep=grasp.jaw_contact_counts(model,d,bottle,fixed,moving)
        if dep is not None:max_pen=max(max_pen,max(0.,-float(dep)*1000))
        bpos=np.asarray(d.qpos[bq:bq+3]).copy()
        fingertip=np.asarray(d.xpos[bid]).copy()
        # For each stage track actual position and object-to-gripper relative shift.
        sample={'time_s':float(d.time),'stage':name,
          'object_x_m':float(bpos[0]),'object_y_m':float(bpos[1]),'object_z_m':float(bpos[2]),
          'object_rise_mm':float((bpos[2]-initial_bottle[2])*1000),
          'object_xy_initial_displacement_mm':float(np.linalg.norm(bpos[:2]-initial_bottle[:2])*1000),
          'box_xy_error_mm':float(np.linalg.norm(bpos[:2]-[args.box_x,args.box_y])*1000),
          'finger_x_m':float(fingertip[0]),'finger_y_m':float(fingertip[1]),
          'finger_z_m':float(fingertip[2]),
          'fixed_contacts':int(fc),'moving_contacts':int(mc),
          'bilateral':int(fc>0 and mc>0),
          'penetration_mm':None if dep is None else max(0.,-float(dep)*1000),
          'tracking_error_max_deg':float(np.max(np.abs(np.rad2deg(goal-d.qpos[qadr[:5]]))))}
        samples.append(sample)
        if name!=current_stage or k==0:
            current_stage=name
            stage_entered.append(name)
            stage_mark.append({'stage':name,'at_s':float(d.time),
                  'object_xyz_m':bpos.tolist(),'finger_xyz_m':fingertip.tolist()})
        if k%stride==0 or k==nsteps:
            frames.append((d.qpos.copy(),d.qvel.copy(),float(d.time),name,int(fc),int(mc)))
    with (args.out/'phase9_trace.csv').open('w',newline='',encoding='utf-8') as f:
        if samples:
            w=csv.DictWriter(f,fieldnames=samples[0].keys());w.writeheader();w.writerows(samples)
    mp4_state=p7v.generate_video(model,frames,args.out,args.fps,bq,stages) if frames else 'NO_FRAMES'
    old=args.out/('phase7v_grasp_4views.mp4' if args.wrist_camera else 'phase7v_grasp_3views.mp4')
    video=None
    if old.exists():
        final=args.out/('phase9_pick_transport_4views.mp4' if args.wrist_camera else 'phase9_pick_transport_3views.mp4')
        old.rename(final)
        video=final.name
    last=samples[-1] if samples else {}
    finalstage=[r for r in samples if r['stage']=='above_box']
    liftstage=[r for r in samples if r['stage']=='lift']
    summary={'status':'SIMULATION_COMPLETED' if not invalid else invalid,
        'note':'NOT a validated grasp or real-box pick-and-place: box is uncalibrated visual proxy',
        'source_pose':args.pose,'forward_mm':args.forward_mm,
        'stages_entered':stage_entered,'stage_markers':stage_mark,
        'video':video,'video_status':mp4_state,
        'max_finger_bottle_penetration_mm':max_pen,
        'lift_peak_object_rise_mm':max((r['object_rise_mm'] for r in liftstage),default=None),
        'final_object_rise_mm':last.get('object_rise_mm'),
        'final_box_xy_error_mm':last.get('box_xy_error_mm'),
        'final_stage_bilateral_fraction':sum(r['bilateral'] for r in finalstage)/max(1,len(finalstage)),
        'final_stage_bottle_above_box':bool(finalstage and last.get('box_xy_error_mm',1e10)<=args.box_xy_tolerance_mm
            and last.get('object_z_m',0)>args.table_z+args.box_height),
        'strict_transport_success':bool(not invalid and finalstage and max_pen<=3.0
            and last.get('box_xy_error_mm',1e10)<=args.box_xy_tolerance_mm
            and last.get('object_z_m',0)>args.table_z+args.box_height
            and sum(r['bilateral'] for r in finalstage)/max(1,len(finalstage))>=.8),
        'limitations':['The starting arm posture is a derived collision-unverified offset, not a measured real home.',
            'The planner uses a visual box proxy with no wall contacts; physical box must be measured.',
            'The Phase8C candidate failed strict penetration criterion; transfer success is not presumed.',
            'Only robot finger-to-object and object-to-ground collision are enabled by the baseline builder.']}
    save_json(args.out/'phase9_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','video','lift_peak_object_rise_mm',
        'final_box_xy_error_mm','strict_transport_success')},ensure_ascii=False,indent=2))
    print('RESULT_DIR:',args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase8c-summary',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pose',default='middle_j4_nonnegative',choices=['middle','middle_j4_nonnegative'])
    p.add_argument('--forward-mm',type=float,default=15.)
    p.add_argument('--bottle-x',type=float,default=.22)
    p.add_argument('--bottle-y',type=float,default=0.)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--box-x',type=float,default=.18,help='Provisional visual box marker X, NOT calibrated')
    p.add_argument('--box-y',type=float,default=-.12,help='Provisional visual box marker Y, NOT calibrated')
    p.add_argument('--box-half-x',type=float,default=.05)
    p.add_argument('--box-half-y',type=float,default=.05)
    p.add_argument('--box-height',type=float,default=.12)
    p.add_argument('--box-xy-tolerance-mm',type=float,default=30.)
    p.add_argument('--hover-clearance-mm',type=float,default=45.)
    p.add_argument('--pregrasp-mm',type=float,default=35.)
    p.add_argument('--home-retreat-mm',type=float,default=60.)
    p.add_argument('--home-raise-mm',type=float,default=80.)
    p.add_argument('--lift-mm',type=float,default=50.)
    p.add_argument('--ik-max-error-mm',type=float,default=15.)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--wrist-camera',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if not a.phase8c_summary.is_file() or not a.urdf.is_file():p.error('Missing input JSON / URDF')
    if a.dt<=0 or a.fps<=0 or a.box_height<=0 or a.box_half_x<=0 or a.box_half_y<=0:p.error('Invalid scene parameters')
    run(a)

if __name__=='__main__':main()
