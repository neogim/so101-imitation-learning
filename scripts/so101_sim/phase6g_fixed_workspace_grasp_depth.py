#!/usr/bin/env python3
"""Phase 6G: same tabletop Muhi, root/middle/tip jaw-depth comparisons.

Simulation-only. Fixed target pose; IK moves the *arm*, never the target per
profile. All tests start from independent model/data and close J6 in dynamics.
Candidate depths are provisional gripper-frame positions requiring mesh review.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import shutil
import sys
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

import physical_grasp_j6_bench as grasp
import physical_workcell_sim as physical
import phase6c_upright_reach_grasp as phase6c

JOINTS = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')
# Provisional local Z-depth positions, not measured finger-arc percentages.
DEPTHS = (('root',-.045),('middle',-.070),('tip',-.095))
REFERENCE_LOCAL_XY=(.030,.036)  # fixed-arm bench geometry; lateral refinement is deliberately NOT automatic


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=float)+'\n',encoding='utf-8')


def fk(model, data, qadr, bid, q):
    data.qpos[qadr[:5]] = q
    mujoco.mj_forward(model,data)
    return np.asarray(data.xpos[bid]).copy(), np.asarray(data.xmat[bid]).reshape(3,3).copy()


def solve_ik(model,data,qadr,bid,lo,hi,xyz,anchor,bench):
    bp,br = fk(model,data,qadr,bid,bench)
    desired_origin = xyz-br@anchor
    seeds=[bench.copy()]
    for delta in (-.65,.65):
        s=bench.copy();s[1]=np.clip(s[1]+delta,lo[1]+.0001,hi[1]-.0001)
        s[2]=np.clip(s[2]-delta,lo[2]+.0001,hi[2]-.0001);seeds.append(s)
    for j4 in (-.4,.4):
        s=bench.copy();s[3]=np.clip(j4,lo[3]+.0001,hi[3]-.0001);seeds.append(s)
    solutions=[]
    for seed in seeds:
        def residual(q):
            p,r=fk(model,data,qadr,bid,q)
            err_rot=Rotation.from_matrix(br@r.T).as_rotvec()
            return np.r_[(p-desired_origin)/.008,err_rot/.4,
                         (q-bench)*.12, (q[3]-bench[3])*.25]
        try:
            sol=least_squares(residual, np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=180)
        except ValueError:
            continue
        pos,rot=fk(model,data,qadr,bid,sol.x)
        anchor_xyz=pos+rot@anchor
        pos_err=float(np.linalg.norm(anchor_xyz-xyz)*1000)
        rot_err=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(br@rot.T).as_rotvec())))
        solutions.append((pos_err + .15*rot_err,sol.x.copy(),{
            'solver_success':bool(sol.success),'nfev':int(sol.nfev),
            'anchor_error_mm':pos_err,'orientation_error_deg':rot_err,
            'grasp_reference_world_xyz_m':anchor_xyz.tolist(),
            'ik_angles_deg':dict(zip(JOINTS,np.rad2deg(sol.x).tolist()))
        }))
    if not solutions:raise RuntimeError('No IK solutions')
    solutions.sort(key=lambda t:t[0])
    return solutions[0][1],solutions[0][2]


def jaw_geometry(model,data,qadr,q,xyz,bq,fixed,moving,bottle,lo,hi):
    data.qpos[qadr[:5]]=q
    data.qpos[bq:bq+3]=xyz; data.qpos[bq+3:bq+7]=[1,0,0,0]
    data.qpos[qadr[5]]=hi[5];mujoco.mj_forward(model,data)
    gf=grasp.geometry_gaps(model,data,bottle,fixed)
    gm=grasp.geometry_gaps(model,data,bottle,moving)
    sweep=[]
    for angle in np.linspace(hi[5],lo[5],17):
        data.qpos[qadr[5]]=angle;mujoco.mj_forward(model,data)
        sweep.append(grasp.geometry_gaps(model,data,bottle,moving))
    return {'open_fixed_gap_mm':gf*1000,'open_moving_gap_mm':gm*1000,
            'closing_moving_min_gap_mm':min(sweep)*1000,
            'open_jaws_overlap':bool(min(gf,gm)<-0.0005),
            'closing_sweeps_bottle':bool(min(sweep)<-0.00035)}


def physics(model,qadr,vadr,aids,lo,hi,xyz,bq,bd,fixed,moving,bottle,floor,q,args,out):
    d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
    d.qpos[qadr[:5]]=q;d.qpos[qadr[5]]=hi[5]
    d.qpos[bq:bq+3]=xyz;d.qpos[bq+3:bq+7]=[1,0,0,0]
    d.qvel[:]=0
    d.ctrl[aids]=np.r_[q,hi[5]]
    mujoco.mj_forward(model,d)
    support_start=float(xyz[2]-physical.MUHI_LENGTH/2)
    phases=[('settle',args.settle_s),('close',args.close_s),('hold',args.hold_s)]
    total=sum(t for _,t in phases);steps=int(math.ceil(total/args.dt))
    stride=max(1,int(round(1/(args.dt*args.fps))))
    rows=[];frames=[]
    renderer=None
    if shutil.which('ffmpeg'):
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,640)
        model.vis.global_.offheight=max(model.vis.global_.offheight,460)
        try:renderer=mujoco.Renderer(model,width=640,height=460)
        except Exception as e:(out/'render_error.txt').write_text(repr(e))
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=xyz;camera.distance=.56;camera.azimuth=110;camera.elevation=-20
    deep=False;bad=False;maxcontact=0
    try:
        for k in range(steps+1):
            t=min(k*args.dt,total)
            if t<args.settle_s:phase='settle';j6=hi[5]
            elif t<args.settle_s+args.close_s:
                phase='close';u=(t-args.settle_s)/args.close_s
                j6=hi[5]+(lo[5]-hi[5])*u
            else:phase='hold';j6=lo[5]
            ctrl=np.r_[q,j6];d.ctrl[aids]=ctrl
            if k:mujoco.mj_step(model,d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():bad=True;break
            fc,mc,depth=grasp.jaw_contact_counts(model,d,bottle,fixed,moving)
            bilateral=int(fc>0 and mc>0)
            if depth is not None and depth<-0.003:deep=True
            fnew=[];mnew=[]
            for ix,c in enumerate(d.contact[:d.ncon]):
                geomids={int(c.geom1),int(c.geom2)}
                if not geomids.intersection(bottle):continue
                role='fixed' if geomids.intersection(fixed) else ('moving' if geomids.intersection(moving) else None)
                if role is None:continue
                outforce=np.zeros(6)
                mujoco.mj_contactForce(model,d,ix,outforce)
                # Contact normal is the first axis of MuJoCo contact frame.
                normal=np.asarray(c.frame).reshape(3,3)[0].copy()
                entry={'role':role,'point_xyz_m':np.asarray(c.pos).tolist(),
                       'normal_mj_frame_world':normal.tolist(),
                       'normal_force_n':float(outforce[0]),
                       'tangent_force_n':float(np.linalg.norm(outforce[1:3]))}
                (fnew if role=='fixed' else mnew).append(entry)
            maxcontact=max(maxcontact,len(fnew)+len(mnew))
            def metric(entries,key,mode='sum'):
                v=[a[key] for a in entries]
                return float(sum(v)) if mode=='sum' else (v[0] if v else None)
            def avgpt(entries):
                return np.mean([a['point_xyz_m'] for a in entries],axis=0).tolist() if entries else None
            row={'time_s':float(d.time),'phase':phase,'fixed_contacts':fc,'moving_contacts':mc,
                 'bilateral':bilateral,'min_contact_dist_mm':None if depth is None else 1000*depth,
                 'fixed_normal_force_n':metric(fnew,'normal_force_n'),
                 'moving_normal_force_n':metric(mnew,'normal_force_n'),
                 'fixed_tangent_force_n':metric(fnew,'tangent_force_n'),
                 'moving_tangent_force_n':metric(mnew,'tangent_force_n'),
                 'fixed_contact_point_xyz_m':avgpt(fnew),
                 'moving_contact_point_xyz_m':avgpt(mnew),
                 'fixed_normal_mj_frame_world':fnew[0]['normal_mj_frame_world'] if fnew else None,
                 'moving_normal_mj_frame_world':mnew[0]['normal_mj_frame_world'] if mnew else None,
                 'bottle_x_m':float(d.qpos[bq]),'bottle_y_m':float(d.qpos[bq+1]),
                 'bottle_z_m':float(d.qpos[bq+2]),
                 'bottle_xy_displacement_mm':float(1000*np.linalg.norm(d.qpos[bq:bq+2]-xyz[:2])),
                 'bottle_tilt_deg':float(np.rad2deg(2*math.acos(min(1,abs(float(d.qpos[bq+3])))))),
                 'j6_target_deg':float(np.rad2deg(j6)),'j6_actual_deg':float(np.rad2deg(d.qpos[qadr[5]])),
                 'max_arm_error_deg':float(np.max(np.abs(np.rad2deg(q-d.qpos[qadr[:5]]))))}
            rows.append(row)
            if renderer is not None and (k%stride==0 or k==steps):
                renderer.update_scene(d,camera=camera)
                im=Image.fromarray(renderer.render().copy())
                canvas=Image.new('RGB',(640,496),'#192532');canvas.paste(im,(0,30))
                ImageDraw.Draw(canvas).text((10,9),f'Phase6G {out.name} | {phase} | {t:.2f}s | Jaws F{fc}/M{mc}',fill='white')
                frames.append(canvas)
    finally:
        if renderer:renderer.close()
    with (out/'joint_contact_trace.csv').open('w',newline='',encoding='utf-8') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    if frames and shutil.which('ffmpeg'):
        fd=out/'frames';fd.mkdir()
        for i,im in enumerate(frames):im.save(fd/f'{i:05d}.png')
        p=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(args.fps),'-i',str(fd/'%05d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(out/'grasp.mp4')],capture_output=True,text=True)
        if p.returncode:(out/'ffmpeg_error.txt').write_text(p.stderr[-2000:])
        shutil.rmtree(fd)
    holdrows=[r for r in rows if r['phase']=='hold']
    bilateral=sum(r['bilateral'] for r in holdrows)
    initial_bottom=support_start
    summary={'status':'NUMERIC_FAILURE' if bad else 'DYNAMIC_TEST_COMPLETED',
       'bilateral_hold_steps':bilateral,'hold_steps':len(holdrows),
       'bilateral_hold_fraction':bilateral/max(1,len(holdrows)),
       'max_contact_count':maxcontact,'deep_penetration_over_3mm':deep,
       'max_xy_displacement_mm':max((r['bottle_xy_displacement_mm'] for r in rows),default=None),
       'final_xy_displacement_mm':rows[-1]['bottle_xy_displacement_mm'] if rows else None,
       'final_bottle_z_m':rows[-1]['bottle_z_m'] if rows else None,
       'support_plane_z_m':initial_bottom,
       'maximum_arm_tracking_error_deg':max((r['max_arm_error_deg'] for r in rows),default=None),
       'max_normal_contact_force_n':max((r['fixed_normal_force_n']+r['moving_normal_force_n'] for r in rows),default=None),
       'video':'grasp.mp4' if (out/'grasp.mp4').exists() else None,
       'note':'Bilateral contact is NOT stable lift. World normal sign depends on contact geom ordering.'}
    dump(out/'physical_summary.json',summary)
    return summary


def run(args):
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    xyz=np.array([args.target_x,args.target_y,args.table_z+physical.MUHI_LENGTH/2],dtype=float)
    if not np.isfinite(xyz).all():raise ValueError('Non-finite target')
    args.support_plane_z_m=float(args.table_z)
    model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor=phase6c.model_build(args)
    bj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,grasp.FINGER_BODY)
    fj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    bq=int(model.jnt_qposadr[fj]);bd=int(model.jnt_dofadr[fj])
    bench=np.zeros(5);bench[4]=np.clip(-math.pi/2,lo[4],hi[4])
    results=[]
    for i,(name,local_z) in enumerate(DEPTHS,1):
        print(f'[{i}/3] {name}: same object xyz={np.round(xyz,4).tolist()}',flush=True)
        sub=args.out/name;sub.mkdir()
        anchor=np.array([REFERENCE_LOCAL_XY[0],REFERENCE_LOCAL_XY[1],local_z])
        data=mujoco.MjData(model);mujoco.mj_resetData(model,data)
        try:
            q,ik=solve_ik(model,data,qadr,bj,lo[:5],hi[:5],xyz,anchor,bench)
            geo=jaw_geometry(model,data,qadr,q,xyz,bq,fixed,moving,bottle,lo,hi)
            phys=physics(model,qadr,vadr,aids,lo,hi,xyz,bq,bd,fixed,moving,bottle,floor,q,args,sub) if ik['anchor_error_mm']<=args.max_ik_error_mm and not geo['open_jaws_overlap'] else {'status':'GEOMETRY_DIAGNOSTIC_ONLY','reason':'IK error or initial open-jaw overlap','bilateral_hold_steps':0}
        except Exception as exc:
            ik={};geo={};phys={'status':'EXECUTION_ERROR','reason':repr(exc),'bilateral_hold_steps':0}
        info={'name':name,'depth_local_z_m':local_z,'anchor_local_m':anchor.tolist(),
              'ik':ik,'jaw_geometry':geo,'physics':phys}
        dump(sub/'result.json',info)
        row={'depth':name,'local_z_mm':1000*local_z,
             'ik_error_mm':ik.get('anchor_error_mm'),'open_fixed_gap_mm':geo.get('open_fixed_gap_mm'),
             'open_moving_gap_mm':geo.get('open_moving_gap_mm'),
             'sweep_moving_min_gap_mm':geo.get('closing_moving_min_gap_mm'),
             'status':phys.get('status'),'bilateral_hold_steps':phys.get('bilateral_hold_steps'),
             'max_xy_displacement_mm':phys.get('max_xy_displacement_mm'),
             'max_normal_contact_force_n':phys.get('max_normal_contact_force_n'),
             'video':name+'/grasp.mp4' if (sub/'grasp.mp4').exists() else None}
        results.append((row,info))
        print(f" {name}: IK={row['ik_error_mm']} mm; status={row['status']}; bilateral={row['bilateral_hold_steps']}",flush=True)
    rows=[r for r,_ in results]
    with (args.out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
        labels=[r['depth'] for r in rows]
        ik_errors=[r['ik_error_mm'] if r['ik_error_mm'] is not None else float('nan') for r in rows]
        axes[0].bar(labels,ik_errors,color='#4c78a8')
        axes[0].axhline(args.max_ik_error_mm,color='#c44e52',linestyle='--',label=f'IK threshold: {args.max_ik_error_mm:g} mm')
        axes[0].set_ylabel('IK grasp-anchor position error (mm)')
        axes[0].set_title('Phase 6G geometric screening: physical runs are conditional')
        axes[0].legend(loc='upper left')
        axes[0].set_ylim(bottom=0)
        for x,(v,row) in enumerate(zip(ik_errors,rows)):
            label='No IK solution' if not np.isfinite(v) else f'{v:.1f} mm'
            axes[0].annotate(label,(x,0 if not np.isfinite(v) else v),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
        status_text=[]
        for row in rows:
            status=row['status'] or 'NO_RESULT'
            if status=='GEOMETRY_DIAGNOSTIC_ONLY':
                status_text.append('SKIPPED (IK / jaw geometry)')
            elif status=='DYNAMIC_TEST_COMPLETED':
                status_text.append('PHYSICS RUN (not grasp success)')
            else:
                status_text.append(status)
        axes[1].axis('off')
        axes[1].set_title('Execution status and physical measurements',loc='left')
        lines=['Position    Result                                   Bilateral steps    XY displacement (mm)']
        for row,label in zip(rows,status_text):
            steps=str(row['bilateral_hold_steps']) if row['status']=='DYNAMIC_TEST_COMPLETED' else 'N/A'
            displ=row['max_xy_displacement_mm']
            displ_text=f'{displ:.1f}' if displ is not None and np.isfinite(displ) and row['status']=='DYNAMIC_TEST_COMPLETED' else 'NOT MEASURED'
            lines.append(f"{row['depth']:<11} {label:<40} {steps:<18} {displ_text}")
        if not any(row['status']=='DYNAMIC_TEST_COMPLETED' for row in rows):
            lines.extend(['', 'NO PHYSICS TEST WAS RUN.', 'Zero contact bars would be misleading: data is unavailable, not a measured failure.',
                          'Review IK error and open-jaw geometry before choosing the next target pose.'])
        axes[1].text(.01,.93,'\n'.join(lines),va='top',ha='left',fontfamily='monospace',fontsize=10,transform=axes[1].transAxes)
        fig.savefig(args.out/'comparison.png',dpi=150);plt.close(fig)
    except Exception as e:(args.out/'plot_error.txt').write_text(repr(e))
    summary={'schema':'phase6g_fixed_workcell_root_middle_tip_v1','simulation_only':True,
       'target_world_xyz_m':xyz.tolist(),'support_z_m':args.table_z,
       'target_reused_across_cases':True,
       'target_is_provisional_not_measured':True,
       'depth_local_z_provisional_not_actual_finger_arc_percent':True,
       'method':'5DoF soft IK per jaw-depth, stationary-arm J6 physical close and hold; no approach/lift',
       'grasp_not_verified_by_contact_alone':True,
       'contact_normal_sign_warning':'MuJoCo contact normals use geom order; interpret axis after checking geom IDs',
       'results':[info for _,info in results]}
    dump(args.out/'summary.json',summary)
    print('RESULT_DIR:',args.out,flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True,type=Path)
    p.add_argument('--urdf',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--target-x',type=float,default=.220)
    p.add_argument('--target-y',type=float,default=0.)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--max-ik-error-mm',type=float,default=20.)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--settle-s',type=float,default=.4)
    p.add_argument('--close-s',type=float,default=1.3)
    p.add_argument('--hold-s',type=float,default=1.0)
    p.add_argument('--arm-kp',type=float,default=18.)
    p.add_argument('--arm-kd',type=float,default=1.8)
    p.add_argument('--arm-torque',type=float,default=3.)
    p.add_argument('--j6-kp',type=float,default=2.)
    p.add_argument('--j6-kd',type=float,default=.08)
    p.add_argument('--j6-torque',type=float,default=.4)
    args=p.parse_args()
    if args.dt<=0 or args.fps<=0 or args.max_ik_error_mm<=0:p.error('Invalid args')
    run(args)

if __name__=='__main__':main()
