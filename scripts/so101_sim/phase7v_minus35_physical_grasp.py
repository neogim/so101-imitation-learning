#!/usr/bin/env python3
"""Phase7V: experimental -35 mm tabletop, sideways Muhi, open->close J6, DLS lift.

Simulation-only pragmatic grasp bench. LeRobot motor->URDF mapping, tabletop and
bottle pose are NOT independently calibrated. No real robot access.
All motions are integrated with mj_step (no teleports during integration).
"""
from __future__ import annotations
import argparse, csv, json, math, shutil, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import phase6c_upright_reach_grasp as phase6c
import phase7q_recorded_success_physics as p7q
import physical_workcell_sim as physical


def freejoint_adr(model):
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    if jid<0:raise ValueError('Dynamic Muhi freejoint missing')
    return int(model.jnt_qposadr[jid])


def jaw_gap(model,data,jaw,bottle):
    return p7q.geom_gap(model,data,bottle,jaw)*1000


def bottle_center_grasp_offset_mm(model,d,fixed,bottle,quat):
    """Closest bottle surface point to fixed jaw, projected onto bottle's long axis."""
    fromto=np.zeros(6)
    best=(float('inf'),None)
    for jaw_geom in fixed:
        for bottle_geom in bottle:
            separation=float(mujoco.mj_geomDistance(
                model,d,int(jaw_geom),int(bottle_geom),.20,fromto))
            if separation<best[0]:
                # geomDistance writes closest points for geom1 and geom2.
                best=(separation,np.asarray(fromto[3:6]).copy())
    if best[1] is None or not np.isfinite(best[0]):
        return float('inf')
    # The Muhi's long axis is the rotated local +Z.
    w,x,y,z=map(float,quat)
    axis=np.array([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)])
    axis/=np.linalg.norm(axis)
    body_center=np.asarray(d.qpos[freejoint_adr(model):freejoint_adr(model)+3])
    return float(1000*np.dot(best[1]-body_center,axis))


def scan_sideways(model,d,bq,qadr,q0,fixed,moving,bottle,table_z,
                  step_mm=5,radius_mm=60,center_tolerance_mm=12.0):
    """Select horizontal Muhi whose *actual closest jaw contact* is near long-axis center.

    Never prefer an end cap just because the fixed jaw is nearby.
    Selection uses MuJoCo closest points; if no central solution, report blocked.
    """
    d.qpos[qadr]=q0
    mujoco.mj_forward(model,d)
    origin=np.mean(d.geom_xpos[list(fixed)],axis=0)
    z=table_z+physical.MUHI_RADIUS+.0005
    orientations=[('along_x',[2**-.5,0.,2**-.5,0.]),
                  ('along_y',[.5,-.5,.5,.5])]
    offs=np.arange(-radius_mm,radius_mm+step_mm/2,step_mm)*.001
    ranked=[]
    candidate_count=0
    for orientation,quat in orientations:
        for dx in offs:
            for dy in offs:
                xy=origin[:2]+[dx,dy]
                if not (0.03<=xy[0]<=.42 and -.30<=xy[1]<=.30):continue
                d.qpos[bq:bq+3]=[xy[0],xy[1],z]
                d.qpos[bq+3:bq+7]=quat
                mujoco.mj_forward(model,d)
                fg=jaw_gap(model,d,fixed,bottle)
                mg=jaw_gap(model,d,moving,bottle)
                if not (.6<=fg<=8. and .6<=mg<=45.):continue
                candidate_count+=1
                center_offset=bottle_center_grasp_offset_mm(model,d,fixed,bottle,quat)
                if not np.isfinite(center_offset) or abs(center_offset)>center_tolerance_mm:
                    continue
                # Prefer centered surface contact over a superficially close end grip.
                score=(abs(center_offset)*0.45 + abs(fg-1.5) +
                       .045*abs(mg-12) + .003*math.hypot(dx,dy))
                item=dict(xyz_m=[float(xy[0]),float(xy[1]),float(z)],
                    quat_wxyz=quat,orientation=orientation,
                    placement='sideways_on_table',
                    fixed_gap_mm=float(fg),moving_gap_mm=float(mg),
                    fixed_jaw_contact_long_axis_offset_mm=center_offset,
                    grasp_center_tolerance_mm=float(center_tolerance_mm),
                    eligible_geometric_candidates=candidate_count,
                    offset_from_fixed_center_mm=[float(dx),float(dy)])
                ranked.append((score,item))
    return min(ranked,key=lambda x:x[0])[1] if ranked else None


def lift_target(model,d,qadr,ref_body,base,amount_mm):
    """Iterative position IK in isolated scratch MjData; actual physics remains mj_step.

    Recomputes the Jacobian along the 50 mm path (unlike the one-shot 15mm IK).
    """
    scratch=mujoco.MjData(model)
    scratch.qpos[:]=d.qpos
    scratch.qvel[:]=0
    mujoco.mj_forward(model,scratch)
    ref_start=np.asarray(scratch.xpos[ref_body]).copy()
    target=ref_start+np.array([0.,0.,amount_mm/1000])
    cols=[]
    for k in range(5):
        joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,p7q.NAMES[k])
        cols.append(int(model.jnt_dofadr[joint]))
    q=base[:5].copy()
    jids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,p7q.NAMES[k]) for k in range(5)]
    limits=np.asarray(model.jnt_range[jids],float)
    history=[]
    for step in range(45):
        scratch.qpos[qadr[:5]]=q
        mujoco.mj_forward(model,scratch)
        error=target-np.asarray(scratch.xpos[ref_body])
        error_mm=float(np.linalg.norm(error)*1000)
        history.append(error_mm)
        if error_mm<.5:break
        jp=np.zeros((3,model.nv));jr=np.zeros((3,model.nv))
        mujoco.mj_jacBody(model,scratch,jp,jr,ref_body)
        J=jp[:,cols]
        damp=.04
        dq=J.T@np.linalg.solve(J@J.T+damp*damp*np.eye(3),error)
        dq=np.clip(dq,-np.deg2rad(4),np.deg2rad(4))
        q=np.clip(q+dq,limits[:,0]+.001,limits[:,1]-.001)
    scratch.qpos[qadr[:5]]=q
    mujoco.mj_forward(model,scratch)
    achieved=(float(scratch.xpos[ref_body,2])-ref_start[2])*1000
    return q,dict(target_lift_mm=float(amount_mm),
       predicted_fk_vertical_lift_mm=achieved,
       predicted_final_xyz_error_mm=float(np.linalg.norm(target-scratch.xpos[ref_body])*1000),
       iterations=len(history),last_error_mm=history[-1],
       delta_joint_deg=np.rad2deg(q-base[:5]).tolist())


def generate_video(model,frames,out,fps,bq,stages):
    if shutil.which('ffmpeg') is None:return 'FFMPEG_MISSING'
    tmp=out/'_frame_tmp';tmp.mkdir(exist_ok=True)
    model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),640)
    model.vis.global_.offheight=max(int(model.vis.global_.offheight),410)
    renderer=None
    try:
        renderer=mujoco.Renderer(model,width=640,height=410)
        rd=mujoco.MjData(model)
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance=.50
        for i,(qpos,qvel,tt,phase,fc,mc) in enumerate(frames):
            rd.qpos[:]=qpos;rd.qvel[:]=qvel
            mujoco.mj_forward(model,rd)
            cam.lookat[:]=[float(qpos[bq]),float(qpos[bq+1]),float(qpos[bq+2])]
            views=[]
            for az,el,name in [(90,-15,'SIDE'),(270,-18,'OPPOSITE'),(105,-68,'OVERHEAD')]:
                cam.azimuth=az;cam.elevation=el
                renderer.update_scene(rd,camera=cam)
                img=Image.fromarray(renderer.render().copy())
                tile=Image.new('RGB',(640,450),'#111827');tile.paste(img,(0,40))
                ImageDraw.Draw(tile).text((12,10),f'{name} | {phase} | t={tt:.2f}s | contacts F={fc} M={mc}',fill='white')
                views.append(tile)
            combined=Image.new('RGB',(1920,450))
            for k,tile in enumerate(views):combined.paste(tile,(k*640,0))
            combined.save(tmp/f'{i:05d}.png')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(tmp/'%05d.png'),
                        '-c:v','libx264','-pix_fmt','yuv420p',str(out/'phase7v_grasp_3views.mp4')],check=True)
        return 'SAVED'
    except Exception as exc:
        (out/'render_error.txt').write_text(repr(exc))
        return 'ERROR'
    finally:
        if renderer is not None:renderer.close()
        shutil.rmtree(tmp,ignore_errors=True)


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    summary=dict(status='STARTED',table_z_mm=args.table_z_mm,episode_time_s=args.time,
                 physics_integrated=True,real_table_calibrated=False,
                 motor_to_urdf_mapping_verified=False,real_bottle_pose_measured=False,
                 objective='center-grasp sideways bottle, J6 close -> 50mm lift; experimental -35mm table',
                 limitations=['Table height is experimental, not actual robot calibration',
                              'Sideways bottle XY and horizontal yaw are synthetic candidates near fixed jaw',
                              'Unverified motor sign/offset and J6 degrees',
                              'Base/link mounting may differ from physical 3D printed parts'])
    try:
        times,states,_=p7q.read_recorded_csv(args.csv)
        mapping=json.loads(args.mapping.read_text())
        qstates=p7q.joint_mapping(states,mapping,args.j6_closed_deg,args.j6_open_deg)
        q0=np.array([np.interp(args.time,times,qstates[:,j]) for j in range(6)])
        opts=SimpleNamespace(urdf=args.urdf,out=args.out,dt=args.dt,
            jaw_collision_mode='axial_cross_convex_patches',jaw_collision_slabs=12,
            jaw_collision_cross_bands=3,jaw_table_contact=True,
            support_plane_z_m=args.table_z_mm/1000.,arm_kp=18.,arm_kd=1.8,
            arm_torque=3.,j6_kp=2.,j6_kd=.08,j6_torque=.4)
        model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor=phase6c.model_build(opts)
        model.opt.timestep=args.dt
        d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
        bq=freejoint_adr(model)
        # Initial J6 is intentionally open so placing a bottle does not assume prior grasp.
        open_r=np.clip(np.deg2rad(args.j6_initial_open_deg),lo[5]+.001,hi[5]-.001)
        close_r=np.clip(np.deg2rad(args.j6_command_closed_deg),lo[5]+.001,hi[5]-.001)
        q0[5]=open_r
        if args.j5_preset_deg is not None:
            j5_value=np.deg2rad(args.j5_preset_deg)
            if not (lo[4]+.001 <= j5_value <= hi[4]-.001):
                raise ValueError('J5 preset outside URDF limits')
            q0[4]=j5_value
        d.qpos[qadr]=q0
        d.ctrl[aids]=q0
        mujoco.mj_forward(model,d)
        if args.fixed_pose_json is None:
            pose=scan_sideways(model,d,bq,qadr,q0,fixed,moving,bottle,
                args.table_z_mm/1000.,args.scan_step_mm,args.scan_radius_mm,args.center_tolerance_mm)
        else:
            pose=json.loads(args.fixed_pose_json.read_text())
            if pose.get('placement')!='sideways_on_table' or len(pose.get('xyz_m',[]))!=3 or len(pose.get('quat_wxyz',[]))!=4:
                raise ValueError('fixed_pose_json must specify sideways_on_table and xyz_m/quat_wxyz')
            d.qpos[bq:bq+3]=pose['xyz_m'];d.qpos[bq+3:bq+7]=pose['quat_wxyz']
            mujoco.mj_forward(model,d)
            pose['fixed_gap_mm']=jaw_gap(model,d,fixed,bottle)
            pose['moving_gap_mm']=jaw_gap(model,d,moving,bottle)
            pose['fixed_jaw_contact_long_axis_offset_mm']=bottle_center_grasp_offset_mm(
                model,d,fixed,bottle,pose['quat_wxyz'])
            pose['grasp_center_tolerance_mm']=args.center_tolerance_mm
            if (abs(pose['fixed_jaw_contact_long_axis_offset_mm'])>args.center_tolerance_mm or
                not(.6<=pose['fixed_gap_mm']<=8. and .6<=pose['moving_gap_mm']<=45.)):
                summary.update(status='BLOCKED_FIXED_BOTTLE_J5_GEOMETRY',
                    initial_bottle_pose=pose)
                return
        if pose is None:
            summary.update(status='BLOCKED_NO_CENTRAL_SIDEWAYS_BOTTLE_POSE',note='No nonpenetrating central fixed-jaw target. Do not substitute end grip.')
            return
        d.qpos[bq:bq+3]=pose['xyz_m'];d.qpos[bq+3:bq+7]=pose['quat_wxyz']
        d.qvel[:]=0
        mujoco.mj_forward(model,d)
        table_dist=[min(float(mujoco.mj_geomDistance(model,d,int(g),int(floor),.15,np.zeros(6))) for g in geoms)*1000 for geoms in (fixed,moving)]
        summary.update(initial_bottle_pose=pose,initial_jaw_table_gap_mm=table_dist,initial_arm_joints_deg=np.rad2deg(q0).tolist())
        if min(table_dist)<-.5:
            summary['status']='BLOCKED_INITIAL_JAW_TABLE_PENETRATION';return
        if pose['fixed_gap_mm']<0 or pose['moving_gap_mm']<0:
            summary['status']='BLOCKED_INITIAL_JAW_BOTTLE_PENETRATION';return
        base=q0.copy()
        ref=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'gripper_link')
        if ref<0:raise RuntimeError('gripper_link body not found')
        lift_angles,lift_meta=lift_target(model,d,qadr,ref,base,args.lift_mm)
        lift_angles=np.clip(lift_angles,lo[:5],hi[:5])
        lift_meta['command_joint_deg']=np.rad2deg(lift_angles).tolist()
        summary['lift_dls']=lift_meta
        phases=[('settle',args.settle_s),('close',args.closing_s),('hold',args.hold_s),('lift',args.lift_s),('retain',args.retain_s)]
        total=sum(duration for _,duration in phases)
        steps=math.ceil(total/args.dt)
        stride=max(1,round(1/(args.dt*args.fps)))
        trace=[];frames=[];t=0.;start_bottle=np.array(d.qpos[bq:bq+3]);initial_z=float(start_bottle[2]);start_jaw_z=float(d.xpos[ref,2]);last_phase=''
        for k in range(steps+1):
            elapsed=0.;phase='retain';u=1.
            for name,duration in phases:
                if t<=elapsed+duration+1e-9:
                    phase=name;u=np.clip((t-elapsed)/max(duration,1e-9),0.,1.);break
                elapsed+=duration
            target=base.copy()
            if phase in ('close','hold','lift','retain'):
                target[5]=open_r+(close_r-open_r)*(u if phase=='close' else 1.)
            if phase in ('lift','retain'):
                target[:5]=base[:5]+(lift_angles-base[:5])*(u if phase=='lift' else 1.)
            d.ctrl[aids]=target
            if k>0:
                mujoco.mj_step(model,d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                summary.update(status='ABORT_NONFINITE_SIM_STATE',phase=phase);break
            fc,mc,force,depth=p7q.contact_force(model,d,fixed,moving,bottle)
            table_now=[min(float(mujoco.mj_geomDistance(model,d,int(g),int(floor),.15,np.zeros(6))) for g in geoms)*1000 for geoms in (fixed,moving)]
            row=dict(time_s=t,phase=phase,j6_target_deg=float(np.rad2deg(target[5])),
                j6_actual_deg=float(np.rad2deg(d.qpos[qadr[5]])),
                j2_actual_deg=float(np.rad2deg(d.qpos[qadr[1]])),
                j3_actual_deg=float(np.rad2deg(d.qpos[qadr[2]])),
                fixed_contact_count=fc,moving_contact_count=mc,
                bilateral_contact=int(fc>0 and mc>0),
                max_contact_normal_force_N=force,
                min_contact_dist_mm=depth*1000 if depth is not None else '',
                jaw_rise_mm=float((d.xpos[ref,2]-start_jaw_z)*1000),
                bottle_rise_mm=float((d.qpos[bq+2]-initial_z)*1000),
                bottle_xy_displacement_mm=float(np.linalg.norm(d.qpos[bq:bq+2]-start_bottle[:2])*1000),
                fixed_table_gap_mm=table_now[0],moving_table_gap_mm=table_now[1])
            trace.append(row)
            if k%stride==0 or (phase!=last_phase):
                frames.append((d.qpos.copy(),d.qvel.copy(),t,phase,fc,mc))
            last_phase=phase;t=min(total,(k+1)*args.dt)
        if trace:
            with (args.out/'phase7v_contact_trace.csv').open('w',newline='') as fp:
                w=csv.DictWriter(fp,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
            tt=[row['time_s'] for row in trace]
            for key,label in [('fixed_contact_count','Fixed jaw'),('moving_contact_count','Moving jaw')]:
                axes[0].plot(tt,[r[key] for r in trace],label=label)
            axes[0].set_ylabel('Contact count')
            axes[1].plot(tt,[r['max_contact_normal_force_N'] for r in trace],label='Jaw-bottle max normal')
            axes[1].set_ylabel('Contact normal force (N)')
            for key,label in [('jaw_rise_mm','Gripper link'),('bottle_rise_mm','Muhi bottle')]:
                axes[2].plot(tt,[r[key] for r in trace],label=label)
            axes[2].set_ylabel('Rise relative to initial (mm)')
            axes[2].set_xlabel('Simulation time (s)')
            for ax in axes:ax.grid(alpha=.2);ax.legend()
            fig.tight_layout();fig.savefig(args.out/'phase7v_contact_lift_graph.png',dpi=170);plt.close(fig)
        bilateral=sum(row['bilateral_contact'] for row in trace if row['phase'] in ('hold','lift','retain'))
        held=[row for row in trace if row['phase']=='retain']
        held_bilateral=bool(held and sum(row['bilateral_contact'] for row in held)>=max(1,int(len(held)*.3)))
        bottle_lift=max((r['bottle_rise_mm'] for r in trace if r['phase'] in ('lift','retain')),default=0.)
        final_bottle=held[-1]['bottle_rise_mm'] if held else None
        max_force=max((r['max_contact_normal_force_N'] for r in trace),default=0.)
        metrics=dict(steps=len(trace),bilateral_contact_steps=bilateral,hold_bilateral=held_bilateral,
           max_jaw_normal_force_N=max_force,max_bottle_rise_mm=bottle_lift,
           final_bottle_rise_mm=final_bottle,
           max_bottle_xy_displacement_mm=max((r['bottle_xy_displacement_mm'] for r in trace),default=0.),
           min_fixed_table_gap_mm=min((r['fixed_table_gap_mm'] for r in trace),default=None),
           min_moving_table_gap_mm=min((r['moving_table_gap_mm'] for r in trace),default=None))
        if summary['status']=='STARTED':
            summary['status']=('SIMULATED_LIFT_WITH_BILATERAL_CONTACT' if held_bilateral and final_bottle is not None and final_bottle>=args.lift_mm*.35
                 else 'PHYSICS_COMPLETE_GRASP_OR_LIFT_NOT_CONFIRMED')
        summary['metrics']=metrics
        if not args.no_video and frames:
            summary['video_status']=generate_video(model,frames,args.out,args.fps,bq,phases)
    except Exception as exc:
        summary.update(status='ERROR',error=repr(exc))
        raise
    finally:
        (args.out/'phase7v_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'status':summary['status'],'metrics':summary.get('metrics'),
                          'video_status':summary.get('video_status'),'out':str(args.out)},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['csv','mapping','urdf','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--time',type=float,default=14.8)
    p.add_argument('--table-z-mm',type=float,default=-35.)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--j6-closed-deg',type=float,default=-10.)
    p.add_argument('--j6-open-deg',type=float,default=100.)
    p.add_argument('--j6-initial-open-deg',type=float,default=80.)
    p.add_argument('--j6-command-closed-deg',type=float,default=0.)
    p.add_argument('--scan-step-mm',type=int,default=5)
    p.add_argument('--scan-radius-mm',type=int,default=60)
    p.add_argument('--center-tolerance-mm',type=float,default=12.)
    p.add_argument('--settle-s',type=float,default=.4)
    p.add_argument('--closing-s',type=float,default=1.4)
    p.add_argument('--hold-s',type=float,default=.5)
    p.add_argument('--lift-s',type=float,default=2.4)
    p.add_argument('--retain-s',type=float,default=.9)
    p.add_argument('--lift-mm',type=float,default=50.)
    p.add_argument('--fixed-pose-json',type=Path,default=None,
        help='Exact bottle pose for Phase7W J5 yaw tests; bypass XY/orientation scan')
    p.add_argument('--j5-preset-deg',type=float,default=None,
        help='Change wrist_roll only, preserving Episode015 J1-J4 initial positions')
    p.add_argument('--no-video',action='store_true')
    args=p.parse_args()
    if args.table_z_mm!=-35.:print('NOTE: overriding the Phase7V experimental table -35mm',file=sys.stderr)
    run(args)

if __name__=='__main__':main()
