#!/usr/bin/env python3
"""Phase7Q: recorded SO-101 episode 015 -> MuJoCo contact-rich diagnostic.

Runs only on recorded data; no robot hardware.  The FK joint sign/zero, normalized
J6->URDF angle and bottle world pose are hypotheses, NOT real calibrations.
An automatic jaw-proximity placement is labelled synthetic/provisional, NOT an
estimated object pose from images.  Its success must never imply real parity.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import phase6c_upright_reach_grasp as phase6c
import physical_grasp_j6_bench as grasp
import physical_workcell_sim as physical

NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def read_recorded_csv(path: Path):
    data=[]
    with path.open(newline='', encoding='utf8') as fp:
        for rec in csv.DictReader(fp):
            st=np.asarray(json.loads(rec['observation.state']), dtype=float)
            action=np.asarray(json.loads(rec['action']), dtype=float)
            t=float(rec['timestamp'])
            if st.shape != (6,) or action.shape != (6,) or not np.isfinite(np.r_[t,st,action]).all():
                raise ValueError('Malformed or nonfinite episode row')
            data.append((t,st,action))
    times=np.asarray([v[0] for v in data])
    if len(data)<3 or np.any(np.diff(times)<=0):
        raise ValueError('Recorded timestamps must be strictly increasing')
    return times, np.stack([r[1] for r in data]), np.stack([r[2] for r in data])


def joint_mapping(values, config, closed_deg, open_deg):
    """Explicit, UNVALIDATED motor-to-URDF hypothesis. No joint angle clamp."""
    v=np.asarray(values,dtype=float)
    sign=np.asarray(config['sign'],float)
    offset=np.asarray(config['offset_deg'],float)
    if sign.shape!=(5,) or offset.shape!=(5,) or not np.all(np.isin(sign,[-1,1])):
        raise ValueError('Invalid 5-axis mapping')
    if v.shape[-1]!=6 or np.min(v[...,5])<0 or np.max(v[...,5])>100:
        raise ValueError('Six joint values with J6 0..100 required')
    if closed_deg>=open_deg:
        raise ValueError('J6 must close when normalized value decreases')
    q=np.empty(v.shape,float)
    q[...,:5]=np.deg2rad(v[...,:5]*sign+offset)
    q[...,5]=np.deg2rad(closed_deg+(open_deg-closed_deg)*v[...,5]/100.)
    return q


def contact_force(model, d, fixed, moving, bottle):
    fc=mc=0; max_n=0.; any_depth=None
    F=set(fixed);M=set(moving);B=set(bottle)
    for i,c in enumerate(d.contact[:d.ncon]):
        pair={int(c.geom1),int(c.geom2)}
        if not(pair & B):continue
        role='fixed' if pair & F else ('moving' if pair & M else None)
        if role is None:continue
        fc+=role=='fixed';mc+=role=='moving'
        out=np.zeros(6);mujoco.mj_contactForce(model,d,i,out)
        max_n=max(max_n,abs(float(out[0])))
        any_depth=float(c.dist) if any_depth is None else min(any_depth,float(c.dist))
    return int(fc),int(mc),max_n,any_depth


def geom_gap(model, d, objects, jaws):
    # True signed convex distance; negative implies penetration.
    return float(grasp.geometry_gaps(model,d,objects,jaws))


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf8') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def yaw_quat(deg):
    q=(Rotation.from_euler('z',deg,degrees=True)*Rotation.from_euler('y',90,degrees=True)).as_quat()
    return np.asarray([q[3],*q[:3]],float)


def locate_provisional_bottle(model, data, qadr, bq, fixed, moving, bottle, floor,
                              table_z, grid_radius_mm=50, grid_step_mm=10):
    """Grid around the *actual FK jaw* to find a nonpenetrating initial geometry.

    This is a synthetic scan, NOT computer vision or object-pose estimation.
    Only XY is scanned; bottle is held horizontal and resting on tabletop.
    """
    center=np.asarray(data.geom_xpos[list(fixed)],float).mean(axis=0)
    # Real bottle orientation is unmeasured; test a small, explicit yaw set.
    yaw_candidates=(0.,45.,90.,135.)
    offsets=np.arange(-grid_radius_mm,grid_radius_mm+grid_step_mm/2,grid_step_mm)/1000
    records=[];acceptable=[]
    z=float(table_z+physical.MUHI_RADIUS+0.0005)
    for yaw in yaw_candidates:
        quat=yaw_quat(yaw)
        for dx in offsets:
            for dy in offsets:
                xyz=np.array([center[0]+dx,center[1]+dy,z],float)
                # The source table is in robot base coordinates; keep bottle in front work area.
                if not (0.03<=xyz[0]<=0.42 and -0.30<=xyz[1]<=0.30):continue
                data.qpos[bq:bq+3]=xyz
                data.qpos[bq+3:bq+7]=quat
                mujoco.mj_forward(model,data)
                fg=geom_gap(model,data,bottle,fixed)
                mg=geom_gap(model,data,bottle,moving)
                # Contact at the fixed jaw, moving jaw initially clear; no penetration.
                score=abs(fg-0.0015)+0.12*abs(mg-0.012)+0.025*np.hypot(dx,dy)
                good=(fg>=-0.00025 and mg>=-0.00025 and fg<0.010 and mg<0.035)
                row=dict(yaw_deg=float(yaw),x_m=float(xyz[0]),y_m=float(xyz[1]),
                         z_m=z,initial_fixed_gap_mm=fg*1000,
                         initial_moving_gap_mm=mg*1000,score_m=float(score),
                         admissible_geometry=bool(good),offset_from_fixed_center_mm=float(1000*np.hypot(dx,dy)))
                records.append(row)
                if good:acceptable.append(row)
    if not acceptable:return None,records
    best=min(acceptable,key=lambda r:r['score_m'])
    result=dict(xyz_m=[best['x_m'],best['y_m'],best['z_m']],
                quat_wxyz=yaw_quat(best['yaw_deg']).tolist(),
                table_z_m=float(table_z),
                source='synthetic_FK_relative_grid_NOT_measured_real_bottle',
                initial_fixed_gap_mm=best['initial_fixed_gap_mm'],
                initial_moving_gap_mm=best['initial_moving_gap_mm'],
                yaw_deg=best['yaw_deg'])
    return result,records


def run(args):
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    result={'status':'NOT_STARTED','episode_index':15,'time_window_s':[args.start_s,args.end_s],
            'physics_engine':'MuJoCo mj_step','recorded_actions_used':True,
            'motor_to_urdf_mapping_verified':False,
            'j6_norm_to_urdf_mapping_verified':False,
            'bottle_pose_measured_from_real':False,
            'contact_model':'original_URDF_visual_plus_12x3_patch_collision',
            'caveats':['Camera extrinsics uncalibrated','No measured gripper force/current/torque',
                       'Bilateral contacts are necessary but not sufficient for stable grasp']}
    try:
        times,states,actions=read_recorded_csv(args.csv)
        config=json.loads(args.mapping.read_text(encoding='utf8'))
        qstate=joint_mapping(states,config,args.j6_closed_deg,args.j6_open_deg)
        qaction=joint_mapping(actions,config,args.j6_closed_deg,args.j6_open_deg)
        result['mapping_hypothesis']=dict(sign=config['sign'],offset_deg=config['offset_deg'],
            j6_closed_deg_at_norm0=args.j6_closed_deg,j6_open_deg_at_norm100=args.j6_open_deg)
        spec_args=SimpleNamespace(urdf=args.urdf,out=args.out,dt=args.dt,
            jaw_collision_mode='axial_cross_convex_patches',
            jaw_collision_slabs=12,jaw_collision_cross_bands=3,
            jaw_table_contact=True,support_plane_z_m=args.table_z,
            arm_kp=args.arm_kp,arm_kd=args.arm_kd,arm_torque=args.arm_torque,
            j6_kp=args.j6_kp,j6_kd=args.j6_kd,j6_torque=args.j6_torque)
        model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor=phase6c.model_build(spec_args)
        free=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
        if free<0:raise RuntimeError('Missing dynamic Muhi free joint')
        bq=int(model.jnt_qposadr[free]);d=mujoco.MjData(model)
        # Audit every recorded goal within replay window, no hidden clipping to URDF limits.
        mask=(times>=args.start_s)&(times<=args.end_s)
        if not np.any(mask):raise ValueError('No samples in requested interval')
        errors=[]
        for i in np.flatnonzero(mask):
            for j,name in enumerate(NAMES):
                if qaction[i,j]<lo[j] or qaction[i,j]>hi[j]:
                    errors.append(dict(t=float(times[i]),joint=name,q_deg=float(np.rad2deg(qaction[i,j])),
                                       bound_deg=np.rad2deg([lo[j],hi[j]]).tolist()))
        if errors:
            result.update(status='BLOCKED_ACTION_OUT_OF_URDF_LIMITS',joint_errors=errors[:30]);return result
        begin=float(args.start_s);end=float(args.end_s)
        q0=np.array([np.interp(begin,times,qstate[:,j]) for j in range(6)])
        mujoco.mj_resetData(model,d)
        d.qpos[qadr]=q0
        d.ctrl[aids]=np.array([np.interp(begin,times,qaction[:,j]) for j in range(6)])
        mujoco.mj_forward(model,d)
        # Two ways to specify bottle position. Neither silently becomes measured truth.
        if args.bottle_pose_json is not None:
            supplied=json.loads(args.bottle_pose_json.read_text())
            if supplied.get('coordinate_frame')!='urdf_base_world':
                raise ValueError('bottle_pose_json requires coordinate_frame=urdf_base_world')
            pose=dict(xyz_m=supplied['xyz_m'],quat_wxyz=supplied['quat_wxyz'],
                      table_z_m=args.table_z,source='user_supplied_not_independently_verified')
            grid=[]
        else:
            pose,grid=locate_provisional_bottle(model,d,qadr,bq,fixed,moving,bottle,floor,args.table_z,
                      grid_radius_mm=args.grid_radius_mm,grid_step_mm=args.grid_step_mm)
        write_csv(args.out/'initial_pose_scan.csv',grid)
        if pose is None:
            result.update(status='BLOCKED_NO_NONPENETRATING_SYNTHETIC_INITIAL_PLACEMENT',
                scan_candidates=len(grid),note='This is a pose/model mismatch, not friction evidence.')
            return result
        result['initial_bottle_pose']=pose
        d.qpos[bq:bq+3]=pose['xyz_m'];d.qpos[bq+3:bq+7]=pose['quat_wxyz']
        mujoco.mj_forward(model,d)
        fg=geom_gap(model,d,bottle,fixed);mg=geom_gap(model,d,bottle,moving)
        # Also check both jaws against the real physical tabletop collision.
        floor_f=min(float(mujoco.mj_geomDistance(model,d,int(g),int(floor),.15,np.zeros(6))) for g in fixed)
        floor_m=min(float(mujoco.mj_geomDistance(model,d,int(g),int(floor),.15,np.zeros(6))) for g in moving)
        initial=dict(fixed_gap_mm=fg*1000,moving_gap_mm=mg*1000,
                     fixed_table_gap_mm=floor_f*1000,moving_table_gap_mm=floor_m*1000)
        result['initial_geom_audit']=initial
        if fg<-args.penetration_tolerance_mm/1000 or mg<-args.penetration_tolerance_mm/1000:
            result['status']='BLOCKED_INITIAL_JAW_BOTTLE_PENETRATION';return result
        if min(floor_f,floor_m)<-args.table_tolerance_mm/1000:
            result['status']='BLOCKED_INITIAL_JAW_TABLE_PENETRATION';return result
        if args.stop_before_physics:
            result['status']='INITIAL_POSE_DIAGNOSTIC_ONLY';return result
        # Real joint action goal commands, interpolated from recorded 20Hz timestamps.
        # We NEVER overwrite qpos once mj_step starts.
        d.qvel[:]=0
        mujoco.mj_forward(model,d)
        original_xy=np.asarray(pose['xyz_m'][:2],float)
        fixed_z0=float(np.mean(d.geom_xpos[list(fixed),2]))
        trace=[];snapshots=[]
        n=int(math.ceil((end-begin)/args.dt))
        stride=max(1,int(round(1/(args.dt*args.fps))))
        for k in range(n+1):
            t=min(begin+k*args.dt,end)
            target=np.array([np.interp(t,times,qaction[:,j]) for j in range(6)])
            d.ctrl[aids]=target
            if k>0:mujoco.mj_step(model,d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                result['status']='ABORT_NONFINITE_SIM_STATE';break
            fc,mc,fn,depth=contact_force(model,d,fixed,moving,bottle)
            fixed_z=float(np.mean(d.geom_xpos[list(fixed),2]))
            err=np.rad2deg(target[:5]-d.qpos[qadr[:5]])
            trace.append(dict(time_s=t,j2_target_deg=float(np.rad2deg(target[1])),
                 j2_actual_deg=float(np.rad2deg(d.qpos[qadr[1]])),j2_error_deg=float(err[1]),
                 j3_target_deg=float(np.rad2deg(target[2])),
                 j3_actual_deg=float(np.rad2deg(d.qpos[qadr[2]])),j3_error_deg=float(err[2]),
                 max_arm_error_deg=float(np.max(np.abs(err))),
                 j6_target_deg=float(np.rad2deg(target[5])),j6_actual_deg=float(np.rad2deg(d.qpos[qadr[5]])),
                 fixed_jaw_delta_z_mm=float(1000*(fixed_z-fixed_z0)),
                 bottle_xy_displacement_mm=float(1000*np.linalg.norm(d.qpos[bq:bq+2]-original_xy)),
                 bottle_z_m=float(d.qpos[bq+2]),fixed_jaw_contacts=fc,moving_jaw_contacts=mc,
                 bilateral_contact=int(fc>0 and mc>0),
                 max_jaw_bottle_contact_normal_n=fn,
                 min_jaw_bottle_contact_dist_mm=depth*1000 if depth is not None else ''))
            if k%stride==0:
                # mj_copyData is unavailable in several Python bindings; use mj_state vectors.
                snapshots.append((np.asarray(d.qpos).copy(),np.asarray(d.qvel).copy(),t))
        write_csv(args.out/'physical_joint_trace.csv',trace)
        if trace:
            from matplotlib import pyplot as plt
            for filename,series,ylabel in (
              ('arm_tracking.png',(('j2_target_deg','J2 target'),('j2_actual_deg','J2 actual'),('j3_target_deg','J3 target'),('j3_actual_deg','J3 actual')),'Joint angle (deg)'),
              ('jaw_rise.png',(('fixed_jaw_delta_z_mm','Fixed-jaw centroid ÎZ'),),'Vertical displacement (mm)'),
              ('bottle_displacement.png',(('bottle_xy_displacement_mm','Muhi XY displacement'),),'Horizontal displacement (mm)'),
              ('contact_counts.png',(('fixed_jaw_contacts','Fixed-jaw contacts'),('moving_jaw_contacts','Moving-jaw contacts')),'Number of contacts')):
                fig,ax=plt.subplots(figsize=(9,4))
                tt=[row['time_s'] for row in trace]
                for key,label in series:ax.plot(tt,[row[key] for row in trace],label=label)
                ax.set_xlabel('Episode time (s)');ax.set_ylabel(ylabel)
                ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(args.out/filename,dpi=130);plt.close(fig)
        result['metrics']=dict(steps=len(trace),max_fixed_jaw_rise_mm=max([r['fixed_jaw_delta_z_mm'] for r in trace],default=None),
             max_muhi_xy_displacement_mm=max([r['bottle_xy_displacement_mm'] for r in trace],default=None),
             max_arm_tracking_error_deg=max([r['max_arm_error_deg'] for r in trace],default=None),
             max_jaw_contact_normal_n=max([r['max_jaw_bottle_contact_normal_n'] for r in trace],default=None),
             bilateral_contact_fraction=float(np.mean([r['bilateral_contact'] for r in trace])) if trace else None,
             lift_from_initial_m=float(d.qpos[bq+2]-pose['xyz_m'][2]))
        if result['status']=='NOT_STARTED':result['status']='PROVISIONAL_PHYSICAL_REPLAY_COMPLETE_NOT_REAL_PARITY'
        if snapshots and not args.no_video:
            result['video']=render_snapshots(model,snapshots,args.out,args.fps,bq)
        return result
    except Exception as e:
        result['status']='ERROR_DURING_RUN'
        result['error']=repr(e)
        raise
    finally:
        (args.out/'phase7q_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(dict(status=result.get('status'),metrics=result.get('metrics'),out=str(args.out)),ensure_ascii=False))


def render_snapshots(model,snapshots,out,fps,bq):
    from PIL import Image,ImageDraw
    if not shutil.which('ffmpeg'):return 'FFMPEG_NOT_INSTALLED'
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,540)
    model.vis.global_.offheight=max(model.vis.global_.offheight,350)
    r=None
    tmp=out/'_video_frames'
    try:
        r=mujoco.Renderer(model,width=540,height=350)
        d=mujoco.MjData(model)
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance=.57
        tmp.mkdir()
        for i,(qpos,qvel,tt) in enumerate(snapshots):
            d.qpos[:]=qpos;d.qvel[:]=qvel
            mujoco.mj_forward(model,d)
            # Camera is a diagnostic view, NOT physical Side camera calibration.
            cam.lookat[:]=[float(qpos[bq]),float(qpos[bq+1]),.065]
            tiles=[]
            for az,el,tag in ((90,-16,'SIDE'),(270,-16,'OPPOSITE'),(105,-78,'OVERHEAD')):
                cam.azimuth=az;cam.elevation=el
                r.update_scene(d,camera=cam)
                pic=Image.fromarray(r.render().copy())
                tile=Image.new('RGB',(540,380),'#17212b');tile.paste(pic,(0,30))
                ImageDraw.Draw(tile).text((8,8),f'MuJoCo mj_step {tag} | t={tt:.2f}s',fill='white')
                tiles.append(tile)
            frame=Image.new('RGB',(1620,380))
            for j,tile in enumerate(tiles):frame.paste(tile,(j*540,0))
            frame.save(tmp/f'{i:05d}.png')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(tmp/'%05d.png'),
                        '-c:v','libx264','-pix_fmt','yuv420p',str(out/'physical_replay_3views.mp4')],check=True)
        return 'SAVED'
    except Exception as e:
        (out/'render_error.txt').write_text(repr(e))
        return 'ERROR'
    finally:
        if r is not None:r.close()
        if tmp.exists():shutil.rmtree(tmp)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bottle-pose-json',type=Path)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--start-s',type=float,default=14.8)
    p.add_argument('--end-s',type=float,default=18.3)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--grid-radius-mm',type=float,default=50.)
    p.add_argument('--grid-step-mm',type=float,default=10.)
    p.add_argument('--penetration-tolerance-mm',type=float,default=.25)
    p.add_argument('--table-tolerance-mm',type=float,default=.5)
    p.add_argument('--j6-closed-deg',type=float,default=-10.)
    p.add_argument('--j6-open-deg',type=float,default=100.)
    p.add_argument('--arm-kp',type=float,default=18.)
    p.add_argument('--arm-kd',type=float,default=1.8)
    p.add_argument('--arm-torque',type=float,default=3.)
    p.add_argument('--j6-kp',type=float,default=2.)
    p.add_argument('--j6-kd',type=float,default=.08)
    p.add_argument('--j6-torque',type=float,default=.4)
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--stop-before-physics',action='store_true')
    args=p.parse_args(argv)
    if not(args.start_s<args.end_s and args.dt>0 and args.fps>0):p.error('Bad time, dt or fps')
    if args.grid_step_mm<=0 or args.grid_radius_mm<=0:p.error('Invalid pose scan grid')
    run(args)


if __name__=='__main__':main()
