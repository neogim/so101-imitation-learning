#!/usr/bin/env python3
"""Phase 6C candidate: Episode 4 approach -> upright Muhi grasp in MuJoCo.

A *new* geometric integration test, not proof that earlier simulated grasp passed.
A single world-upright bottle is placed BEFORE integration and stays dynamic.
Its initial place is selected from the recorded approach END pose; the real
Episode 4 lying-object geometry is not silently treated as upright.

Robot J1-J6 follow PD position servos, implicitfast. Robot self-collisions and
robot-table collisions are disabled in this FIRST integration to isolate jaw-
bottle contact; finger-bottle and bottle support-plane contacts stay enabled.
No trajectory qpos overwrite, no object teleport. No real robot or hardware.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw

import real_to_sim as r2s
import physical_workcell_sim as physical
import physical_grasp_j6_bench as grasp
import phase6b_episode4_physical_approach as phase6b

NAMES = (*r2s.JOINTS, 'gripper')
DEFAULT_PROFILE_CONFIG = Path(__file__).resolve().parents[2] / 'configs/so101/phase6c_approach_profiles.json'


def load_approach_profile(config_path: Path, profile: str):
    """Read a named, *joint-space* endpoint adjustment (not world-frame IK)."""
    payload = json.loads(config_path.resolve(strict=True).read_text(encoding='utf-8'))
    profiles = payload.get('profiles', {})
    if profile not in profiles:
        raise ValueError(f'Unknown approach profile {profile!r}; available: {sorted(profiles)}')
    item = profiles[profile]
    offset4 = float(item['j4_wrist_flex_delta_deg'])
    offset5 = float(item.get('j5_wrist_roll_delta_deg', 0.0))
    start = float(item.get('blend_start_fraction', 0.75))
    if not (np.isfinite([offset4, offset5, start]).all()
            and abs(offset4) <= 45 and abs(offset5) <= 45
            and 0.0 <= start < 1.0):
        raise ValueError('Invalid joint-space approach profile parameters')
    return {'name': profile, 'j4_wrist_flex_delta_deg': offset4,
            'j5_wrist_roll_delta_deg': offset5, 'blend_start_fraction': start,
            'warning': 'Joint-angle offsets, NOT verified world-Z tilt or Cartesian IK.'}


def apply_endpoint_joint_offsets(joints, times, profile):
    """Smoothly change only the final section; preserve episode start pose."""
    result = np.asarray(joints, dtype=float).copy()
    if result.ndim != 2 or result.shape[1] != 5:
        raise ValueError('Expected Episode joint array of shape (frames, 5)')
    start = float(profile['blend_start_fraction'])
    duration = float(times[-1] - times[0])
    if duration <= 0:
        raise ValueError('Reference segment duration must be positive')
    u = np.clip((np.asarray(times) - times[0] - start*duration)
                / ((1-start)*duration), 0.0, 1.0)
    eased = u*u*(3-2*u)
    result[:, 3] += eased*profile['j4_wrist_flex_delta_deg']
    result[:, 4] += eased*profile['j5_wrist_roll_delta_deg']
    return result



def emit(out, summary):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:summary.get(k) for k in ('status','stage','reason','max_error_deg','bilateral_hold_steps','min_jaw_distance_mm')}, ensure_ascii=False, indent=2))


def model_build(args):
    spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    spec.compiler.discardvisual=False
    physical.install_dynamic_muhi(spec)
    # A temporary horizontal SUPPORT plane below the upright bottle, whose
    # height is set AFTER choosing the pose; this is NOT the measured workcell.
    spec.worldbody.add_geom(name='phase6c_support', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            pos=[0,0,0],size=[1,1,.1],rgba=[.64,.71,.76,1],
                            contype=1,conaffinity=1,friction=[.9,.01,.001])
    for n in NAMES:
        if spec.joint(n) is None: raise ValueError('Missing joint: '+n)
        a=spec.add_actuator(name='phase6c_pd_'+n,target=n)
        a.trntype=mujoco.mjtTrn.mjTRN_JOINT
        a.dyntype=mujoco.mjtDyn.mjDYN_NONE
        a.gaintype=mujoco.mjtGain.mjGAIN_FIXED
        a.biastype=mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0]=args.arm_kp if n!='gripper' else args.j6_kp
        a.biasprm[1]=-a.gainprm[0]
        a.biasprm[2]=-(args.arm_kd if n!='gripper' else args.j6_kd)
        cap=args.arm_torque if n!='gripper' else args.j6_torque
        a.forcelimited=True;a.forcerange[:]=[-cap,cap]
        a.ctrllimited=False
    model=spec.compile()
    model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.timestep=args.dt
    qadr=[];vadr=[];aids=[];low=[];high=[]
    for n in NAMES:
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
        aid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'phase6c_pd_'+n)
        if j<0 or aid<0:raise ValueError('Joint/actuator not found: '+n)
        qadr.append(int(model.jnt_qposadr[j]));vadr.append(int(model.jnt_dofadr[j]));aids.append(aid)
        low.append(float(model.jnt_range[j,0])+.0001)
        high.append(float(model.jnt_range[j,1])-.0001)
    qadr=np.array(qadr);vadr=np.array(vadr);aids=np.array(aids)
    # Isolate contacts: only finger->bottle and bottle->support. Preserve
    # verified original finger collision geometry. No artificial proxy jaws.
    fixed=grasp.owned_collision_geoms(model,grasp.FINGER_BODY)
    moving=grasp.owned_collision_geoms(model,grasp.MOVING_BODY)
    bottle=tuple(physical.geom_id(model,n) for n in physical.DYNAMIC_GEOMS)
    floor=physical.geom_id(model,'phase6c_support')
    if not fixed or not moving:raise ValueError('Finger collision mesh not found')
    model.geom_contype[:]=0;model.geom_conaffinity[:]=0
    # Collision bitmasks: jaw <-> bottle, bottle <-> support only.
    # Jaw <-> support and robot self collision remain OFF in Phase6C.
    for g in (*fixed,*moving):
        model.geom_contype[g]=2;model.geom_conaffinity[g]=1
    for g in bottle:
        model.geom_contype[g]=1;model.geom_conaffinity[g]=2|4
    model.geom_contype[floor]=4;model.geom_conaffinity[floor]=1
    return model,qadr,vadr,aids,np.asarray(low),np.asarray(high),fixed,moving,bottle,floor


def run(args):
    if args.out.exists():raise FileExistsError(args.out)
    if args.dt<=0 or args.approach_fraction<=0 or args.fps<1:raise ValueError('Invalid options')
    ep,joints,times,seg=phase6b.load_reference(args.dataset_root,args.episode,args.approach_fraction)
    profile = load_approach_profile(args.profile_config, args.approach_profile)
    joints = apply_endpoint_joint_offsets(joints, times, profile)
    model,qadr,vadr,aids,low,high,fixed,moving,bottle,floor=model_build(args)
    gripper_open=high[-1]
    gripper_closed=low[-1]
    target_raw=np.deg2rad(np.column_stack([joints,np.full(len(joints),math.degrees(gripper_open))]))
    target=np.clip(target_raw,low,high)
    clip=np.max(np.abs(np.rad2deg(target_raw-target)),axis=0)
    meta={'experiment':'phase6c_episode4_upright_muhi_approach_then_J6',
          'status':'PREFLIGHT','stage':'geometry','simulation_only':True,
          'source_episode':args.episode,'approach_segment':seg,
          'approach_profile':profile,
          'muhi_upright_world_Z':True,'muhi_placement':'fixed before first physics step, selected from final recorded jaw geometry',
          'object_model':'provisional 50g cylindrical Muhi; URDF finger collision meshes',
          'integrator':'implicitfast','dt_s':args.dt,
          'arm_pd':[args.arm_kp,args.arm_kd,args.arm_torque],
          'j6_pd':[args.j6_kp,args.j6_kd,args.j6_torque],
          'self_collision':'off','robot_floor_collision':'off', 'jaw_jaw_collision':'off',
          'bottle_floor_and_finger_contact':'on',
          'real_table_and_laying_episode4_layout':'not reused',
          'joint_mapping':'raw recorded degrees; uncalibrated; optional J4/J5 endpoint offsets',
          'limit_clipping_deg':dict(zip(NAMES,clip.tolist())),
          'assumptions':['The Episode4 approach target was recorded with a lying object, so this does not prove upright pre-grasp alignment.',
                         'World support plane is virtual, placed at bottle bottom after preflight.',
                         'No object or joint qpos reset/teleport during time stepping.',
                         'No pickup or Lift has been attempted.'],
          'warnings':[]}
    args.out.mkdir(parents=True,exist_ok=False)
    if float(np.max(clip[:5]))>5:
        meta.update(status='GEOMETRY_NOT_READY',reason='Recorded joint angles outside URDF limits >5 deg')
        emit(args.out,meta);return
    d=mujoco.MjData(model)
    mujoco.mj_resetData(model,d)
    d.qpos[qadr]=target[-1];d.qvel[:]=0
    mujoco.mj_forward(model,d)
    free_joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    bq=int(model.jnt_qposadr[free_joint]);bd=int(model.jnt_dofadr[free_joint])
    candidate,diag=grasp.select_bench_pose(model,d,bq,gripper_open,gripper_closed,
                                            qadr[-1],fixed,moving,bottle,
                                            min_fixed_gap_m=.0005,max_fixed_gap_m=.007)
    meta['endpoint_candidate_search']=diag
    if candidate is None:
        meta.update(status='UPRIGHT_ALIGNMENT_NOT_FOUND',reason='Recorded endpoint wrist/hand pose is not a compatible upright two-jaw grasp with current geometry. Needs an IK pregrasp pose, not arbitrary target teleport.')
        emit(args.out,meta);return
    center,quat,gf,gm,local,label,cross_angle,cross_gap=candidate
    meta['candidate_world_xyz_m']=center.tolist()
    meta['candidate_local_xyz_m']=list(local)
    meta['candidate_world_quat_wxyz']=quat.tolist()
    meta['candidate_open_jaw_gap_mm']={'fixed':gf*1000,'moving':gm*1000}
    meta['predicted_contact_angle_deg']=math.degrees(cross_angle)
    # 97 mm bottle: support plane under bottom of assembled two cylinders.
    bottom_z=float(center[2])-physical.MUHI_LENGTH/2.
    if bottom_z < args.minimum_floor_z:
        meta.update(status='UPRIGHT_ALIGNMENT_NOT_FOUND',reason=f'Candidate bottle bottom z={bottom_z:.4f}m below allowed support plane minimum {args.minimum_floor_z:.4f}m; need IK target or corrected scene calibration.')
        emit(args.out,meta);return
    model.geom_pos[floor,2]=bottom_z
    meta['virtual_support_z_m']=bottom_z
    d.qpos[qadr]=target[0];d.qvel[:]=0
    d.qpos[bq:bq+3]=center;d.qpos[bq+3:bq+7]=quat
    d.ctrl[aids]=target[0]
    mujoco.mj_forward(model,d)
    before_contact=sum(1 for c in d.contact[:d.ncon] if set((int(c.geom1),int(c.geom2))) & set(bottle))
    meta['initial_bottle_related_contact_count']=before_contact
    # Any initial collision of the bottle with the approaching gripper is a
    # geometry failure. Floor-only support contact is expected.
    initial_f,initial_m,initial_depth=grasp.jaw_contact_counts(model,d,bottle,fixed,moving)
    if initial_f or initial_m:
        meta.update(status='INITIAL_JAW_BOTTLE_OVERLAP',reason='Episode4 starting fingers already touch the candidate standing bottle; move the pre-grasp pose rather than forcing it.')
        emit(args.out,meta);return
    # Preflight: test *contact free* approach with the bottle present in the
    # same location throughout. No controller hand-over at approach end.
    pregrasp_hold=args.pregrasp_hold_s
    closing=args.closing_s
    retention=args.retention_s
    total=times[-1]+pregrasp_hold+closing+retention
    steps=int(math.ceil(total/args.dt))
    rt=max(1,int(round(1/(args.dt*args.fps))))
    rows=[];frames=[]
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,480)
    model.vis.global_.offheight=max(model.vis.global_.offheight,360)
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=[.1,0,float(center[2])];cam.distance=.7;cam.azimuth=120;cam.elevation=-25
    ffmpeg=shutil.which('ffmpeg')
    renderer=mujoco.Renderer(model,width=480,height=360) if ffmpeg else None
    deep_pen=False;numeric_fail=False
    try:
        for k in range(steps+1):
            clock=min(k*args.dt,total)
            approach_time=min(clock,float(times[-1]))
            arm=np.array([np.interp(approach_time,times,target[:,j]) for j in range(5)])
            if clock<=times[-1]+pregrasp_hold:
                goal_j6=gripper_open;phase='approach' if clock<=times[-1] else 'hold'
            elif clock<times[-1]+pregrasp_hold+closing:
                u=(clock-times[-1]-pregrasp_hold)/closing
                goal_j6=gripper_open+u*(gripper_closed-gripper_open);phase='close'
            else:
                goal_j6=gripper_closed;phase='retain'
            ctl=np.r_[arm,goal_j6]
            d.ctrl[aids]=ctl
            if k:mujoco.mj_step(model,d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                numeric_fail=True;break
            f,m,depth=grasp.jaw_contact_counts(model,d,bottle,fixed,moving)
            if depth is not None and depth*1000 < -args.deep_penetration_mm:deep_pen=True
            q=np.asarray(d.qpos[qadr]).copy()
            err=np.rad2deg(ctl-q)
            row={'step':k,'time_s':float(d.time),'phase':phase,
                 'fixed_jaw_contacts':f,'moving_jaw_contacts':m,
                 'bilateral_contact':int(f>0 and m>0),
                 'min_jaw_distance_mm':depth*1000 if depth is not None else '',
                 'bottle_z_m':float(d.qpos[bq+2]),
                 'bottle_horizontal_displacement_mm':1000*float(np.linalg.norm(d.qpos[bq:bq+2]-center[:2])),
                 'bottle_angular_velocity_rad_s':float(np.linalg.norm(d.qvel[bd+3:bd+6]))}
            for i,n in enumerate(NAMES):
                row[n+'_target_deg']=math.degrees(ctl[i]);row[n+'_actual_deg']=math.degrees(q[i]);row[n+'_error_deg']=float(err[i])
            rows.append(row)
            if renderer is not None and (k%rt==0 or k==steps):
                renderer.update_scene(d,camera=cam)
                pic=Image.fromarray(renderer.render().copy())
                panel=Image.new('RGB',(480,390),(20,27,38));panel.paste(pic,(0,30))
                ImageDraw.Draw(panel).text((10,8),f'Phase6C UPRIGHT | {phase} | {clock:.2f}s | contacts F{f}/M{m}',fill='white')
                frames.append(panel)
    finally:
        if renderer is not None:renderer.close()
    if rows:
        with (args.out/'joint_trace.csv').open('w',newline='',encoding='utf-8') as fp:
            w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        seconds=[r['time_s'] for r in rows]
        for n in r2s.JOINTS:
            axes[0].plot(seconds,[r[n+'_error_deg'] for r in rows],label=n,linewidth=1)
        axes[0].set_ylabel('J1-J5 tracking error (deg)')
        axes[0].legend(ncol=3,fontsize=8)
        axes[1].plot(seconds,[r['gripper_error_deg'] for r in rows],label='J6 error')
        axes[1].plot(seconds,[r['bilateral_contact'] for r in rows],label='bilateral contact (0/1)')
        axes[1].set_ylabel('J6 error / bilateral contact')
        axes[1].legend(fontsize=9)
        axes[2].plot(seconds,[r['bottle_horizontal_displacement_mm'] for r in rows],label='Muhi XY displacement')
        axes[2].set_ylabel('Muhi displacement (mm)')
        axes[2].set_xlabel('Simulation time (s)')
        for ax in axes:ax.grid(alpha=.2)
        approach_end=float(times[-1])
        close_start=approach_end+pregrasp_hold
        for ax in axes:
            ax.axvline(approach_end,ls=':',color='gray')
            ax.axvline(close_start,ls='--',color='gray')
        fig.tight_layout()
        fig.savefig(args.out/'tracking_and_contact.png',dpi=160)
        plt.close(fig)
    if frames and ffmpeg:
        frame_dir=args.out/'frames';frame_dir.mkdir()
        for k,pic in enumerate(frames):pic.save(frame_dir/f'{k:05d}.png')
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-framerate',str(args.fps),'-i',str(frame_dir/'%05d.png'),'-c:v','libx264','-profile:v','baseline','-pix_fmt','yuv420p','-movflags','+faststart',str(args.out/'upright_reach_grasp.mp4')],check=True)
        shutil.rmtree(frame_dir)
    release=[r for r in rows if r['phase']=='retain']
    bilateral=sum(r['bilateral_contact'] for r in release)
    maxerr={n:max(abs(r[n+'_error_deg']) for r in rows) for n in NAMES} if rows else {}
    depths=[float(r['min_jaw_distance_mm']) for r in rows if r['min_jaw_distance_mm']!='']
    meta.update(stage='completed',status=('NUMERIC_FAILURE' if numeric_fail else
                   'DEEP_PENETRATION' if deep_pen else
                   'BILATERAL_CONTACT_OBSERVED' if bilateral>0 else
                   'REACH_GRASP_NO_BILATERAL_CONTACT'),
                max_error_deg=maxerr,
                bilateral_hold_steps=bilateral,
                hold_steps=len(release),
                min_jaw_distance_mm=min(depths) if depths else None,
                final_bottle_horizontal_displacement_mm=rows[-1]['bottle_horizontal_displacement_mm'] if rows else None,
                final_bottle_z_m=rows[-1]['bottle_z_m'] if rows else None,
                frames=len(rows),video='upright_reach_grasp.mp4' if frames and ffmpeg else None)
    # Contact alone does not prove safe retention, and this test does NOT lift.
    meta['note']='Bilateral contact observed != validated grasp. No lift; check bottle motion and contact forces before PASS.'
    emit(args.out,meta)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--urdf',type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episode',type=int,default=4)
    p.add_argument('--approach-fraction',type=float,default=.8)
    p.add_argument('--profile-config',type=Path,default=DEFAULT_PROFILE_CONFIG)
    p.add_argument('--approach-profile',default='baseline',
                   help='Named J4/J5 endpoint adjustment; not a world-frame approach angle')
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--fps',type=float,default=8)
    p.add_argument('--minimum-floor-z',type=float,default=-.05)
    p.add_argument('--arm-kp',type=float,default=18)
    p.add_argument('--arm-kd',type=float,default=1.8)
    p.add_argument('--arm-torque',type=float,default=3)
    p.add_argument('--j6-kp',type=float,default=2)
    p.add_argument('--j6-kd',type=float,default=.08)
    p.add_argument('--j6-torque',type=float,default=.4)
    p.add_argument('--pregrasp-hold-s',type=float,default=.35)
    p.add_argument('--closing-s',type=float,default=1.3)
    p.add_argument('--retention-s',type=float,default=.8)
    p.add_argument('--deep-penetration-mm',type=float,default=3)
    args=p.parse_args()
    run(args)

if __name__=='__main__':main()
