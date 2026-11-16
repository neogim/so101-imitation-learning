#!/usr/bin/env python3
"""Phase 6B: offline, physical PD tracking of recorded SO-101 Episode 4 arm approach.

Uses LeRobot v3 observation.state J1..J5 (degrees) as a recorded kinematic
reference. J6 is held open: NO grasp, no bottle contact, no real hardware.
Model dynamics (inertia, gravity, PD actuation, actuator limits) run with mj_step.
Self collision is disabled to isolate the control instability seen in Phase6A.
Camera is a diagnostic free camera, NOT the calibrated physical wrist camera.
"""
from __future__ import annotations
import argparse, csv, json, math, shutil, subprocess
from pathlib import Path

import numpy as np
import mujoco
import real_to_sim as r2s

NAMES = (*r2s.JOINTS, 'gripper')


def load_reference(root: Path, episode: int, fraction: float):
    ep = r2s.read_episode(root, episode, stride=1, max_frames=100000)
    rows = ep['rows']
    if len(rows) < 15:
        raise ValueError('Episode too short for approach analysis')
    states = np.array([r['state'] for r in rows], dtype=float)
    raw_t = np.array([r['timestamp'] for r in rows], dtype=float)
    if np.any(np.diff(raw_t) <= 0):
        raise ValueError('Non-monotonic recording timestamps')
    gripper = states[:, 5]
    initial_j6 = float(np.median(gripper[:5]))
    # First sustained gripper change is a *candidate* grasp onset, not verified.
    moved = np.abs(gripper - initial_j6) > 12.0  # LeRobot normalized J6 units
    onset = next((i for i in range(20, len(rows)-3) if moved[i:i+4].all()), None)
    if onset is not None:
        end = max(14, onset - 2)
        segment_mode = 'before_first_sustained_J6_change_heuristic'
    else:
        end = max(14, min(len(rows)-1, int((len(rows)-1)*fraction)))
        segment_mode = 'fraction_fallback_no_clear_J6_onset'
    end = min(end, len(rows)-1)
    target = states[:end+1, :5]
    t = raw_t[:end+1] - raw_t[0]
    if not (0.3 <= t[-1] <= 40):
        raise ValueError(f'Unexpected approach duration {t[-1]:.3f}s')
    return ep, target, t, {
        'segment_mode': segment_mode,
        'episode_frames_total': len(rows),
        'approach_frames': len(t),
        'recording_fps': ep['info']['fps'],
        'start_frame': 0, 'end_frame_inclusive': end,
        'candidate_j6_onset_frame': onset,
        'normalized_j6_at_start': initial_j6,
        'duration_s': float(t[-1]),
        'recorded_joint_range_deg': {n: float(np.ptp(target[:,i])) for i,n in enumerate(r2s.JOINTS)},
        'source_parquet': ep['source_parquet'],
    }


def compile_robot(urdf, kp, kd, torque, dt, integrator, ref_first):
    spec = mujoco.MjSpec.from_file(str(urdf.resolve(strict=True)))
    spec.compiler.discardvisual = False
    for n in NAMES:
        if spec.joint(n) is None:
            raise ValueError(f'Missing joint {n} in URDF')
        a = spec.add_actuator(name=f'phase6b_pd_{n}', target=n)
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = kp
        a.biasprm[1] = -kp
        a.biasprm[2] = -kd
        a.ctrllimited = False
        a.forcelimited = True
        a.forcerange[:] = [-torque, torque]
    model = spec.compile()
    # Exactly the Phase 6A isolated-actuator contact setting.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.opt.timestep = dt
    model.opt.integrator = {'euler':mujoco.mjtIntegrator.mjINT_EULER,
                            'implicitfast':mujoco.mjtIntegrator.mjINT_IMPLICITFAST}[integrator]
    qadr=[]; vadr=[]; actuator=[]; low=[]; high=[]
    for n in NAMES:
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
        a=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,f'phase6b_pd_{n}')
        if j < 0 or a < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f'Unexpected joint/actuator {n}')
        qadr.append(int(model.jnt_qposadr[j]));vadr.append(int(model.jnt_dofadr[j]));actuator.append(a)
        if model.jnt_limited[j]:
            low.append(float(model.jnt_range[j,0])+1e-4)
            high.append(float(model.jnt_range[j,1])-1e-4)
        else:
            low.append(-math.inf);high.append(math.inf)
    return model, np.array(qadr), np.array(vadr), np.array(actuator), np.array(low), np.array(high)


def run_trial(args, label, targets_deg, t_ref, out, make_video=False):
    model,qadr,vadr,act,low,high=compile_robot(args.urdf,args.kp,args.kd,args.max_torque_nm,
                                               args.dt,label,targets_deg[0])
    open_j6_deg = args.j6_open_deg
    requested=np.deg2rad(np.column_stack([targets_deg,np.full(len(t_ref),open_j6_deg)]))
    clamped=np.clip(requested,low,high)
    clipping=np.rad2deg(np.max(np.abs(requested-clamped),axis=0))
    # Prevent silently replacing recorded robot motion with a clipped trajectory.
    if np.max(clipping[:5]) > 5:
        raise ValueError('Recorded J1-J5 differ from URDF joint limits by >5deg; check calibration/sign/offset: '
                         +repr(dict(zip(NAMES,clipping.round(2).tolist()))))
    data=mujoco.MjData(model)
    mujoco.mj_resetData(model,data)
    data.qpos[qadr] = clamped[0]
    data.qvel[vadr] = 0
    data.ctrl[act] = clamped[0]
    mujoco.mj_forward(model,data)
    target_data=mujoco.MjData(model)
    mujoco.mj_resetData(model,target_data)
    target_data.qpos[qadr] = clamped[0]
    mujoco.mj_forward(model,target_data)
    out.mkdir(parents=True,exist_ok=False)
    rows=[];images=[]
    nsteps=int(math.ceil(float(t_ref[-1])/args.dt))
    # Render only a few frames per second. Both panels use the SAME MuJoCo model.
    render_interval=max(1,int(round(1/(args.fps*args.dt))))
    renderer=None
    if make_video:
        from PIL import Image, ImageDraw
        width,height=420,320
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,width)
        model.vis.global_.offheight=max(model.vis.global_.offheight,height)
        renderer=mujoco.Renderer(model,width=width,height=height)
        cam=mujoco.MjvCamera()
        cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]=[.12,0,.15]
        cam.distance=.66
        cam.azimuth=135
        cam.elevation=-25
    finite=True
    try:
        for step in range(nsteps+1):
            time=min(step*args.dt,float(t_ref[-1]))
            target=np.array([np.interp(time,t_ref,clamped[:,i]) for i in range(6)])
            data.ctrl[act]=target
            if step:
                mujoco.mj_step(model,data)
            q=np.asarray(data.qpos[qadr]).copy()
            vel=np.asarray(data.qvel[vadr]).copy()
            if not(np.isfinite(q).all() and np.isfinite(vel).all()):
                finite=False
                break
            trace={'time_s':float(data.time),'step':step,'ncon':int(data.ncon)}
            for i,n in enumerate(NAMES):
                trace[f'{n}_target_deg']=float(np.rad2deg(target[i]))
                trace[f'{n}_actual_deg']=float(np.rad2deg(q[i]))
                trace[f'{n}_error_deg']=float(np.rad2deg(target[i]-q[i]))
                trace[f'{n}_velocity_deg_s']=float(np.rad2deg(vel[i]))
            rows.append(trace)
            if renderer is not None and (step%render_interval==0 or step==nsteps):
                target_data.qpos[qadr]=target
                target_data.qvel[:]=0
                mujoco.mj_forward(model,target_data)
                renderer.update_scene(target_data,camera=cam)
                left=Image.fromarray(renderer.render().copy())
                renderer.update_scene(data,camera=cam)
                right=Image.fromarray(renderer.render().copy())
                panel=Image.new('RGB',(width*2,height+42),(19,26,36))
                panel.paste(left,(0,42));panel.paste(right,(width,42))
                dr=ImageDraw.Draw(panel)
                dr.text((8,8),f'RECORDED TARGET | Episode {args.episode}',fill='white')
                dr.text((width+8,8),f'MUJOCO PHYSICAL | {label} | {time:.2f}s',fill='white')
                images.append(panel)
    finally:
        if renderer is not None: renderer.close()
    if not rows:
        raise RuntimeError('No numeric trajectory frames')
    with (out/'joint_trace.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    max_error={n:float(max(abs(r[f'{n}_error_deg']) for r in rows)) for n in NAMES}
    rms_error={n:float(np.sqrt(np.mean([r[f'{n}_error_deg']**2 for r in rows]))) for n in NAMES}
    # For diagnostics only: no arbitrary 0.5degree stop or actuator cutoff.
    summary={'status':'COMPLETED_TRACKING_UNVERIFIED' if finite else 'NUMERIC_FAILURE',
             'integrator':label,'simulation_only':True,'dt_s':args.dt,
             'kp':args.kp,'kd':args.kd,'max_torque_nm':args.max_torque_nm,
             'max_error_deg':max_error,'rms_error_deg':rms_error,
             'max_error_all_deg':max(max_error.values()),'steps':len(rows),
             'physics_simulated_s':float(rows[-1]['time_s']),
             'recorded_approach_s':float(t_ref[-1]),
             'initial_joint_limit_clipping_deg':dict(zip(NAMES,clipping.round(3).tolist())),
             'J6_mode':'held at fixed simulated open angle, NOT replayed or grasped',
             'self_collision':'off as in Phase 6A; dynamic robot joints and gravity ON',
             'muhi':'not physically simulated; no contact/grasp in Phase 6B',
             'reference':'LeRobot observation.state J1-J5 raw degrees; calibration unverified',
             'video':'target_vs_physical.mp4' if make_video else None}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if make_video and images:
        frame_dir=out/'frames'
        frame_dir.mkdir()
        for k,img in enumerate(images): img.save(frame_dir/f'{k:05d}.png')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(args.fps),
                        '-i',str(frame_dir/'%05d.png'),'-c:v','libx264','-profile:v','baseline',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'target_vs_physical.mp4')],check=True)
        shutil.rmtree(frame_dir)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--episode',type=int,default=4)
    p.add_argument('--urdf',type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--approach-fraction',type=float,default=.8)
    p.add_argument('--j6-open-deg',type=float,default=60.)
    p.add_argument('--kp',type=float,default=18.)
    p.add_argument('--kd',type=float,default=1.8)
    p.add_argument('--max-torque-nm',type=float,default=3.)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--fps',type=int,default=8)
    a=p.parse_args()
    if not(.1 <= a.approach_fraction<=1 and .0005<=a.dt<=.01 and 1<=a.fps<=20):
        p.error('Invalid approach/fps/dt settings')
    if a.out.exists(): raise FileExistsError(a.out)
    ep,targets,t,segment=load_reference(a.dataset_root,a.episode,a.approach_fraction)
    a.out.mkdir(parents=True)
    (a.out/'segment.json').write_text(json.dumps(segment,indent=2,ensure_ascii=False)+'\n')
    results={}
    for label in ['euler','implicitfast']:
        results[label]=run_trial(a,label,targets,t,a.out/label,make_video=(label=='implicitfast'))
        print(label,results[label]['status'],'max_err_deg=',round(results[label]['max_error_all_deg'],3),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(5,1,figsize=(11,11),sharex=True)
    for i,name in enumerate(r2s.JOINTS):
        for label in results:
            with (a.out/label/'joint_trace.csv').open(newline='') as f:
                rec=list(csv.DictReader(f))
            ax[i].plot([float(x['time_s']) for x in rec],
                       [float(x[f'{name}_error_deg']) for x in rec],label=label,alpha=.9)
        ax[i].set_ylabel(name+'\nerror (deg)');ax[i].grid(alpha=.25)
    ax[0].legend();ax[-1].set_xlabel('Episode 4 time (s)')
    fig.tight_layout();fig.savefig(a.out/'tracking_error.png',dpi=150);plt.close(fig)
    manifest={'status':'SIM_COMPLETED_NOT_FULL_GRASP_VALIDATION','episode':a.episode,
              'segment':segment,'trials':results,
              'next_gate':'Verify moving trajectory and compare target_vs_physical.mp4 before restoring Muhi/contact.'}
    (a.out/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('RESULT_DIR:',a.out,flush=True)
    print('APPROACH_DURATION_S:',segment['duration_s'],flush=True)
    print('RECORDED_JOINT_RANGE_DEG:',segment['recorded_joint_range_deg'],flush=True)

if __name__=='__main__':main()
