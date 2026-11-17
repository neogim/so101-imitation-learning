#!/usr/bin/env python3
"""Phase 6F (simulation only): prove static J6 bilateral hold FIRST,
then reverse-plan collision-free retreat and replay approach to the same pose.

No original Episode 4 approach, no joint qpos teleport during physics, no lift.
The object starts at the same benchmark XYZ in each independent physics trial.
"""
from __future__ import annotations
import argparse, csv, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import phase6c_upright_reach_grasp as phase6c
import physical_grasp_j6_bench as grasp
import physical_workcell_sim as physical

NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')


def save(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=float)+'\n',encoding='utf-8')


def tip_fk(model, data, qadr, gripper_bid, q):
    data.qpos[qadr[:5]]=q
    mujoco.mj_forward(model,data)
    return np.asarray(data.xpos[gripper_bid]).copy(),np.asarray(data.xmat[gripper_bid]).reshape(3,3).copy()


def ease(a,b,T, fps=20):
    n=max(25,int(round(T*fps)))
    u=np.linspace(0.,1.,n+1)
    s=u**3*(10-15*u+6*u*u)
    return a[None,:]+s[:,None]*(b-a)[None,:],u*T


def physics_trial(args, case_dir, joints_deg, times, xyz, support_z, tag):
    case_dir.mkdir(parents=True,exist_ok=True)
    np.savez(case_dir/'arm_trajectory.npz',joints_deg=joints_deg,times_s=times)
    output=case_dir/'physical'
    cmd=[sys.executable,'-u',str(Path(phase6c.__file__).resolve()),
         '--dataset-root',str(args.dataset_root),'--urdf',str(args.urdf),
         '--episode',str(args.episode),'--out',str(output),
         '--arm-trajectory-npz',str(case_dir/'arm_trajectory.npz'),
         '--fixed-bottle-xyz-m',*(str(float(x)) for x in xyz),
         '--support-plane-z-m',str(float(support_z)),
         '--minimum-floor-z',str(args.minimum_floor_z),
         '--fps',str(args.fps),
         '--arm-kp',str(args.arm_kp),'--arm-kd',str(args.arm_kd),
         '--arm-torque',str(args.arm_torque),
         '--j6-kp',str(args.j6_kp),'--j6-kd',str(args.j6_kd),
         '--j6-torque',str(args.j6_torque),
         '--closing-s',str(args.closing_s),'--retention-s',str(args.retention_s)]
    env=os.environ.copy();env.setdefault('MUJOCO_GL','osmesa')
    proc=subprocess.run(cmd,text=True,capture_output=True,
                        cwd=str(Path(phase6c.__file__).parent),env=env)
    (case_dir/'console.log').write_text('COMMAND: '+' '.join(cmd)+'\n'+proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    try: summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    except (OSError,ValueError): summary={'status':'EXECUTION_ERROR','reason':(proc.stderr or proc.stdout)[-4000:]}
    if proc.returncode:
        summary['status']='EXECUTION_ERROR';summary['returncode']=proc.returncode
    summary['phase6f_trial']=tag
    save(case_dir/'trial_result.json',summary)
    return summary


def find_retreat(model, data, qadr, wrist_bid, q_target, low, high, xyz,
                 bq, fixed, moving, bottle, retreat_m):
    """Reverse-plan from verified grasp. Only open-jaw paths are assessed."""
    wrist_xyz,R=tip_fk(model,data,qadr,wrist_bid,q_target)
    local_to_bottle=xyz-wrist_xyz
    local_to_bottle/=max(np.linalg.norm(local_to_bottle),1e-12)
    candidates=[('back_from_object',-local_to_bottle),
                ('up_world_z',np.array([0.,0.,1.])),
                ('back_and_up',(-local_to_bottle+np.array([0.,0.,.8])) /
                 np.linalg.norm(-local_to_bottle+np.array([0.,0.,.8])))]
    out=[]
    for label,direction in candidates:
        target=wrist_xyz+retreat_m*direction
        def residual(q):
            p,r=tip_fk(model,data,qadr,wrist_bid,q)
            rerr=Rotation.from_matrix(R@r.T).as_rotvec()
            return np.r_[(p-target)/.004, rerr/.25,
                          (q-q_target)/1.5*.2,
                          (q[3:]-q_target[3:])/.25*.35]
        sol=least_squares(residual,np.clip(q_target,low+1e-6,high-1e-6),
                          bounds=(low,high),max_nfev=150)
        q=sol.x
        pos,rot=tip_fk(model,data,qadr,wrist_bid,q)
        xyzerr=float(np.linalg.norm(pos-target)*1000)
        # Reverse from grasp to retreat, then reverse exactly for approach.
        traj,t=ease(np.rad2deg(q_target),np.rad2deg(q),args_placeholder_retreat_s)
        d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
        d.qpos[bq:bq+3]=xyz;d.qpos[bq+3:bq+7]=[1,0,0,0]
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'gripper')
        d.qpos[qadr[5]]=model.jnt_range[jid,1]-1e-4
        min_gap=1e9;min_index=-1
        for i,q_deg in enumerate(traj):
            d.qpos[qadr[:5]]=np.deg2rad(q_deg);mujoco.mj_forward(model,d)
            gap=min(grasp.geometry_gaps(model,d,bottle,fixed),
                    grasp.geometry_gaps(model,d,bottle,moving))
            if gap<min_gap:min_gap,min_index=gap,i
        first_gap=min(grasp.geometry_gaps(model,d,bottle,fixed),
                      grasp.geometry_gaps(model,d,bottle,moving))
        out.append({'direction':label,'retreat_xyz':target.tolist(),
                    'q_pregrasp_deg':np.rad2deg(q).tolist(),
                    'retreat_tip_error_mm':xyzerr,
                    'minimum_open_jaw_gap_mm':1000*min_gap,
                    'minimum_gap_index':min_index,
                    'sample_count':len(traj),
                    'feasible':bool(xyzerr<8 and min_gap>=-.0005 and sol.success),
                    'q':q,'trajectory':traj,'time':t})
    return out


# fixed to 2.0 seconds in the purely kinematic reverse/forward trajectory.
args_placeholder_retreat_s=2.0


def run(args):
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    manifest={'schema':'phase6f_grasp_first_reverse_v1','simulation_only':True,
              'phase':'bench_grasp_first','status':'INITIALIZED',
              'goal':'Find physically verified fixed-arm grasp, then reverse-retreat and replay',
              'no_lift':True,
              'contact_provenance':'2026-11-15 bench bilateral contacts observed with assistance, not proven dynamic pickup',
              'warnings':[]}
    model,qadr,vadr,act,low,high,fixed,moving,bottle,floor=phase6c.model_build(args)
    data=mujoco.MjData(model);mujoco.mj_resetData(model,data)
    q_bench=np.zeros(5);q_bench[4]=np.clip(-math.pi/2,low[4],high[4])
    data.qpos[qadr[:5]]=q_bench;data.qpos[qadr[5]]=high[5]
    mujoco.mj_forward(model,data)
    free=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    bq=int(model.jnt_qposadr[free])
    wrist_bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,grasp.FINGER_BODY)
    candidate,diagnostics=grasp.select_bench_pose(model,data,bq,high[5],low[5],qadr[5],fixed,moving,bottle)
    manifest['bench_search']=diagnostics
    if candidate is None:
        manifest.update(status='NO_BENCH_GEOMETRY',reason='No collision-free two-jaw closing geometry found')
        save(args.out/'summary.json',manifest);return manifest
    xyz=np.asarray(candidate[0],dtype=float)
    support_z=float(xyz[2]-physical.MUHI_LENGTH/2)
    manifest.update(bench_joints_deg=np.rad2deg(q_bench).tolist(),
                    bottle_xyz_m=xyz.tolist(),support_z_m=support_z,
                    note='Fixed bottle world coordinates come from simulated bench geometry, not measurements')
    # 1: A stationary arm with independent J6 closing. Must genuinely contact in physics.
    qdeg=np.rad2deg(q_bench)
    joints=np.stack([qdeg,qdeg]);times=np.array([0.,args.bench_hold_s])
    grasp_trial=physics_trial(args,args.out/'01_fixed_arm_J6',joints,times,xyz,support_z,'fixed_arm_grasp_only')
    hold_steps=int(grasp_trial.get('bilateral_hold_steps') or 0)
    xy=float(grasp_trial.get('final_bottle_horizontal_displacement_mm') or 0)
    manifest['fixed_grasp_trial']={'status':grasp_trial.get('status'),
        'bilateral_hold_steps':hold_steps,'horizontal_displacement_mm':xy,
        'support_method':grasp_trial.get('support_geometry_method'),
        'last_bottle_z_m':grasp_trial.get('final_bottle_z_m')}
    # The contact hold must be independently reproduced with arm physically held
    # before building any moving-arm trajectory. Deliberately no arbitrary 0.5° abort.
    required=max(25,round(args.retention_s/args.dt*.25))
    if (grasp_trial.get('status') in ('EXECUTION_ERROR','GEOMETRY_NOT_READY','NUMERIC_FAILURE')
            or hold_steps<required or xy>20):
        manifest.update(status='FIXED_ARM_GRASP_NOT_ESTABLISHED',
          reason='First reproduce stable two-jaw contact with dynamic Muhi and compile-time support; no trajectory planning yet',
          required_bilateral_hold_steps=int(required))
        save(args.out/'summary.json',manifest);return manifest
    manifest['phase']='reverse_retreat_planning'
    # 2: Invert the trajectory: solve a nearby, collision-free pregrasp away from bottle.
    global args_placeholder_retreat_s
    args_placeholder_retreat_s=args.retreat_s
    routes=find_retreat(model,data,qadr,wrist_bid,q_bench,low[:5],high[:5],xyz,
                        bq,fixed,moving,bottle,args.retreat_m)
    safe=[]
    for r in routes:
        simple={k:v for k,v in r.items() if k not in ('q','trajectory','time')}
        (args.out/f"retreat_{r['direction']}.json").write_text(json.dumps(simple,indent=2)+'\n')
        if r['feasible']:safe.append(r)
    manifest['retreat_candidates']=[{k:v for k,v in r.items() if k not in ('q','trajectory','time')} for r in routes]
    if not safe:
        manifest.update(status='NO_CLEAR_REVERSE_RETREAT',
                        reason='Fixed-arm bilateral contact established, but no tested collision-free retreat; keep grasp result')
        save(args.out/'summary.json',manifest);return manifest
    best=min(safe,key=lambda c:(c['retreat_tip_error_mm'], -c['minimum_open_jaw_gap_mm']))
    manifest['chosen_retreat_direction']=best['direction']
    # 3: Replay exact reverse of retreat trajectory, qpregrasp -> qbench.
    forward=best['trajectory'][::-1].copy()
    # Start state is open jaw at qpregrasp; approach then close J6 at qbench.
    # Phase6C/6F runner never teleports joints or object during stepping.
    t=np.linspace(0.,args.approach_s,len(forward))
    approach=physics_trial(args,args.out/'02_retreat_reverse_approach',forward,t,xyz,support_z,'replay_approach_then_J6')
    manifest['approach_trial']={'status':approach.get('status'),
        'bilateral_hold_steps':approach.get('bilateral_hold_steps'),
        'bottle_horizontal_displacement_mm':approach.get('final_bottle_horizontal_displacement_mm'),
        'support_method':approach.get('support_geometry_method'),
        'max_joint_error_deg':approach.get('max_error_deg')}
    manifest['status']='APPROACH_CONTACT_OBSERVED' if (approach.get('bilateral_hold_steps') or 0)>=required else 'APPROACH_GRASP_NOT_REPRODUCED'
    manifest['note']='Contact hold in both trials does NOT establish Lift/pick success; evaluate retained object movement before claiming pickup.'
    save(args.out/'summary.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True,type=Path)
    p.add_argument('--urdf',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--episode',type=int,default=4)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--approach-fraction',type=float,default=.8)
    p.add_argument('--arm-kp',type=float,default=18)
    p.add_argument('--arm-kd',type=float,default=1.8)
    p.add_argument('--arm-torque',type=float,default=3)
    p.add_argument('--j6-kp',type=float,default=2)
    p.add_argument('--j6-kd',type=float,default=.08)
    p.add_argument('--j6-torque',type=float,default=.4)
    p.add_argument('--closing-s',type=float,default=1.3)
    p.add_argument('--retention-s',type=float,default=1.0)
    p.add_argument('--bench-hold-s',type=float,default=.5)
    p.add_argument('--retreat-m',type=float,default=.05)
    p.add_argument('--retreat-s',type=float,default=2.)
    p.add_argument('--approach-s',type=float,default=3.)
    p.add_argument('--minimum-floor-z',type=float,default=-.05)
    args=p.parse_args()
    if not 0.02 <= args.retreat_m <= .12:p.error('--retreat-m must be 0.02..0.12')
    if args.dt<=0 or args.approach_s<=0:p.error('Invalid timing')
    result=run(args)
    print(json.dumps({'status':result['status'],'out':str(args.out)},ensure_ascii=False))

if __name__=='__main__':main()
