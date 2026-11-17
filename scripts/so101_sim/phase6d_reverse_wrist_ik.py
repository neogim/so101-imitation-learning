#!/usr/bin/env python3
"""Phase 6D: fixed upright Muhi; reverse-J4 wrist candidates via MuJoCo FK / IK.

Simulation-only. Reuses Episode4 approach, then builds joint-space transition
onto an IK solution for the SAME world-space bottle in all candidates. Legacy
J4=0, J5=-90 came from a fixed-arm contact test, not proven grasp retention.
No hardware control; dynamic grasp evaluated by phase6c_upright_reach_grasp.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import phase6b_episode4_physical_approach as phase6b
import phase6c_upright_reach_grasp as phase6c
import physical_grasp_j6_bench as grasp
import physical_workcell_sim as physical
import real_to_sim as r2s

HERE = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = HERE / 'configs/so101/phase6d_fixed_muhi.json'
GRIPPER = grasp.FINGER_BODY
JOINTS = r2s.JOINTS


def fk(model, data, qadr, wrist_bid, angles):
    data.qpos[qadr[:5]] = angles
    mujoco.mj_forward(model, data)
    pos = np.asarray(data.xpos[wrist_bid]).copy()
    rot = np.asarray(data.xmat[wrist_bid]).reshape(3, 3).copy()
    return pos, rot


def ik_solution(model, data, qadr, wrist_bid, q_original, low, high,
                bottle_center, anchor_local, j4_deg, j5_deg):
    """Find full 5-DoF joint configuration for a fixed world target and wrist pose.

    Desired wrist rotation comes from setting J4/J5 in the recorded endpoint,
    while the wrist origin is shifted so the SAME local grasp anchor points to
    the SAME bottle center. Orientation is soft (5 DoF cannot span SE(3)).
    """
    seed = np.clip(np.asarray(q_original, dtype=float).copy(), low, high)
    seed[3] = np.clip(math.radians(j4_deg), low[3], high[3])
    seed[4] = np.clip(math.radians(j5_deg), low[4], high[4])
    _, goal_rot = fk(model, data, qadr, wrist_bid, seed)
    goal_origin = bottle_center - goal_rot @ anchor_local

    def residual(q):
        origin, rot = fk(model, data, qadr, wrist_bid, q)
        # 1 cm positional error = 1 residual unit; 0.4 rad angular = 1 unit.
        pos = (origin - goal_origin) / .01
        relative = goal_rot @ rot.T
        orient = Rotation.from_matrix(relative).as_rotvec() / .4
        # Prioritize real bottle alignment, but do not force artificial IK success.
        prior = (q - seed) / 1.25
        j4_prior = np.array([(q[3]-seed[3]) / .18])
        return np.r_[pos, orient, prior*.12, j4_prior*.65]

    result = least_squares(residual, seed, bounds=(low, high), max_nfev=160,
                           xtol=1e-7, ftol=1e-7, gtol=1e-7)
    q = result.x
    pos, rot = fk(model, data, qadr, wrist_bid, q)
    anchor_world = pos + rot @ anchor_local
    residual_xyz_mm = 1000 * (anchor_world - bottle_center)
    rot_deg = math.degrees(np.linalg.norm(Rotation.from_matrix(goal_rot @ rot.T).as_rotvec()))
    return q, {
        'solver_success': bool(result.success), 'solver_message': str(result.message),
        'function_evaluations': int(result.nfev), 'objective_cost': float(result.cost),
        'target_wrist_origin_m': goal_origin.tolist(), 'actual_wrist_origin_m': pos.tolist(),
        'anchor_world_xyz_m': anchor_world.tolist(),
        'bottle_world_xyz_m': bottle_center.tolist(),
        'anchor_position_error_mm': float(np.linalg.norm(residual_xyz_mm)),
        'anchor_error_xyz_mm': residual_xyz_mm.tolist(),
        'wrist_orientation_error_deg': rot_deg,
        'initial_requested_j4_deg': float(j4_deg), 'actual_ik_j4_deg': math.degrees(q[3]),
        'initial_requested_j5_deg': float(j5_deg), 'actual_ik_j5_deg': math.degrees(q[4]),
        'ik_joint_angles_deg': dict(zip(JOINTS, np.rad2deg(q).tolist())),
        'note': '5-DoF soft wrist orientation + fixed world xyz; not full 6D IK and not proof of grasp.'
    }


def generate_trajectory(original_deg, times, end_deg, transition_seconds):
    original_deg = np.asarray(original_deg, dtype=float)
    times = np.asarray(times, dtype=float)
    # Preserve Episode4 recorded approach, then add a smooth 3-s wrist
    # repositioning segment. Do not instantly jump joint angles.
    N = max(35, int(round(transition_seconds * 20)))
    add = np.linspace(0, transition_seconds, N+1)[1:]
    u = add / transition_seconds
    ease = u**3 * (10 - 15*u + 6*u*u)
    end = np.asarray(end_deg, dtype=float)
    tail = original_deg[-1][None, :] + ease[:, None] * (end-original_deg[-1])[None, :]
    return np.vstack([original_deg, tail]), np.r_[times, times[-1]+add]


def sweep_geometry(model, data, qadr, q5, low, high, bottle_xyz,
                   bq, fixed, moving, bottle):
    data.qpos[qadr[:5]] = q5
    data.qpos[qadr[5]] = high[5]
    data.qpos[bq:bq+3] = bottle_xyz
    data.qpos[bq+3:bq+7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    gf = grasp.geometry_gaps(model, data, bottle, fixed)
    gm = grasp.geometry_gaps(model, data, bottle, moving)
    gaps = []
    for j6 in np.linspace(high[5], low[5], 13):
        data.qpos[qadr[5]] = j6
        mujoco.mj_forward(model, data)
        gaps.append(grasp.geometry_gaps(model, data, bottle, moving))
    return {
        'open_gap_fixed_mm': gf*1000, 'open_gap_moving_mm': gm*1000,
        'moving_sweep_min_gap_mm': min(gaps)*1000,
        'kinematic_preflight': ('CANDIDATE_NEAR_JAWS' if -1.0 <= gf*1000 <= 7.0
                               and -1.0 <= gm*1000 <= 10.0 and min(gaps)<-.00035
                               else 'JAW_ALIGNMENT_NOT_CONFIRMED'),
    }


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    cfg = json.loads(args.config.resolve(strict=True).read_text(encoding='utf-8'))
    xyz = np.array(cfg['fixed_bottle_world_xyz_m'], dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('Invalid fixed bottle world coordinates')
    ep, reference, t_ref, segment = phase6b.load_reference(args.dataset_root,args.episode,args.approach_fraction)
    model,qadr,vadr,act,low,high,fixed,moving,bottle,floor = phase6c.model_build(args)
    data = mujoco.MjData(model)
    free_joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    bq=int(model.jnt_qposadr[free_joint])
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,GRIPPER)
    end_reference = np.clip(np.deg2rad(reference[-1]),low[:5],high[:5])
    # Anchor is measured using same baseline's saved Muhi pose and recorded
    # endpoint. The bottle is then held fixed for ALL reverse-wrist candidates.
    end_origin,end_rotation=fk(model,data,qadr,bid,end_reference)
    anchor_local = end_rotation.T @ (xyz-end_origin)
    q_end_deg = np.rad2deg(end_reference)
    j4 = float(q_end_deg[3]); j5=float(q_end_deg[4])
    # -J4 = true bend reversal; historical contact pose uses 0° J4 / -90° J5.
    reverse_j4 = float(np.clip(-j4,math.degrees(low[3])+.01, math.degrees(high[3])-.01))
    historical_j5 = float(np.clip(float(cfg['historical_reference_j5_deg']),
                                  math.degrees(low[4])+.01,math.degrees(high[4])-.01))
    cases = [
        ('reverse_j4',reverse_j4,historical_j5),
        ('historical_j4_zero',float(np.clip(0.,math.degrees(low[3])+.01,math.degrees(high[3])-.01)),historical_j5),
        ('half_reverse_j4',reverse_j4*.5,historical_j5),
    ]
    manifest={'experiment':'phase6d_reverse_wrist_fixed_world_ik',
              'simulation_only':True, 'status':'DIAGNOSTIC_NOT_GRASP_SUCCESS',
              'source_episode':args.episode,'segment':segment,
              'baseline_source':cfg['fixed_bottle_provenance'],
              'fixed_bottle_world_xyz_m':xyz.tolist(),
              'bottle_upright_quat_wxyz':[1,0,0,0],
              'reference_endpoint_joint_deg':dict(zip(JOINTS,q_end_deg.tolist())),
              'historical_j4_j5':{'J4_deg':0,'J5_deg':-90,
                   'qualification':'Fixed-arm CONTACT trial, NOT verified retained grasp'},
              'anchor_local_m':anchor_local.tolist(),
              'notes':['Same physical Muhi xyz across all candidates; dynamic object free during time stepping.',
                       'IK wrist orientation is soft due to 5 arm joints, not 6D pose guarantee.',
                       'Recorded Episode4 assumed lying object. Joint-space transition is not guaranteed collision-free.',
                       'No artificial 0.5degree abort or object teleport; contact metrics must be inspected.',
                       'No lift; bilateral contact is not grasp proof.'], 'results':[]}
    rows=[]
    for i,(name,j4_target,j5_target) in enumerate(cases,1):
        print(f'[{i}/{len(cases)}] {name}: IK toward SAME Muhi pose',flush=True)
        result_dir=args.out/name
        result_dir.mkdir()
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model,data)
        q,diag=ik_solution(model,data,qadr,bid,end_reference,low[:5],high[:5],xyz,
                           anchor_local,j4_target,j5_target)
        diag['profile']=name
        diag['gripper_clearance']=sweep_geometry(model,data,qadr,q,low,high,xyz,bq,fixed,moving,bottle)
        (result_dir/'ik_diagnostics.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n')
        end_deg=np.rad2deg(q)
        joints,times=generate_trajectory(reference,t_ref,end_deg,args.transition_s)
        np.savez(result_dir/'ik_trajectory.npz',joints_deg=joints,times_s=times)
        cmd=[sys.executable,'-u',str(Path(phase6c.__file__).resolve()),
             '--dataset-root',str(args.dataset_root),'--urdf',str(args.urdf),
             '--out',str(result_dir/'physical'),
             '--episode',str(args.episode),
             '--arm-trajectory-npz',str(result_dir/'ik_trajectory.npz'),
             '--fixed-bottle-xyz-m',*(str(v) for v in xyz),
             '--fps',str(args.fps), '--minimum-floor-z',str(args.minimum_floor_z)]
        env=os.environ.copy()
        env.setdefault('MUJOCO_GL','osmesa')
        process=subprocess.run(cmd,text=True,capture_output=True,env=env,cwd=Path(phase6c.__file__).parent)
        (result_dir/'console.log').write_text('COMMAND: '+' '.join(cmd)+'\n'+process.stdout+'\n'+process.stderr)
        sum_path=result_dir/'physical'/'summary.json'
        try:
            phys=json.loads(sum_path.read_text())
        except (OSError,ValueError):
            phys={'status':'EXECUTION_ERROR','reason':(process.stderr or process.stdout)[-3000:]}
        if process.returncode:
            phys['status']='EXECUTION_ERROR';phys['exit_code']=process.returncode
        row={'profile':name,'requested_j4_deg':j4_target,'requested_j5_deg':j5_target,
             'ik_j4_deg':diag['actual_ik_j4_deg'],
             'ik_anchor_error_mm':diag['anchor_position_error_mm'],
             'ik_orientation_error_deg':diag['wrist_orientation_error_deg'],
             'open_fixed_gap_mm':diag['gripper_clearance']['open_gap_fixed_mm'],
             'moving_sweep_min_gap_mm':diag['gripper_clearance']['moving_sweep_min_gap_mm'],
             'ik_preflight':diag['gripper_clearance']['kinematic_preflight'],
             'simulation_status':phys.get('status','UNKNOWN'),
             'bilateral_hold_steps':phys.get('bilateral_hold_steps',None),
             'bottle_final_xy_displacement_mm':phys.get('final_bottle_horizontal_displacement_mm',None),
             'max_j1_j5_tracking_error_deg':max((abs(float(phys.get('max_error_deg',{}).get(n,0))) for n in JOINTS),default=None)
                 if phys.get('max_error_deg') else None,
             'video':str(Path(name)/'physical'/'upright_reach_grasp.mp4') if (result_dir/'physical'/'upright_reach_grasp.mp4').is_file() else None,
             'graph':str(Path(name)/'physical'/'tracking_and_contact.png') if (result_dir/'physical'/'tracking_and_contact.png').is_file() else None}
        rows.append(row)
        manifest['results'].append({'profile':name,'ik':diag,'physics':phys,'comparison':row})
        print(f"[{i}/{len(cases)}] {name}: IK xyz err {row['ik_anchor_error_mm']:.2f}mm, phys {row['simulation_status']}, bilateral {row['bilateral_hold_steps']}",flush=True)
    with (args.out/'comparison.csv').open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    manifest['completed_cases']=len(rows)
    manifest['status']='BATCH_COMPLETED_WITH_ERRORS' if any(r['simulation_status']=='EXECUTION_ERROR' for r in rows) else 'BATCH_COMPLETED_DIAGNOSTIC_ONLY'
    (args.out/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
        names=[r['profile'] for r in rows]
        axes[0].barh(names,[r['ik_anchor_error_mm'] for r in rows]);axes[0].set_xlabel('IK fixed-world anchor error (mm), lower is better')
        axes[1].barh(names,[r['bottle_final_xy_displacement_mm'] or 0 for r in rows])
        axes[1].set_xlabel('Dynamic Muhi displacement XY (mm); NOT grasp success')
        for ax in axes:ax.invert_yaxis();ax.grid(alpha=.2)
        fig.savefig(args.out/'comparison.png',dpi=150);plt.close(fig)
    except Exception as e:
        (args.out/'plot_error.txt').write_text(str(e))
    print('RESULT_DIR:',args.out,flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--urdf',type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument('--episode',type=int,default=4)
    p.add_argument('--approach-fraction',type=float,default=.8)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--transition-s',type=float,default=3.)
    p.add_argument('--fps',type=int,default=6)
    p.add_argument('--minimum-floor-z',type=float,default=-.05)
    # Fields used by phase6c.model_build and shared PD actuation.
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--arm-kp',type=float,default=18.)
    p.add_argument('--arm-kd',type=float,default=1.8)
    p.add_argument('--arm-torque',type=float,default=3.)
    p.add_argument('--j6-kp',type=float,default=2.)
    p.add_argument('--j6-kd',type=float,default=.08)
    p.add_argument('--j6-torque',type=float,default=.4)
    args=p.parse_args()
    if not 1<=args.transition_s<=8: p.error('--transition-s must be 1..8')
    run(args)

if __name__=='__main__':main()
