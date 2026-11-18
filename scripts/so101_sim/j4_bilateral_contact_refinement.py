#!/usr/bin/env python3
"""Refine nonnegative-J4 grasp using paired jaw geometry, not a proxy TCP only.

Simulation-only. The moving/fixed jaw witnesses are measured in world space with
MuJoCo mj_geomDistance; friction and the baseline candidate are unchanged.
The score is a geometric *hypothesis*, not proof of stable dynamics.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import physical_grasp_j6_bench as grasp

JOINTS = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')


def nearest_witness(model, data, bottle_geoms, jaw_geoms):
    """Return signed gap and two world-frame witness points.

    For mj_geomDistance(model, data, g_bottle, g_jaw, ..., points),
    points[:3] is on the bottle and points[3:] is on the jaw.
    """
    best = None
    for b in bottle_geoms:
        for g in jaw_geoms:
            witness = np.zeros(6, dtype=float)
            d = float(mujoco.mj_geomDistance(
                model, data, int(b), int(g), .20, witness))
            if best is None or d < best['gap_m']:
                best = {'gap_m':d, 'bottle_xyz':witness[:3].copy(),
                        'jaw_xyz':witness[3:].copy(), 'bottle_geom':int(b),
                        'jaw_geom':int(g)}
    if best is None:
        raise ValueError('No collision geoms for jaw/bottle')
    return best


def contact_alignment(fixed, moving, center_xyz):
    """Check opposing XY radial contact locations and cylinder-height alignment."""
    c = np.asarray(center_xyz, dtype=float)
    rf = fixed['bottle_xyz'][:2] - c[:2]
    rm = moving['bottle_xyz'][:2] - c[:2]
    nf, nm = float(np.linalg.norm(rf)), float(np.linalg.norm(rm))
    opposition = float(np.dot(rf, rm)/(nf*nm)) if min(nf,nm) > 1e-5 else 1.
    zf = float(abs(fixed['bottle_xyz'][2]-c[2])*1000)
    zm = float(abs(moving['bottle_xyz'][2]-c[2])*1000)
    return opposition, zf, zm


def pose_metrics(model, data, qadr, bq, xyz, fixed_geoms,
                 moving_geoms, bottle_geoms, jaw_open, jaw_closed, q):
    data.qpos[qadr[:5]] = q
    data.qpos[bq:bq+3] = xyz
    data.qpos[bq+3:bq+7] = [1, 0, 0, 0]
    # First guard against a collision already present before J6 closes.
    data.qpos[qadr[5]] = jaw_open
    mujoco.mj_forward(model, data)
    open_f = nearest_witness(model, data, bottle_geoms, fixed_geoms)
    open_m = nearest_witness(model, data, bottle_geoms, moving_geoms)
    # Evaluate end-closed and just before the end, not two independently
    # selected contact times (both jaws must approach at the SAME J6 angle).
    stages=[]
    for frac in (.88, 1.):
        angle = jaw_open + frac*(jaw_closed-jaw_open)
        data.qpos[qadr[5]]=angle
        mujoco.mj_forward(model, data)
        f=nearest_witness(model,data,bottle_geoms,fixed_geoms)
        m=nearest_witness(model,data,bottle_geoms,moving_geoms)
        cos,zf,zm=contact_alignment(f,m,xyz)
        stages.append({'fraction_closed':frac,'j6_deg':float(np.rad2deg(angle)),
                       'fixed_gap_mm':1000*f['gap_m'],
                       'moving_gap_mm':1000*m['gap_m'],
                       'radial_opposition_cos':cos,
                       'fixed_center_height_offset_mm':zf,
                       'moving_center_height_offset_mm':zm,
                       'fixed_contact_world_m':f['bottle_xyz'].tolist(),
                       'moving_contact_world_m':m['bottle_xyz'].tolist()})
    return {'open_fixed_gap_mm':1000*open_f['gap_m'],
            'open_moving_gap_mm':1000*open_m['gap_m'],
            'stages':stages}


def quality(metrics, q, seed):
    # Millimeter-valued penalties; allow light compression on the final
    # closed side. Reject configurations that require the bottle to teleport
    # into the open jaw. The best "near close" stage wins; at each stage
    # the fixed+moving gaps and both radial witness points are simultaneous.
    def stage_cost(stage):
        gf,gm=stage['fixed_gap_mm'],stage['moving_gap_mm']
        # Prefer gap near zero for BOTH jaws at same joint angle.
        # Negative overlap is allowed to ~0.7 mm then penalized.
        def gap_penalty(g):
            return max(0,g-0.8)*1.2 + max(0,-g-0.7)*4. + .12*abs(g)
        radial = 5*max(0, stage['radial_opposition_cos']+0.5)
        # Only penalize contacts well above/below the mid-body area.
        height = .12*(max(0,stage['fixed_center_height_offset_mm']-24) +
                       max(0,stage['moving_center_height_offset_mm']-24))
        return gap_penalty(gf)+gap_penalty(gm)+radial+height

    open_f,open_m=metrics['open_fixed_gap_mm'],metrics['open_moving_gap_mm']
    open_penalty=10*(max(0,-open_f-0.3)+max(0,-open_m-0.3))
    movement = 1.0*float(np.linalg.norm((q-seed)/math.radians(55)))
    return float(min(stage_cost(x) for x in metrics['stages'])+
                 open_penalty+movement)


def refine(model, qadr, body_id, bq, xyz, fixed, moving, bottle,
           low, high, jaw_open, jaw_closed, candidates, max_ik_error_mm,
           out, j4_min_deg=0., max_starts=4):
    """Return best refined q, its local reference anchor, and JSON diagnostics.

    `candidates` are the existing 18 scored anchor points, each holding
    q and anchor. Narrow local constrained refinement is computationally
    bounded; do not rerun unrestricted exhaustive simulation for all points.
    """
    out = Path(out)
    bounds_lo = np.asarray(low[:5],dtype=float).copy()
    bounds_hi = np.asarray(high[:5],dtype=float).copy()
    bounds_lo[3]=max(bounds_lo[3], math.radians(j4_min_deg))
    if bounds_lo[3]>=bounds_hi[3]:
        raise ValueError('No feasible nonnegative J4 range')
    data=mujoco.MjData(model)
    valid = [c for c in candidates if c.get('q') is not None
             and c.get('valid_open')
             and c['ik_error_mm'] <= max_ik_error_mm]
    valid.sort(key=lambda x:x['score'])
    starts = valid[:max_starts]
    if not starts:raise ValueError('No valid J4 seed candidates')
    measured=[]
    winners=[]
    for k,entry in enumerate(starts):
        seed=np.asarray(entry['q'],dtype=float).copy()
        trace={'seed_rank':k+1,'seed_anchor_local_mm':[
            entry['local_x_mm'],entry['local_y_mm'],entry['local_z_mm']],
            'initial_j4_deg':float(np.rad2deg(seed[3])),
            'seed_heuristic_score':entry['score']}
        def evaluate(q):
            return pose_metrics(model,data,qadr,bq,xyz,fixed,moving,
                                bottle,jaw_open,jaw_closed,q)
        initial=evaluate(seed)
        trace['before']=initial
        def objective(q):
            return quality(evaluate(q),q,seed)
        opt=minimize(objective,seed,method='L-BFGS-B',
                     bounds=list(zip(bounds_lo,bounds_hi)),
                     options={'maxiter':28,'maxfun':220,'ftol':2e-5})
        refined=np.clip(opt.x,bounds_lo,bounds_hi)
        final=evaluate(refined)
        before_score=quality(initial,seed,seed)
        after_score=quality(final,refined,seed)
        # An optimizer that worsens a metric must not replace the seed.
        if after_score>before_score:
            refined=seed.copy();final=initial;after_score=before_score
        trace.update({'optimizer_success':bool(opt.success),
                      'optimizer_message':str(opt.message),
                      'optimizer_evaluations':int(opt.nfev),
                      'j4_deg':float(np.rad2deg(refined[3])),
                      'before_score':before_score,'after_score':after_score,
                      'after':final})
        measured.append(trace)
        open_clear=min(final['open_fixed_gap_mm'],
                       final['open_moving_gap_mm'])>=-0.5
        stage=min(final['stages'],key=lambda s:
                  abs(s['fixed_gap_mm'])+abs(s['moving_gap_mm']))
        paired=bool(open_clear and
                    -1.5<=stage['fixed_gap_mm']<=2. and
                    -1.5<=stage['moving_gap_mm']<=2. and
                    stage['radial_opposition_cos']<=-0.25)
        winners.append((not paired,not open_clear,after_score,k,
                        refined,entry,trace))
        trace['preflight_paired_contact']=paired
        trace['preflight_open_clear']=bool(open_clear)
    winners.sort(key=lambda x:x[:4])
    _,_,_,idx,q,entry,trace=winners[0]
    # We do NOT assert success here; only a MuJoCo dynamic close/hold can
    # validate grasping. Selection conservatively favors a predicted pair.
    output={'method':'Local bounded J4>=0 IK refinement using same-J6-angle bilateral signed gaps and opposite radial contact witnesses; friction unchanged',
            'chosen_seed_rank':idx+1,'chosen_predicted_paired':trace['preflight_paired_contact'],
            'trials':measured,
            'notes':['No approach or lift; collision witness points approximate the contact location.',
                     'The optimizer may fail to find a bilateral solution; do not label a geometry-only result as grasp success.',
                     'The original middle candidate is unmodified.']}
    (out/'j4_bilateral_refinement.json').write_text(
        json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=('seed_rank','initial_j4_deg','j4_deg','seed_heuristic_score',
            'before_score','after_score','preflight_paired_contact',
            'preflight_open_clear','optimizer_evaluations','optimizer_success')
    with (out/'j4_bilateral_refinement.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(measured)
    return q, np.asarray(entry['anchor'],dtype=float), output
