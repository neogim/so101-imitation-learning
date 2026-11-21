#!/usr/bin/env python3
"""Phase 8 (integration stage 1): route an object pose to proven experimental runners.

This is an orientation-based orchestration interface, NOT a trained pose
estimator and NOT a shared IK implementation yet.  Horizontal runner replays
saved successful yaw cases; upright runner runs the Phase6G contact baseline.
Neither path should be described as real-robot-ready.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

HORIZONTAL_ANGLES = (0, 15, 30, 45, 90, 95, 135, 150, 165)


def orientation_from_quaternion(quat_wxyz, *, upright_deg=20.0, horizontal_deg=70.0):
    """Classify local +Z bottle long axis against world +Z; no numpy needed."""
    if len(quat_wxyz) != 4:
        raise ValueError('Expected quaternion [w,x,y,z]')
    w, x, y, z = map(float, quat_wxyz)
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if not math.isfinite(norm) or norm < 1e-10:
        raise ValueError('Invalid quaternion')
    w, x, y, z = (v/norm for v in (w, x, y, z))
    abs_vertical_dot = min(1., abs(1.-2.*(x*x+y*y)))
    tilt_deg = math.degrees(math.acos(abs_vertical_dot))
    if tilt_deg <= upright_deg:
        return 'upright', tilt_deg
    if tilt_deg >= horizontal_deg:
        return 'horizontal', tilt_deg
    return 'unsupported_tilt', tilt_deg


def command_for_strategy(args, orientation):
    here = Path(__file__).resolve().parent
    if orientation == 'horizontal':
        if args.yaw_deg is None:
            raise ValueError('For horizontal grasp provide --yaw-deg (object yaw in world XY)')
        nearest = min(HORIZONTAL_ANGLES, key=lambda a: abs(float(args.yaw_deg)-a))
        if abs(float(args.yaw_deg)-nearest) > 0.25:
            raise ValueError('Only recorded successful yaw cases are replayable; '
                             'do not silently interpolate untested yaw angles')
        required = (args.results_root, args.csv, args.mapping, args.urdf)
        if any(v is None for v in required):
            raise ValueError('Horizontal runner requires --results-root --csv --mapping --urdf')
        cmd = [sys.executable, str(here/'phase7z_rerender_success_world_fixed.py'),
               '--results-root', str(args.results_root), '--csv', str(args.csv),
               '--mapping', str(args.mapping), '--urdf', str(args.urdf),
               '--out', str(args.out/'horizontal'), '--angles', str(nearest)]
    elif orientation == 'upright':
        if args.dataset_root is None or args.urdf is None:
            raise ValueError('Upright runner requires --dataset-root and --urdf')
        cmd = [sys.executable, str(here/'phase6g_fixed_workspace_grasp_depth.py'),
               '--dataset-root', str(args.dataset_root), '--urdf', str(args.urdf),
               '--out', str(args.out/'upright'), '--depth-only', 'middle',
               '--lift-mm', str(getattr(args,'lift_mm',50.))]
    else:
        raise ValueError('Unsupported posture: tilted object needs a separate strategy')
    if args.wrist_camera:
        cmd.append('--wrist-camera')
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--orientation', choices=['auto','horizontal','upright'], default='auto')
    p.add_argument('--quat-wxyz', nargs=4, type=float, help='Bottle long axis is local Z')
    p.add_argument('--yaw-deg', type=float, default=None)
    p.add_argument('--results-root', type=Path)
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--csv', type=Path)
    p.add_argument('--mapping', type=Path)
    p.add_argument('--urdf', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--wrist-camera', action='store_true', help='Visualization-only J5/J6 fixture')
    p.add_argument('--lift-mm',type=float,default=50.0,help='Upright target lift; success requires physical evidence')
    p.add_argument('--dry-run', action='store_true', help='Inspect routing without MuJoCo execution')
    args = p.parse_args(argv)
    if args.orientation == 'auto':
        if args.quat_wxyz is None:
            p.error('auto requires --quat-wxyz')
        orientation, tilt = orientation_from_quaternion(args.quat_wxyz)
    else:
        orientation, tilt = args.orientation, None
    if orientation == 'unsupported_tilt':
        payload = dict(status='UNSUPPORTED_TILT', tilt_deg=tilt,
                       explanation='No intermediate-pose strategy has been validated.')
        if not args.dry_run:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out/'phase8_summary.json').write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps(payload, ensure_ascii=False))
        return 2
    try:
        cmd = command_for_strategy(args, orientation)
    except ValueError as exc:
        p.error(str(exc))
    payload = dict(status='PLANNED' if args.dry_run else 'RUNNING',
                   orientation=orientation, tilt_deg=tilt,
                   strategy='phase7z_saved_success_replay' if orientation=='horizontal'
                            else 'phase6g_upright_contact_baseline',
                   wrist_camera=bool(args.wrist_camera), command=cmd,
                   limitations=['No shared IK yet: first integration stage only',
                                'Upright is contact baseline, stable lift unconfirmed',
                                'Horizontal success source uses synthetically adjusted bottle XY',
                                'Wrist mount model is photo-estimated, not calibrated'])
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, text=True)
    payload.update(status='RUNNER_COMPLETED' if result.returncode == 0 else 'RUNNER_FAILED',
                   exit_code=result.returncode)
    if orientation == 'upright':
        report_path=args.out/'upright'/'summary.json'
        if report_path.is_file():
            report=json.loads(report_path.read_text())
            upright=[]
            for case in report.get('results',[]):
                physics=case.get('physics',{})
                upright.append({
                    'name':case.get('name'),
                    'j4_deg':case.get('ik',{}).get('j4_wrist_flex_deg'),
                    'j5_deg':case.get('ik',{}).get('ik_angles_deg',{}).get('wrist_roll'),
                    'bilateral_hold_steps':physics.get('bilateral_hold_steps'),
                    'lift_status':physics.get('lift_status','NOT_TESTED'),
                    'lift_success':bool(physics.get('lift_success',False)),
                    'final_rise_mm':physics.get('lift_final_bottle_rise_mm'),
                    'final_xy_drift_mm':physics.get('lift_final_xy_drift_mm'),
                    'deep_penetration':physics.get('deep_penetration_over_3mm'),
                    'video':str(args.out/'upright'/case.get('name','')/'grasp.mp4')})
            payload['upright_cases']=upright
            payload['physical_success_count']=sum(x['lift_success'] for x in upright)
            payload['physical_success_confirmed']=payload['physical_success_count']>0
            if result.returncode==0 and not payload['physical_success_confirmed']:
                payload['status']='PHYSICS_RAN_NO_STABLE_LIFT'
    (args.out/'phase8_summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:payload[k] for k in ('status','orientation','strategy','exit_code')}, ensure_ascii=False))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
