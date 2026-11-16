#!/usr/bin/env python3
"""Execute Phase 6C joint-space approach candidates once each in a single batch.

Each case runs the existing Phase 6C simulation in a separate process.  Status
values are geometric/physical diagnostics, NOT verified grasps or world-frame
approach angles. The batch continues after preflight failures or exceptions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

PROFILE_SCRIPT = Path(__file__).with_name('phase6c_upright_reach_grasp.py')
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs/so101/phase6c_approach_profiles.json'


def read_profiles(config: Path, names: list[str] | None) -> list[str]:
    payload = json.loads(config.read_text(encoding='utf-8'))
    available = payload['profiles']
    selected = list(available) if not names else names
    if len(selected) != len(set(selected)) or any(x not in available for x in selected):
        raise ValueError('Unknown/duplicate profile. Available: ' + ', '.join(available))
    if not selected:
        raise ValueError('No profiles selected')
    return selected


def summarize_one(name: str, summary: dict, case_dir: Path, returncode: int) -> dict:
    errors = summary.get('max_error_deg') or {}
    return {
        'profile': name,
        'status': summary.get('status', 'MISSING_STATUS'),
        'stage': summary.get('stage', ''),
        'reason': summary.get('reason', ''),
        'exit_code': returncode,
        'j4_offset_deg': (summary.get('approach_profile') or {}).get('j4_wrist_flex_delta_deg', ''),
        'j5_offset_deg': (summary.get('approach_profile') or {}).get('j5_wrist_roll_delta_deg', ''),
        'max_j1_j5_error_deg': round(max((abs(float(errors[k])) for k in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll') if k in errors), default=0.), 3) if errors else '',
        'bilateral_hold_steps': summary.get('bilateral_hold_steps', ''),
        'hold_steps': summary.get('hold_steps', ''),
        'min_jaw_distance_mm': summary.get('min_jaw_distance_mm', ''),
        'final_bottle_xy_mm': summary.get('final_bottle_horizontal_displacement_mm', ''),
        'video': str((case_dir/'upright_reach_grasp.mp4').relative_to(case_dir.parent)) if (case_dir/'upright_reach_grasp.mp4').exists() else '',
        'graph': str((case_dir/'tracking_and_contact.png').relative_to(case_dir.parent)) if (case_dir/'tracking_and_contact.png').exists() else '',
        'summary': str((case_dir/'summary.json').relative_to(case_dir.parent)),
    }


def plot_comparison(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [r['profile'].replace('wrist_flex_', 'J4 ') for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    y = np.arange(len(rows))
    # No success score is manufactured when geometry preflight rejects a case.
    vals = [float(r['max_j1_j5_error_deg']) if r['max_j1_j5_error_deg'] != '' else 0. for r in rows]
    ax1.barh(y, vals)
    ax1.set_yticks(y, labels)
    ax1.set_xlabel('Max J1–J5 tracking error (degrees), completed trials only')
    ax1.invert_yaxis()
    for i,r in enumerate(rows):
        if r['max_j1_j5_error_deg'] == '':
            ax1.text(0, i, ' not run: '+str(r['status']), va='center', fontsize=9)
        else:
            ax1.text(vals[i], i, '  '+f'{vals[i]:.2f}°', va='center', fontsize=9)
    ax2.axis('off')
    table = ax2.table(cellText=[[r['profile'],r['status'],str(r['bilateral_hold_steps']),
                                (f"{float(r['min_jaw_distance_mm']):.2f}" if isinstance(r['min_jaw_distance_mm'], (int,float)) else '—')]
                               for r in rows],
                      colLabels=['Profile','Status (diagnostic only)','Bilateral hold steps','Min jaw distance (mm)'],
                      cellLoc='left',loc='center',colWidths=[.24,.47,.14,.15])
    table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.7)
    fig.suptitle('SO-101 Phase 6C: one-run, five-profile comparison (NOT grasp success)')
    fig.savefig(output,dpi=140);plt.close(fig)


def execute(args: argparse.Namespace) -> dict:
    if args.out.exists():
        raise FileExistsError(f'Output already exists: {args.out}')
    profiles = read_profiles(args.profile_config, args.profiles)
    args.out.mkdir(parents=True)
    rows = []
    environment = os.environ.copy()
    for index, name in enumerate(profiles, start=1):
        print(f'[{index}/{len(profiles)}] {name}: starting', flush=True)
        case_dir = args.out/name
        cmd = [sys.executable, '-u', str(PROFILE_SCRIPT),
               '--dataset-root', str(args.dataset_root), '--urdf', str(args.urdf),
               '--profile-config',str(args.profile_config), '--approach-profile',name,
               '--episode',str(args.episode),'--out',str(case_dir),
               '--fps',str(args.fps)]
        result = subprocess.run(cmd,cwd=PROFILE_SCRIPT.parent,env=environment,
                                capture_output=True,text=True,check=False)
        case_dir.mkdir(exist_ok=True)
        (case_dir/'console.log').write_text('COMMAND: '+' '.join(cmd)+'\n\nSTDOUT:\n'+result.stdout+'\nSTDERR:\n'+result.stderr,encoding='utf-8')
        path = case_dir/'summary.json'
        if path.exists():
            try:
                summary = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError) as exc:
                summary={'status':'INVALID_SUMMARY','reason':str(exc),'stage':'exception'}
        else:
            summary={'status':'EXECUTION_ERROR','reason':(result.stderr or result.stdout)[-1800:], 'stage':'exception'}
        if result.returncode != 0:
            summary['status']='EXECUTION_ERROR'
            summary['reason']=(result.stderr or result.stdout)[-1800:]
            summary['stage']='exception'
        summary['batch_profile']=name
        summary['process_exit_code']=result.returncode
        path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        row = summarize_one(name,summary,case_dir,result.returncode)
        rows.append(row)
        print(f"[{index}/{len(profiles)}] {name}: {row['status']}; contacts={row['bilateral_hold_steps']} exit={result.returncode}",flush=True)
    with (args.out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest={'schema':'phase6c_batch_v1','status':'BATCH_COMPLETED_WITH_ERRORS' if any(r['status']=='EXECUTION_ERROR' for r in rows) else 'BATCH_COMPLETED_DIAGNOSTIC_ONLY',
              'profile_config':str(args.profile_config),'profiles':profiles,'results':rows,
              'notes':['All profiles executed with independent MuJoCo states and virtual bottle placements.',
                       'Different profile candidates may produce different bottle locations: NOT a controlled same-target grasp comparison.',
                       'J4 offsets are joint-space offsets, NOT world-Z approach angles.',
                       'Bilateral contact and completed simulation are NOT proof of retained grasp.',
                       'Self/robot-table collision disabled; object shapes and support surface provisional.'],
              'video_policy':'Each completed trial saves a separate video; preflight rejections have no video.'}
    (args.out/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    try:
        plot_comparison(rows,args.out/'comparison.png')
    except Exception as exc:
        (args.out/'plot_error.txt').write_text(str(exc),encoding='utf-8')
        print('Graph generation failed; CSV and case summaries saved:',exc,flush=True)
    print('RESULT_DIR:',args.out,'\nCSV:',args.out/'comparison.csv',flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',required=True,type=Path)
    parser.add_argument('--urdf',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--profile-config',default=DEFAULT_CONFIG,type=Path)
    parser.add_argument('--profiles',nargs='*',default=None,
                        help='Omit to run every profile in the profile JSON, preserving order.')
    parser.add_argument('--episode',type=int,default=4)
    parser.add_argument('--fps',type=float,default=6)
    args=parser.parse_args()
    if args.fps<1 or args.fps>20: parser.error('--fps must be 1..20')
    execute(args)


if __name__=='__main__':
    main()
