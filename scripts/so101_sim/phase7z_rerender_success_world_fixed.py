#!/usr/bin/env python3
"""Re-simulate nine confirmed horizontal-Muhi grasps with a world-fixed camera.

Consumes recorded Phase7W/X/Y *successful cases*, not the failure diagnostics.
Replays the physical grasp through Phase7V, which must have the world-locked
camera rendering fix. Original success outputs and milestone remain untouched.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOTS = {
    0: ('phase7w-j5-yaw-eight-angles-20261121-*', 'yaw_000'),
    15: ('phase7y-j5xy-recovery-*', 'recovery_yaw_015'),
    30: ('phase7x-j5xy-recovery-fail-videos-*', 'recovery_yaw_030'),
    45: ('phase7x-j5xy-recovery-fail-videos-*', 'recovery_yaw_045'),
    90: ('phase7y-j5xy-recovery-*', 'recovery_yaw_090'),
    95: ('phase7y-j5xy-recovery-*', 'recovery_yaw_095'),
    135: ('phase7y-j5xy-recovery-*', 'recovery_yaw_135'),
    150: ('phase7w-j5-yaw-eight-angles-20261121-*', 'yaw_150'),
    165: ('phase7w-j5-yaw-eight-angles-20261121-*', 'yaw_165'),
}


def source_for(root: Path, yaw: int) -> Path:
    pattern, child = ROOTS[yaw]
    matches = sorted((x / child for x in root.glob(pattern)
                      if (x / child / 'phase7v_summary.json').is_file()))
    if not matches:
        raise FileNotFoundError(f'Missing successful source folder for yaw {yaw}: {pattern}/{child}')
    return matches[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--angles', nargs='+', type=int, default=list(ROOTS))
    p.add_argument('--fps', type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p7v = Path(__file__).with_name('phase7v_minus35_physical_grasp.py')
    if not p7v.is_file():
        raise FileNotFoundError(p7v)
    source = p7v.read_text()
    if 'camera_lookat=np.array(' not in source or 'cam.lookat[:]=camera_lookat' not in source:
        raise RuntimeError('Phase7V is missing world-locked camera implementation')

    results = []
    for yaw in args.angles:
        if yaw not in ROOTS:
            raise ValueError(f'Unknown yaw: {yaw}')
        old = source_for(args.results_root, yaw)
        prior = json.loads((old/'phase7v_summary.json').read_text())
        if prior.get('status') != 'SIMULATED_LIFT_WITH_BILATERAL_CONTACT':
            raise RuntimeError(f'Previous run was NOT confirmed success: {old}')
        old_pose = prior.get('initial_bottle_pose')
        if not old_pose:
            raise RuntimeError(f'Missing initial bottle pose: {old}')
        j5 = prior['initial_arm_joints_deg'][4]
        out = args.out/f'yaw_{yaw:03d}'
        out.mkdir(exist_ok=True)
        pose_path = out/'bottle_pose.json'
        pose_path.write_text(json.dumps(old_pose, indent=2, ensure_ascii=False)+'\n')
        cmd = [sys.executable, str(p7v), '--csv', str(args.csv),
               '--mapping', str(args.mapping), '--urdf', str(args.urdf),
               '--out', str(out), '--time', str(prior['episode_time_s']),
               '--table-z-mm', str(prior['table_z_mm']), '--lift-mm', '50',
               '--fps', str(args.fps), '--center-tolerance-mm', '12',
               '--j5-preset-deg', str(j5), '--fixed-pose-json', str(pose_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        (out/'process_stdout.txt').write_text(proc.stdout)
        (out/'process_stderr.txt').write_text(proc.stderr)
        report_path = out/'phase7v_summary.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        video = out/'phase7v_grasp_3views.mp4'
        record = {'yaw_deg':yaw, 'source_dir':str(old), 'j5_deg':j5,
                  'exit_code':proc.returncode, 'status':report.get('status','ERROR'),
                  'rise_mm':report.get('metrics',{}).get('final_bottle_rise_mm'),
                  'video':str(video) if video.is_file() else None,
                  'camera':'world_fixed_initial_bottle',
                  'video_status':report.get('video_status')}
        results.append(record)
        (args.out/'phase7z_summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False)+'\n')
        print(f"yaw={yaw:3d} status={record['status']} rise={record['rise_mm']} video={record['video_status']}", flush=True)
        if proc.returncode or not video.is_file():
            raise RuntimeError(f'Replay failed for yaw {yaw}: see {out}/process_stderr.txt')
    print(f'Completed {len(results)} world-fixed camera videos at {args.out}', flush=True)


if __name__ == '__main__':
    main()
