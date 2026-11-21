#!/usr/bin/env python3
"""Phase7X: recover failed bottle yaw cases with small XY offsets, and create failure videos.

Use the successful Phase7V center-grasp / 50 mm lift condition as baseline.
For target yaw angles (default 30 and 45 deg), keep J1..J4 fixed, allow J5 and
small bottle XY offsets to vary, then run the standard Phase7V physics if geometry passes.
Also generate *failure videos* for baseline J5-only failed cases so the failure mode is visible.

Simulation only. -35 mm table remains a synthetic setting.
"""
from __future__ import annotations

import argparse, csv, json, math, subprocess, sys, shutil
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import phase7v_minus35_physical_grasp as p7v
import phase7q_recorded_success_physics as p7q
import phase6c_upright_reach_grasp as p6c
import physical_workcell_sim as physical

BASE_XYZ=(0.1879926284382279,0.17363603141077924,-0.017)
BASE_J5_DEG=-127.7802200317383


def quat_for_yaw(yaw_deg: float):
    rot = Rotation.from_euler('z', yaw_deg, degrees=True) * Rotation.from_euler('y', 90, degrees=True)
    x, y, z, w = rot.as_quat()
    return [float(w), float(x), float(y), float(z)]


def candidate_pose(xyz, yaw_deg):
    return {
        'xyz_m': [float(x) for x in xyz],
        'quat_wxyz': quat_for_yaw(yaw_deg),
        'orientation': f'yaw_{int(yaw_deg):03d}deg',
        'placement': 'sideways_on_table',
        'yaw_deg': int(yaw_deg),
        'fixed_from': 'Phase7V_center50_baseline',
        'pose_measured_from_real_camera': False,
    }


def build_model(urdf: Path, out: Path, table_z_mm: float):
    opts=SimpleNamespace(urdf=urdf,out=out,dt=.002,
        jaw_collision_mode='axial_cross_convex_patches',jaw_collision_slabs=12,
        jaw_collision_cross_bands=3,jaw_table_contact=True,
        support_plane_z_m=table_z_mm/1000.,arm_kp=18.,arm_kd=1.8,
        arm_torque=3.,j6_kp=2.,j6_kd=.08,j6_torque=.4)
    return p6c.model_build(opts)


def init_q0(csv_path, mapping_path, time_s, j6_closed_deg, j6_open_deg):
    times,states,_=p7q.read_recorded_csv(csv_path)
    mapping=json.loads(mapping_path.read_text())
    qstates=p7q.joint_mapping(states,mapping,j6_closed_deg,j6_open_deg)
    q0=np.array([np.interp(time_s,times,qstates[:,j]) for j in range(6)])
    return q0


def metric_for_pose(model, d, qadr, q0, bq, fixed, moving, bottle, floor, pose, j5_deg, tol_mm):
    q=q0.copy(); q[4]=math.radians(j5_deg)
    d.qpos[qadr]=q
    d.qpos[bq:bq+3]=pose['xyz_m']
    d.qpos[bq+3:bq+7]=pose['quat_wxyz']
    mujoco.mj_forward(model,d)
    fg=p7v.jaw_gap(model,d,fixed,bottle)
    mg=p7v.jaw_gap(model,d,moving,bottle)
    offset=p7v.bottle_center_grasp_offset_mm(model,d,fixed,bottle,pose['quat_wxyz'])
    ft=min(float(mujoco.mj_geomDistance(model,d,int(geom),int(floor),.2,np.zeros(6))) for geom in fixed)*1000.
    mt=min(float(mujoco.mj_geomDistance(model,d,int(geom),int(floor),.2,np.zeros(6))) for geom in moving)*1000.
    eligible=(.6<=fg<=8. and .6<=mg<=45. and abs(offset)<=tol_mm and ft>=-.5 and mt>=-.5)
    return {
        'j5_deg': float(j5_deg),
        'fixed_gap_mm': float(fg),
        'moving_gap_mm': float(mg),
        'center_offset_mm': float(offset),
        'fixed_table_gap_mm': float(ft),
        'moving_table_gap_mm': float(mt),
        'eligible': bool(eligible),
        'geometry_violation_mm': float(
            max(0., .6-fg)+max(0., fg-8.)+max(0., .6-mg)+max(0., mg-45.)+
            max(0., abs(offset)-tol_mm)+max(0., -.5-ft)+max(0., -.5-mt)
        )
    }


def search_j5_only_baseline(model, d, qadr, q0, bq, fixed, moving, bottle, floor, yaw_deg, tol_mm, lim_deg, step_deg=5.0):
    pose=candidate_pose(BASE_XYZ, yaw_deg)
    pred=BASE_J5_DEG+yaw_deg
    trial_angles=np.unique(np.concatenate([
        np.arange(lim_deg[0]+0.5, lim_deg[1]-0.5, step_deg),
        [BASE_J5_DEG, pred], np.arange(pred-5., pred+5.1, 1.)
    ]))
    candidates=[]
    for j5_deg in trial_angles:
        c=metric_for_pose(model,d,qadr,q0,bq,fixed,moving,bottle,floor,pose,float(j5_deg),tol_mm)
        candidates.append(c)
    good=[c for c in candidates if c['eligible']]
    for item in good[:4]:
        for j5_deg in np.arange(item['j5_deg']-3., item['j5_deg']+3.1, 1.):
            candidates.append(metric_for_pose(model,d,qadr,q0,bq,fixed,moving,bottle,floor,pose,float(j5_deg),tol_mm))
    good=[c for c in candidates if c['eligible']]
    good.sort(key=lambda c: abs(c['j5_deg']-pred)+0.6*abs(c['center_offset_mm']))
    nearest=min(candidates,key=lambda c:c['geometry_violation_mm'])
    return pose, (good[0] if good else None), nearest, candidates


def search_xy_recovery(model, d, qadr, q0, bq, fixed, moving, bottle, floor, yaw_deg, tol_mm, lim_deg,
                       xy_radius_mm=10.0, xy_step_mm=2.5, j5_radius_deg=18.0, j5_step_deg=2.5):
    pred=BASE_J5_DEG+yaw_deg
    rows=[]
    for dx_mm in np.arange(-xy_radius_mm, xy_radius_mm+xy_step_mm/2, xy_step_mm):
        for dy_mm in np.arange(-xy_radius_mm, xy_radius_mm+xy_step_mm/2, xy_step_mm):
            xyz=[BASE_XYZ[0]+dx_mm/1000., BASE_XYZ[1]+dy_mm/1000., BASE_XYZ[2]]
            pose=candidate_pose(xyz, yaw_deg)
            for j5_deg in np.arange(pred-j5_radius_deg, pred+j5_radius_deg+j5_step_deg/2, j5_step_deg):
                if not (lim_deg[0]+0.5 < j5_deg < lim_deg[1]-0.5):
                    continue
                c=metric_for_pose(model,d,qadr,q0,bq,fixed,moving,bottle,floor,pose,float(j5_deg),tol_mm)
                c['dx_mm']=float(dx_mm); c['dy_mm']=float(dy_mm); c['xyz_m']=xyz; c['quat_wxyz']=pose['quat_wxyz']
                c['yaw_deg']=int(yaw_deg)
                c['score']=float(
                    2.0*abs(c['center_offset_mm']) + abs(c['fixed_gap_mm']-1.5) + 0.045*abs(c['moving_gap_mm']-12.) +
                    0.08*math.hypot(dx_mm,dy_mm) + 0.02*abs(j5_deg-pred)
                )
                rows.append(c)
    good=[r for r in rows if r['eligible']]
    good.sort(key=lambda r:r['score'])
    nearest=min(rows,key=lambda r:r['geometry_violation_mm']) if rows else None
    return (good[0] if good else None), nearest, rows


def generate_failure_video(model, d, qadr, aids, bq, fixed, moving, bottle, floor, q0, pose, j5_deg,
                           out: Path, dt=0.002, fps=8, open_deg=80., close_deg=10., settle_s=0.6, closing_s=1.3, hold_s=0.8):
    out.mkdir(parents=True, exist_ok=True)
    lo = model.jnt_range[[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,p7q.NAMES[k]) for k in range(6)],0]
    hi = model.jnt_range[[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,p7q.NAMES[k]) for k in range(6)],1]
    q=q0.copy(); q[4]=math.radians(j5_deg); q[5]=np.clip(math.radians(open_deg), lo[5]+.001, hi[5]-.001)
    open_r=float(q[5]); close_r=float(np.clip(math.radians(close_deg), lo[5]+.001, hi[5]-.001))
    mujoco.mj_resetData(model,d)
    d.qpos[qadr]=q
    d.ctrl[aids]=q
    d.qpos[bq:bq+3]=pose['xyz_m']; d.qpos[bq+3:bq+7]=pose['quat_wxyz']
    d.qvel[:]=0
    mujoco.mj_forward(model,d)
    phases=[('settle',settle_s),('close',closing_s),('hold',hold_s)]
    total=sum(x[1] for x in phases)
    steps=math.ceil(total/dt)
    stride=max(1,round(1/(dt*fps)))
    trace=[]; frames=[]; last_phase=''; t=0.
    start=np.array(d.qpos[bq:bq+3])
    for k in range(steps+1):
        elapsed=0.; phase='hold'; u=1.
        for name,dur in phases:
            if t<=elapsed+dur+1e-9:
                phase=name; u=np.clip((t-elapsed)/max(dur,1e-9),0.,1.); break
            elapsed+=dur
        target=q.copy()
        if phase in ('close','hold'):
            target[5]=open_r+(close_r-open_r)*(u if phase=='close' else 1.)
        d.ctrl[aids]=target
        if k>0:
            mujoco.mj_step(model,d)
        fc,mc,force,depth=p7q.contact_force(model,d,fixed,moving,bottle)
        row=dict(time_s=t,phase=phase,j5_deg=j5_deg,j6_actual_deg=float(np.rad2deg(d.qpos[qadr[5]])),
                 fixed_contact_count=fc,moving_contact_count=mc,bilateral_contact=int(fc>0 and mc>0),
                 max_contact_normal_force_N=float(force),bottle_rise_mm=float((d.qpos[bq+2]-start[2])*1000),
                 bottle_xy_displacement_mm=float(np.linalg.norm(d.qpos[bq:bq+2]-start[:2])*1000),
                 fixed_gap_mm=float(p7v.jaw_gap(model,d,fixed,bottle)),moving_gap_mm=float(p7v.jaw_gap(model,d,moving,bottle)))
        trace.append(row)
        if k%stride==0 or phase!=last_phase:
            frames.append((d.qpos.copy(),d.qvel.copy(),t,phase,fc,mc))
        last_phase=phase; t=min(total,(k+1)*dt)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            break
    if trace:
        with (out/'failure_trace.csv').open('w',newline='') as fp:
            w=csv.DictWriter(fp,fieldnames=list(trace[0].keys())); w.writeheader(); w.writerows(trace)
    video_status=p7v.generate_video(model, frames, out, fps, bq, phases) if frames else 'NO_FRAMES'
    summary={
        'status':'FAILURE_DIAGNOSTIC_COMPLETE',
        'yaw_deg': pose['yaw_deg'],
        'j5_deg': j5_deg,
        'initial_pose': pose,
        'initial_metrics': metric_for_pose(model,d,qadr,q0,bq,fixed,moving,bottle,floor,pose,j5_deg,12.),
        'max_bottle_rise_mm': max((r['bottle_rise_mm'] for r in trace), default=None),
        'max_xy_displacement_mm': max((r['bottle_xy_displacement_mm'] for r in trace), default=None),
        'max_force_N': max((r['max_contact_normal_force_N'] for r in trace), default=None),
        'video_status': video_status,
    }
    (out/'failure_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    return summary


def write_case_sheet(rows, out_png: Path):
    cellw, cellh = 540, 360
    cols=2
    sheet=Image.new('RGB', (cellw*cols, cellh*math.ceil(len(rows)/cols)), 'white')
    for i,row in enumerate(rows):
        cell=Image.new('RGB',(cellw,cellh),'#f8fafc')
        dr=ImageDraw.Draw(cell)
        title=f"Yaw {row['yaw_deg']}Â° | {row['status']}"
        dr.text((10,8), title, fill='#111827')
        if row.get('selected_j5_deg') is not None:
            dr.text((10,28), f"J5={row['selected_j5_deg']:.1f}Â°, lift={row.get('final_bottle_rise_mm','--')}", fill='#111827')
        img_path=row.get('snapshot')
        if img_path and Path(img_path).is_file():
            with Image.open(img_path) as im:
                im=im.convert('RGB'); im.thumbnail((520,300))
                cell.paste(im, (10,50))
        sheet.paste(cell, ((i%cols)*cellw, (i//cols)*cellh))
    sheet.save(out_png)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('csv','mapping','urdf','out'): ap.add_argument('--'+key, required=True, type=Path)
    ap.add_argument('--time', type=float, default=14.8)
    ap.add_argument('--table-z-mm', type=float, default=-35.0)
    ap.add_argument('--recovery-yaws', nargs='*', type=int, default=[30,45])
    ap.add_argument('--failure-yaws', nargs='*', type=int, default=[15,30,45,95,135])
    ap.add_argument('--center-tolerance-mm', type=float, default=12.0)
    ap.add_argument('--xy-radius-mm', type=float, default=10.0)
    ap.add_argument('--xy-step-mm', type=float, default=2.5)
    ap.add_argument('--j5-radius-deg', type=float, default=18.0)
    ap.add_argument('--j5-step-deg', type=float, default=2.5)
    ap.add_argument('--lift-mm', type=float, default=50.0)
    ap.add_argument('--fps', type=int, default=8)
    args=ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    q0=init_q0(args.csv, args.mapping, args.time, -10.0, 100.0)
    q0[5]=math.radians(80.)
    model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor = build_model(args.urdf, args.out, args.table_z_mm)
    lim_deg=np.rad2deg(model.jnt_range[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'wrist_roll')])
    d=mujoco.MjData(model)
    bq=p7v.freejoint_adr(model)

    summary={
        'status':'COMPLETED',
        'table_z_mm': args.table_z_mm,
        'base_xyz_m': BASE_XYZ,
        'base_j5_deg': BASE_J5_DEG,
        'recovery_yaws': args.recovery_yaws,
        'failure_yaws': args.failure_yaws,
        'failure_diagnostics': [],
        'recovery_cases': [],
        'limitations': [
            'J1-J4 initial pose remains fixed during pre-grasp; only J5 and bottle XY offset are varied',
            'Table height -35 mm remains synthetic and not calibrated to the real robot',
            'Failure videos are exploratory simulations for visualization and may begin from geometrically imperfect states',
        ]
    }

    # Baseline failure diagnostics.
    for yaw in args.failure_yaws:
        pose, good, nearest, candidates = search_j5_only_baseline(model,d,qadr,q0,bq,fixed,moving,bottle,floor,yaw,args.center_tolerance_mm,lim_deg)
        chosen = good if good is not None else nearest
        fail_dir=args.out/f'failure_yaw_{yaw:03d}'
        failure_summary=generate_failure_video(model,d,qadr,aids,bq,fixed,moving,bottle,floor,q0,pose,chosen['j5_deg'],fail_dir,fps=args.fps)
        failure_summary['baseline_j5_only_success']=bool(good is not None)
        failure_summary['nearest_geometry']=nearest
        summary['failure_diagnostics'].append(failure_summary)

    # Recovery search.
    for yaw in args.recovery_yaws:
        case_dir=args.out/f'recovery_yaw_{yaw:03d}'
        case_dir.mkdir(exist_ok=True)
        best, nearest, rows = search_xy_recovery(model,d,qadr,q0,bq,fixed,moving,bottle,floor,yaw,args.center_tolerance_mm,
                                                 lim_deg,args.xy_radius_mm,args.xy_step_mm,args.j5_radius_deg,args.j5_step_deg)
        rows_csv=[]
        for r in rows:
            rows_csv.append({k:r[k] for k in ['yaw_deg','j5_deg','dx_mm','dy_mm','fixed_gap_mm','moving_gap_mm','center_offset_mm','fixed_table_gap_mm','moving_table_gap_mm','eligible','score','geometry_violation_mm']})
        if rows_csv:
            with (case_dir/'search_candidates.csv').open('w',newline='') as fp:
                w=csv.DictWriter(fp,fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)
        case={'yaw_deg': yaw, 'best_recovery': best, 'nearest_candidate': nearest, 'status': 'NO_RECOVERY_GEOMETRY'}
        if best is not None:
            pose={'xyz_m':[float(x) for x in best['xyz_m']], 'quat_wxyz':[float(x) for x in best['quat_wxyz']],
                  'orientation':f'yaw_{yaw:03d}deg', 'placement':'sideways_on_table', 'yaw_deg': int(yaw),
                  'fixed_gap_mm': best['fixed_gap_mm'], 'moving_gap_mm': best['moving_gap_mm'],
                  'fixed_jaw_contact_long_axis_offset_mm': best['center_offset_mm'],
                  'grasp_center_tolerance_mm': args.center_tolerance_mm,
                  'offset_from_fixed_center_mm':[best['dx_mm'], best['dy_mm']]}
            pose_path=case_dir/'bottle_pose.json'
            pose_path.write_text(json.dumps(pose, indent=2)+'\n')
            cmd=[sys.executable, str(Path(p7v.__file__).resolve()),
                 '--csv', str(args.csv), '--mapping', str(args.mapping), '--urdf', str(args.urdf), '--out', str(case_dir),
                 '--time', str(args.time), '--table-z-mm', str(args.table_z_mm), '--lift-mm', str(args.lift_mm), '--fps', str(args.fps),
                 '--center-tolerance-mm', str(args.center_tolerance_mm), '--j5-preset-deg', str(best['j5_deg']), '--fixed-pose-json', str(pose_path)]
            proc=subprocess.run(cmd,capture_output=True,text=True)
            (case_dir/'process_stdout.txt').write_text(proc.stdout)
            (case_dir/'process_stderr.txt').write_text(proc.stderr)
            case['physics_exit_code']=proc.returncode
            if (case_dir/'phase7v_summary.json').is_file():
                sim=json.loads((case_dir/'phase7v_summary.json').read_text())
                case['sim_summary']=sim
                case['status']=sim.get('status','UNKNOWN')
            else:
                case['status']='PHYSICS_ERROR'
        summary['recovery_cases'].append(case)

    # Summary plots.
    fig,ax=plt.subplots(figsize=(10,5.5))
    for case in summary['recovery_cases']:
        yaw=case['yaw_deg']
        nearest=case.get('nearest_candidate') or {}
        best=case.get('best_recovery') or {}
        if nearest:
            ax.scatter([yaw],[nearest.get('fixed_gap_mm',np.nan)],marker='x',s=90,label='Baseline nearest fixed-gap' if yaw==summary['recovery_cases'][0]['yaw_deg'] else None)
        if best:
            ax.scatter([yaw],[best.get('fixed_gap_mm',np.nan)],marker='o',s=90,label='Recovered fixed-gap' if yaw==summary['recovery_cases'][0]['yaw_deg'] else None)
            ax.annotate(f"dx={best['dx_mm']:.1f}, dy={best['dy_mm']:.1f}, J5={best['j5_deg']:.1f}", (yaw,best['fixed_gap_mm']), xytext=(8,8), textcoords='offset points', fontsize=8)
    ax.axhline(0.6,color='gray',ls='--',label='Min fixed-gap threshold')
    ax.set_xlabel('Muhi yaw (deg)')
    ax.set_ylabel('Fixed-jaw bottle gap (mm)')
    ax.set_title('Phase7X recovery: J5 + small XY offsets for failed yaw angles')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(args.out/'phase7x_recovery_fixed_gap.png',dpi=170); plt.close(fig)

    # Capture video thumbnails if present.
    sheet_rows=[]
    for case in summary['recovery_cases']:
        img=None
        v=Path(args.out)/f"recovery_yaw_{case['yaw_deg']:03d}"/'phase7v_grasp_3views.mp4'
        if v.is_file() and shutil.which('ffmpeg'):
            png=v.parent/'last_frame_side.png'
            proc=subprocess.run(['ffmpeg','-y','-loglevel','error','-ss','5.0','-i',str(v),'-frames:v','1','-vf','crop=640:450:0:0,scale=520:300',str(png)])
            if proc.returncode==0 and png.is_file(): img=str(png)
        rise = None
        if case.get('sim_summary') and case['sim_summary'].get('metrics'):
            rise=case['sim_summary']['metrics'].get('final_bottle_rise_mm')
        sheet_rows.append({'yaw_deg':case['yaw_deg'],'status':case['status'],'selected_j5_deg': case.get('best_recovery',{}).get('j5_deg'), 'final_bottle_rise_mm': rise, 'snapshot': img})
    if sheet_rows:
        write_case_sheet(sheet_rows, args.out/'phase7x_recovery_contact_sheet.png')

    (args.out/'phase7x_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': summary['status'], 'out': str(args.out),
                      'recovery_statuses': {c['yaw_deg']: c['status'] for c in summary['recovery_cases']}}, ensure_ascii=False))


if __name__=='__main__':
    main()
