#!/usr/bin/env python3
"""Phase7W: can J5-only pre-grasp adaptation handle 8 Muhi tabletop yaw angles?

Experimental 50mm lift uses the same Phase7V physics. Hold Muhi center XYZ and
J1..J4 initial angles FIXED; vary only Muhi yaw and initial J5. J6 opens/closes,
then J1..J4 may lift. No robot hardware. -35mm table is a synthetic setting.
"""
from __future__ import annotations
import argparse,csv,json,math,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import phase7v_minus35_physical_grasp as p7v
import phase7q_recorded_success_physics as p7q
import phase6c_upright_reach_grasp as p6c
import physical_workcell_sim as physical

YAW_DEG=(0,15,30,45,95,135,150,165)
BASE_XYZ=(0.1879926284382279,0.17363603141077924,-0.017)
BASE_J5_DEG=-127.7802200317383


def quat_for_yaw(yaw):
    rot=Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_euler('y',90,degrees=True)
    x,y,z,w=rot.as_quat()
    return [float(w),float(x),float(y),float(z)]


def candidate_pose(xyz,yaw):
    return {'xyz_m':list(xyz),'quat_wxyz':quat_for_yaw(yaw),
        'orientation':f'yaw_{yaw:03d}deg', 'placement':'sideways_on_table',
        'yaw_deg':int(yaw),'fixed_from':'successful_Phase7V_bottle_center',
        'pose_measured_from_real_camera':False}


def metrics(model,d,qadr,q0,bq,pose,j5_deg,j5_joint,limits,fixed,moving,bottle,floor,tol):
    if not (limits[0]+0.001 < math.radians(j5_deg) < limits[1]-0.001):return None
    q=q0.copy();q[4]=math.radians(j5_deg)
    d.qpos[qadr]=q
    d.qpos[bq:bq+3]=pose['xyz_m']
    d.qpos[bq+3:bq+7]=pose['quat_wxyz']
    mujoco.mj_forward(model,d)
    fg=p7v.jaw_gap(model,d,fixed,bottle)
    mg=p7v.jaw_gap(model,d,moving,bottle)
    offset=p7v.bottle_center_grasp_offset_mm(model,d,fixed,bottle,pose['quat_wxyz'])
    ft=min(float(mujoco.mj_geomDistance(model,d,int(geom),int(floor),.2,np.zeros(6)))
           for geom in fixed)*1000.
    mt=min(float(mujoco.mj_geomDistance(model,d,int(geom),int(floor),.2,np.zeros(6)))
           for geom in moving)*1000.
    # J5 axes in world space; near-vertical indicates yaw-only motion might be plausible.
    j5axis=np.asarray(d.xaxis[j5_joint]).copy()
    alignment=float(np.dot(j5axis,[0.,0.,1.]))
    eligible=(.6<=fg<=8. and .6<=mg<=45. and abs(offset)<=tol and ft>=-.5 and mt>=-.5)
    # Never mark a geometrically unsuitable candidate as physically graspable.
    score=(.45*abs(offset)+abs(fg-1.5)+.045*abs(mg-12.)
           +.001*abs(j5_deg-BASE_J5_DEG))
    violation=(max(0.,.6-fg)+max(0.,fg-8.) +
               max(0.,.6-mg)+max(0.,mg-45.)+
               max(0.,abs(offset)-tol)+
               max(0.,-.5-ft)+max(0.,-.5-mt))
    return dict(j5_deg=float(j5_deg),fixed_gap_mm=float(fg),moving_gap_mm=float(mg),
        center_offset_mm=float(offset),fixed_table_gap_mm=float(ft),
        moving_table_gap_mm=float(mt),j5_axis_dot_world_z=alignment,
        eligible=bool(eligible),score=float(score),geometry_violation_mm=float(violation))


def write_rows(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf8') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_graphs(rows,out):
    yaw=[r['yaw_deg'] for r in rows]
    j5=[r.get('selected_j5_deg',float('nan')) for r in rows]
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.plot(yaw,j5,'o-',label='Selected J5 (pre-grasp)')
    ax.axhline(BASE_J5_DEG,color='gray',ls=':',label='Successful Phase7V J5 baseline')
    ax.set(xlabel='Muhi tabletop yaw angle (deg)',ylabel='J5 / wrist_roll angle (deg)',
           title='Phase7W: J5-only adaptation to bottle yaw (J1-J4 fixed)')
    ax.set_xticks(yaw);ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(out/'phase7w_j5_vs_yaw.png',dpi=160);plt.close(fig)
    lift=[r.get('final_bottle_rise_mm',float('nan')) for r in rows]
    contact=[r.get('hold_bilateral',False) for r in rows]
    fig,ax=plt.subplots(figsize=(10,5.5))
    for xx,yy,cc in zip(yaw,lift,contact):
        if yy is None or not np.isfinite(yy):
            ax.annotate('no eligible J5', (xx,3),rotation=90,fontsize=8,ha='center',color='gray')
        else:
            ax.scatter([xx],[yy],s=100,c='#16803c' if cc else '#c2410c',zorder=4)
            ax.annotate(f'{yy:.1f}',(xx,yy),xytext=(0,6),textcoords='offset points',ha='center')
    ax.axhline(50,color='#475569',ls='--',label='Target 50 mm')
    ax.set(xlabel='Muhi tabletop yaw angle (deg)',ylabel='Final bottle vertical displacement (mm)',
           title='Phase7W: MuJoCo lift by yaw after J5-only pre-grasp alignment')
    ax.set_xticks(yaw);ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(out/'phase7w_lift_vs_yaw.png',dpi=160);plt.close(fig)


def make_contact_sheet(rows,out):
    cellw,cellh=480,380
    sheet=Image.new('RGB',(cellw*4,cellh*2),'white')
    for k,row in enumerate(rows):
        cell=Image.new('RGB',(cellw,cellh),'#eef2f7');draw=ImageDraw.Draw(cell)
        yaw=row['yaw_deg'];draw.text((10,6),f'Yaw {yaw} deg  J5={row.get("selected_j5_deg", "none")}',fill='#111827')
        video=out/f'yaw_{yaw:03d}'/'phase7v_grasp_3views.mp4'
        if video.is_file():
            png=out/f'yaw_{yaw:03d}'/'last_frame_side.png'
            proc=subprocess.run(['ffmpeg','-y','-loglevel','error','-ss','5.2','-i',str(video),
                '-frames:v','1','-vf','crop=640:450:0:0,scale=480:335',str(png)],capture_output=True)
            if proc.returncode==0 and png.is_file():
                with Image.open(png) as image:cell.paste(image.convert('RGB'),(0,40))
        else:
            draw.text((12,90),'Not geometrically reachable using J5 only',fill='#991b1b')
        sheet.paste(cell,((k%4)*cellw,(k//4)*cellh))
    sheet.save(out/'phase7w_lastframe_comparison.png')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('csv','mapping','urdf','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--time',type=float,default=14.8)
    p.add_argument('--table-z-mm',type=float,default=-35.0)
    p.add_argument('--lift-mm',type=float,default=50.0)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--center-tolerance-mm',type=float,default=12.0)
    p.add_argument('--coarse-j5-step-deg',type=float,default=5.0)
    p.add_argument('--geometry-only',action='store_true')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.table_z_mm!=-35.0:print('WARNING: changing experimental baseline table height')
    times,states,_=p7q.read_recorded_csv(args.csv)
    mapping=json.loads(args.mapping.read_text())
    qs=p7q.joint_mapping(states,mapping,-10.0,100.0)
    q0=np.array([np.interp(args.time,times,qs[:,i]) for i in range(6)])
    q0[5]=math.radians(80.)
    options=SimpleNamespace(urdf=args.urdf,out=args.out,dt=.002,
        jaw_collision_mode='axial_cross_convex_patches',jaw_collision_slabs=12,
        jaw_collision_cross_bands=3,jaw_table_contact=True,
        support_plane_z_m=args.table_z_mm/1000.,arm_kp=18.,arm_kd=1.8,
        arm_torque=3.,j6_kp=2.,j6_kd=.08,j6_torque=.4)
    model,qadr,_,_,lo,hi,fixed,moving,bottle,floor=p6c.model_build(options)
    j5=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'wrist_roll')
    if j5<0:raise RuntimeError('J5 joint wrist_roll not found')
    lim=model.jnt_range[j5]
    j6_limit=(math.degrees(lo[5]),math.degrees(hi[5]))
    bq=p7v.freejoint_adr(model)
    d=mujoco.MjData(model)
    xyz=list(BASE_XYZ)
    # Keep the known successful center, not a fresh per-yaw XY placement search.
    rows=[]
    all_candidates=[]
    for yaw in YAW_DEG:
        pose=candidate_pose(xyz,yaw)
        candidates=[]
        sweep=np.arange(math.degrees(lim[0])+0.5,math.degrees(lim[1])-.5,
                        args.coarse_j5_step_deg)
        trial_angles=np.unique(np.concatenate([sweep,[BASE_J5_DEG]]))
        for v in trial_angles:
            c=metrics(model,d,qadr,q0,bq,pose,float(v),j5,lim,fixed,moving,bottle,floor,args.center_tolerance_mm)
            if c is not None:candidates.append(c)
        good=sorted((c for c in candidates if c['eligible']),key=lambda c:c['score'])
        # Refine around the best coarse feasible candidates using +/-3 deg in steps of 1 deg.
        near=[]
        for item in good[:4]:
            for v in np.arange(item['j5_deg']-3,item['j5_deg']+3.1,1.):
                c=metrics(model,d,qadr,q0,bq,pose,float(v),j5,lim,fixed,moving,bottle,floor,args.center_tolerance_mm)
                if c is not None:near.append(c)
        candidates+=near
        good=sorted((c for c in candidates if c['eligible']),key=lambda c:c['score'])
        nearest=min(candidates,key=lambda c:c['geometry_violation_mm']) if candidates else None
        case=dict(yaw_deg=int(yaw),j1_j4_constant=True,muhi_xyz_constant=True,
                  geometric_candidates_tested=len(candidates),eligible_j5_candidates=len(good),
                  closest_infeasible_candidate=nearest,chosen_geometric_candidate=good[0] if good else None,
                  initial_pose=pose,selected_j5_deg=good[0]['j5_deg'] if good else None,
                  status='J5_GEOMETRY_OK' if good else 'J5_ONLY_NO_CENTRAL_GEOMETRY')
        sub=args.out/f'yaw_{yaw:03d}';sub.mkdir(exist_ok=True)
        (sub/'bottle_pose.json').write_text(json.dumps(pose,indent=2)+'\n')
        if good and not args.geometry_only:
            cmd=[sys.executable,str(Path(p7v.__file__).resolve()),'--csv',str(args.csv),
                '--mapping',str(args.mapping),'--urdf',str(args.urdf),'--out',str(sub),
                '--time',str(args.time),'--table-z-mm',str(args.table_z_mm),
                '--lift-mm',str(args.lift_mm),'--fps',str(args.fps),
                '--center-tolerance-mm',str(args.center_tolerance_mm),
                '--j5-preset-deg',str(good[0]['j5_deg']),
                '--fixed-pose-json',str(sub/'bottle_pose.json')]
            proc=subprocess.run(cmd,capture_output=True,text=True)
            (sub/'process_stdout.txt').write_text(proc.stdout)
            (sub/'process_stderr.txt').write_text(proc.stderr)
            case['physics_exit_code']=proc.returncode
            if (sub/'phase7v_summary.json').is_file():
                sim=json.loads((sub/'phase7v_summary.json').read_text())
                case['sim_status']=sim.get('status')
                met=sim.get('metrics',{})
                case['final_bottle_rise_mm']=met.get('final_bottle_rise_mm')
                case['hold_bilateral']=met.get('hold_bilateral',False)
                case['bilateral_contact_steps']=met.get('bilateral_contact_steps',0)
                case['max_jaw_force_N']=met.get('max_jaw_normal_force_N',0)
                case['max_bottle_xy_displacement_mm']=met.get('max_bottle_xy_displacement_mm')
                case['video_status']=sim.get('video_status')
                if proc.returncode==0 and sim.get('status')=='SIMULATED_LIFT_WITH_BILATERAL_CONTACT':
                    case['status']='PHYSICS_LIFT_WITH_CONTACT'
                elif proc.returncode!=0:
                    case['status']='PHYSICS_ERROR'
                else:case['status']='PHYSICS_NO_CONFIRMED_LIFT'
            else:
                case['status']='PHYSICS_ERROR'
        rows.append(case)
        print(f'Yaw {yaw:3d}: {case["status"]} J5={case["selected_j5_deg"]}, lift={case.get("final_bottle_rise_mm")}')
        # write results after EACH yaw to preserve partial progress in Drive
        (args.out/'phase7w_j5_yaw_summary.json').write_text(json.dumps({
             'status':'PARTIAL' if len(rows)<len(YAW_DEG) else 'COMPLETED',
             'angles_deg':list(YAW_DEG),'table_z_mm':args.table_z_mm,'bottle_xyz_m':xyz,
             'j1_to_j4_initial_deg':np.rad2deg(q0[:4]).tolist(),
             'initial_j5_deg':BASE_J5_DEG,'j5_limits_deg':np.rad2deg(lim).tolist(),
             'j5_world_axis_is_not_assumed_vertical':True,
             'physics_enabled':not args.geometry_only,
             'real_to_sim_calibrated':False,'cases':rows},ensure_ascii=False,indent=2)+'\n')
    simple=[{k:case.get(k) for k in ('yaw_deg','selected_j5_deg','eligible_j5_candidates',
             'status','sim_status','final_bottle_rise_mm','hold_bilateral','max_jaw_force_N')}
            for case in rows]
    write_rows(args.out/'phase7w_j5_yaw_results.csv',simple)
    make_graphs(rows,args.out)
    if not args.geometry_only:make_contact_sheet(rows,args.out)
    print('DONE',str(args.out))

if __name__=='__main__':main()
