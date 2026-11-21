#!/usr/bin/env python3
"""Phase7U: fast A/B table-height comparison; same Episode015 FK pose in all cases.

Diagnostic experiments, NOT calibration. No real robot, no mj_step, no URDF edits.
The tabletop and collision contact model are changed only between independent
compiled MuJoCo models. Results do not imply real-world table heights.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import phase6c_upright_reach_grasp as phase6c
import phase7q_recorded_success_physics as phase7q


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--time', type=float, default=14.8)
    p.add_argument('--table-z-mm', nargs='+', type=float, default=[0.0,-2.4,-35.0,-45.0])
    args=p.parse_args()
    if len(args.table_z_mm)<2 or len(args.table_z_mm)>8:
        raise ValueError('Supply 2 to 8 table heights')
    if len(set(args.table_z_mm)) != len(args.table_z_mm):
        raise ValueError('Table height values must be distinct')
    args.out.mkdir(parents=True, exist_ok=True)
    times,states,_=phase7q.read_recorded_csv(args.csv)
    mapping=json.loads(args.mapping.read_text())
    converted=phase7q.joint_mapping(states,mapping,-10.0,100.0)
    q0=np.array([np.interp(args.time,times,converted[:,j]) for j in range(6)])
    cases=[];tiles=[]
    for i,mm in enumerate(args.table_z_mm):
        case_dir=args.out/f'case_{i:02d}_{mm:+.1f}mm'
        case_dir.mkdir(exist_ok=True)
        opts=SimpleNamespace(urdf=args.urdf,out=case_dir,dt=.002,
            jaw_collision_mode='axial_cross_convex_patches',jaw_collision_slabs=12,
            jaw_collision_cross_bands=3,jaw_table_contact=True,
            support_plane_z_m=mm/1000.0,arm_kp=18.,arm_kd=1.8,arm_torque=3.,
            j6_kp=2.,j6_kd=.08,j6_torque=.4)
        model,qadr,_,_,_,_,fixed,moving,bottle,floor=phase6c.model_build(opts)
        data=mujoco.MjData(model)
        mujoco.mj_resetData(model,data);data.qpos[qadr]=q0
        mujoco.mj_forward(model,data)
        def min_gap(group):
            g=-1;z=np.inf
            for geom in group:
                sep=float(mujoco.mj_geomDistance(model,data,int(geom),int(floor),.4,np.zeros(6)))
                if sep<z:z=sep;g=int(geom)
            return z*1000,mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g)
        fixed_gap,fg=min_gap(fixed);moving_gap,mg=min_gap(moving)
        status='CLEAR' if min(fixed_gap,moving_gap)>=0 else 'PENETRATING'
        case=dict(case=i,table_z_mm=mm,fixed_gap_mm=fixed_gap,moving_gap_mm=moving_gap,
                  status=status,fixed_worst_geom=fg,moving_worst_geom=mg,
                  episode_time_s=args.time,urdf_joint_deg=np.rad2deg(q0).tolist())
        cases.append(case)
        # Side-by-side MuJoCo render: same camera target & distance in every case.
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,720)
        model.vis.global_.offheight=max(model.vis.global_.offheight,500)
        model.geom_rgba[list(bottle),3]=0
        for geom in fixed:model.geom_rgba[geom]=[.96,.42,.17,.68]
        for geom in moving:model.geom_rgba[geom]=[.61,.30,.87,.68]
        model.geom_rgba[floor]=[.54,.77,.89,.68]
        renderer=mujoco.Renderer(model,width=720,height=500)
        try:
            cam=mujoco.MjvCamera()
            cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:]=[.19,.18,.015]
            cam.azimuth=100.;cam.elevation=-15.;cam.distance=.34
            renderer.update_scene(data,camera=cam)
            rendered=Image.fromarray(renderer.render().copy())
        finally:renderer.close()
        tile=Image.new('RGB',(720,565),'#ffffff')
        tile.paste(rendered,(0,65))
        dr=ImageDraw.Draw(tile)
        dr.text((12,8),f'Table Z = {mm:+.1f} mm | {status}',fill='#171717')
        dr.text((12,32),f'Fixed {fixed_gap:+.2f} mm | Moving {moving_gap:+.2f} mm',fill='#18181b')
        tiles.append(tile)
    montage=Image.new('RGB',(720*2,565*((len(tiles)+1)//2)),'white')
    for i,tile in enumerate(tiles):montage.paste(tile,((i%2)*720,(i//2)*565))
    montage.save(args.out/'phase7u_table_ab_mujoco_3d.png')
    fig,ax=plt.subplots(figsize=(11,5.5),facecolor='white')
    xx=np.arange(len(cases));width=.34
    f=[c['fixed_gap_mm'] for c in cases];m=[c['moving_gap_mm'] for c in cases]
    ax.bar(xx-width/2,f,width,label='Fixed jaw',color='#c2410c')
    ax.bar(xx+width/2,m,width,label='Moving jaw',color='#7e22ce')
    ax.axhline(0,color='black',lw=1.8)
    for i in range(len(cases)):
        for pos,val in ((xx[i]-width/2,f[i]),(xx[i]+width/2,m[i])):
            ax.annotate(f'{val:+.1f}',(pos,val),xytext=(0,5 if val>=0 else -15),
                        textcoords='offset points',ha='center',fontsize=9)
    ax.set_xticks(xx,[f'{c["table_z_mm"]:+.1f}' for c in cases])
    ax.set_xlabel('Experimental tabletop Z in URDF base/world (mm)')
    ax.set_ylabel('Signed jaw-to-table gap (mm); positive = clearance')
    ax.set_title('Episode015 t=14.8 s: same arm pose, different tabletop heights')
    ax.grid(axis='y',alpha=.2);ax.legend()
    fig.tight_layout()
    fig.savefig(args.out/'phase7u_table_ab_gaps.png',dpi=180)
    plt.close(fig)
    with (args.out/'phase7u_table_ab_cases.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=['case','table_z_mm','fixed_gap_mm','moving_gap_mm','status'])
        w.writeheader();w.writerows([{k:x[k] for k in w.fieldnames} for x in cases])
    summary=dict(status='TABLE_AB_GEOMETRY_COMPARISON_COMPLETE',physics_integrated=False,
                 real_table_calibrated=False, joint_mapping_verified=False,
                 same_episode_pose=True,table_only_varied=True,cases=cases,
                 interpretation='A lowered table can remove table penetration while not improving real-to-sim pose accuracy. Do not use a low table as physical calibration.',
                 next_step='Compare realistic side-camera arm pose vs recorded frame; run contact-rich grasp only after selecting a deliberate experimental table setting.')
    (args.out/'phase7u_table_ab_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':summary['status'],'cases':[{'table_z_mm':c['table_z_mm'],
        'fixed_gap_mm':round(c['fixed_gap_mm'],2),'moving_gap_mm':round(c['moving_gap_mm'],2),
        'status':c['status']} for c in cases]},ensure_ascii=False))

if __name__=='__main__':
    main()
