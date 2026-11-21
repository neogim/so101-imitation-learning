#!/usr/bin/env python3
"""Phase 7R - visual-vs-collision inspection of Phase7Q's stopped initial state.

Uses identical Episode015 actions, unverified motor/URDF map, jaw proxy model,
and synthetic bottle locator as Phase7Q. No physics integration, robot hardware,
parameter tuning, or automatic correction of table height. The output is a
GEOMETRY DIAGNOSTIC, not calibrated real-to-sim reproduction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import phase6c_upright_reach_grasp as phase6c
import physical_workcell_sim as physical
import phase7q_recorded_success_physics as phase7q


def json_safe(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v


def closest(model, d, a, b):
    # Signed separation: negative is interpenetration of MuJoCo collision geoms.
    fromto=np.zeros(6)
    distance=float(mujoco.mj_geomDistance(model,d,int(a),int(b),0.5,fromto))
    return distance, fromto[:3].copy(), fromto[3:].copy()


def nearest_to_floor(model, d, group, floor):
    vals=[]
    for geom in group:
        dist,p1,p2=closest(model,d,geom,floor)
        vals.append(dict(geom_id=int(geom),geom_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(geom)),
                         gap_mm=round(1000*dist,4),jaw_point_world_m=p1.tolist(),
                         table_point_world_m=p2.tolist(),jaw_point_z_m=float(p1[2]),
                         table_point_z_m=float(p2[2])))
    if not vals:raise ValueError('No jaw collision geoms were returned by existing model.')
    return sorted(vals,key=lambda x:x['gap_mm'])


def nearest_to_bottle(model,d,group,bottle):
    candidates=[]
    for geom in group:
        for obj in bottle:
            gap,a,b=closest(model,d,geom,obj)
            candidates.append((gap,geom,obj,a,b))
    gap,geom,obj,a,b=min(candidates,key=lambda x:x[0])
    return dict(gap_mm=round(1000*gap,4),jaw_geom=int(geom),bottle_geom=int(obj),
                jaw_point_world_m=a.tolist(),bottle_point_world_m=b.tolist())


def mark_points(scene, entries):
    # MuJoCo scene markers are render-only and do not change collision geometry.
    for xyz, rgba, radius in entries:
        if scene.ngeom >= scene.maxgeom: break
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([radius,radius,radius]),np.asarray(xyz,float),
                           np.eye(3).reshape(-1),np.asarray(rgba,float))
        scene.ngeom+=1


def draw_tile(img, title, subtitle):
    w,h=img.size
    tile=Image.new('RGB',(w,h+58),'#182532');tile.paste(img,(0,58))
    draw=ImageDraw.Draw(tile)
    draw.text((10,8),title,fill='white')
    draw.text((10,31),subtitle,fill='#c5d5e4')
    return tile


def make_montage(images,cols,out):
    if not images:return
    w,h=images[0].size
    result=Image.new('RGB',(w*cols,h*math.ceil(len(images)/cols)),'#182532')
    for idx,item in enumerate(images):result.paste(item,((idx%cols)*w,(idx//cols)*h))
    result.save(out)


def run(args):
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    report=dict(status='STARTED',episode=15,episode_time_s=args.time,
        physics_integrated=False,sim_configuration='Phase7Q 12x3 collision patches and support plane',
        real_measurements_verified=False,
        caveat='Synthetic bottle XY/yaw; motor-deg and J6 mappings unverified; table z=0 is a model assumption, not real measured tabletop.')
    try:
        times,states,actions=phase7q.read_recorded_csv(args.csv)
        mapping=json.loads(args.mapping.read_text(encoding='utf8'))
        qstate=phase7q.joint_mapping(states,mapping,args.j6_closed_deg,args.j6_open_deg)
        q0=np.array([np.interp(args.time,times,qstate[:,j]) for j in range(6)])
        settings=SimpleNamespace(urdf=args.urdf,out=args.out,dt=.002,
            jaw_collision_mode='axial_cross_convex_patches',jaw_collision_slabs=12,
            jaw_collision_cross_bands=3,jaw_table_contact=True,support_plane_z_m=args.table_z,
            arm_kp=18.,arm_kd=1.8,arm_torque=3.,j6_kp=2.,j6_kd=.08,j6_torque=.4)
        model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor=phase6c.model_build(settings)
        dyn=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
        if dyn<0:raise RuntimeError('Dynamic bottle freejoint missing')
        bq=int(model.jnt_qposadr[dyn]); d=mujoco.MjData(model)
        mujoco.mj_resetData(model,d)
        d.qpos[qadr]=q0
        mujoco.mj_forward(model,d)
        pose,scan=phase7q.locate_provisional_bottle(model,d,qadr,bq,fixed,moving,bottle,floor,args.table_z,
            grid_radius_mm=args.grid_radius_mm,grid_step_mm=args.grid_step_mm)
        phase7q.write_csv(args.out/'initial_pose_scan.csv',scan)
        if not pose:
            report['status']='NO_SYNTHETIC_BOTTLE_PLACEMENT'
            return report
        d.qpos[bq:bq+3]=np.asarray(pose['xyz_m'],float)
        d.qpos[bq+3:bq+7]=np.asarray(pose['quat_wxyz'],float)
        mujoco.mj_forward(model,d)
        fixed_table=nearest_to_floor(model,d,fixed,floor)
        moving_table=nearest_to_floor(model,d,moving,floor)
        closest_fixed=nearest_to_bottle(model,d,fixed,bottle)
        closest_moving=nearest_to_bottle(model,d,moving,bottle)
        report.update(status='GEOMETRY_VISUALIZATION_READY',
            initial_bottle_pose=pose,
            fixed_jaw_table_worst=fixed_table[0],
            moving_jaw_table_worst=moving_table[0],
            fixed_jaw_bottle_worst=closest_fixed,
            moving_jaw_bottle_worst=closest_moving,
            table_z_m=float(args.table_z),
            jaw_collision_geoms_count=dict(fixed=len(fixed),moving=len(moving)),
            arm_initial_joint_urdf_deg=np.rad2deg(q0).tolist(),
            used_mapping=mapping)
        for filename,vals in [('fixed_collision_geoms.csv',fixed_table),('moving_collision_geoms.csv',moving_table)]:
            phase7q.write_csv(args.out/filename,vals)
        # Show explicit min-gap geoms as highlights; no force or dynamics evaluation.
        original_rgba=model.geom_rgba.copy()
        shown=list(fixed)+list(moving)+list(bottle)+[floor]
        fixed_worst=fixed_table[0]['geom_id'];moving_worst=moving_table[0]['geom_id']
        marker_positions=[]
        for item,color in ((fixed_table[0],[1.0,0.3,0.1,1.0]),(moving_table[0],[.8,0.2,1.,1.])):
            marker_positions.append((item['jaw_point_world_m'],color,.004))
            marker_positions.append((item['table_point_world_m'],color,.003))
        cams=[(90.,-10.,'SIDE'),(180.,-20.,'FRONT'),(100.,-75.,'TOP')]
        imgs=[]
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,800)
        model.vis.global_.offheight=max(model.vis.global_.offheight,520)
        renderer=mujoco.Renderer(model,width=800,height=520)
        try:
            cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            center=np.array([*pose['xyz_m'][:2],.025])
            cam.distance=.38
            for view in ('visual_with_collision','collision_geometry_only'):
                for az,el,name in cams:
                    model.geom_rgba[:]=original_rgba
                    if view=='collision_geometry_only':
                        model.geom_rgba[:,3]=0
                        for g in shown:model.geom_rgba[g]=[.68,.72,.74,.68]
                        for g in fixed:model.geom_rgba[g]=[.98,.48,.19,.80]
                        for g in moving:model.geom_rgba[g]=[.67,.36,.93,.80]
                        for g in bottle:model.geom_rgba[g]=[.25,.78,.9,.85]
                        model.geom_rgba[floor]=[.55,.63,.68,.60]
                    else:
                        for g in fixed:model.geom_rgba[g]=[.98,.48,.19,.55]
                        for g in moving:model.geom_rgba[g]=[.67,.36,.93,.55]
                        model.geom_rgba[fixed_worst]=[1.,.15,.0,.92]
                        model.geom_rgba[moving_worst]=[.86,.10,1.,.92]
                    cam.azimuth=az;cam.elevation=el;cam.lookat[:]=center
                    renderer.update_scene(d,camera=cam)
                    mark_points(renderer.scene,marker_positions)
                    img=Image.fromarray(renderer.render().copy())
                    desc=f"Fixed/table: {fixed_table[0]['gap_mm']:.2f} mm | Moving/table: {moving_table[0]['gap_mm']:.2f} mm"
                    imgs.append(draw_tile(img,f'{view.upper()} / {name} / t={args.time:.2f}s',desc))
            model.geom_rgba[:]=original_rgba
        finally:
            renderer.close()
            model.geom_rgba[:]=original_rgba
        make_montage(imgs,3,args.out/'initial_geometry_visual_vs_collision_3views.png')
        # Gap summaries, with per-patch extremes to explain which collision proxy penetrates the plane.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels=['Fixed jaw / bottle','Moving jaw / bottle','Fixed jaw / table','Moving jaw / table']
        values=[closest_fixed['gap_mm'],closest_moving['gap_mm'],fixed_table[0]['gap_mm'],moving_table[0]['gap_mm']]
        fig,ax=plt.subplots(figsize=(11,5))
        y=np.arange(4)
        bars=ax.barh(y,values,color=['tab:blue' if v>=0 else 'tab:red' for v in values])
        ax.invert_yaxis();ax.set_yticks(y,labels)
        ax.axvline(0,color='black',linewidth=1)
        ax.axvline(-.5,color='tab:orange',linestyle='--',label='Jaw/table limit: -0.5 mm')
        ax.axvline(-.25,color='tab:purple',linestyle=':',label='Jaw/bottle limit: -0.25 mm')
        ax.set_xlabel('Signed gap (mm): positive = clearance, negative = penetration')
        ax.set_title('Phase7R initial geometry: same stopped state as Phase7Q')
        ax.grid(axis='x',alpha=.25);ax.legend(loc='lower right')
        for bar,value in zip(bars,values):
            ax.text(value+(1 if value>=0 else -1),bar.get_y()+bar.get_height()/2,
                    f'{value:+.2f} mm',va='center',ha='left' if value>=0 else 'right')
        fig.tight_layout();fig.savefig(args.out/'initial_geometry_gaps_english.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4.5))
        for vals,label in ((fixed_table,'Fixed jaw patches'),(moving_table,'Moving jaw patches')):
            ax.scatter(np.arange(len(vals)),[x['gap_mm'] for x in vals],s=8,label=label,alpha=.6)
        ax.axhline(0,color='black',linewidth=1)
        ax.axhline(-.5,color='tab:red',ls='--',label='Stop threshold -0.5 mm')
        ax.set_xlabel('Collision patch index (sorted by penetration)')
        ax.set_ylabel('Minimum patch-to-table signed gap (mm)')
        ax.set_title('Which jaw collision patches intersect the tabletop?')
        ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(args.out/'patch_table_penetration_distribution.png',dpi=150);plt.close(fig)
        return report
    except Exception as exc:
        report.update(status='ERROR',error=repr(exc))
        raise
    finally:
        (args.out/'phase7r_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=json_safe)+'\n')
        print(json.dumps(dict(status=report['status'],fixed_table=report.get('fixed_jaw_table_worst',{}).get('gap_mm'),
                              moving_table=report.get('moving_jaw_table_worst',{}).get('gap_mm'),out=str(args.out))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--time',type=float,default=14.8)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--j6-closed-deg',type=float,default=-10.)
    p.add_argument('--j6-open-deg',type=float,default=100.)
    p.add_argument('--grid-radius-mm',type=float,default=50.)
    p.add_argument('--grid-step-mm',type=float,default=10.)
    a=p.parse_args()
    run(a)

if __name__=='__main__':main()
