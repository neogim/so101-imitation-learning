#!/usr/bin/env python3
"""A/B the ORIGINAL URDF collision meshes versus split convex jaw patches.

Exactly the SAME final arm q and Muhi pose from an existing Phase 7 result.
Static forward-kinematic J6 sweep only. No IK, no mj_step, no claim of grasp.
Saves side/opposite/overhead diagnostics and distance profiles for each mode.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import phase6c_upright_reach_grasp as phase6c
import phase7c_horizontal_from_upright_baseline as phase7
import physical_grasp_j6_bench as grasp
import physical_workcell_sim as physical


def minimum_distance(model, data, first, second):
    best=(float('inf'), '', '')
    for a in first:
        for b in second:
            dist=float(mujoco.mj_geomDistance(model,data,int(a),int(b),.5,
                                               np.zeros(6)))
            if dist<best[0]:
                best=(dist,
                    mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(a)) or str(a),
                    mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(b)) or str(b))
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--result-json',type=Path,required=True,
                   help='Existing Phase7 middle/result.json with actual solved q')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--target-x',type=float,default=.220)
    p.add_argument('--target-y',type=float,default=0.)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--yaw-deg',type=float,default=0.)
    p.add_argument('--height-clearance-mm',type=float,default=.5)
    p.add_argument('--fps',type=int,default=8)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Output already exists: {args.out}')
    args.out.mkdir(parents=True)
    original=json.loads(args.result_json.read_text(encoding='utf-8'))
    qmap=original['ik']['ik_angles_deg']
    q=np.deg2rad(np.asarray([qmap[n] for n in phase7.JOINTS],dtype=float))
    rot=Rotation.from_euler('z',args.yaw_deg,degrees=True)*Rotation.from_euler(
        'y',90.,degrees=True)
    xyzw=rot.as_quat()
    quat=np.asarray([xyzw[3],xyzw[0],xyzw[1],xyzw[2]],dtype=float)
    xyz=np.asarray([args.target_x,args.target_y,
       args.table_z+physical.MUHI_RADIUS+args.height_clearance_mm/1000.])
    rows=[]
    reports=[]
    for mode in ('original','axial_cross_convex_patches'):
        folder=args.out/mode
        folder.mkdir()
        setup=SimpleNamespace(
            urdf=args.urdf.resolve(),out=folder,
            jaw_collision_mode=mode,jaw_collision_slabs=12,
            jaw_collision_cross_bands=3,support_plane_z_m=args.table_z,
            arm_kp=18.,arm_kd=1.8,arm_torque=3.,
            j6_kp=2.,j6_kd=.08,j6_torque=.4,dt=.002)
        (model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor)=(
            phase6c.model_build(setup))
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
        bq=int(model.jnt_qposadr[j])
        d=mujoco.MjData(model)
        for percent in (0,25,50,75,100):
            d.qpos[qadr[:5]]=q
            d.qpos[qadr[5]]=hi[5]+(percent/100.)*(lo[5]-hi[5])
            d.qpos[bq:bq+3]=xyz
            d.qpos[bq+3:bq+7]=quat
            mujoco.mj_forward(model,d)
            fg=minimum_distance(model,d,fixed,bottle)
            mg=minimum_distance(model,d,moving,bottle)
            floor_f=minimum_distance(model,d,(floor,),fixed)
            floor_m=minimum_distance(model,d,(floor,),moving)
            row=dict(mode=mode,j6_percent=percent,
                fixed_to_muhi_signed_mm=1000*fg[0],
                moving_to_muhi_signed_mm=1000*mg[0],
                fixed_to_table_signed_mm=1000*floor_f[0],
                moving_to_table_signed_mm=1000*floor_m[0],
                fixed_nearest_geom=fg[1],moving_nearest_geom=mg[1])
            rows.append(row)
        # Render original visuals plus collision-highlighted overlay.
        try:
            phase7.horizontal_pose_preview(model,qadr,bq,q,xyz,quat,lo,hi,
                                           folder,args.fps)
            from PIL import Image, ImageDraw
            renderer=mujoco.Renderer(model,width=640,height=460)
            if True:
                cam=mujoco.MjvCamera()
                cam.type=mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:]=xyz
                cam.distance=.5
                for gid in (*fixed,*moving):
                    model.geom_rgba[int(gid)]=[1.,.38,.07,.92]
                    model.geom_group[int(gid)]=0
                d.qpos[qadr[:5]]=q
                d.qpos[qadr[5]]=hi[5]
                d.qpos[bq:bq+3]=xyz
                d.qpos[bq+3:bq+7]=quat
                mujoco.mj_forward(model,d)
                images=[]
                for az,el in ((90.,-15.),(270.,-15.),(120.,-80.)):
                    cam.azimuth=az
                    cam.elevation=el
                    renderer.update_scene(d,camera=cam)
                    images.append(Image.fromarray(renderer.render().copy()))
                strip=Image.new('RGB',(1920,490),'#15222c')
                for k,img in enumerate(images):
                    strip.paste(img,(640*k,30))
                ImageDraw.Draw(strip).text((10,8),
                    mode.upper()+' COLLISION GEOMS HIGHLIGHTED IN ORANGE',
                    fill='white')
                strip.save(folder/'collision_geoms_3views.png')
                renderer.close()
        except Exception as exc:
            (folder/'render_error.txt').write_text(repr(exc),encoding='utf-8')
        reports.append(dict(mode=mode,fixed_geoms=len(fixed),
                            moving_geoms=len(moving),
                            jaw_table_physical_contact_enabled=False,
                            actual_physics=False,ik_reoptimized=False))
    with (args.out/'same_pose_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out/'comparison.json').write_text(json.dumps(dict(
        q_deg=np.rad2deg(q).tolist(),muhi_center_m=xyz.tolist(),
        muhi_quat_wxyz=quat.tolist(),
        method='same joint pose, same object pose, same J6 sweep',
        important='Signed distance <0 means geometric penetration, NOT physical collision-solving',
        reports=reports,measurements=rows),indent=2),encoding='utf-8')
    print('RESULT_DIR:',args.out)
    print('This A/B is GEOMETRY ONLY; do not call it a physical grasp success.')


if __name__=='__main__':
    main()
