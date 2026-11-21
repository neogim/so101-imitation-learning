#!/usr/bin/env python3
"""SO-101 Phase 7T: infer *CAD candidate*, not physical measurement, for tabletop Z.

Use all base_link collision meshes, report their transformed raw STL minima,
and rerun Episode015 t=14.8 s with the CAD floor candidate. No hardware and
no physics stepping. Never treat this as calibrated motor-to-URDF mapping.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
import jaw_collision_geometry as jaw
import phase7q_recorded_success_physics as phase7q
import phase6c_upright_reach_grasp as phase6c


def base_mesh_extrema(urdf):
    root = ET.parse(urdf).getroot()
    base = next((link for link in root.findall('link') if link.get('name')=='base_link'), None)
    if base is None:
        raise ValueError('base_link is missing')
    out=[]
    for collision in base.findall('collision'):
        mesh=collision.find('./geometry/mesh')
        if mesh is None:
            continue
        src=(urdf.parent / mesh.attrib['filename']).resolve(strict=True)
        origin=collision.find('origin')
        xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        R=Rotation.from_euler('xyz',rpy).as_matrix()
        triangles=jaw.read_stl_triangles(src)*scale
        transformed=triangles @ R.T + xyz
        z=transformed[:,:,2]
        lowest=float(z.min())
        # Favor large, approximately horizontal bottom faces when assessing
        # whether the low point could actually be a supporting contact.
        a=transformed[:,1]-transformed[:,0]
        b=transformed[:,2]-transformed[:,0]
        normal=np.cross(a,b)
        length=np.linalg.norm(normal,axis=1)
        horiz=(np.abs(normal[:,2]) / np.maximum(length,1.e-12) > .995)
        near=np.all(np.abs(z-lowest)<.0005,axis=1)
        contact_area=float(.5*length[horiz & near].sum())
        atbottom=transformed[z <= lowest+.0005]
        out.append(dict(mesh=src.name,origin_xyz_m=xyz.tolist(),origin_rpy_rad=rpy.tolist(),
                        world_base_min_z_mm=lowest*1000,
                        world_base_max_z_mm=float(z.max())*1000,
                        near_bottom_horizontal_face_area_mm2=contact_area*1e6,
                        near_bottom_xy_range_mm=[(atbottom[:,0].min()*1000,atbottom[:,0].max()*1000),
                                                 (atbottom[:,1].min()*1000,atbottom[:,1].max()*1000)] if len(atbottom) else None))
    if not out:raise ValueError('base_link has no collision STL')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('csv','mapping','urdf','out'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--time',type=float,default=14.8)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    summary=dict(status='STARTED',time_s=a.time,real_table_measured=False,
                 motor_to_urdf_calibrated=False,physics_integrated=False,
                 caveat='CAD sole floor is a candidate for a robot directly on the table, not a measurement or a verified support fixture.')
    try:
        meshes=base_mesh_extrema(a.urdf.resolve(strict=True))
        base_part=next((r for r in meshes if r['mesh']=='base_so101_v2.stl'),None)
        if base_part is None:raise ValueError('Base STL base_so101_v2.stl not found')
        # Report both the nominal base-so101 sole and the lowest of ANY base
        # mesh. A lower fixture must be investigated before choosing the floor.
        sole=base_part['world_base_min_z_mm']
        min_all=min(r['world_base_min_z_mm'] for r in meshes)
        candidate=min_all
        summary.update(base_meshes=meshes,base_sole_z_mm=sole,base_all_parts_min_z_mm=min_all,
                       candidate_table_z_mm=candidate,
                       base_part_mismatch_warning=bool(min_all < sole-1),
                       candidate_source='minimum transformed raw STL vertex among base_link collision meshes; not independent real measurement')
        times,states,_=phase7q.read_recorded_csv(a.csv)
        config=json.loads(a.mapping.read_text())
        qs=phase7q.joint_mapping(states,config,-10,100)
        q0=np.array([np.interp(a.time,times,qs[:,i]) for i in range(6)])
        opts=SimpleNamespace(urdf=a.urdf,out=a.out,dt=.002,jaw_collision_mode='axial_cross_convex_patches',
             jaw_collision_slabs=12,jaw_collision_cross_bands=3,jaw_table_contact=True,
             support_plane_z_m=candidate/1000,arm_kp=18.,arm_kd=1.8,arm_torque=3.,
             j6_kp=2.,j6_kd=.08,j6_torque=.4)
        model,qadr,_,_,_,_,fixed,moving,_,floor=phase6c.model_build(opts)
        data=mujoco.MjData(model);mujoco.mj_resetData(model,data)
        data.qpos[qadr]=q0;mujoco.mj_forward(model,data)
        def worst(geoms):
            vals=[float(mujoco.mj_geomDistance(model,data,int(i),int(floor),.2,np.zeros(6))) for i in geoms]
            j=int(np.argmin(vals));return dict(gap_mm=vals[j]*1000,
                geom=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(geoms[j])))
        f,m=worst(fixed),worst(moving)
        summary.update(fixed_jaw=f,moving_jaw=m,urdf_joint_deg=np.rad2deg(q0).tolist(),
                       mapping=config,remaining_penetration_mm=max(0,-min(f['gap_mm'],m['gap_mm'])),
                       status='CAD_TABLE_CANDIDATE_STILL_PENETRATES' if min(f['gap_mm'],m['gap_mm'])<-.5
                       else 'CAD_TABLE_CANDIDATE_NONPENETRATING_BUT_NOT_REAL_CALIBRATED')
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,5.5),facecolor='white')
        yy=['Table candidate (base min)','Fixed jaw (CAD floor)','Moving jaw (CAD floor)']
        vals=[candidate,f['gap_mm'],m['gap_mm']]
        colors=['#64748b','#dc2626' if vals[1]<0 else '#16a34a','#7c3aed' if vals[2]<0 else '#16a34a']
        bars=ax.barh(yy,vals,color=colors)
        ax.axvline(0,color='#111827',lw=2,label='Zero reference (mm)')
        for bar,v in zip(bars,vals):
            ax.text(v+(1 if v>=0 else -1),bar.get_y()+bar.get_height()/2,f'{v:+.2f} mm',
                    va='center',ha='left' if v>=0 else 'right',fontweight='bold')
        ax.set_xlabel('Table Z in base frame (first row), or jaw/table signed gap (remaining rows), mm')
        ax.set_title('Phase 7T: CAD tabletop candidate and remaining jaw/table penetration')
        ax.grid(axis='x',alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(a.out/'phase7t_cad_table_vs_penetration.png',dpi=170);plt.close(fig)
    except Exception as e:
        summary.update(status='ERROR',error=repr(e));raise
    finally:
        (a.out/'phase7t_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:summary.get(k) for k in ('status','candidate_table_z_mm','base_sole_z_mm','fixed_jaw','moving_jaw','error')},ensure_ascii=False))
if __name__=='__main__':main()
