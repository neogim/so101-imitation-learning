#!/usr/bin/env python3
"""Simulation-only root-aperture checks for SO-101 fixed and moving jaws.

Two independent observations:
 (1) local CAD-versus-generated-collision sections around the thick/root end
 (2) MuJoCo signed clearance between root jaw geom groups as J6 is closed.
The result is NOT a measured real-world gap and a min pair-distance is NOT
a guarantee of unobstructed aperture throughout the whole section.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import mujoco

from jaw_collision_geometry import read_stl_triangles


def _root_indices(part):
    pieces=part['parts']
    slabs=max(p['slab'] for p in pieces)+1
    count=max(2,slabs//4)
    side=part['root_side_heuristic']
    return set(range(count)) if side=='low_index' else set(range(slabs-count,slabs))


def _geom_sets(model,report,all_fixed,all_moving):
    groups={}
    for body,original in [('gripper_link',all_fixed),
                          ('moving_jaw_so101_v1_link',all_moving)]:
        if report is None:
            groups[body]=tuple(original)
            continue
        jaw=report['jaw_parts'][body]
        indices=_root_indices(jaw)
        selected=[]
        for i,entry in enumerate(jaw['parts']):
            if entry['slab'] not in indices:
                continue
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,
                                 f'{body}_collision_patch_{i:03d}')
            if gid>=0:
                selected.append(gid)
        if not selected:
            raise ValueError(f'Cannot map root collision patches in {body}')
        groups[body]=tuple(selected)
    return groups


def _pair_distance(model,data,aa,bb):
    buf=np.zeros(6,dtype=float)
    return min(float(mujoco.mj_geomDistance(model,data,int(a),int(b),0.20,buf))
               for a in aa for b in bb)


def gap_profile(model,qadr,angles,pose5,root_fixed,root_moving,
                fixed,moving):
    data=mujoco.MjData(model)
    rows=[]
    for angle in angles:
        mujoco.mj_resetData(model,data)
        data.qpos[qadr[:5]]=pose5
        data.qpos[qadr[5]]=angle
        mujoco.mj_forward(model,data)
        root_dist=_pair_distance(model,data,root_fixed,root_moving)
        full_dist=_pair_distance(model,data,fixed,moving)
        rows.append({'j6_deg':float(np.rad2deg(angle)),
                     'root_min_pair_distance_mm':float(root_dist*1000),
                     'all_jaw_min_pair_distance_mm':float(full_dist*1000),
                     'root_gap_or_overlap':'overlap' if root_dist<0 else 'clearance'})
    return rows


def draw_section(report,generated_dir,out):
    """2-D projected root-end silhouette of original versus generated collision."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from jaw_collision_geometry import read_stl_triangles
    fig,axs=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    for row,body in enumerate(('gripper_link','moving_jaw_so101_v1_link')):
        part=report['jaw_parts'][body]
        source=Path(report['source_urdf']).parent/'assets'/part['visual_stl']
        source_points=read_stl_triangles(source).reshape(-1,3)
        axis=np.asarray(part['principal_axis_mesh_local'])
        transverse=np.asarray(part['transverse_axis_mesh_local'])
        other=np.cross(axis,transverse)
        root_slabs=_root_indices(part)
        chosen=[p for p in part['parts'] if p['slab'] in root_slabs]
        lo=min(p['axial_range_mesh_units'][0] for p in chosen)
        hi=max(p['axial_range_mesh_units'][1] for p in chosen)
        mask=(source_points@axis>=lo-1e-8)&(source_points@axis<=hi+1e-8)
        orig=source_points[mask]
        collision=np.concatenate([read_stl_triangles(generated_dir/'jaw_collision'/p['file']).reshape(-1,3)
                                  for p in chosen],axis=0)
        # Two orthogonal projections are both necessary: a gap may be invisible
        # in one view. This is not a boolean solid-volume cross-section.
        for col,(secondary,label) in enumerate(((transverse,'transverse'),(other,'third axis'))):
            ax=axs[row,col]
            for cloud,color,name in ((orig,'tab:blue','original STL'),
                                     (collision,'tab:red','convex patches')):
                plotted=cloud[::max(1,len(cloud)//15000)]
                ax.scatter(plotted@axis*1000,plotted@secondary*1000,
                           s=.25,alpha=.22,c=color,label=name)
            ax.set_title(f'{body}: root-end projection / {label}')
            ax.set_xlabel('jaw-longitudinal coordinate (mm)')
            ax.set_ylabel('jaw-cross coordinate (mm)')
            ax.set_aspect('equal',adjustable='datalim')
            if row==0 and col==0:ax.legend(loc='best',markerscale=9,fontsize=8)
    fig.suptitle('Root-end CAD versus compound collision projections (independent jaw-local frames)')
    fig.savefig(out/'jaw_root_section_comparison.png',dpi=135)
    plt.close(fig)


def audit(model,qadr,low,high,pose5,fixed,moving,out,report_file=None):
    out=Path(out)
    report=json.loads(Path(report_file).read_text()) if report_file and Path(report_file).exists() else None
    root=_geom_sets(model,report,fixed,moving)
    rows=gap_profile(model,qadr,np.linspace(high[5],low[5],9),pose5,
                     root['gripper_link'],
                     root['moving_jaw_so101_v1_link'],fixed,moving)
    result={'description':'Root jaw clearance in MuJoCo; positive=clearance, negative=penetration',
            'root_selection':('thicker axial end of jaw CAD' if report else 'all jaw geoms; no root segmentation'),
            'is_true_cross_section':False,
            'physical_root_gap_measured':False,
            'root_fixed_geom_count':len(root['gripper_link']),
            'root_moving_geom_count':len(root['moving_jaw_so101_v1_link']),
            'samples':rows}
    (out/'jaw_root_gap_profile.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([r['j6_deg'] for r in rows],[r['root_min_pair_distance_mm'] for r in rows],
            'o-',label='Root-end pair min gap')
    ax.plot([r['j6_deg'] for r in rows],[r['all_jaw_min_pair_distance_mm'] for r in rows],
            's--',label='Whole jaw pair min gap')
    ax.axhline(0,color='black',linewidth=.8)
    ax.set_xlabel('J6 joint angle (deg)   [open to closed]')
    ax.set_ylabel('Signed minimum surface distance (mm)')
    ax.set_title('SO-101 root-end gap across J6')
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(out/'jaw_root_gap_profile.png',dpi=140,bbox_inches='tight')
    plt.close(fig)
    if report is not None:
        try:
            draw_section(report,Path(report_file).parent,out)
        except Exception as exc:
            (out/'jaw_root_section_error.txt').write_text(repr(exc))
    return result
