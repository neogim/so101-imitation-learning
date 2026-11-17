#!/usr/bin/env python3
"""Phase 6G: same tabletop Muhi, root/middle/tip jaw-depth comparisons.

Simulation-only. Fixed target pose; IK moves the *arm*, never the target per
profile. All tests start from independent model/data and close J6 in dynamics.
Candidate depths are provisional gripper-frame positions requiring mesh review.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import shutil
import sys
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

import physical_grasp_j6_bench as grasp
import physical_workcell_sim as physical
import phase6c_upright_reach_grasp as phase6c

JOINTS = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')
# Provisional local Z-depth positions, not measured finger-arc percentages.
DEPTHS = (('root',-.045),('middle',-.070),('tip',-.095))
REFERENCE_LOCAL_XY=(.030,.036)  # fixed-arm bench geometry; lateral refinement is deliberately NOT automatic


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=float)+'\n',encoding='utf-8')


def fk(model, data, qadr, bid, q):
    data.qpos[qadr[:5]] = q
    mujoco.mj_forward(model,data)
    return np.asarray(data.xpos[bid]).copy(), np.asarray(data.xmat[bid]).reshape(3,3).copy()


def solve_ik(model,data,qadr,bid,lo,hi,xyz,anchor,bench):
    _,br = fk(model,data,qadr,bid,bench)
    # Optimize the *actual* grasp anchor at every candidate configuration.
    # The old fixed desired_origin used the bench rotation and ignored the
    # change of the local anchor's world position when the wrist rotated.
    seeds=[bench.copy()]
    for delta in (-.65,.65):
        s=bench.copy();s[1]=np.clip(s[1]+delta,lo[1]+.0001,hi[1]-.0001)
        s[2]=np.clip(s[2]-delta,lo[2]+.0001,hi[2]-.0001);seeds.append(s)
    for j4 in (-.4,.4):
        s=bench.copy();s[3]=np.clip(j4,lo[3]+.0001,hi[3]-.0001);seeds.append(s)
    solutions=[]
    for seed in seeds:
        def residual(q):
            p,r=fk(model,data,qadr,bid,q)
            anchor_world=p+r@anchor
            err_rot=Rotation.from_matrix(br@r.T).as_rotvec()
            return np.r_[(anchor_world-xyz)/.008,err_rot/.4,
                         (q-bench)*.12, (q[3]-bench[3])*.25]
        try:
            sol=least_squares(residual, np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=180)
        except ValueError:
            continue
        pos,rot=fk(model,data,qadr,bid,sol.x)
        anchor_xyz=pos+rot@anchor
        pos_err=float(np.linalg.norm(anchor_xyz-xyz)*1000)
        rot_err=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(br@rot.T).as_rotvec())))
        solutions.append((pos_err + .15*rot_err,sol.x.copy(),{
            'solver_success':bool(sol.success),'nfev':int(sol.nfev),
            'reference_j2_deg':float(np.rad2deg(bench[1])),
            'solved_j2_deg':float(np.rad2deg(sol.x[1])),
            'anchor_error_mm':pos_err,'orientation_error_deg':rot_err,
            'grasp_reference_world_xyz_m':anchor_xyz.tolist(),
            'ik_angles_deg':dict(zip(JOINTS,np.rad2deg(sol.x).tolist()))
        }))
    if not solutions:raise RuntimeError('No IK solutions')
    solutions.sort(key=lambda t:t[0])
    return solutions[0][1],solutions[0][2]


def search_nonreverse_contact_pose(model,qadr,bj,lo,hi,xyz,bq,
                                   fixed,moving,bottle,bench,out,args):
    """Search a physically meaningful gripper placement with J4 >= 0.

    Prior version optimized only one arbitrary grasp-body anchor and ended
    with both jaws about 20-27 mm from the bottle at fully open. Evaluate
    actual jaw-to-cylinder distances at open and closed J6, while preserving
    the bottle's fixed XYZ and the original collision/friction settings.
    The tested offsets are *gripper-frame target points*, not target moves.
    """
    import itertools
    low=lo[:5].copy()
    low[3]=max(low[3],math.radians(args.j4_nonnegative_min_deg))
    if low[3]>=hi[3]:
        raise ValueError('J4 bound does not intersect the URDF range')
    options=[]
    pose_data=mujoco.MjData(model)
    for x,y,z in itertools.product(
            (.005,.030,.055),(.011,.036,.061),(-.070,-.090)):
        anchor=np.array([x,y,z],dtype=float)
        try:
            mujoco.mj_resetData(model,pose_data)
            q,ik=solve_ik(model,pose_data,qadr,bj,low,hi[:5],xyz,anchor,bench)
            pose_data.qpos[qadr[:5]]=q
            pose_data.qpos[bq:bq+3]=xyz
            pose_data.qpos[bq+3:bq+7]=[1,0,0,0]
            pose_data.qpos[qadr[5]]=hi[5]
            mujoco.mj_forward(model,pose_data)
            open_fixed=grasp.geometry_gaps(model,pose_data,bottle,fixed)
            open_moving=grasp.geometry_gaps(model,pose_data,bottle,moving)
            pose_data.qpos[qadr[5]]=lo[5]
            mujoco.mj_forward(model,pose_data)
            closed_fixed=grasp.geometry_gaps(model,pose_data,bottle,fixed)
            closed_moving=grasp.geometry_gaps(model,pose_data,bottle,moving)
            # Prefer pre-contact clearance of roughly 1 mm for the fixed
            # jaw and a closing sweep that reaches the opposing jaw. Values
            # are heuristic, not hard-coded as a physically proven fit.
            # Penalize any initial overlap, excessive far-side gap,
            # and arm-anchor IK error. Do not penalize all penetration as
            # "missed": a small negative distance predicts contact.
            score=(1000*abs(open_fixed-.0015) +
                   1000*abs(closed_moving+.001) +
                   2500*max(0.,-open_fixed-.00025) +
                   2500*max(0.,-open_moving-.00025) +
                   400*max(0.,.004-open_moving) +
                   .4*ik['anchor_error_mm'] +
                   .008*ik['orientation_error_deg'])
            row={'local_x_mm':1000*x,'local_y_mm':1000*y,
                 'local_z_mm':1000*z,
                 'j2_deg':float(np.rad2deg(q[1])),
                 'j4_deg':float(np.rad2deg(q[3])),
                 'ik_error_mm':ik['anchor_error_mm'],
                 'orientation_error_deg':ik['orientation_error_deg'],
                 'open_fixed_gap_mm':1000*open_fixed,
                 'open_moving_gap_mm':1000*open_moving,
                 'closed_fixed_gap_mm':1000*closed_fixed,
                 'closed_moving_gap_mm':1000*closed_moving,
                 'score':float(score),
                 'valid_open':bool(min(open_fixed,open_moving)>-.0005),
                 'anchor':anchor.tolist(),
                 'q':q,'ik':ik}
            options.append(row)
        except Exception as exc:
            options.append({'local_x_mm':1000*x,'local_y_mm':1000*y,
                            'local_z_mm':1000*z,'score':None,
                            'error':repr(exc)})
    fields=('local_x_mm','local_y_mm','local_z_mm','j2_deg','j4_deg',
            'ik_error_mm','orientation_error_deg','open_fixed_gap_mm',
            'open_moving_gap_mm','closed_fixed_gap_mm','closed_moving_gap_mm',
            'score','valid_open','error')
    with (out/'j4_nonnegative_contact_search.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(options)
    # Do not silently select an overlapping open jaw, or an IK point far
    # away from the user-provided fixed bottle location.
    feasible=[r for r in options if r.get('score') is not None
              and r['valid_open'] and r['ik_error_mm']<=args.max_ik_error_mm]
    if not feasible:
        raise RuntimeError('No nonreverse contact-aware J4 candidate passed geometry/IK preflight; review search CSV')
    chosen=min(feasible,key=lambda r:r['score'])
    short={k:v for k,v in chosen.items() if k not in ('q','ik')}
    dump(out/'j4_nonnegative_selected_pose.json',
         {'chosen':short,'feasible_count':len(feasible),
          'candidate_count':len(options),
          'method':'18 local finger-anchor placements scored on actual open/closed jaw-to-bottle distances; fixed bottle XYZ; no friction edits',
          'caveat':'Geometric preflight; best score does not imply physical grasp success'})
    return chosen['q'],chosen['ik'],np.asarray(chosen['anchor']),short


def jaw_geometry(model,data,qadr,q,xyz,bq,fixed,moving,bottle,lo,hi):
    data.qpos[qadr[:5]]=q
    data.qpos[bq:bq+3]=xyz; data.qpos[bq+3:bq+7]=[1,0,0,0]
    data.qpos[qadr[5]]=hi[5];mujoco.mj_forward(model,data)
    gf=grasp.geometry_gaps(model,data,bottle,fixed)
    gm=grasp.geometry_gaps(model,data,bottle,moving)
    sweep=[]
    for angle in np.linspace(hi[5],lo[5],17):
        data.qpos[qadr[5]]=angle;mujoco.mj_forward(model,data)
        sweep.append(grasp.geometry_gaps(model,data,bottle,moving))
    return {'open_fixed_gap_mm':gf*1000,'open_moving_gap_mm':gm*1000,
            'closing_moving_min_gap_mm':min(sweep)*1000,
            'open_jaws_overlap':bool(min(gf,gm)<-0.0005),
            'closing_sweeps_bottle':bool(min(sweep)<-0.00035)}


def audit_jaw_meshes(model,fixed,moving,out):
    """Compare compiled jaw mesh surface/vertex cloud with its convex hull.

    MuJoCo mesh collision operates on convex geometry. This shows where the
    convex approximation differs, without claiming that a visual mesh is an
    independent collision mesh or that a concavity necessarily caused ejection.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    report=[]
    geoms=[('fixed',g) for g in fixed]+[('moving',g) for g in moving]
    eligible=[]
    for role,gid in geoms:
        meshid=int(model.geom_dataid[gid])
        if int(model.geom_type[gid])!=int(mujoco.mjtGeom.mjGEOM_MESH) or meshid<0:
            continue
        st=int(model.mesh_vertadr[meshid]);num=int(model.mesh_vertnum[meshid])
        verts=np.asarray(model.mesh_vert[st:st+num],dtype=float)
        if len(verts)<4:continue
        try:
            hull=ConvexHull(verts)
            volume=float(hull.volume)
            vertex_fraction=float(len(np.unique(hull.simplices))/len(verts))
            eligible.append((role,gid,verts,hull))
            report.append({'role':role,'geom_id':int(gid),
                'geom_name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(gid)),
                'mesh_id':meshid,'vertices':len(verts),
                'convex_hull_vertices':int(len(np.unique(hull.simplices))),
                'hull_vertex_fraction':vertex_fraction,'hull_volume_m3':volume,
                'note':'Mesh vertices are in compiled local mesh coordinates. Contact is evaluated on the convex representation; the hull includes concavities.'})
        except Exception as exc:
            report.append({'role':role,'geom_id':int(gid),'error':str(exc)})
    if eligible:
        # A compound collision model may contain dozens of convex slabs.
        # Plot one panel per jaw ROLE, never one huge subplot per geom.
        fig=plt.figure(figsize=(14,6))
        for k,role in enumerate(('fixed','moving')):
            shapes=[(gid,verts,hull) for label,gid,verts,hull in eligible if label==role]
            if not shapes:continue
            ax=fig.add_subplot(1,2,k+1,projection='3d')
            total=sum(len(v) for _,v,_ in shapes)
            for gid,verts,hull in shapes:
                sampled=verts[::max(1,len(verts)//max(100,1800//len(shapes)))]
                ax.scatter(sampled[:,0],sampled[:,1],sampled[:,2],s=.35,alpha=.13)
                edges=set()
                for tri in hull.simplices:
                    for j in range(3):
                        edges.add(tuple(sorted((int(tri[j]),int(tri[(j+1)%3])))))
                # Edge budget controls plot complexity in segmented collision.
                edge_list=sorted(edges)
                for a,b in edge_list[::max(1,len(edge_list)//max(150,1500//len(shapes)))]:
                    ax.plot(verts[[a,b],0],verts[[a,b],1],verts[[a,b],2],
                            linewidth=.3,alpha=.4,color='tab:red')
            allv=np.concatenate([v for _,v,_ in shapes],axis=0)
            span=np.ptp(allv,axis=0)
            center=(allv.min(axis=0)+allv.max(axis=0))/2
            radius=max(1e-5,float(max(span))/2)
            ax.set_xlim(center[0]-radius,center[0]+radius)
            ax.set_ylim(center[1]-radius,center[1]+radius)
            ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_title(f'{role} collision geoms: {len(shapes)} (vertices={total})')
            ax.set_xlabel('local X (m)');ax.set_ylabel('local Y (m)');ax.set_zlabel('local Z (m)')
        fig.suptitle('Compiled collision vertices and convex hulls (mesh-local frames)')
        fig.savefig(out/'jaw_mesh_vs_convex_hull.png',dpi=125,bbox_inches='tight')
        plt.close(fig)
    dump(out/'jaw_mesh_audit.json',{'geoms':report,
         'interpretation':'A compiled mesh and its convex hull can differ at concavities. This visualization is local mesh coordinates, not a proof of which specific jaw surface pushed the object.'})
    return report


def plot_contact_push(rows,out):
    """World-frame bottle motion and *signed* jaw-on-bottle force."""
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array([r['time_s'] for r in rows])
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    axes[0].plot(t,[1000*(r['bottle_x_m']-rows[0]['bottle_x_m']) for r in rows],label='Bottle delta world X')
    axes[0].plot(t,[1000*(r['bottle_y_m']-rows[0]['bottle_y_m']) for r in rows],label='Bottle delta world Y')
    axes[0].set_ylabel('Displacement (mm)');axes[0].legend()
    axes[1].plot(t,[r['jaw_on_bottle_force_world_x_n'] for r in rows],label='Jaw force on bottle world X')
    axes[1].plot(t,[r['jaw_on_bottle_force_world_y_n'] for r in rows],label='Jaw force on bottle world Y')
    axes[1].set_ylabel('Net jaw force (N)');axes[1].set_xlabel('Simulation time (s)')
    axes[1].legend()
    for ax in axes:
        ax.axvline(rows[0]['time_s'],color='gray',alpha=.2)
        ax.grid(alpha=.25)
    fig.suptitle('Phase 6G: bottle displacement and jaw contact force (world axes)')
    fig.savefig(out/'jaw_push_direction.png',dpi=140);plt.close(fig)


def physics(model,qadr,vadr,aids,lo,hi,xyz,bq,bd,fixed,moving,bottle,floor,q,args,out):
    d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
    d.qpos[qadr[:5]]=q;d.qpos[qadr[5]]=hi[5]
    d.qpos[bq:bq+3]=xyz;d.qpos[bq+3:bq+7]=[1,0,0,0]
    d.qvel[:]=0
    d.ctrl[aids]=np.r_[q,hi[5]]
    mujoco.mj_forward(model,d)
    support_start=float(xyz[2]-physical.MUHI_LENGTH/2)
    phases=[('settle',args.settle_s),('close',args.close_s),('hold',args.hold_s)]
    total=sum(t for _,t in phases);steps=int(math.ceil(total/args.dt))
    stride=max(1,int(round(1/(args.dt*args.fps))))
    rows=[];frames=[]
    renderer=None
    if shutil.which('ffmpeg'):
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,640)
        model.vis.global_.offheight=max(model.vis.global_.offheight,460)
        try:renderer=mujoco.Renderer(model,width=640,height=460)
        except Exception as e:(out/'render_error.txt').write_text(repr(e))
    scene_options=mujoco.MjvOption()
    scene_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True
    scene_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=True
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=xyz;camera.distance=.56;camera.azimuth=110;camera.elevation=-20
    deep=False;bad=False;maxcontact=0
    try:
        for k in range(steps+1):
            t=min(k*args.dt,total)
            if t<args.settle_s:phase='settle';j6=hi[5]
            elif t<args.settle_s+args.close_s:
                phase='close';u=(t-args.settle_s)/args.close_s
                j6=hi[5]+(lo[5]-hi[5])*u
            else:phase='hold';j6=lo[5]
            ctrl=np.r_[q,j6];d.ctrl[aids]=ctrl
            if k:mujoco.mj_step(model,d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():bad=True;break
            fc,mc,depth=grasp.jaw_contact_counts(model,d,bottle,fixed,moving)
            bilateral=int(fc>0 and mc>0)
            if depth is not None and depth<-0.003:deep=True
            fnew=[];mnew=[]
            jaw_force_world=np.zeros(3)
            contact_points_world=[]
            for ix,c in enumerate(d.contact[:d.ncon]):
                geomids={int(c.geom1),int(c.geom2)}
                if not geomids.intersection(bottle):continue
                role='fixed' if geomids.intersection(fixed) else ('moving' if geomids.intersection(moving) else None)
                if role is None:continue
                outforce=np.zeros(6)
                mujoco.mj_contactForce(model,d,ix,outforce)
                # Contact normal is the first axis of MuJoCo contact frame.
                normal=np.asarray(c.frame).reshape(3,3)[0].copy()
                # Rows of c.frame are the world-space contact-frame axes.
                # Positive local force acts from geom1 toward geom2.
                force_on_bottle=(1 if int(c.geom2) in bottle else -1)*(np.asarray(c.frame).reshape(3,3).T@outforce[:3])
                jaw_force_world+=force_on_bottle
                contact_points_world.append({'role':role,
                     'point_world_m':np.asarray(c.pos).tolist(),
                     'force_on_bottle_world_n':force_on_bottle.tolist(),
                     'normal_force_n':float(outforce[0])})
                entry={'role':role,'point_xyz_m':np.asarray(c.pos).tolist(),
                       'normal_mj_frame_world':normal.tolist(),
                       'normal_force_n':float(outforce[0]),
                       'tangent_force_n':float(np.linalg.norm(outforce[1:3]))}
                (fnew if role=='fixed' else mnew).append(entry)
            maxcontact=max(maxcontact,len(fnew)+len(mnew))
            def metric(entries,key,mode='sum'):
                v=[a[key] for a in entries]
                return float(sum(v)) if mode=='sum' else (v[0] if v else None)
            def avgpt(entries):
                return np.mean([a['point_xyz_m'] for a in entries],axis=0).tolist() if entries else None
            row={'time_s':float(d.time),'phase':phase,'fixed_contacts':fc,'moving_contacts':mc,
                 'bilateral':bilateral,'min_contact_dist_mm':None if depth is None else 1000*depth,
                 'jaw_on_bottle_force_world_x_n':float(jaw_force_world[0]),
                 'jaw_on_bottle_force_world_y_n':float(jaw_force_world[1]),
                 'jaw_on_bottle_force_world_z_n':float(jaw_force_world[2]),
                 'jaw_contact_points_world_json':json.dumps(contact_points_world),
                 'fixed_normal_force_n':metric(fnew,'normal_force_n'),
                 'moving_normal_force_n':metric(mnew,'normal_force_n'),
                 'fixed_tangent_force_n':metric(fnew,'tangent_force_n'),
                 'moving_tangent_force_n':metric(mnew,'tangent_force_n'),
                 'fixed_contact_point_xyz_m':avgpt(fnew),
                 'moving_contact_point_xyz_m':avgpt(mnew),
                 'fixed_normal_mj_frame_world':fnew[0]['normal_mj_frame_world'] if fnew else None,
                 'moving_normal_mj_frame_world':mnew[0]['normal_mj_frame_world'] if mnew else None,
                 'bottle_x_m':float(d.qpos[bq]),'bottle_y_m':float(d.qpos[bq+1]),
                 'bottle_z_m':float(d.qpos[bq+2]),
                 'bottle_xy_displacement_mm':float(1000*np.linalg.norm(d.qpos[bq:bq+2]-xyz[:2])),
                 'bottle_tilt_deg':float(np.rad2deg(2*math.acos(min(1,abs(float(d.qpos[bq+3])))))),
                 'j6_target_deg':float(np.rad2deg(j6)),'j6_actual_deg':float(np.rad2deg(d.qpos[qadr[5]])),
                 'max_arm_error_deg':float(np.max(np.abs(np.rad2deg(q-d.qpos[qadr[:5]]))))}
            rows.append(row)
            if renderer is not None and (k%stride==0 or k==steps):
                renderer.update_scene(d,camera=camera,scene_option=scene_options)
                im=Image.fromarray(renderer.render().copy())
                canvas=Image.new('RGB',(640,496),'#192532');canvas.paste(im,(0,30))
                ImageDraw.Draw(canvas).text((10,9),f'{out.name} {phase} {t:.2f}s F{fc}/M{mc} | jaw->bottle Fxy=({jaw_force_world[0]:+.2f},{jaw_force_world[1]:+.2f})N',fill='white')
                frames.append(canvas)
    finally:
        if renderer:renderer.close()
    with (out/'joint_contact_trace.csv').open('w',newline='',encoding='utf-8') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    if frames and shutil.which('ffmpeg'):
        fd=out/'frames';fd.mkdir()
        for i,im in enumerate(frames):im.save(fd/f'{i:05d}.png')
        p=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(args.fps),'-i',str(fd/'%05d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(out/'grasp.mp4')],capture_output=True,text=True)
        if p.returncode:(out/'ffmpeg_error.txt').write_text(p.stderr[-2000:])
        shutil.rmtree(fd)
    try:plot_contact_push(rows,out)
    except Exception as exc:(out/'push_plot_error.txt').write_text(repr(exc))
    holdrows=[r for r in rows if r['phase']=='hold']
    bilateral=sum(r['bilateral'] for r in holdrows)
    initial_bottom=support_start
    summary={'status':'NUMERIC_FAILURE' if bad else 'DYNAMIC_TEST_COMPLETED',
       'bilateral_hold_steps':bilateral,'hold_steps':len(holdrows),
       'bilateral_hold_fraction':bilateral/max(1,len(holdrows)),
       'max_contact_count':maxcontact,'deep_penetration_over_3mm':deep,
       'max_xy_displacement_mm':max((r['bottle_xy_displacement_mm'] for r in rows),default=None),
       'final_xy_displacement_mm':rows[-1]['bottle_xy_displacement_mm'] if rows else None,
       'final_bottle_z_m':rows[-1]['bottle_z_m'] if rows else None,
       'support_plane_z_m':initial_bottom,
       'maximum_arm_tracking_error_deg':max((r['max_arm_error_deg'] for r in rows),default=None),
       'max_normal_contact_force_n':max((r['fixed_normal_force_n']+r['moving_normal_force_n'] for r in rows),default=None),
       'video':'grasp.mp4' if (out/'grasp.mp4').exists() else None,
       'contact_force_plot':'jaw_push_direction.png' if (out/'jaw_push_direction.png').exists() else None,
       'force_frame':'world XYZ; signed force on bottle, excluding table support',
       'note':'Bilateral contact is NOT stable lift. World normal sign depends on contact geom ordering.'}
    dump(out/'physical_summary.json',summary)
    return summary


def run(args):
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    xyz=np.array([args.target_x,args.target_y,args.table_z+physical.MUHI_LENGTH/2],dtype=float)
    if not np.isfinite(xyz).all():raise ValueError('Non-finite target')
    args.support_plane_z_m=float(args.table_z)
    model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor=phase6c.model_build(args)
    collision_build=args.out/'collision_urdf'/'jaw_collision_build.json'
    try: mesh_report=audit_jaw_meshes(model,fixed,moving,args.out)
    except Exception as exc:
        mesh_report=[]
        (args.out/'jaw_mesh_audit_error.txt').write_text(repr(exc))
    bj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,grasp.FINGER_BODY)
    fj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    bq=int(model.jnt_qposadr[fj]);bd=int(model.jnt_dofadr[fj])
    bench=np.zeros(5);bench[4]=np.clip(-math.pi/2,lo[4],hi[4])
    # The sign of J2 is URDF-dependent. Pick the candidate that physically
    # raises the finger body in world Z, rather than guessing a motor sign.
    reference_j2_deg=float(np.rad2deg(bench[1]))
    reference_height_m=None
    preferred_height_m=None
    j2_raising_direction='none'
    if args.j2_raise_deg > 0:
        probe=mujoco.MjData(model)
        mujoco.mj_resetData(model,probe)
        reference_height_m=float(fk(model,probe,qadr,bj,bench)[0][2])
        candidates=[]
        delta=math.radians(args.j2_raise_deg)
        for sign in (-1,1):
            candidate=bench.copy()
            candidate[1]=float(np.clip(bench[1]+sign*delta,lo[1]+1e-5,hi[1]-1e-5))
            if abs(candidate[1]-bench[1])<1e-6:continue
            height=float(fk(model,probe,qadr,bj,candidate)[0][2])
            candidates.append((height,candidate,sign))
        if candidates:
            preferred_height_m,preferred,sign=max(candidates,key=lambda x:x[0])
            if preferred_height_m > reference_height_m + 1e-5:
                bench=preferred
                j2_raising_direction='positive' if sign>0 else 'negative'
            else:
                preferred_height_m=reference_height_m
    raised_seed_details={
       'requested_j2_raise_deg':args.j2_raise_deg,
       'original_reference_j2_deg':reference_j2_deg,
       'preferred_j2_deg':float(np.rad2deg(bench[1])),
       'j2_raising_direction':j2_raising_direction,
       'original_finger_body_world_z_m':reference_height_m,
       'preferred_finger_body_world_z_m':preferred_height_m,
       'preferred_finger_body_raise_mm':None if reference_height_m is None else
            1000*(preferred_height_m-reference_height_m),
       'note':'J2-only FK rise is a seed diagnostic, not the final IK finger height or a measured successful pose.'
    }
    print('J2 raised IK seed:',json.dumps(raised_seed_details,ensure_ascii=False),flush=True)
    results=[]
    root_gap_audit_status='NOT_RUN'
    depths=[d for d in DEPTHS if args.depth_only=='all' or d[0]==args.depth_only]
    if args.j4_compare and args.depth_only in ('all','middle'):
        # Controlled wrist comparison: same Muhi XYZ, same finger depth, same
        # collision and friction model. Only the allowable wrist-flex sign
        # changes; all remaining arm joints may re-optimize through IK.
        depths.append(('middle_j4_nonnegative',-.070))
    for i,(name,local_z) in enumerate(depths,1):
        print(f'[{i}/{len(depths)}] {name}: same object xyz={np.round(xyz,4).tolist()}',flush=True)
        sub=args.out/name;sub.mkdir()
        anchor=np.array([REFERENCE_LOCAL_XY[0],REFERENCE_LOCAL_XY[1],local_z])
        data=mujoco.MjData(model);mujoco.mj_resetData(model,data)
        try:
            ik_lo=lo[:5].copy()
            if name=='middle_j4_nonnegative':
                # 0 deg is included; cannot use the baseline ~-95 deg posture.
                ik_lo[3]=max(ik_lo[3],math.radians(args.j4_nonnegative_min_deg))
                if ik_lo[3]>=hi[3]:
                    raise ValueError('Requested nonnegative J4 minimum exceeds URDF joint limits')
            if name=='middle_j4_nonnegative' and args.j4_contact_search:
                q,ik,anchor,search_choice=search_nonreverse_contact_pose(
                    model,qadr,bj,lo,hi,xyz,bq,fixed,moving,bottle,
                    bench,args.out,args)
                ik['contact_aware_search_selected']=search_choice
            else:
                q,ik=solve_ik(model,data,qadr,bj,ik_lo,hi[:5],xyz,anchor,bench)
            ik['j4_wrist_flex_deg']=float(np.rad2deg(q[3]))
            ik['j4_constraint']='J4 >= %.3f deg' % args.j4_nonnegative_min_deg if name=='middle_j4_nonnegative' else 'original URDF limits'
            if name=='middle':
                # Audit the physically relevant center-depth pose before any
                # bottle physics, without changing any arm or object state.
                try:
                    import jaw_root_gap_diagnostics
                    jaw_root_gap_diagnostics.audit(
                        model,qadr,lo,hi,q,fixed,moving,args.out,
                        collision_build if collision_build.exists() else None)
                    root_gap_audit_status='COMPLETED'
                except Exception as exc:
                    root_gap_audit_status='ERROR: '+repr(exc)
                    (args.out/'jaw_root_gap_error.txt').write_text(repr(exc))
            geo=jaw_geometry(model,data,qadr,q,xyz,bq,fixed,moving,bottle,lo,hi)
            phys=physics(model,qadr,vadr,aids,lo,hi,xyz,bq,bd,fixed,moving,bottle,floor,q,args,sub) if ik['anchor_error_mm']<=args.max_ik_error_mm and not geo['open_jaws_overlap'] else {'status':'GEOMETRY_DIAGNOSTIC_ONLY','reason':'IK error or initial open-jaw overlap','bilateral_hold_steps':0}
        except Exception as exc:
            ik={};geo={};phys={'status':'EXECUTION_ERROR','reason':repr(exc),'bilateral_hold_steps':0}
        info={'name':name,'depth_local_z_m':local_z,'anchor_local_m':anchor.tolist(),
              'ik':ik,'jaw_geometry':geo,'physics':phys}
        dump(sub/'result.json',info)
        row={'depth':name,'local_z_mm':1000*local_z,
             'j2_seed_deg':raised_seed_details['preferred_j2_deg'],
             'j2_solved_deg':ik.get('solved_j2_deg'),
             'j4_solved_deg':ik.get('j4_wrist_flex_deg'),
             'j4_constraint':ik.get('j4_constraint'),
             'ik_error_mm':ik.get('anchor_error_mm'),'open_fixed_gap_mm':geo.get('open_fixed_gap_mm'),
             'open_moving_gap_mm':geo.get('open_moving_gap_mm'),
             'sweep_moving_min_gap_mm':geo.get('closing_moving_min_gap_mm'),
             'status':phys.get('status'),'bilateral_hold_steps':phys.get('bilateral_hold_steps'),
             'max_xy_displacement_mm':phys.get('max_xy_displacement_mm'),
             'max_normal_contact_force_n':phys.get('max_normal_contact_force_n'),
             'video':name+'/grasp.mp4' if (sub/'grasp.mp4').exists() else None}
        results.append((row,info))
        print(f" {name}: IK={row['ik_error_mm']} mm; status={row['status']}; bilateral={row['bilateral_hold_steps']}",flush=True)
    rows=[r for r,_ in results]
    with (args.out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
        labels=[r['depth'] for r in rows]
        ik_errors=[r['ik_error_mm'] if r['ik_error_mm'] is not None else float('nan') for r in rows]
        axes[0].bar(labels,ik_errors,color='#4c78a8')
        axes[0].axhline(args.max_ik_error_mm,color='#c44e52',linestyle='--',label=f'IK threshold: {args.max_ik_error_mm:g} mm')
        axes[0].set_ylabel('IK grasp-anchor position error (mm)')
        axes[0].set_title('Phase 6G geometric screening: physical runs are conditional')
        axes[0].legend(loc='upper left')
        axes[0].set_ylim(bottom=0)
        for x,(v,row) in enumerate(zip(ik_errors,rows)):
            label='No IK solution' if not np.isfinite(v) else f'{v:.1f} mm'
            axes[0].annotate(label,(x,0 if not np.isfinite(v) else v),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
        status_text=[]
        for row in rows:
            status=row['status'] or 'NO_RESULT'
            if status=='GEOMETRY_DIAGNOSTIC_ONLY':
                status_text.append('SKIPPED (IK / jaw geometry)')
            elif status=='DYNAMIC_TEST_COMPLETED':
                status_text.append('PHYSICS RUN (not grasp success)')
            else:
                status_text.append(status)
        axes[1].axis('off')
        axes[1].set_title('Execution status and physical measurements',loc='left')
        lines=['Position    Result                                   Bilateral steps    XY displacement (mm)']
        for row,label in zip(rows,status_text):
            steps=str(row['bilateral_hold_steps']) if row['status']=='DYNAMIC_TEST_COMPLETED' else 'N/A'
            displ=row['max_xy_displacement_mm']
            displ_text=f'{displ:.1f}' if displ is not None and np.isfinite(displ) and row['status']=='DYNAMIC_TEST_COMPLETED' else 'NOT MEASURED'
            lines.append(f"{row['depth']:<11} {label:<40} {steps:<18} {displ_text}")
        if not any(row['status']=='DYNAMIC_TEST_COMPLETED' for row in rows):
            lines.extend(['', 'NO PHYSICS TEST WAS RUN.', 'Zero contact bars would be misleading: data is unavailable, not a measured failure.',
                          'Review IK error and open-jaw geometry before choosing the next target pose.'])
        axes[1].text(.01,.93,'\n'.join(lines),va='top',ha='left',fontfamily='monospace',fontsize=10,transform=axes[1].transAxes)
        fig.savefig(args.out/'comparison.png',dpi=150);plt.close(fig)
    except Exception as e:(args.out/'plot_error.txt').write_text(repr(e))
    summary={'schema':'phase6g_fixed_workcell_root_middle_tip_v1','simulation_only':True,
       'target_world_xyz_m':xyz.tolist(),'support_z_m':args.table_z,
       'target_reused_across_cases':True,
       'depth_subset':args.depth_only,
       'j4_sign_comparison_enabled':args.j4_compare,
       'j4_contact_search_enabled':args.j4_contact_search,
       'j4_contact_search_csv':'j4_nonnegative_contact_search.csv' if (args.out/'j4_nonnegative_contact_search.csv').exists() else None,
       'j4_contact_selected_pose':'j4_nonnegative_selected_pose.json' if (args.out/'j4_nonnegative_selected_pose.json').exists() else None,
       'j4_nonnegative_min_deg':args.j4_nonnegative_min_deg,
       'j4_comparison_note':'middle and middle_j4_nonnegative share target, depth, jaw and physics parameters; J1-J5 are re-solved; this is a wrist-sign-constrained IK A/B, not isolated single-joint motion',
       'jaw_collision_mode':args.jaw_collision_mode,
       'jaw_collision_slabs':args.jaw_collision_slabs if args.jaw_collision_mode!='original' else None,
       'jaw_collision_cross_bands':args.jaw_collision_cross_bands if args.jaw_collision_mode=='axial_cross_convex_patches' else 1,
       'root_gap_audit_status':root_gap_audit_status,
       'root_gap_profile':'jaw_root_gap_profile.json' if (args.out/'jaw_root_gap_profile.json').exists() else None,
       'root_gap_plot':'jaw_root_gap_profile.png' if (args.out/'jaw_root_gap_profile.png').exists() else None,
       'root_section_plot':'jaw_root_section_comparison.png' if (args.out/'jaw_root_section_comparison.png').exists() else None,
       'jaw_collision_build_report':str(collision_build.relative_to(args.out)) if collision_build.is_file() else None,
       'j2_raised_seed':raised_seed_details,
       'jaw_mesh_audit':'jaw_mesh_audit.json' if (args.out/'jaw_mesh_audit.json').exists() else None,
       'jaw_mesh_figure':'jaw_mesh_vs_convex_hull.png' if (args.out/'jaw_mesh_vs_convex_hull.png').exists() else None,
       'target_is_provisional_not_measured':True,
       'depth_local_z_provisional_not_actual_finger_arc_percent':True,
       'method':'5DoF soft IK per jaw-depth, stationary-arm J6 physical close and hold; no approach/lift',
       'grasp_not_verified_by_contact_alone':True,
       'contact_normal_sign_warning':'MuJoCo contact normals use geom order; interpret axis after checking geom IDs',
       'results':[info for _,info in results]}
    dump(args.out/'summary.json',summary)
    print('RESULT_DIR:',args.out,flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True,type=Path)
    p.add_argument('--urdf',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--target-x',type=float,default=.220)
    p.add_argument('--target-y',type=float,default=0.)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--max-ik-error-mm',type=float,default=20.)
    p.add_argument('--depth-only',choices=('all','root','middle','tip'),default='all',
                   help='Restrict physics and geometry to one depth; middle recommended for root-gap regression')
    p.add_argument('--no-j4-contact-search',dest='j4_contact_search',action='store_false',
                   help='Restore old single-anchor IK for J4 nonnegative, for A/B comparison')
    p.set_defaults(j4_contact_search=True)
    p.add_argument('--no-j4-compare',dest='j4_compare',action='store_false',
                   help='Disable automatic extra middle candidate with J4 constrained nonnegative')
    p.set_defaults(j4_compare=True)
    p.add_argument('--j4-nonnegative-min-deg',type=float,default=0.,
                   help='Lower bound in degrees for J4 in additional nonreverse middle IK candidate')
    p.add_argument('--jaw-collision-mode',choices=('original','axial_convex_slabs','axial_cross_convex_patches'),
                   default='axial_cross_convex_patches',
                   help='Simulation only: preserve authentic visual jaw meshes; split collision longitudinally and transversely to reduce filling of root aperture; original restores baseline')
    p.add_argument('--jaw-collision-slabs',type=int,default=12,
                   help='Number of axial convex slices per original jaw mesh (2..32)')
    p.add_argument('--jaw-collision-cross-bands',type=int,default=3,
                   help='Crosswise convex partitions per axial slab in axial_cross_convex_patches mode (1..8)')
    p.add_argument('--j2-raise-deg',type=float,default=15.,
                   help='Prefer an IK seed raised at J2 by this angle (auto-select world-Z-positive sign); 0 disables')
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--settle-s',type=float,default=.4)
    p.add_argument('--close-s',type=float,default=1.3)
    p.add_argument('--hold-s',type=float,default=1.0)
    p.add_argument('--arm-kp',type=float,default=18.)
    p.add_argument('--arm-kd',type=float,default=1.8)
    p.add_argument('--arm-torque',type=float,default=3.)
    p.add_argument('--j6-kp',type=float,default=2.)
    p.add_argument('--j6-kd',type=float,default=.08)
    p.add_argument('--j6-torque',type=float,default=.4)
    args=p.parse_args()
    if args.dt<=0 or args.fps<=0 or args.max_ik_error_mm<=0 or args.j2_raise_deg<0 or not 2<=args.jaw_collision_slabs<=32 or not 1<=args.jaw_collision_cross_bands<=8 or not 0<=args.j4_nonnegative_min_deg<=95:p.error('Invalid args')
    run(args)

if __name__=='__main__':main()
