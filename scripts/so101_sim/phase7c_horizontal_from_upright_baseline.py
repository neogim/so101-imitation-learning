#!/usr/bin/env python3
"""Phase 7C: horizontal bottle physics from proven Phase6G convex jaw benchmark.

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
import real_pose_safety as real_safety

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
    baseline_choice={k:v for k,v in chosen.items() if k not in ('q','ik')}
    chosen_q=chosen['q'].copy()
    chosen_anchor=np.asarray(chosen['anchor'],dtype=float)
    refined_report=None
    if args.j4_bilateral_refine:
        import j4_bilateral_contact_refinement
        chosen_q,chosen_anchor,refined_report=j4_bilateral_contact_refinement.refine(
            model,qadr,bj,bq,xyz,fixed,moving,bottle,
            lo,hi,hi[5],lo[5],options,args.max_ik_error_mm,
            out,j4_min_deg=args.j4_nonnegative_min_deg)
    # Recompute the IK diagnostics for the ACTUAL selected/refined posture.
    # The refinement may choose a different starting seed.
    origin=(sorted(feasible,key=lambda r:r['score'])[
        int(refined_report['chosen_seed_rank'])-1] if refined_report else chosen)
    pos,rot=fk(model,pose_data,qadr,bj,chosen_q)
    anchor_xyz=pos+rot@chosen_anchor
    _,bench_rot=fk(model,pose_data,qadr,bj,bench)
    ik=dict(origin['ik'])
    ik['refinement_source_seed_rank']=(
        int(refined_report['chosen_seed_rank']) if refined_report else None)
    ik.update({'solved_j2_deg':float(np.rad2deg(chosen_q[1])),
               'anchor_error_mm':float(np.linalg.norm(anchor_xyz-xyz)*1000),
               'orientation_error_deg':float(np.rad2deg(
                   np.linalg.norm(Rotation.from_matrix(bench_rot@rot.T).as_rotvec()))),
               'grasp_reference_world_xyz_m':anchor_xyz.tolist(),
               'ik_angles_deg':dict(zip(JOINTS,np.rad2deg(chosen_q).tolist())),
               'ik_refinement':'paired contact witnesses, J4 >= 0' if refined_report else '18-anchor screening'})
    short=dict(baseline_choice)
    short.update({'selected_anchor_local_mm':(1000*chosen_anchor).tolist(),
                  'selected_j4_deg':float(np.rad2deg(chosen_q[3])),
                  'selected_j2_deg':float(np.rad2deg(chosen_q[1])),
                  'selected_ik_error_mm':ik['anchor_error_mm'],
                  'bilateral_refinement_used':bool(refined_report),
                  'predicted_bilateral_contact':(
                      refined_report['chosen_predicted_paired'] if refined_report else None),
                  'bilateral_refinement_report':(
                      'j4_bilateral_refinement.json' if refined_report else None)})
    dump(out/'j4_nonnegative_selected_pose.json',
         {'chosen':short,'unrefined_seed':baseline_choice,
          'feasible_count':len(feasible),'candidate_count':len(options),
          'method':'18-anchor IK preflight, then optional bounded 5-joint refinement of simultaneous fixed/moving jaw contact witnesses; bottle and friction unchanged',
          'caveat':'Predicted bilateral proximity is NOT confirmed grasp; inspect dynamic contacts and motion'})
    return chosen_q,ik,chosen_anchor,short


def jaw_geometry(model,data,qadr,q,xyz,bq,fixed,moving,bottle,lo,hi,object_quat):
    data.qpos[qadr[:5]]=q
    data.qpos[bq:bq+3]=xyz; data.qpos[bq+3:bq+7]=object_quat
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


def actual_finger_long_axis_local(args):
    """Read the REAL STL principal axis instead of assuming link Y/Z."""
    import xml.etree.ElementTree as ET
    report_path=args.out/'collision_urdf'/'jaw_collision_build.json'
    report=json.loads(report_path.read_text())
    part=report['jaw_parts'][grasp.FINGER_BODY]
    direction=np.asarray(part['principal_axis_mesh_local'],float)
    root=ET.parse(args.urdf).getroot()
    link=next(x for x in root.findall('link')
              if x.get('name')==grasp.FINGER_BODY)
    mesh_name=part['visual_stl']
    collision=next(x for x in link.findall('collision')
        if x.find('./geometry/mesh') is not None
        and Path(x.find('./geometry/mesh').get('filename','')).name==mesh_name)
    origin=collision.find('origin')
    rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
    direction=Rotation.from_euler('xyz',rpy).apply(direction)
    return direction/np.linalg.norm(direction)


def finger_vs_bottle_axis_angle(model,data,qadr,bid,q,finger_local,object_axis):
    """0 degrees means fingers lie PERPENDICULAR to Muhi long axis."""
    _,rot=fk(model,data,qadr,bid,q)
    f=rot@finger_local
    cosine=abs(float(np.dot(f,object_axis)))
    return float(np.rad2deg(np.arcsin(np.clip(cosine,0.,1.))))


def jaw_centerline_bracketing(model,data,fixed,moving,bottle,xyz):
    """Object must lie on the INSIDE of the fixed jaw, facing the moving jaw.

    This uses collision-patch geometry centers rather than the artificial
    IK anchor or the unsigned nearest-geom distance. It is a positional
    bracketing constraint, NOT a physical contact success claim.
    """
    fc=np.mean(np.asarray(data.geom_xpos)[list(fixed)],axis=0)
    mc=np.mean(np.asarray(data.geom_xpos)[list(moving)],axis=0)
    v=mc-fc
    length=float(np.linalg.norm(v))
    if length<1e-7:
        return dict(valid=False,reason='JAW_CENTROIDS_COINCIDE',
                    along_mm=0.,jaw_separation_mm=length*1000.,
                    lateral_mm=float('inf'))
    direction=v/length
    delta=xyz-fc
    along=float(np.dot(delta,direction))
    lateral=float(np.linalg.norm(delta-along*direction))
    # A real cylinder between the claws should have its center beyond
    # the fixed jaw's inner face, and before the moving jaw's inner face.
    # Closest point pair: fixed collision patch -> bottle side surface.
    # The object can be 1.5mm from the OUTER face, so the sign matters.
    nearest=(float('inf'),None,None)
    for fgid in fixed:
        for bgid in bottle:
            witness=np.zeros(6,dtype=float)
            distance=float(mujoco.mj_geomDistance(
                model,data,int(fgid),int(bgid),.3,witness))
            if distance<nearest[0]:
                nearest=(distance,witness[:3].copy(),witness[3:].copy())
    fx=nearest[1]
    bx=nearest[2]
    if fx is None or bx is None:
        inward=0.
    else:
        from_fixed_to_muhi=bx-fx
        inward=float(np.dot(from_fixed_to_muhi,direction))
    return dict(valid=bool(0.<along<length and inward>0.),
                along_mm=along*1000.,jaw_separation_mm=length*1000.,
                lateral_mm=lateral*1000.,
                fixed_surface_inward_projection_mm=inward*1000.,
                fixed_surface_inward=bool(inward>0.),
                fixed_nearest_surface_world_m=None if fx is None else fx.tolist(),
                muhi_nearest_surface_world_m=None if bx is None else bx.tolist(),
                fixed_center_world_m=fc.tolist(),
                moving_center_world_m=mc.tolist(),
                between_centroids=bool(0.<along<length))


def center_band_contact_witness(model,data,patches,bottle,xyz,axis):
    """Nearest real-patch / real-Muhi witnesses and axial position.
    End-face contacts are rejected; no artificial central collision geom.
    """
    best=(float('inf'),None,None)
    for gid in patches:
        for bgid in bottle:
            x=np.zeros(6,dtype=float)
            dist=float(mujoco.mj_geomDistance(
                model,data,int(gid),int(bgid),.3,x))
            if dist<best[0]:
                best=(dist,x[:3].copy(),x[3:].copy())
    if best[1] is None:
        return dict(distance_mm=1e6,axial_offset_mm=1e6,
                    side_vector_world=[0.,0.,0.],point_world_m=None)
    finger_point,bottle_point=best[1],best[2]
    axial=float(np.dot(bottle_point-xyz,axis))
    radial=finger_point-xyz
    radial=radial-np.dot(radial,axis)*axis
    mag=np.linalg.norm(radial)
    unit=radial/mag if mag>1e-8 else np.zeros(3)
    return dict(distance_mm=best[0]*1000.,
        axial_offset_mm=abs(axial)*1000.,
        signed_axial_offset_mm=axial*1000.,
        side_vector_world=unit.tolist(),
        point_world_m=bottle_point.tolist())


def arm_tool_floor_clearance(model,data,floor,tool_geoms):
    """Signed geomDistance floor-to-jaws and wrist/tool collision geoms.

    Evaluate independently from the collision bit masks, which deliberately
    differ between upright Phase6G and this horizontal diagnostic.
    """
    nearest=(float('inf'),None)
    for gid in tool_geoms:
        val=float(mujoco.mj_geomDistance(model,data,int(floor),int(gid),.25,
                                         np.zeros(6,dtype=float)))
        if val<nearest[0]:
            nearest=(val,int(gid))
    return dict(gap_mm=nearest[0]*1000.,
                nearest_geom=(None if nearest[1] is None else
                    mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,nearest[1])))


def horizontal_jaw_clearance_search(model,data,qadr,bq,lo,hi,xyz,quat,
                                    fixed,moving,bottle,floor,tool_geoms,q_initial,anchor,bid,args,out,
                                    real_samples):
    """Contact-first 5-DOF IK for the horizontal Muhi grasp -> J6.

    Critical: J5 is NOT traded away for smaller nearest-geom distance.
    The horizontal projection of the actual STL finger axis must be 90 deg
    from the horizontal bottle long axis, and have nonzero projection.
    After using J1/J5 for initialization, refine ALL five arm joints against
    the fixed jaw's actual inner collision face. J6 stays open until geometry
    and measured joint-path preflight pass.
    """
    from scipy.optimize import least_squares
    finger_local=actual_finger_long_axis_local(args)
    object_axis=Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).apply([0.,0.,1.])
    obj_xy=np.asarray(object_axis[:2],float)
    obj_xy/=max(1e-10,float(np.linalg.norm(obj_xy)))
    angle_limit=args.max_finger_perpendicular_error_deg
    central_half_mm=args.central_grasp_half_mm
    trials=[]
    stages=[]

    def top_view(q):
        p,r=fk(model,data,qadr,bid,q)
        world=r@finger_local
        horizontal=np.asarray(world[:2],float)
        proj=float(np.linalg.norm(horizontal))
        if proj<1e-8:
            return 90.,proj,world.tolist()
        unit=horizontal/proj
        error=float(np.degrees(np.arcsin(
            np.clip(abs(np.dot(unit,obj_xy)),0.,1.))))
        return error,proj,world.tolist()

    def heading_to_muhi(q):
        """Signed plan-view heading error of base->FIXED JAW versus base->Muhi.

        Use the real collision patch centers instead of the synthetic IK
        anchor. Zero error means the J1 ray faces Muhi; it does NOT mean
        that the jaws themselves touch the cylinder.
        """
        data.qpos[qadr[:5]]=q
        data.qpos[qadr[5]]=hi[5]
        data.qpos[bq:bq+3]=xyz
        data.qpos[bq+3:bq+7]=quat
        mujoco.mj_forward(model,data)
        base=np.asarray(data.xanchor[mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_JOINT,'shoulder_pan')],float)[:2]
        jaw=np.mean([np.asarray(data.geom_xpos[int(g)],float)
                     for g in fixed],axis=0)[:2]
        target=np.asarray(xyz[:2],float)
        ray=jaw-base
        goal=target-base
        if min(np.linalg.norm(ray),np.linalg.norm(goal))<.005:
            return 180.,jaw.tolist(),target.tolist()
        ray/=np.linalg.norm(ray)
        goal/=np.linalg.norm(goal)
        angle=np.degrees(np.arctan2(
            ray[0]*goal[1]-ray[1]*goal[0],np.dot(ray,goal)))
        return float(angle),jaw.tolist(),target.tolist()

    def aim_j1(seed):
        """Stage 1: adjust ONLY J1, preserving recorded J2/J3/J4/J5."""
        from scipy.optimize import minimize_scalar
        width=np.deg2rad(args.max_j1_heading_from_recorded_deg)
        lower=max(float(lo[0]+.0002),float(seed[0]-width))
        upper=min(float(hi[0]-.0002),float(seed[0]+width))
        if upper<=lower:raise ValueError('J1 heading interval empty')
        def objective(j1):
            q=seed.copy();q[0]=j1
            error,_,_=heading_to_muhi(q)
            # Small tie-break favoring less change from observed pose.
            return error*error + .002*np.rad2deg(j1-seed[0])**2
        grid=np.linspace(lower,upper,25)
        ranked=sorted((objective(float(v)),float(v)) for v in grid)
        step=(upper-lower)/24.
        candidates=[ranked[0][1]]
        for _,v in ranked[:3]:
            fit=minimize_scalar(objective,
                 bounds=(max(lower,v-step),min(upper,v+step)),
                 method='bounded',options={'maxiter':40,'xatol':1e-4})
            candidates.append(float(fit.x))
        best=min(candidates,key=objective)
        result=seed.copy();result[0]=best
        return result

    def align_j5(q):
        """Stage 1: only J5 is swept and adjusted; J1..J4 remain fixed."""
        # Limit J5 rotation around the actually observed joint configuration.
        # Never use a +/-150deg sweep disconnected from the real pose.
        width=np.deg2rad(args.max_j5_from_recorded_deg)
        lower=float(max(lo[4]+.0002,q[4]-width))
        upper=float(min(hi[4]-.0002,q[4]+width))
        if upper<=lower:raise ValueError('No J5 motion range at recorded seed')
        grid=np.linspace(lower,upper,33)
        possibilities=[]
        for j5 in grid:
            candidate=q.copy();candidate[4]=float(j5)
            err,proj,_=top_view(candidate)
            cost=err+180.*max(0.,.55-proj)
            possibilities.append((cost,candidate,err,proj))
        possibilities.sort(key=lambda v:v[0])
        result=possibilities[0][1]
        # Small local refinement of J5 only, around grid minima.
        from scipy.optimize import minimize_scalar
        step=(upper-lower)/32
        for _,pre,_,_ in possibilities[:4]:
            center=float(pre[4])
            a=max(lower,center-step)
            b=min(upper,center+step)
            if b<=a:continue
            def obj(j5):
                candidate=q.copy();candidate[4]=j5
                err,proj,_=top_view(candidate)
                return err+180.*max(0.,.55-proj)
            fit=minimize_scalar(obj,bounds=(a,b),method='bounded',
                                options={'xatol':.0001,'maxiter':35})
            candidate=q.copy();candidate[4]=fit.x
            if obj(candidate[4])<obj(result[4]):result=candidate
        return result

    def measure(q):
        data.qpos[qadr[:5]]=q
        data.qpos[qadr[5]]=hi[5]
        data.qpos[bq:bq+3]=xyz
        data.qpos[bq+3:bq+7]=quat
        mujoco.mj_forward(model,data)
        fg=float(grasp.geometry_gaps(model,data,bottle,fixed))*1000.
        mg=float(grasp.geometry_gaps(model,data,bottle,moving))*1000.
        bracket=jaw_centerline_bracketing(model,data,fixed,moving,bottle,xyz)
        open_floor=arm_tool_floor_clearance(model,data,floor,tool_geoms)
        fw=center_band_contact_witness(model,data,fixed,bottle,xyz,object_axis)
        data.qpos[qadr[5]]=hi[5]+.8*(lo[5]-hi[5])
        mujoco.mj_forward(model,data)
        m80=float(grasp.geometry_gaps(model,data,bottle,moving))*1000.
        data.qpos[qadr[5]]=lo[5]
        mujoco.mj_forward(model,data)
        m100=float(grasp.geometry_gaps(model,data,bottle,moving))*1000.
        closed_floor=arm_tool_floor_clearance(model,data,floor,tool_geoms)
        mw=center_band_contact_witness(model,data,moving,bottle,xyz,object_axis)
        opposition=float(np.dot(fw['side_vector_world'],mw['side_vector_world']))
        pos,rot=fk(model,data,qadr,bid,q)
        anchor_err=float(np.linalg.norm((pos+rot@anchor)-xyz)*1000.)
        err,projection,_=top_view(q)
        return (fg,mg,m80,m100,anchor_err,err,projection,
                bracket,fw,mw,opposition,open_floor,closed_floor)

    # Initial states are ACTUAL recorded follower joint vectors. No invented
    # +/- J2/J3 shifts, forced positive J4, or distant upright IK anchors.
    # Rank by FK of the physical jaw reference and take diverse neighbors.
    possible=[]
    for i,real_deg in enumerate(real_samples):
        seed=np.deg2rad(np.asarray(real_deg[:5],dtype=float))
        if np.any(seed<lo[:5]+.001) or np.any(seed>hi[:5]-.001):
            continue
        p,r=fk(model,data,qadr,bid,seed)
        distance=float(np.linalg.norm(p+r@anchor-xyz)*1000.)
        possible.append((distance,i,seed))
    possible.sort(key=lambda t:t[0])
    seeds=[]
    for distance,i,q in possible:
        if any(np.max(np.abs(np.rad2deg(q[1:4]-prev[1:4])))<args.seed_diversity_deg
               for _,prev in seeds):
            continue
        seeds.append((f'recorded_state_{i:05d}_fkerr_{distance:.1f}mm',q))
        if len(seeds)>=args.real_seed_count:break
    if not seeds:
        raise RuntimeError('No recorded follower poses inside ORIGINAL URDF joint limits')
    dump(out/'real_recorded_seed_selection.json',
         [dict(seed=label,q_deg=np.rad2deg(q).tolist(),
               source='LeRobot observation.state; measured follower, NOT IK')
          for label,q in seeds])
    best=None

    for label,seed in seeds:
        try:
            # STEP 1: J1 aims the whole arm at Muhi, before reach IK.
            # Keep J2/J3/J4/J5 at recorded values during rotation.
            # Reject recorded seed with nonadjacent self-intersection.
            recorded_safety=real_safety.audit_candidate(model,data,qadr,seed,
                real_samples,ignored_geom_ids=bottle,
                max_axis_deg=args.real_pose_max_axis_deg,
                max_l2_deg=args.real_pose_max_l2_deg,
                penetration_tolerance_mm=args.self_collision_tolerance_mm)
            if not recorded_safety['valid']:
                trials.append(dict(seed=label,status='RECORDED_SEED_SELF_COLLISION_OR_LIMIT',
                    real_pose_safety=recorded_safety,valid_for_physics=False))
                continue
            q_heading=aim_j1(seed)
            heading_error,jaw_xy,goal_xy=heading_to_muhi(q_heading)
            heading_safety=real_safety.audit_candidate(model,data,qadr,q_heading,
                real_samples,ignored_geom_ids=bottle,
                max_axis_deg=args.real_pose_max_axis_deg,
                max_l2_deg=args.real_pose_max_l2_deg,
                penetration_tolerance_mm=args.self_collision_tolerance_mm)
            if not heading_safety['valid']:
                trials.append(dict(seed=label,status='J1_HEADING_COLLISION',
                    heading_error_deg=heading_error,
                    real_pose_safety=heading_safety,valid_for_physics=False))
                continue
            stages.append(dict(seed=label,stage='1_J1_heading',
                heading_error_deg=heading_error,
                fixed_jaw_xy_m=jaw_xy,muhi_center_xy_m=goal_xy,
                j1_start_deg=float(np.rad2deg(seed[0])),
                j1_aligned_deg=float(np.rad2deg(q_heading[0])),
                q_deg=np.rad2deg(q_heading).tolist()))
            # STEP 2: J5 re-aligns the finger after the J1 yaw.
            q_first=align_j5(q_heading)
            aligned_safety=real_safety.audit_candidate(model,data,qadr,q_first,
                real_samples,ignored_geom_ids=bottle,
                max_axis_deg=args.real_pose_max_axis_deg,
                max_l2_deg=args.real_pose_max_l2_deg,
                penetration_tolerance_mm=args.self_collision_tolerance_mm)
            if not aligned_safety['valid']:
                trials.append(dict(seed=label,status='J5_ALIGNMENT_COLLISION',
                    real_pose_safety=aligned_safety,valid_for_physics=False))
                continue
            first_err,first_proj,first_axis=top_view(q_first)
            stages.append(dict(seed=label,stage='1_J5_top_view_align',
                  j5_deg=float(np.rad2deg(q_first[4])),
                  top_view_perpendicular_error_deg=first_err,
                  finger_horizontal_projection=first_proj,
                  finger_world_axis=first_axis,
                  q_deg=np.rad2deg(q_first).tolist()))
            # STEP 3: first solve ONE explicit 3-D Cartesian target.
            # Unlike mj_geomDistance against the whole bottle, the target
            # cannot drift from the cylindrical center toward an end cap.
            import fixed_jaw_center_target_ik as center_ik
            q, center_target, center_trials = center_ik.solve(
                model, data, qadr, bq, seed, q_first, lo, hi,
                xyz, quat, object_axis, fixed, moving, bottle,
                physical.MUHI_RADIUS,
                max_j1_deg=args.max_j1_heading_from_recorded_deg,
                max_j23_deg=args.max_approach_j123_deg,
                max_j4_deg=35.,
                max_j5_deg=args.max_j5_from_recorded_deg,
                max_nfev=max(120,args.horizontal_ik_max_nfev))
            target_error_mm = center_target['error_mm']
            stages.append(dict(seed=label, stage='3_center_side_cartesian_target',
                target_world_m=center_target['target_world_m'],
                fixed_surface_world_m=center_target['fixed_surface_world_m'],
                position_error_mm=target_error_mm,
                q_deg=np.rad2deg(q).tolist()))
            (fg,mg,m80,m100,err,angle,proj,bracket,fw,mw,opposition,
                  open_floor,closed_floor)=measure(q)
            # Hardware-observed J2/J3/J4 envelope + nonadjacent self collision.
            # The object (Muhi) is excluded; bottle contact is intentional.
            safety=real_safety.audit_candidate(model,data,qadr,q,real_samples,
                ignored_geom_ids=bottle,
                max_axis_deg=args.real_pose_max_axis_deg,
                max_l2_deg=args.real_pose_max_l2_deg,
                penetration_tolerance_mm=args.self_collision_tolerance_mm)
            # Audit the initialization and final 5-DOF IK path.
            path_audit=[]
            for stage,start,finish in (
                ('J1_heading',seed,q_heading),('J5_align',q_heading,q_first),
                ('five_dof_inner_face_IK',q_first,q)):
                for alpha in np.linspace(0.,1.,args.preflight_path_steps):
                    intermediate=start+(finish-start)*float(alpha)
                    checkpoint=real_safety.audit_candidate(model,data,qadr,intermediate,
                        real_samples,ignored_geom_ids=bottle,
                        max_axis_deg=args.real_pose_max_axis_deg,
                        max_l2_deg=args.real_pose_max_l2_deg,
                        penetration_tolerance_mm=args.self_collision_tolerance_mm)
                    path_audit.append(dict(stage=stage,alpha=float(alpha),
                        valid=checkpoint['valid'],
                        penetrating_pairs=checkpoint['nonadjacent_self_collision']['penetrating_pairs'],
                        real_pose_distance_deg=checkpoint['observed_real_pose_proximity']['joint234_euclidean_difference_deg']))
            path_valid=all(check['valid'] for check in path_audit)
            final_heading_error,final_jaw_xy,_=heading_to_muhi(q)
            valid=(safety['valid'] and path_valid
                   and target_error_mm<=2. and 1.<=fg<=2. and fw['axial_offset_mm']<=2.
                   and mg>=1. and angle<=angle_limit
                   and proj>=.55 and (mg-m80)>=2.
                   and bracket['between_centroids']
                   and bracket['fixed_surface_inward']
                   and fw['axial_offset_mm']<=central_half_mm
                   and mw['axial_offset_mm']<=central_half_mm
                   and opposition<=-.35
                   and min(open_floor['gap_mm'],closed_floor['gap_mm'])
                       >=args.min_tool_floor_gap_mm)
            score=(25.*target_error_mm+3.*abs(fg-1.5)+20.*max(0.,1.-mg)+
                   .3*abs(m80)+.1*err+2.*angle+
                   80.*max(0.,.55-proj)+
                   70.*float(not bracket['fixed_surface_inward'])+
                   30.*float(not bracket['between_centroids'])+
                   5.*max(0.,fw['axial_offset_mm']-central_half_mm)+
                   5.*max(0.,mw['axial_offset_mm']-central_half_mm)+
                   50.*max(0.,opposition+.35)+
                   8.*max(0.,args.min_tool_floor_gap_mm-open_floor['gap_mm'])+
                   8.*max(0.,args.min_tool_floor_gap_mm-closed_floor['gap_mm'])+
                   0.*max(0.,abs(final_heading_error)-args.max_j1_heading_error_deg))
            row=dict(seed=label,score=float(score),
                 stage='2_five_dof_inner_face_contact_IK',
                 open_fixed_gap_mm=fg,open_moving_gap_mm=mg,
                 fixed_jaw_inside_bracketing=bracket,
                 fixed_contact_witness=fw,moving_contact_witness=mw,
                 opposite_radial_sides_cos=opposition,
                 open_tool_floor_clearance=open_floor,
                 closed_tool_floor_clearance=closed_floor,
                 tool_floor_ok=bool(min(open_floor['gap_mm'],
                    closed_floor['gap_mm'])>=args.min_tool_floor_gap_mm),
                 center_band_half_mm=central_half_mm,
                 both_contacts_central=bool(fw['axial_offset_mm']<=central_half_mm
                       and mw['axial_offset_mm']<=central_half_mm),
                 fixed_jaw_inside_valid=bool(bracket['between_centroids']
                       and bracket['fixed_surface_inward']),
                 closing_80pct_moving_gap_mm=m80,
                 closed_moving_gap_mm=m100,anchor_error_mm=err,
                 finger_perpendicular_error_deg=angle,
                 top_view_perpendicular_error_deg=angle,
                 finger_horizontal_projection=proj,
                 real_pose_safety=safety,
                 recorded_real_seed_deg=np.rad2deg(seed).tolist(),
                 joint_path_validation=path_audit,
                 joint_path_valid=bool(path_valid),
                 j4_reverse_bend_deg=float(np.rad2deg(q[3])),
                 j4_branch='recorded_follower_bounded_5dof_contact_ik',
                 j1_j4_j5_locked_during_reach=False,
                 ik_method='explicit_center_side_cartesian_fixed_jaw_IK',
                 center_target_ik=center_target,
                 center_target_all_anchors=center_trials,
                 center_target_error_mm=target_error_mm,
                 center_target_valid=bool(target_error_mm<=2.),
                 fixed_target_inner_gap_mm=[1.,2.],
                 fixed_target_axial_center_mm=[-2.,2.],
                 heading_error_deg=float(final_heading_error),
                 heading_valid=bool(abs(final_heading_error)<=args.max_j1_heading_error_deg),
                 heading_jaw_xy_m=final_jaw_xy,
                 j1_heading_deg=float(np.rad2deg(q_heading[0])),
                 approach_changed_joints=list(JOINTS),
                 reach_joint_deltas_deg=np.rad2deg(q[1:3]-seed[1:3]).tolist(),
                 all_joint_deltas_deg=np.rad2deg(q-seed).tolist(),
                 j5_initial_alignment_deg=float(np.rad2deg(q_first[4])),
                 j5_solved_deg=float(np.rad2deg(q[4])),
                 j5_locked_at_deg=float(np.rad2deg(q[4])),
                 perpendicular_tolerance_deg=angle_limit,
                 open_clear=bool(1.<=fg<=2. and mg>=1.),
                 direction_clear=bool(angle<=angle_limit and proj>=.55),
                 moving_closes_toward_muhi=bool((mg-m80)>=2.),
                 valid_for_physics=bool(valid),
                 ik_angles_deg=np.rad2deg(q).tolist(),
                 optimizer_evaluations=int(sum(t['nfev'] for t in center_trials)))
            trials.append(row)
            stages.append(dict(seed=label,stage='3_five_dof_fixed_jaw_contact_IK',
                 j5_deg=row['j5_locked_at_deg'],
                 top_view_perpendicular_error_deg=angle,
                 fixed_gap_mm=fg,moving_gap_mm=mg,
                 center_target_error_mm=target_error_mm,
                 j4_deg=float(np.rad2deg(q[3])),
                 candidate_valid=bool(valid),
                 real_pose_safety_valid=bool(safety['valid']),
                 q_deg=row['ik_angles_deg']))
            if best is None or (valid,-score)>(best[0],-best[1]):
                best=(valid,score,q,row,q_first.copy(),q_heading.copy())
        except Exception as exc:
            trials.append(dict(seed=label,error=repr(exc),
                               valid_for_physics=False))
    dump(out/'horizontal_jaw_ik_candidates.json',trials)
    dump(out/'ordered_j5_then_approach_stages.json',stages)
    if best is None:
        raise RuntimeError('Ordered J5 alignment did not yield a candidate')
    report=dict(method='explicit_center_side_3D_fixed_jaw_IK_then_J6',
        chosen=best[3],n_candidates=len(seeds),
        stage1_j1_heading_q_deg=np.rad2deg(best[5]).tolist(),
        stage1_j5_aligned_q_deg=np.rad2deg(best[4]).tolist(),
        stage2_approach_q_deg=np.rad2deg(best[2]).tolist(),
        actual_finger_axis_local=finger_local.tolist(),
        bottle_axis_world=object_axis.tolist(),
        top_view_angle_limit_deg=angle_limit,
        seed_strategy='actual_recorded_follower_kinematically_nearest_diverse',
        j4_sign_override=False,
        note='J1/J5 initialize posture. Stage3 optimizes a body-fixed collision-surface point to an explicit fixed 3-D point 1.5mm outside Muhi center-side cylinder, never end cap; J6 physics only after center-target error <=2mm AND full geometry/path gates pass.')
    return best[2],report



def render_j5_stage_top_view(model,qadr,bq,q,xyz,quat,hi,out):
    """Overhead frame showing J5-aligned pose BEFORE any approach / J6."""
    d=mujoco.MjData(model)
    d.qpos[qadr[:5]]=q
    d.qpos[qadr[5]]=hi[5]
    d.qpos[bq:bq+3]=xyz
    d.qpos[bq+3:bq+7]=quat
    mujoco.mj_forward(model,d)
    model.vis.global_.offwidth=max(640,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(460,model.vis.global_.offheight)
    renderer=None
    try:
        renderer=mujoco.Renderer(model,width=640,height=460)
        camera=mujoco.MjvCamera()
        camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:]=xyz
        camera.distance=.48
        camera.azimuth=120.
        camera.elevation=-89.
        renderer.update_scene(d,camera=camera)
        panel=Image.new('RGB',(640,492),'#17212c')
        panel.paste(Image.fromarray(renderer.render().copy()),(0,30))
        ImageDraw.Draw(panel).text((10,9),
              'STAGE 1: J5 TOP VIEW ALIGN / J6 OPEN / NO PHYSICS',
              fill='white')
        panel.save(out/'stage1_j5_top_view.png')
        # Show the J4 bend from the side BEFORE approach, too.
        camera.azimuth=90.
        camera.elevation=-15.
        renderer.update_scene(d,camera=camera)
        side=Image.new('RGB',(640,492),'#17212c')
        side.paste(Image.fromarray(renderer.render().copy()),(0,30))
        ImageDraw.Draw(side).text((10,9),
              f'STAGE 1: J4={np.rad2deg(q[3]):+.1f} deg / J5={np.rad2deg(q[4]):+.1f} deg',
              fill='white')
        side.save(out/'stage1_j4_side_view.png')
    except Exception as exc:
        (out/'stage1_view_error.txt').write_text(repr(exc))
    finally:
        if renderer is not None:renderer.close()


def horizontal_pose_preview(model,qadr,bq,q,xyz,quat,lo,hi,out,fps=8):
    """Save three-view still + visually labeled kinematic J6 diagnostic video.

    This manually changes qpos and calls mj_forward: NEVER physical grasp.
    Actual MuJoCo mj_step contact is solely middle/grasp.mp4.
    """
    d=mujoco.MjData(model)
    d.qpos[qadr[:5]]=q
    d.qpos[qadr[5]]=hi[5]
    d.qpos[bq:bq+3]=xyz
    d.qpos[bq+3:bq+7]=quat
    mujoco.mj_forward(model,d)
    try:
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,640)
        model.vis.global_.offheight=max(model.vis.global_.offheight,460)
        renderer=mujoco.Renderer(model,width=640,height=460)
        from PIL import ImageDraw
        camera=mujoco.MjvCamera()
        camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:]=xyz
        camera.distance=.5
        def triptych(fraction):
            # Diagnostic ONLY: writes the J6 angle directly.
            d.qpos[qadr[5]]=hi[5]+fraction*(lo[5]-hi[5])
            mujoco.mj_forward(model,d)
            imgs=[]
            for az,el,view in ((90.,-15.,'SIDE'),(270.,-15.,'OPPOSITE'),
                               (120.,-80.,'OVERHEAD')):
                camera.azimuth=az
                camera.elevation=el
                renderer.update_scene(d,camera=camera)
                pane=Image.new('RGB',(640,490),'#192532')
                pane.paste(Image.fromarray(renderer.render().copy()),(0,30))
                ImageDraw.Draw(pane).text((8,8),
                    f'KINEMATIC ONLY / NO CONTACT SOLVER | {view} | J6 {fraction*100:.0f}% | CLOSE THEN HOLD',
                    fill='white')
                imgs.append(pane)
            canvas=Image.new('RGB',(1920,490))
            for i,pane in enumerate(imgs):canvas.paste(pane,(640*i,0))
            return canvas
        try:
            triptych(0.).save(out/'open_jaw_pose_3views.png')
            if shutil.which('ffmpeg'):
                frame_dir=out/'kinematic_frames'
                frame_dir.mkdir(exist_ok=True)
                frames=max(20,2*fps)
                for i in range(frames):
                    # Close and HOLD. The earlier reopen was misleading.
                    frac=min(1.,(i/max(1,frames-1))/.7)
                    triptych(frac).save(frame_dir/f'{i:05d}.png')
                proc=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error',
                      '-y','-framerate',str(fps),'-i',
                      str(frame_dir/'%05d.png'),'-c:v','libx264',
                      '-pix_fmt','yuv420p',str(out/'kinematic_j6_NOT_PHYSICS.mp4')],
                      capture_output=True,text=True)
                if proc.returncode:
                    (out/'kinematic_video_error.txt').write_text(proc.stderr[-2000:])
                shutil.rmtree(frame_dir)
        finally:
            renderer.close()
    except Exception as exc:
        (out/'pose_preview_error.txt').write_text(repr(exc))


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
    d.qpos[bq:bq+3]=xyz;d.qpos[bq+3:bq+7]=args.object_quat
    d.qvel[:]=0
    d.ctrl[aids]=np.r_[q,hi[5]]
    mujoco.mj_forward(model,d)
    support_start=float(args.table_z)
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
                panels=[]
                for az,el,view in ((90.,-15.,'SIDE'),(270.,-15.,'OPPOSITE'),(120.,-80.,'OVERHEAD')):
                    camera.azimuth=az;camera.elevation=el
                    renderer.update_scene(d,camera=camera,scene_option=scene_options)
                    im=Image.fromarray(renderer.render().copy())
                    pane=Image.new('RGB',(640,496),'#192532')
                    pane.paste(im,(0,30))
                    ImageDraw.Draw(pane).text((10,9),
                        f'PHYSICS mj_step {view} {phase} t={t:.2f} J6={np.rad2deg(d.qpos[qadr[5]]):.1f} F{fc}/M{mc}',fill='white')
                    panels.append(pane)
                canvas=Image.new('RGB',(1920,496),'#192532')
                for ix,pane in enumerate(panels):canvas.paste(pane,(640*ix,0))
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
    # Rotate the SAME free-body bottle, preserving Phase6G geom/mass/friction.
    from scipy.spatial.transform import Rotation as _R
    if args.orientation=='horizontal':
        rot=_R.from_euler('z',args.yaw_deg,degrees=True)*_R.from_euler('y',90.,degrees=True)
        xyzw=rot.as_quat()
        args.object_quat=[float(xyzw[3]),float(xyzw[0]),float(xyzw[1]),float(xyzw[2])]
        height=physical.MUHI_RADIUS+args.object_height_clearance_mm/1000.
    else:
        args.object_quat=[1.,0.,0.,0.]
        height=physical.MUHI_LENGTH/2
    xyz=np.array([args.target_x,args.target_y,args.table_z+height],dtype=float)
    if not np.isfinite(xyz).all():raise ValueError('Non-finite target')
    args.support_plane_z_m=float(args.table_z)
    # Fail closed: without actual recorded follower joint states the IK is
    # not allowed to proceed to a physical MuJoCo grasp simulation.
    follower_cal=real_safety.load_follower_calibration(args.follower_calibration)
    real_samples,real_meta=real_safety.recorded_joint_samples(args.dataset_root)
    dump(args.out/'follower_calibration_audit.json',dict(
         follower_calibration_source=str(args.follower_calibration),
         calibration=real_safety.calibration_report(follower_cal),
         actual_recorded_pose_dataset=real_meta,
         caveat='Encoder ranges do not prove mechanical safety or servo zero/sign alignment.'))
    model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor=phase6c.model_build(args)
    collision_build=args.out/'collision_urdf'/'jaw_collision_build.json'
    try: mesh_report=audit_jaw_meshes(model,fixed,moving,args.out)
    except Exception as exc:
        mesh_report=[]
        (args.out/'jaw_mesh_audit_error.txt').write_text(repr(exc))
    bj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,grasp.FINGER_BODY)
    fj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    bq=int(model.jnt_qposadr[fj]);bd=int(model.jnt_dofadr[fj])
    # Check ONLY moving wrist/forearm/tool shapes; not the base mounting
    # hardware which may legitimately intersect the support plane.
    tool_geoms=[]
    for gid in range(model.ngeom):
        if gid==floor or gid in bottle:
            continue
        body=int(model.geom_bodyid[gid])
        if body==0:
            continue
        body_name=(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body) or '').lower()
        if any(key in body_name for key in ('elbow','wrist','gripper','jaw')):
            tool_geoms.append(gid)
    tool_geoms=sorted(set(tool_geoms)|set(fixed)|set(moving))
    if not tool_geoms:
        raise RuntimeError('No tool collision geometry for floor clearance')
    bench=np.zeros(5);bench[4]=np.clip(-math.pi/2,lo[4],hi[4])
    # Legacy J2 raised probe below is DIAGNOSTIC ONLY for horizontal mode.
    # Recorded follower seeds replace it for the actual horizontal search.
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
            elif args.orientation=='horizontal':
                # Do not call the legacy upright-oriented free IK. It can
                # produce a folded and unsafe posture before any safeguards.
                q=np.deg2rad(np.asarray(real_samples[0,:5],dtype=float))
                ik={'method':'recorded_follower_seeds_only_no_legacy_upright_IK',
                    'anchor_error_mm':None}
            else:
                q,ik=solve_ik(model,data,qadr,bj,ik_lo,hi[:5],xyz,anchor,bench)
            if args.orientation=='horizontal':
                q,clearance_report=horizontal_jaw_clearance_search(
                    model,data,qadr,bq,lo,hi,xyz,args.object_quat,
                    fixed,moving,bottle,floor,tool_geoms,q,anchor,bj,args,sub,
                    real_samples)
                # Capture the J1-only posture separately (before J5).
                heading_q=np.deg2rad(np.asarray(
                    clearance_report['stage1_j1_heading_q_deg'],float))
                render_j5_stage_top_view(model,qadr,bq,heading_q,xyz,
                                          args.object_quat,hi,sub)
                for source_name,target_name in (
                    ('stage1_j5_top_view.png','stage1_j1_heading_top_view.png'),
                    ('stage1_j4_side_view.png','stage1_j1_heading_side_view.png')):
                    source_path=sub/source_name
                    if source_path.is_file():
                        source_path.rename(sub/target_name)
                aligned_deg=clearance_report['stage1_j5_aligned_q_deg']
                aligned_q=np.deg2rad(np.asarray(aligned_deg,float))
                render_j5_stage_top_view(model,qadr,bq,aligned_q,xyz,
                                          args.object_quat,hi,sub)
                ik['horizontal_clearance_search']=clearance_report
                ik['ik_angles_deg']=dict(zip(JOINTS,np.rad2deg(q).tolist()))
                ik['anchor_error_mm']=clearance_report['chosen']['anchor_error_mm']
                ip,ir=fk(model,data,qadr,bj,q)
                ik['grasp_reference_world_xyz_m']=(ip+ir@anchor).tolist()
            horizontal_pose_preview(model,qadr,bq,q,xyz,args.object_quat,
                                    lo,hi,sub,args.fps)
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
            geo=jaw_geometry(model,data,qadr,q,xyz,bq,fixed,moving,bottle,lo,hi,args.object_quat)
            # The synthetic IK reference point is not a meaningful gate after
            # directly optimizing physical jaw-to-bottle clearance. Preserve its
            # measured error, but require an open-jaw non-overlap before physics.
            use_refined_geometry=(name=='middle_j4_nonnegative' and
                                  args.j4_contact_search and args.j4_bilateral_refine)
            # Run real contact physics for a non-overlapping horizontal
            # initial posture, even if the soft IK anchor-error is high.
            # A bad candidate is NOT a proven grasp; see contact metrics.
            direction_clear=(args.orientation!='horizontal' or
                 ik.get('horizontal_clearance_search',{}).get('chosen',{})
                   .get('valid_for_physics',False))
            can_run=(not geo['open_jaws_overlap'] and direction_clear and
                (args.orientation=='horizontal' or use_refined_geometry or
                 ik['anchor_error_mm']<=args.max_ik_error_mm))
            phys=physics(model,qadr,vadr,aids,lo,hi,xyz,bq,bd,fixed,moving,bottle,floor,q,args,sub) if can_run else {
                'status':'GEOMETRY_DIAGNOSTIC_ONLY',
                'reason':'Explicit center-side Cartesian target <=2mm, inner fixed-jaw gap 1-2mm, end-cap rejection, floor, orientation, opposition or path not satisfied; inspect horizontal_jaw_ik_candidates.json',
                'bilateral_hold_steps':None}
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
             'predicted_bilateral_contact':ik.get('contact_aware_search_selected',{}).get('predicted_bilateral_contact'),
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
    summary={'schema':'phase7c_horizontal_upright_contact_parity_v1',
       'simulation_only':True,
       'orientation':args.orientation,
       'central_grasp_half_mm':args.central_grasp_half_mm,
       'central_grasp_required':True,
       'follower_calibration_json':str(args.follower_calibration),
       'real_joint_samples':real_meta,
       'real_pose_and_self_collision_preflight_required':True,
       'horizontal_j4_min_deg':args.horizontal_j4_min_deg,
       'reversed_j4_bend_required':bool(args.orientation=='horizontal'),
       'min_tool_floor_gap_mm':args.min_tool_floor_gap_mm,
       'tool_floor_clearance_required':True,
       'kinematic_video_phase':'close_and_hold_NOT_PHYSICS',
       'yaw_deg':args.yaw_deg,
       'object_quat_wxyz':args.object_quat,
       'source_baseline_sha':'f304135959adc5f9e97d693de37474f766c63cc1',
       'j6_pd':[args.j6_kp,args.j6_kd,args.j6_torque],
       'physical_close_is_mj_step':True,
       'lift_executed':False,
       'target_world_xyz_m':xyz.tolist(),'support_z_m':args.table_z,
       'target_reused_across_cases':True,
       'depth_subset':args.depth_only,
       'j4_sign_comparison_enabled':args.j4_compare,
       'j4_contact_search_enabled':args.j4_contact_search,
       'j4_bilateral_refinement_enabled':args.j4_bilateral_refine,
       'j4_bilateral_refinement_report':'j4_bilateral_refinement.json' if (args.out/'j4_bilateral_refinement.json').exists() else None,
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
       'method':'horizontal: explicit Muhi center-side 3D Cartesian surface point IK; stationary-arm J6 physical close and hold only if all gates pass; no approach/lift',
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
    p.add_argument('--orientation',choices=('horizontal','upright'),default='horizontal')
    p.add_argument('--yaw-deg',type=float,default=0.)
    p.add_argument('--object-height-clearance-mm',type=float,default=.5)
    p.add_argument('--horizontal-ik-max-nfev',type=int,default=160)
    p.add_argument('--max-finger-perpendicular-error-deg',type=float,default=8.)
    p.add_argument('--central-grasp-half-mm',type=float,default=10.)
    p.add_argument('--min-tool-floor-gap-mm',type=float,default=2.)
    p.add_argument('--follower-calibration',type=Path,default=Path('data/calibration/so101_follower/my_follower_arm.json'))
    p.add_argument('--real-pose-max-axis-deg',type=float,default=30.)
    p.add_argument('--real-pose-max-l2-deg',type=float,default=45.)
    p.add_argument('--self-collision-tolerance-mm',type=float,default=1.)
    p.add_argument('--real-seed-count',type=int,default=8)
    p.add_argument('--seed-diversity-deg',type=float,default=12.)
    p.add_argument('--max-arm-from-recorded-deg',type=float,default=18.,
                   help='Legacy option, not used for horizontal J1..J3 approach')
    p.add_argument('--max-j1-heading-from-recorded-deg',type=float,default=65.)
    p.add_argument('--max-j1-heading-error-deg',type=float,default=10.)
    p.add_argument('--max-approach-j123-deg',type=float,default=45.,
                   help='Max individual J1/J2/J3 deviation from real recorded pose; J4,J5 remain locked')
    p.add_argument('--max-j5-from-recorded-deg',type=float,default=35.)
    p.add_argument('--preflight-path-steps',type=int,default=5)
    p.add_argument('--horizontal-j4-min-deg',type=float,default=5.,
                   help='For horizontal grasp, force J4 to the opposite bend versus prior -95 deg candidate.')
    p.add_argument('--target-x',type=float,default=.220)
    p.add_argument('--target-y',type=float,default=0.)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--max-ik-error-mm',type=float,default=20.)
    p.add_argument('--depth-only',choices=('all','root','middle','tip'),default='middle',
                   help='Restrict physics and geometry to one depth; middle recommended for root-gap regression')
    p.add_argument('--no-j4-bilateral-refine',dest='j4_bilateral_refine',action='store_false',
                   help='Disable bounded paired-contact refinement; use the prior 18-anchor selector only')
    p.set_defaults(j4_bilateral_refine=True)
    p.add_argument('--no-j4-contact-search',dest='j4_contact_search',action='store_false',
                   help='Restore old single-anchor IK for J4 nonnegative, for A/B comparison')
    p.set_defaults(j4_contact_search=True)
    p.add_argument('--no-j4-compare',dest='j4_compare',action='store_false',
                   help='Disable automatic extra middle candidate with J4 constrained nonnegative')
    p.set_defaults(j4_compare=False)
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
    # Same J6 controller as validated upright Phase6G.
    p.add_argument('--j6-kp',type=float,default=2.)
    p.add_argument('--j6-kd',type=float,default=.08)
    p.add_argument('--j6-torque',type=float,default=.4)
    args=p.parse_args()
    if args.object_height_clearance_mm<0: p.error('Object clearance cannot be negative')
    if args.real_pose_max_axis_deg<=0 or args.real_pose_max_l2_deg<=0 or args.self_collision_tolerance_mm<0:
        p.error('Invalid real pose / self collision thresholds')
    if args.real_seed_count<1 or args.seed_diversity_deg<=0 or args.max_arm_from_recorded_deg<=0 or args.max_approach_j123_deg<=0 or args.max_j5_from_recorded_deg<=0 or args.max_j1_heading_from_recorded_deg<=0 or not 0<args.max_j1_heading_error_deg<90 or args.preflight_path_steps<2:
        p.error('Invalid recorded-seed / path preflight settings')
    if not 0<=args.horizontal_j4_min_deg<=70:
        p.error('horizontal-j4-min-deg must be in [0,70]')
    if args.min_tool_floor_gap_mm<0:
        p.error('min-tool-floor-gap-mm must be nonnegative')
    if not 0<args.central_grasp_half_mm<=15:
        p.error('central-grasp-half-mm must be (0,15]')
    if not 0<args.max_finger_perpendicular_error_deg<=45:
        p.error('Finger perpendicular error limit must be (0,45] degrees')
    if args.dt<=0 or args.fps<=0 or args.max_ik_error_mm<=0 or args.j2_raise_deg<0 or not 2<=args.jaw_collision_slabs<=32 or not 1<=args.jaw_collision_cross_bands<=8 or not 0<=args.j4_nonnegative_min_deg<=95:p.error('Invalid args')
    run(args)

if __name__=='__main__':main()
