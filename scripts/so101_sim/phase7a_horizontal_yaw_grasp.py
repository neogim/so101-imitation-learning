#!/usr/bin/env python3
"""SO-101 Phase 7A-1: horizontal Muhi yaw 0/45/90 grasp proof of concept.

Simulation only. The physically independent Muhi stays in its fixed X/Y
location on the same floor plane; only its world-Z yaw changes. Starting from
the same Home posture, compare a kinematic Home->Pregrasp->Grasp preview,
then stationary-arm J6 close/hold dynamics. NOT an ACT training dataset or
a demonstrated dynamic approach/lift.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

import physical_workcell_sim as physical
import phase6c_upright_reach_grasp as phase6c
import phase6g_fixed_workspace_grasp_depth as phase6g
import physical_grasp_j6_bench as grasp

DEGS = (0, 45, 90)
JOINT_NAMES = phase6g.JOINTS


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))+'\n')


def bottle_pose(x, y, table_z, yaw_deg):
    # Local +Z is Muhi's long axis. Ry(+90 deg) rotates it to world +X;
    # Rz(yaw) then rotates it in the horizontal plane.
    # The segmented cap remains at the local -Z end, so end asymmetry is kept.
    quat_xyzw = (Rotation.from_euler('z', yaw_deg, degrees=True) *
                 Rotation.from_euler('y', 90, degrees=True)).as_quat()
    center = np.array([x, y, table_z + physical.MUHI_RADIUS + .0005], float)
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    axis = Rotation.from_quat(quat_xyzw).apply([0., 0., 1.])
    return center, quat_wxyz, axis


def finger_axis_local(args):
    """Principal longitudinal axis from actual finger STL, transformed to link frame."""
    p=args.out/'collision_urdf'/'jaw_collision_build.json'
    report=json.loads(p.read_text())['jaw_parts'][grasp.FINGER_BODY]
    direction=np.asarray(report['principal_axis_mesh_local'],float)
    link=next(x for x in ET.parse(args.urdf).getroot().findall('link')
              if x.get('name')==grasp.FINGER_BODY)
    original_mesh=report['visual_stl']
    collision=next(x for x in link.findall('collision')
                   if x.find('./geometry/mesh') is not None and
                   Path(x.find('./geometry/mesh').get('filename','')).name==original_mesh)
    origin=collision.find('origin')
    rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
    direction=Rotation.from_euler('xyz',rpy).apply(direction)
    return direction/np.linalg.norm(direction)


def finger_axis_error(model,data,qadr,bid,q,finger_axis,object_axis):
    _,r=phase6g.fk(model,data,qadr,bid,q)
    # 0deg means perpendicular; a 35deg deviation is allowed as a
    # soft starting preference. J6 physics determines actual contact.
    alignment = float(np.dot(r @ finger_axis, object_axis))
    return float(np.degrees(np.arcsin(np.clip(abs(alignment), 0.0, 1.0))))


def tool_table_gap(model,data,qadr,bq,q,xyz,quat,hi,floor,tool_geoms,jaw_angle=None):
    data.qpos[qadr[:5]]=q
    data.qpos[qadr[5]]=hi[5] if jaw_angle is None else jaw_angle
    data.qpos[bq:bq+3]=xyz
    data.qpos[bq+3:bq+7]=quat
    mujoco.mj_forward(model,data)
    distances=[(float(mujoco.mj_geomDistance(model,data,int(floor),int(g),.25,np.zeros(6))),g)
               for g in tool_geoms]
    distance,gid=min(distances)
    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(gid))
    return float(distance*1000),name


def geom_witness(model, data, jaw_ids, bottle_ids):
    best = None
    for j in jaw_ids:
        for b in bottle_ids:
            pts = np.zeros(6, float)
            # In this call, points[0:3] belongs to the bottle.
            d = float(mujoco.mj_geomDistance(model, data, int(b), int(j), .15, pts))
            if best is None or d < best['gap_m']:
                best = dict(gap_m=d, bottle_point=pts[:3].copy(),
                            jaw_point=pts[3:].copy(), jaw_geom=int(j),
                            bottle_geom=int(b))
    return best


def geometry(model, data, qadr, bq, q, xyz, quat, axis,
             fixed, moving, bottle, lo, hi):
    data.qpos[qadr[:5]] = q
    data.qpos[bq:bq+3] = xyz
    data.qpos[bq+3:bq+7] = quat
    out = {}
    for label, u in [('open', 0.), ('near_closed', .90), ('closed', 1.)]:
        data.qpos[qadr[5]] = hi[5]+u*(lo[5]-hi[5])
        mujoco.mj_forward(model, data)
        f = geom_witness(model, data, fixed, bottle)
        m = geom_witness(model, data, moving, bottle)
        # Strip the axial component: opposite sides means cos(theta)=-1
        # in the cross-section perpendicular to the cylinder long axis.
        a = f['bottle_point']-xyz
        b = m['bottle_point']-xyz
        a = a-np.dot(a, axis)*axis
        b = b-np.dot(b, axis)*axis
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(np.dot(a, b)/(na*nb)) if min(na, nb)>1e-6 else 1.
        out[label] = dict(fixed_gap_mm=1000*f['gap_m'],
                          moving_gap_mm=1000*m['gap_m'],
                          radial_opposition_cos=cos,
                          fixed_contact_candidate_world_m=f['bottle_point'].tolist(),
                          moving_contact_candidate_world_m=m['bottle_point'].tolist())
    return out


def score_geo(g):
    # Heuristic only; NOT physical grasp success probability.
    def score_stage(s):
        def gp(x):
            return max(0., x-1.0) + 3.0*max(0., -x-1.0) + .15*abs(x)
        return (gp(s['fixed_gap_mm'])+gp(s['moving_gap_mm'])+
                18.*max(0., s['radial_opposition_cos']+.4))
    open_g = g['open']
    open_overlap = sum(max(0., -open_g[k]-.3) for k in
                       ('fixed_gap_mm', 'moving_gap_mm'))
    return float(min(score_stage(g['near_closed']),
                     score_stage(g['closed'])) + 30*open_overlap)


def ik_target(model, data, qadr, bid, lo, hi, xyz, anchor, ref_q, target_r,
              max_nfev=75, finger_axis=None, object_axis=None):
    def residual(q):
        p, r = phase6g.fk(model, data, qadr, bid, q)
        rot_err = Rotation.from_matrix(target_r.T@r).as_rotvec()
        orth=[] if finger_axis is None else [2.*float(np.dot(r@finger_axis,object_axis))]
        return np.r_[(p+r@anchor-xyz)/.012, rot_err/.55, orth, (q-ref_q)*.18]
    out = least_squares(residual, np.clip(ref_q, lo+1e-5, hi-1e-5),
                        bounds=(lo, hi), max_nfev=max_nfev)
    p, r = phase6g.fk(model, data, qadr, bid, out.x)
    return out.x, float(np.linalg.norm(p+r@anchor-xyz)*1000)


def search_pose(model, data, qadr, bq, bid, lo, hi, xyz, quat, axis,
                fixed, moving, bottle, max_seeds, finger_axis):
    # A few deliberate grasp orientations, not 18*4 local parameter searches
    # carried over from a vertical bottle. Retain the known middle as a
    # pose warm start, and rotate the desired gripper orientation with yaw.
    bench = np.zeros(5)
    bench[4] = np.clip(-math.pi/2, lo[4], hi[4])
    candidates = []
    yaw = math.degrees(math.atan2(axis[1],axis[0]))
    base_p, base_r = phase6g.fk(model,data,qadr,bid,bench)
    # Include two roll branches; J4 is NOT constrained to a sign.
    for orientation_deg in (yaw, yaw+90, yaw-90):
        target_r=Rotation.from_euler('z',orientation_deg,degrees=True).as_matrix()@base_r
        for anchor_z in (-.055,-.070):
            anchor=np.array([.030,.036,anchor_z])
            for offset_z in (0.,.018):
                goal=xyz+np.array([0.,0.,offset_z])
                # Include J5 wrist-roll branches instead of only one
                # posture from the upright-bottle benchmark.
                for j5_seed_deg in (-90.,0.,90.):
                    qseed=bench.copy()
                    qseed[4]=np.clip(math.radians(j5_seed_deg),lo[4]+1e-5,hi[4]-1e-5)
                    try:
                        q,err=ik_target(model,data,qadr,bid,lo[:5],hi[:5],goal,anchor,
                                        qseed,target_r,finger_axis=finger_axis,
                                        object_axis=axis)
                        g=geometry(model,data,qadr,bq,q,xyz,quat,axis,
                                   fixed,moving,bottle,lo,hi)
                        angle_error=finger_axis_error(model,data,qadr,bid,q,finger_axis,axis)
                        score=score_geo(g)+.12*err+.6*max(0.,angle_error-35.)
                        candidates.append(dict(q=q,anchor=anchor,ik_error_mm=err,
                            score=score,geo=g,branch_deg=orientation_deg,
                            j5_seed_deg=j5_seed_deg,
                            finger_perpendicular_error_deg=angle_error,
                            goal_height_offset_mm=1000*offset_z))
                    except (ValueError,RuntimeError):
                        continue
    candidates.sort(key=lambda c:c['score'])
    if not candidates:
        raise RuntimeError('No IK seeds for the horizontal grasp')
    optimized = []
    # Local adjustment against the *true* jaw-to-cylinder distances.
    # Optimization may fail to converge on a nonsmooth collision-distance
    # objective. Retain its best evaluated result and log it transparently.
    for seed_rank, c in enumerate(candidates[:max_seeds],1):
        q0 = c['q'].copy()
        def objective(q):
            g = geometry(model,data,qadr,bq,q,xyz,quat,axis,
                         fixed,moving,bottle,lo,hi)
            angle_error=finger_axis_error(model,data,qadr,bid,q,finger_axis,axis)
            return (score_geo(g)+.6*np.linalg.norm((q-q0)/.5)
                    +.6*max(0.,angle_error-35.))
        opt = minimize(objective,q0,method='Powell',
                       bounds=list(zip(lo[:5]+1e-5,hi[:5]-1e-5)),
                       options=dict(maxiter=10,maxfev=180,xtol=.012,ftol=.03))
        trial = np.clip(opt.x,lo[:5]+1e-5,hi[:5]-1e-5)
        if objective(trial) >= objective(q0):
            trial = q0.copy()
        g=geometry(model,data,qadr,bq,trial,xyz,quat,axis,
                   fixed,moving,bottle,lo,hi)
        valid_open=min(g['open']['fixed_gap_mm'],
                       g['open']['moving_gap_mm'])>=-.5
        close = min([g['near_closed'],g['closed']],
                    key=lambda s:abs(s['fixed_gap_mm'])+
                                 abs(s['moving_gap_mm'])+
                                 10*max(0,s['radial_opposition_cos']+.4))
        opposite = close['radial_opposition_cos']<=-.25
        angle_error=finger_axis_error(model,data,qadr,bid,trial,finger_axis,axis)
        paired = (valid_open and opposite and angle_error<=35. and
                  -1.5<=close['fixed_gap_mm']<=2. and
                  -1.5<=close['moving_gap_mm']<=2.)
        result = dict(q=trial,geometry=g,initial_score=c['score'],
                      final_score=objective(trial),seed_rank=seed_rank,
                      original_anchor=c['anchor'].tolist(),
                      ik_initial_error_mm=c['ik_error_mm'],
                      orientation_branch_deg=c['branch_deg'],
                      optimizer_success=bool(opt.success),
                      optimizer_nfev=int(opt.nfev),
                      j5_solved_deg=float(np.rad2deg(trial[4])),
                      finger_perpendicular_error_deg=angle_error,
                      valid_open=bool(valid_open),
                      predicts_paired=bool(paired))
        optimized.append(result)
    # Prioritize expected two-sided contacts; never claim success before
    # the J6 physical close/hold experiment.
    optimized.sort(key=lambda c:(not c['predicts_paired'],
                                 not c['valid_open'],c['final_score']))
    return optimized[0],optimized


def contact_counts(model, data, fixed, moving, bottle):
    return grasp.jaw_contact_counts(model,data,bottle,fixed,moving)


def save_video(frames, destination, fps):
    if not frames or shutil.which('ffmpeg') is None:
        return None
    p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
                        '-f','rawvideo','-pix_fmt','rgb24','-s','640x460',
                        '-r',str(fps),'-i','-','-c:v','libx264','-pix_fmt','yuv420p',
                        str(destination)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.asarray(fr,dtype=np.uint8).tobytes())
    p.stdin.close()
    stderr=p.stderr.read()
    rc=p.wait()
    if rc:
        (destination.parent/'ffmpeg_error.txt').write_text(
            stderr.decode(errors='replace')[-3000:])
        return None
    return destination.name


def render_scene(model,data,camera,renderer,scene_options,label):
    renderer.update_scene(data,camera=camera,scene_option=scene_options)
    img=Image.fromarray(renderer.render().copy())
    ImageDraw.Draw(img).text((10,10),label,fill=(255,255,255),
                             stroke_width=1,stroke_fill=(0,0,0))
    return img


def physical_close(model,qadr,aids,bq,bd,q,xyz,quat,lo,hi,fixed,moving,
                   bottle,args,out):
    d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
    d.qpos[qadr[:5]]=q
    d.qpos[qadr[5]]=hi[5]
    d.qpos[bq:bq+3]=xyz;d.qpos[bq+3:bq+7]=quat
    d.qvel[:]=0
    d.ctrl[aids]=np.r_[q,hi[5]]
    mujoco.mj_forward(model,d)
    phases=[('settle',.4),('close',1.3),('hold',1.)]
    total=sum(t for _,t in phases)
    steps=int(round(total/args.dt))
    frame_stride=max(1,int(round(1/(args.fps*args.dt))))
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=xyz;camera.distance=.52
    camera.azimuth=115;camera.elevation=-31
    vis=mujoco.MjvOption()
    vis.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True
    model.vis.global_.offwidth=max(640,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(460,model.vis.global_.offheight)
    renderer=None
    if shutil.which('ffmpeg'):
        try:renderer=mujoco.Renderer(model,width=640,height=460)
        except Exception as exc:
            (out/'render_error.txt').write_text(repr(exc))
    frames=[];rows=[];deep=False;bad=False
    try:
        for k in range(steps+1):
            t=min(k*args.dt,total)
            if t<.4:phase='settle';j6=hi[5]
            elif t<1.7:
                phase='close';j6=hi[5]+(lo[5]-hi[5])*((t-.4)/1.3)
            else:phase='hold';j6=lo[5]
            d.ctrl[aids]=np.r_[q,j6]
            if k:mujoco.mj_step(model,d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                bad=True;break
            fc,mc,depth=contact_counts(model,d,fixed,moving,bottle)
            deep |= depth is not None and depth < -.003
            rows.append(dict(time_s=float(d.time),phase=phase,
                             fixed_contacts=int(fc),moving_contacts=int(mc),
                             bilateral=int(fc>0 and mc>0),
                             min_contact_dist_mm=depth*1000 if depth is not None else None,
                             bottle_x_m=float(d.qpos[bq]),
                             bottle_y_m=float(d.qpos[bq+1]),
                             bottle_z_m=float(d.qpos[bq+2]),
                             bottle_xy_displacement_mm=1000*float(
                                 np.linalg.norm(d.qpos[bq:bq+2]-xyz[:2])),
                             j4_target_deg=float(np.rad2deg(q[3])),
                             j6_target_deg=float(np.rad2deg(j6)),
                             j6_actual_deg=float(np.rad2deg(d.qpos[qadr[5]])),
                             max_arm_tracking_error_deg=float(np.max(np.abs(
                                 np.rad2deg(d.qpos[qadr[:5]]-q))))))
            if renderer is not None and (k%frame_stride==0 or k==steps):
                frames.append(render_scene(model,d,camera,renderer,vis,
                        f'Yaw {args.current_yaw_deg} | {phase} | F {fc} M {mc} | t {t:.2f}s'))
    finally:
        if renderer:renderer.close()
    if rows:
        with (out/'joint_contact_trace.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=rows[0].keys())
            wr.writeheader();wr.writerows(rows)
    vid=save_video(frames,out/'grasp.mp4',args.fps)
    hold=[r for r in rows if r['phase']=='hold']
    return dict(status='NUMERIC_FAILURE' if bad else 'DYNAMIC_TEST_COMPLETED',
                hold_steps=len(hold),
                bilateral_hold_steps=sum(r['bilateral'] for r in hold),
                max_xy_displacement_mm=max((r['bottle_xy_displacement_mm'] for r in rows),default=None),
                deep_penetration_over_3mm=bool(deep),
                final_bottle_z_m=rows[-1]['bottle_z_m'] if rows else None,
                video=vid,
                note='Stationary-arm J6 close/hold only. No validated dynamic Home-to-grasp and no lift.')


def approach_preview(model,data,qadr,bq,bid,lo,hi,q,xyz,quat,
                     args,out,bench,floor,tool_geoms):
    # Preview only: interpolation in configuration space, collision NOT checked
    # in the currently isolated (finger-bottle-only) model.
    probe=mujoco.MjData(model)
    p, r=phase6g.fk(model,probe,qadr,bid,q)
    # Move away from the bottle along world +Z while maintaining near-constant
    # gripper orientation. This is a prototype pregrasp, not a safe motion plan.
    home=np.clip(bench,lo[:5]+1e-5,hi[:5]-1e-5)
    anchor=np.array([.03,.036,-.07],dtype=float)
    point=p+r@anchor
    pre,_=ik_target(model,probe,qadr,bid,lo[:5],hi[:5],
                    point+[0,0,.040],anchor,q,r,max_nfev=65)
    path=[]
    for phase,first,last,n in [('home_to_pregrasp',home,pre,24),
                               ('pregrasp_to_grasp',pre,q,24)]:
        for k in range(n):
            u=(k+1)/n
            # smoothstep to avoid instantaneous joint-velocity jumps
            w=u*u*(3-2*u)
            path.append((phase, (1-w)*first+w*last))
    clearances=[]
    for k,(phase,qs) in enumerate(path):
        mm,name=tool_table_gap(model,probe,qadr,bq,qs,xyz,quat,hi,floor,tool_geoms)
        clearances.append(dict(frame=k,phase=phase,table_gap_mm=mm,nearest_geom=name))
    nearest=min(clearances,key=lambda r:r['table_gap_mm'])
    table_ok=nearest['table_gap_mm']>=args.min_table_clearance_mm
    with (out/'table_clearance_trace.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=clearances[0].keys());wr.writeheader();wr.writerows(clearances)
    with (out/'kinematic_preview_joint_targets.csv').open('w',newline='') as f:
        fields=['frame','phase','time_s']+[n+'_target_deg' for n in JOINT_NAMES]+['j6_target_deg']
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for k,(phase,qs) in enumerate(path):
            wr.writerow(dict(frame=k,phase=phase,time_s=k/args.fps,
                             **{n+'_target_deg':float(np.rad2deg(v)) for n,v in zip(JOINT_NAMES,qs)},
                             j6_target_deg=float(np.rad2deg(hi[5]))))
    if not table_ok:
        return dict(status='TABLE_CLEARANCE_FAILED',video=None,
                    min_table_gap_mm=nearest['table_gap_mm'],
                    nearest_geom=nearest['nearest_geom'],table_preflight_ok=False)
    if shutil.which('ffmpeg') is None:
        return dict(status='KINEMATIC_PREVIEW_ONLY',video=None,
                    min_table_gap_mm=nearest['table_gap_mm'],table_preflight_ok=True)
    model.vis.global_.offwidth=max(640,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(460,model.vis.global_.offheight)
    renderer=mujoco.Renderer(model,width=640,height=460)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=xyz;camera.distance=.62;camera.azimuth=110;camera.elevation=-25
    frames=[]
    try:
        for k,(phase,qs) in enumerate(path):
            data.qpos[qadr[:5]]=qs;data.qpos[qadr[5]]=hi[5]
            data.qpos[bq:bq+3]=xyz;data.qpos[bq+3:bq+7]=quat
            mujoco.mj_forward(model,data)
            frames.append(render_scene(model,data,camera,renderer,
                          mujoco.MjvOption(),
                          f'KINEMATIC PREVIEW ONLY | yaw {args.current_yaw_deg} | {phase}'))
    finally:renderer.close()
    vid=save_video(frames,out/'approach_preview.mp4',args.fps)
    return dict(status='KINEMATIC_PREVIEW_ONLY',video=vid,
                min_table_gap_mm=nearest['table_gap_mm'],table_preflight_ok=True,
                note='Tool/forearm plane clearance scanned. Not dynamic tracking, self-collision or complete world collision certification. No ACT dataset yet.')


def run(args):
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    args.support_plane_z_m=args.table_z
    model,qadr,vadr,aids,lo,hi,fixed,moving,bottle,floor=phase6c.model_build(args)
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,grasp.FINGER_BODY)
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    bq=int(model.jnt_qposadr[jid]);bd=int(model.jnt_dofadr[jid])
    bench=np.zeros(5);bench[4]=np.clip(-math.pi/2,lo[4],hi[4])
    faxis=finger_axis_local(args)
    # The older contact bench disabled jaw/table contact. Explicitly
    # query geometry distance to the plane without changing contact masks.
    tool_geoms=[]
    for gid in range(model.ngeom):
        body=int(model.geom_bodyid[gid])
        if body==0 or gid in bottle or gid==floor:continue
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body) or ''
        if any(part in name for part in ('elbow','wrist','gripper','jaw')):
            tool_geoms.append(gid)
    if not tool_geoms:raise ValueError('No forearm/wrist/tool collision geoms found')
    # The Phase6C baseline isolates jaw/bottle and bottle/table contacts.
    # Phase7A additionally enables actual arm/tool/table response.
    # Use a dedicated collision bit (8) for non-jaw tool links, and bit 4
    # for the jaw patches. Avoid accidentally creating tool/bottle contacts.
    jaw_geoms=set(fixed)|set(moving)
    model.geom_conaffinity[floor] |= 8
    for geom in tool_geoms:
        if geom in jaw_geoms:
            model.geom_conaffinity[geom] |= 4
        else:
            model.geom_contype[geom] |= 8
    outputs=[]
    for yaw in args.yaw_deg:
        args.current_yaw_deg=float(yaw)
        out=args.out/f'yaw_{yaw:03g}';out.mkdir()
        xyz,quat,axis=bottle_pose(args.target_x,args.target_y,args.table_z,yaw)
        data=mujoco.MjData(model);mujoco.mj_resetData(model,data)
        try:
            best,options=search_pose(model,data,qadr,bq,bid,lo,hi,xyz,quat,
                                      axis,fixed,moving,bottle,args.max_seeds,faxis)
            q=best['q']
            choices=[]
            for entry in options:
                choices.append({k:v for k,v in entry.items() if k!='q'})
            write_json(out/'grasp_candidates.json',
                       dict(yaw_deg=yaw,world_center_m=xyz.tolist(),
                            world_quat_wxyz=quat.tolist(),axis_world=axis.tolist(),
                            selected=choices[0],candidates=choices))
            preview=approach_preview(model,data,qadr,bq,bid,lo,hi,q,xyz,quat,
                                     args,out,bench,floor,tool_geoms)
            # Keep a visual record even for failed geometry candidates.
            # Reject deeply intersecting jaws before running dynamics.
            table_gaps=[]
            for frac in (0.,.5,.85,1.):
                angle=hi[5]+frac*(lo[5]-hi[5])
                gap,near=tool_table_gap(model,data,qadr,bq,q,xyz,quat,hi,floor,
                                        tool_geoms,jaw_angle=angle)
                table_gaps.append(dict(j6_closed_fraction=frac,
                                       gap_mm=gap,nearest_geom=near))
            shallowest=min(table_gaps,key=lambda x:x['gap_mm'])
            table_gap=shallowest['gap_mm']
            nearest_table_geom=shallowest['nearest_geom']
            preflight=(best['valid_open'] and
                       table_gap>=args.min_table_clearance_mm and
                       preview.get('table_preflight_ok',False))
            physics=physical_close(model,qadr,aids,bq,bd,q,xyz,quat,lo,hi,
                                   fixed,moving,bottle,args,out) if preflight else {
                                       'status':'SKIPPED_GEOMETRY_OR_TABLE_PREFLIGHT',
                                       'bilateral_hold_steps':None,'video':None}
            outcome=dict(yaw_deg=yaw,bottle_world_center_m=xyz.tolist(),
                         bottle_world_quat_wxyz=quat.tolist(),
                         bottle_axis_world=axis.tolist(),
                         j4_solved_deg=float(np.rad2deg(q[3])),
                         j5_solved_deg=float(np.rad2deg(q[4])),
                         finger_perpendicular_error_deg=best['finger_perpendicular_error_deg'],
                         table_grasp_gap_mm=table_gap,
                         table_grasp_sweep=table_gaps,
                         nearest_table_geom=nearest_table_geom,
                         joint_angles_deg=dict(zip(JOINT_NAMES,np.rad2deg(q).tolist())),
                         geometry=best['geometry'],
                         geometric_score=best['final_score'],
                         predicted_bilateral=best['predicts_paired'],
                         dynamic=physics,approach_preview=preview,
                         note='Predicted geometric grasp != dynamic physical grasp success')
        except Exception as exc:
            outcome=dict(yaw_deg=yaw,status='EXECUTION_ERROR',
                         error=repr(exc),dynamic={'status':'EXECUTION_ERROR'})
        write_json(out/'result.json',outcome)
        outputs.append(outcome)
        print(f'Yaw {yaw}: {outcome.get("dynamic",{}).get("status")}; bilateral='
              f'{outcome.get("dynamic",{}).get("bilateral_hold_steps")}',flush=True)
    rows=[dict(yaw_deg=r['yaw_deg'],status=r.get('dynamic',{}).get('status'),
               j4_deg=r.get('j4_solved_deg'),j5_deg=r.get('j5_solved_deg'),
               finger_perpendicular_error_deg=r.get('finger_perpendicular_error_deg'),
               table_grasp_gap_mm=r.get('table_grasp_gap_mm'),
               predicted_bilateral=r.get('predicted_bilateral'),
               bilateral_hold_steps=r.get('dynamic',{}).get('bilateral_hold_steps'),
               max_xy_displacement_mm=r.get('dynamic',{}).get('max_xy_displacement_mm'),
               deep_penetration=r.get('dynamic',{}).get('deep_penetration_over_3mm'),
               geometric_score=r.get('geometric_score'),
               video=r.get('dynamic',{}).get('video'),
               approach_preview=r.get('approach_preview',{}).get('video'),
               error=r.get('error')) for r in outputs]
    with (args.out/'comparison.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    write_json(args.out/'summary.json',dict(
        schema='phase7a1_horizontal_yaw_v1',simulation_only=True,
        target_world_xy_m=[args.target_x,args.target_y],
        horizontal_bottle_center_height_m=args.table_z+physical.MUHI_RADIUS+.0005,
        bottle_length_m=physical.MUHI_LENGTH,bottle_radius_m=physical.MUHI_RADIUS,
        yaw_degrees=list(args.yaw_deg),joint_sign_constraint='URDF limits only',
        finger_long_axis_local=faxis.tolist(),
        permitted_perpendicular_error_deg=35.0,
        table_preflight_min_clearance_mm=args.min_table_clearance_mm,
        dynamic_tool_table_contact_enabled=True,
        table_preflight_j6_samples=[0.0,0.5,0.85,1.0],
        act_dataset_ready=False,
        preview_is_kinematic_only=True,
        validation_note='Collision geometry optimization + stationary-arm J6 close/hold. Physical Home->Pregrasp->Grasp not yet validated. No lift.',
        collision_model=args.jaw_collision_mode,friction_modified=False,
        results=outputs))
    print('RESULT_DIR:',args.out,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--yaw-deg',type=int,nargs='+',default=list(DEGS))
    p.add_argument('--target-x',type=float,default=.220)
    p.add_argument('--target-y',type=float,default=0.)
    p.add_argument('--table-z',type=float,default=0.)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--fps',type=int,default=8)
    p.add_argument('--max-seeds',type=int,default=4)
    p.add_argument('--min-table-clearance-mm',type=float,default=1.0)
    p.add_argument('--jaw-collision-mode',default='axial_cross_convex_patches')
    p.add_argument('--jaw-collision-slabs',type=int,default=12)
    p.add_argument('--jaw-collision-cross-bands',type=int,default=3)
    p.add_argument('--arm-kp',type=float,default=18.)
    p.add_argument('--arm-kd',type=float,default=1.8)
    p.add_argument('--arm-torque',type=float,default=3.)
    p.add_argument('--j6-kp',type=float,default=18.)
    p.add_argument('--j6-kd',type=float,default=1.8)
    p.add_argument('--j6-torque',type=float,default=3.)
    a=p.parse_args()
    if a.max_seeds<1 or any(y<0 or y>=180 for y in a.yaw_deg):
        p.error('yaw must be in [0,180) and max-seeds>=1; 180-fold symmetry audit is separate')
    run(a)


if __name__=='__main__':main()
