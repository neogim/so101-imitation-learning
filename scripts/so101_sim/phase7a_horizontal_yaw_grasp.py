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
        jaw_vector=m['jaw_point']-f['jaw_point']
        jaw_span=float(np.linalg.norm(jaw_vector))
        jaw_dir=jaw_vector/max(jaw_span,1e-6)
        center_between=(float(np.dot(xyz-f['jaw_point'],jaw_dir))>0 and
                        float(np.dot(xyz-m['jaw_point'],jaw_dir))<0)
        fixed_facing=float(np.dot(f['bottle_point']-f['jaw_point'],jaw_dir))
        moving_facing=float(np.dot(m['bottle_point']-m['jaw_point'],-jaw_dir))
        contact_axial_mm=1000*abs(float(np.dot(f['bottle_point']-xyz,axis)))
        out[label] = dict(
                          cylinder_center_between_jaws=bool(center_between),
                          fixed_inner_facing_mm=1000*fixed_facing,
                          moving_inner_facing_mm=1000*moving_facing,
                          fixed_witness_axial_offset_mm=contact_axial_mm,
                          jaw_span_mm=1000*jaw_span,
                          fixed_jaw_nearest_point_world_m=f['jaw_point'].tolist(),
                          moving_jaw_nearest_point_world_m=m['jaw_point'].tolist(),
                          fixed_gap_mm=1000*f['gap_m'],
                          moving_gap_mm=1000*m['gap_m'],
                          radial_opposition_cos=cos,
                          fixed_contact_candidate_world_m=f['bottle_point'].tolist(),
                          moving_contact_candidate_world_m=m['bottle_point'].tolist())
    return out


def score_geo(g):
    """Prioritize the fundamental sequence: fixed inner face first (2-3mm),
    then moving jaw closing from the opposite side. No angle-only shortcut.
    """
    f=g['open']['fixed_gap_mm']
    m=g['open']['moving_gap_mm']
    near=g['near_closed']
    end=g['closed']
    # 2.5mm from the fixed jaw to the outside of the cylinder.
    score=4.*abs(f-2.5)
    if not g['open']['cylinder_center_between_jaws']:
        score+=45.
    if g['open']['fixed_inner_facing_mm']<=0:
        score+=35.
    score+=2.*max(0.,g['open']['fixed_witness_axial_offset_mm']-25.)
    score+=8.*max(0., -f)
    score+=4.*max(0., 4.-m) + .1*max(0.,m-35.)
    # The moving jaw must approach/cross the cylinder surface by J6.
    score+=2.*max(0.,min(near['moving_gap_mm'],end['moving_gap_mm'])-1.)
    score+=7.*max(0.,-min(near['moving_gap_mm'],end['moving_gap_mm'])-2.)
    # Opposite radial faces instead of the previous same-side contact.
    cos=min(near['radial_opposition_cos'],end['radial_opposition_cos'])
    score+=35.*max(0.,cos+.3)
    return float(score)

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


def search_pose(model,data,qadr,bq,bid,lo,hi,xyz,quat,axis,fixed,moving,
                bottle,max_seeds,finger_axis,floor,tool_geoms,args):
    bench=np.zeros(5)
    bench[4]=np.clip(-math.pi/2,lo[4],hi[4])
    _,base_r=phase6g.fk(model,data,qadr,bid,bench)
    yaw=math.degrees(math.atan2(axis[1],axis[0]))
    candidates=[]
    # Candidate generation uses multiple arm approach directions and J5
    # seeds, but every candidate is scored using physical collision meshes.
    for orientation_deg in (yaw,yaw+90,yaw-90):
        target_r=Rotation.from_euler('z',orientation_deg,degrees=True).as_matrix()@base_r
        for anchor_z in (-.045,-.070,-.095):
            anchor=np.array([.030,.036,anchor_z])
            for vertical_offset in (0.,.018,.035):
                for j5_seed in (-90.,0.,90.):
                    ref=bench.copy()
                    ref[4]=np.clip(math.radians(j5_seed),lo[4]+1e-5,hi[4]-1e-5)
                    try:
                        q,err=ik_target(model,data,qadr,bid,lo[:5],hi[:5],
                            xyz+[0,0,vertical_offset],anchor,ref,target_r,
                            finger_axis=finger_axis,object_axis=axis)
                        g=geometry(model,data,qadr,bq,q,xyz,quat,axis,
                                   fixed,moving,bottle,lo,hi)
                        gap,_=tool_table_gap(model,data,qadr,bq,q,xyz,quat,
                                              hi,floor,tool_geoms)
                        angle=finger_axis_error(model,data,qadr,bid,q,finger_axis,axis)
                        score=(score_geo(g)+1.5*max(0.,2.-gap)
                               +.15*err+.10*max(0.,angle-35.))
                        candidates.append((score,q.copy(),anchor,err,
                                           orientation_deg,j5_seed,angle))
                    except (ValueError,RuntimeError):
                        continue
    if not candidates:
        raise RuntimeError('No IK candidate for any horizontal grasp seed')
    candidates.sort(key=lambda item:item[0])
    refined=[]
    for seed_rank,entry in enumerate(candidates[:max_seeds],1):
        score0,q0,anchor,err0,branch,j5seed,angle0=entry
        def objective(q):
            g=geometry(model,data,qadr,bq,q,xyz,quat,axis,
                       fixed,moving,bottle,lo,hi)
            # Static table clearance with jaw open AND during J6 closure.
            gaps=[tool_table_gap(model,data,qadr,bq,q,xyz,quat,hi,
                                 floor,tool_geoms,jaw_angle=hi[5]+u*(lo[5]-hi[5]))[0]
                  for u in (0.,.5,1.)]
            angle=finger_axis_error(model,data,qadr,bid,q,finger_axis,axis)
            return (score_geo(g)+16.*max(0.,2.-min(gaps))
                    +.10*max(0.,angle-35.)
                    +.3*np.linalg.norm((q-q0)/.5))
        opt=minimize(objective,q0,method='Powell',
            bounds=list(zip(lo[:5]+1e-5,hi[:5]-1e-5)),
            options=dict(maxiter=12,maxfev=220,xtol=.008,ftol=.025))
        q1=np.clip(opt.x,lo[:5]+1e-5,hi[:5]-1e-5)
        if objective(q1)>objective(q0):q1=q0
        g=geometry(model,data,qadr,bq,q1,xyz,quat,axis,
                   fixed,moving,bottle,lo,hi)
        gaps=[tool_table_gap(model,data,qadr,bq,q1,xyz,quat,hi,
                             floor,tool_geoms,jaw_angle=hi[5]+u*(lo[5]-hi[5]))[0]
              for u in (0.,.5,.85,1.)]
        angle=finger_axis_error(model,data,qadr,bid,q1,finger_axis,axis)
        op=g['open']
        close=min((g['near_closed'],g['closed']),
                  key=lambda k:abs(k['moving_gap_mm'])
                                    +20.*max(0.,k['radial_opposition_cos']+.3))
        margin=float(op['fixed_gap_mm'])
        # A true geometric preflight, not a declaration of actual grasp.
        valid=(2.<=margin<=3. and op['moving_gap_mm']>=3.
               and op['cylinder_center_between_jaws']
               and op['fixed_inner_facing_mm']>0
               and op['fixed_witness_axial_offset_mm']<=25.
               and -2.<=close['moving_gap_mm']<=2.
               and close['radial_opposition_cos']<=-.3
               and min(gaps)>=args.min_table_clearance_mm
               and angle<=45.)
        refined.append(dict(q=q1,anchor=anchor,geometry=g,
            score=objective(q1),initial_score=score0,
            fixed_inner_surface_gap_mm=margin,
            table_min_gap_mm=min(gaps),table_j6_sweep_gap_mm=gaps,
            finger_perpendicular_error_deg=angle,
            j5_solved_deg=float(np.rad2deg(q1[4])),
            j6_torque_limit_nm=args.j6_torque,
            ik_seed_error_mm=err0,orientation_branch_deg=branch,
            seed_rank=seed_rank,j5_seed_deg=j5seed,
            optimizer_success=bool(opt.success),
            valid_open=bool(margin>=0 and op['moving_gap_mm']>=0),
            predicts_paired=bool(valid)))
    refined.sort(key=lambda x:(not x['predicts_paired'],
                                abs(x['fixed_inner_surface_gap_mm']-2.5),
                                x['score']))
    return refined[0],refined

def contact_counts(model, data, fixed, moving, bottle):
    return grasp.jaw_contact_counts(model,data,bottle,fixed,moving)


def save_video(frames, destination, fps):
    if not frames or shutil.which('ffmpeg') is None:
        return None
    width, height = frames[0].size
    if any(frame.size != (width, height) for frame in frames):
        raise ValueError('Video frames must have identical dimensions')
    p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
                        '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
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


def render_table_collision_diagnostic(model,qadr,bq,q,xyz,quat,lo,hi,
                                      floor,tool_geoms,args,out,bench,bid):
    """Synchronized 3-view MuJoCo geometry diagnostic, including invalid poses.

    Every movie frame shows the SAME robot/object state in 3 FIXED cameras:
    side | opposite side | overhead. No camera switching in the movie.
    This is mj_forward geometry, not a dynamically executed trajectory.
    """
    view_w, view_h = 640, 460
    model.vis.global_.offwidth=max(view_w,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(view_h,model.vis.global_.offheight)
    data=mujoco.MjData(model)
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=xyz
    camera.distance=.43
    scene_options=mujoco.MjvOption()
    original_colors=model.geom_rgba.copy()
    frames=[]
    rows=[]
    renderer=None
    try:
        renderer=mujoco.Renderer(model,width=view_w,height=view_h)
        stages=[('home_open',bench,0.),('grasp_open',q,0.),
                ('grasp_half',q,.5),('grasp_85pct',q,.85),
                ('grasp_closed',q,1.)]
        viewpoints=[('side',85.,-15.),
                    ('opposite_side',265.,-15.),
                    ('overhead',120.,-75.)]
        for stage,posture,closing_fraction in stages:
            jaw_angle=hi[5]+closing_fraction*(lo[5]-hi[5])
            data.qpos[qadr[:5]]=posture
            data.qpos[qadr[5]]=jaw_angle
            data.qpos[bq:bq+3]=xyz
            data.qpos[bq+3:bq+7]=quat
            data.qvel[:]=0
            mujoco.mj_forward(model,data)
            contact=[]
            for gid in tool_geoms:
                dist=float(mujoco.mj_geomDistance(
                    model,data,int(floor),int(gid),.25,np.zeros(6)))
                contact.append((dist,int(gid)))
            gap,gid=min(contact)
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid)
            rows.append(dict(stage=stage,j6_closed_fraction=closing_fraction,
                             min_tool_table_gap_mm=gap*1000,nearest_geom=name))
            model.geom_rgba[:]=original_colors
            if gap<args.min_table_clearance_mm/1000:
                model.geom_rgba[gid]=[.95,.12,.14,1.]
            panels=[]
            for view,azimuth,elevation in viewpoints:
                camera.azimuth=azimuth
                camera.elevation=elevation
                frame=render_scene(model,data,camera,renderer,scene_options,
                                   f'Yaw {args.current_yaw_deg} | {stage} | {view}')
                dr=ImageDraw.Draw(frame)
                dr.rectangle((0,430,640,460),fill=(25,37,50))
                dr.text((10,438),f'TABLE GAP {gap*1000:+.1f} mm | closest: {name}',
                        fill=(255,175,175) if gap<0 else (180,255,195))
                panels.append(frame)
                frame.save(out/f'diagnostic_{stage}_{view}.png')
            # All three viewpoints at precisely the same simulated pose.
            canvas=Image.new('RGB',(view_w*len(panels),view_h))
            for index,panel in enumerate(panels):
                canvas.paste(panel,(view_w*index,0))
            canvas.save(out/f'diagnostic_{stage}_3views.png')
            # Hold this *single synchronized triptych* for one second.
            frames.extend([canvas]*max(1,int(args.fps)))
        with (out/'table_collision_diagnostic.csv').open('w',newline='') as fp:
            wr=csv.DictWriter(fp,fieldnames=rows[0].keys())
            wr.writeheader();wr.writerows(rows)
        video=save_video(frames,out/'table_collision_diagnostic.mp4',args.fps)
        return dict(status='RENDERED_KINEMATIC_DIAGNOSTIC',video=video,
                    layout='synchronized_triptych_1920x460',
                    snapshots=len(stages)*len(viewpoints),
                    stages=len(stages),views=len(viewpoints),
                    minimum_gap_mm=min(r['min_tool_table_gap_mm'] for r in rows),
                    note='All 3 camera angles shown simultaneously at identical pose; static MuJoCo geometry sweep, not dynamic contact.')
    finally:
        model.geom_rgba[:]=original_colors
        if renderer is not None:renderer.close()


def lift_ik_targets(model,qadr,bid,lo,hi,q,args):
    probe=mujoco.MjData(model)
    # The anchor follows the gripper. World +Z is 20mm lift, independent of
    # Muhi horizontal yaw and irrespective of joint sign conventions.
    anchor=np.array([.030,.036,-.070])
    p,R=phase6g.fk(model,probe,qadr,bid,q)
    base=p+R@anchor
    pre,pre_err=ik_target(model,probe,qadr,bid,lo[:5],hi[:5],
                          base+[0,0,.040],anchor,q,R,max_nfev=110)
    raised,raise_err=ik_target(model,probe,qadr,bid,lo[:5],hi[:5],
                          base+[0,0,args.lift_mm/1000.],anchor,q,R,
                          max_nfev=110)
    return pre,raised,dict(pregrasp_ik_error_mm=pre_err,
                           lift_ik_error_mm=raise_err,
                           commanded_lift_mm=args.lift_mm)


def physical_close(model,qadr,aids,bq,bd,q,xyz,quat,lo,hi,fixed,moving,
                   bottle,args,out,bid,pre_q,lift_q):
    """Physical approach -> J6 torque-limited close -> hold -> 20mm lift."""
    d=mujoco.MjData(model)
    mujoco.mj_resetData(model,d)
    d.qpos[qadr[:5]]=pre_q
    d.qpos[qadr[5]]=hi[5]
    d.qpos[bq:bq+3]=xyz
    d.qpos[bq+3:bq+7]=quat
    d.qvel[:]=0
    d.ctrl[aids]=np.r_[pre_q,hi[5]]
    mujoco.mj_forward(model,d)
    phases=[('settle',.4),('approach',1.0),
            ('close',1.3),('hold',.7),('lift',1.),('lift_hold',.6)]
    cutoffs=np.cumsum([duration for _,duration in phases])
    total=float(cutoffs[-1])
    steps=int(math.ceil(total/args.dt))
    stride=max(1,int(round(1/(args.fps*args.dt))))
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=xyz
    camera.distance=.55;camera.azimuth=115;camera.elevation=-25
    vis=mujoco.MjvOption()
    vis.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True
    model.vis.global_.offwidth=max(640,model.vis.global_.offwidth)
    model.vis.global_.offheight=max(460,model.vis.global_.offheight)
    renderer=None
    if shutil.which('ffmpeg'):
        try:renderer=mujoco.Renderer(model,width=640,height=460)
        except Exception as exc:(out/'render_error.txt').write_text(repr(exc))
    frames=[];rows=[];deep=False;bad=False
    lift_permission=None
    lift_start_z=None
    hold_window=[]
    try:
        for k in range(steps+1):
            t=min(k*args.dt,total)
            if t<cutoffs[0]:
                phase='settle';qctl=pre_q;j6=hi[5]
            elif t<cutoffs[1]:
                phase='approach'
                u=(t-cutoffs[0])/phases[1][1]
                u=u*u*(3.-2.*u)
                qctl=(1.-u)*pre_q+u*q;j6=hi[5]
            elif t<cutoffs[2]:
                phase='close';qctl=q
                u=(t-cutoffs[1])/phases[2][1]
                j6=hi[5]+u*(lo[5]-hi[5])
            elif t<cutoffs[3]:
                phase='hold';qctl=q;j6=lo[5]
            else:
                if lift_permission is None:
                    # Require substantial two-jaw contact before lifting.
                    tail=hold_window[-max(1,int(.25/args.dt)):]
                    lift_permission=(len(tail)>0 and
                        sum(v['bilateral'] for v in tail)/len(tail)>=.6 and
                        max(v['bottle_xy_displacement_mm'] for v in tail)<=10. and
                        not deep)
                    lift_start_z=float(d.qpos[bq+2])
                if lift_permission:
                    if t<cutoffs[4]:
                        phase='lift'
                        u=(t-cutoffs[3])/phases[4][1]
                        u=u*u*(3.-2.*u)
                        qctl=(1.-u)*q+u*lift_q
                    else:
                        phase='lift_hold';qctl=lift_q
                else:
                    phase='lift_skipped_no_stable_grasp';qctl=q
                j6=lo[5]
            d.ctrl[aids]=np.r_[qctl,j6]
            if k:mujoco.mj_step(model,d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                bad=True;break
            fc,mc,depth=grasp.jaw_contact_counts(model,d,bottle,fixed,moving)
            deep=deep or (depth is not None and depth<-.003)
            row=dict(time_s=float(d.time),phase=phase,
                     fixed_contacts=int(fc),moving_contacts=int(mc),
                     bilateral=int(fc>0 and mc>0),
                     min_contact_dist_mm=None if depth is None else 1000*depth,
                     bottle_x_m=float(d.qpos[bq]),bottle_y_m=float(d.qpos[bq+1]),
                     bottle_z_m=float(d.qpos[bq+2]),
                     bottle_xy_displacement_mm=1000*float(
                         np.linalg.norm(d.qpos[bq:bq+2]-xyz[:2])),
                     j6_target_deg=float(np.rad2deg(j6)),
                     j6_actual_deg=float(np.rad2deg(d.qpos[qadr[5]])),
                     j6_actual_torque_nm=float(d.actuator_force[aids[5]]),
                     max_arm_tracking_error_deg=float(np.max(np.abs(
                         np.rad2deg(d.qpos[qadr[:5]]-qctl)))))
            rows.append(row)
            if phase=='hold':hold_window.append(row)
            if renderer is not None and (k%stride==0 or k==steps):
                # Same instant, three fixed cameras side by side. No jump cut.
                view_panels=[]
                for name,azimuth,elevation in (
                        ('side',85.,-15.),('opposite',265.,-15.),
                        ('overhead',120.,-75.)):
                    camera.azimuth=azimuth
                    camera.elevation=elevation
                    view_panels.append(render_scene(model,d,camera,renderer,vis,
                        f'Yaw {args.current_yaw_deg} {phase} {name} | F{fc}/M{mc} '
                        f'z={1000*(d.qpos[bq+2]-xyz[2]):+.1f}mm'))
                canvas=Image.new('RGB',(1920,460))
                for ix,panel in enumerate(view_panels):
                    canvas.paste(panel,(640*ix,0))
                frames.append(canvas)
    finally:
        if renderer:renderer.close()
    if rows:
        with (out/'joint_contact_trace.csv').open('w',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=rows[0].keys())
            writer.writeheader();writer.writerows(rows)
    video=save_video(frames,out/'grasp.mp4',args.fps)
    hold=[r for r in rows if r['phase']=='hold']
    lift_end=[r for r in rows if r['phase']=='lift_hold']
    travel=(1000*(lift_end[-1]['bottle_z_m']-lift_start_z)
            if lift_end and lift_start_z is not None else None)
    hold_lift_fraction=(sum(r['bilateral'] for r in lift_end)/len(lift_end)
                        if lift_end else 0.)
    lift_success=bool(lift_permission and travel is not None and
                      travel>=.75*args.lift_mm and
                      hold_lift_fraction>=.8 and not deep)
    return dict(
       status='NUMERIC_FAILURE' if bad else
              ('LIFT_20MM_SUCCEEDED' if lift_success else
               'LIFT_FAILED_OR_SKIPPED'),
       hold_steps=len(hold),
       bilateral_hold_steps=sum(r['bilateral'] for r in hold),
       max_xy_displacement_mm=max((r['bottle_xy_displacement_mm'] for r in rows),default=None),
       deep_penetration_over_3mm=bool(deep),
       requested_lift_mm=args.lift_mm,
       lift_attempted=bool(lift_permission),
       measured_lift_mm=travel,
       lifted_contact_fraction=hold_lift_fraction,
       lift_success=lift_success,
       max_abs_j6_torque_nm=max((abs(r['j6_actual_torque_nm']) for r in rows),default=None),
       final_bottle_z_m=rows[-1]['bottle_z_m'] if rows else None,
       video=video,
       note='Real MuJoCo physics, joint torque capped; 20mm world Z lift, NOT a real-robot demonstration.')


def approach_preview(model,data,qadr,bq,bid,lo,hi,q,xyz,quat,
                     args,out,bench,floor,tool_geoms,pre_q=None):
    # Preview only: interpolation in configuration space, collision NOT checked
    # in the currently isolated (finger-bottle-only) model.
    probe=mujoco.MjData(model)
    p, r=phase6g.fk(model,probe,qadr,bid,q)
    # Move away from the bottle along world +Z while maintaining near-constant
    # gripper orientation. This is a prototype pregrasp, not a safe motion plan.
    home=np.clip(bench,lo[:5]+1e-5,hi[:5]-1e-5)
    anchor=np.array([.03,.036,-.07],dtype=float)
    point=p+r@anchor
    pre=(pre_q.copy() if pre_q is not None else ik_target(
         model,probe,qadr,bid,lo[:5],hi[:5],
         point+[0,0,.040],anchor,q,r,max_nfev=65)[0])
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
                                      axis,fixed,moving,bottle,args.max_seeds,faxis,
                                      floor,tool_geoms,args)
            q=best['q']
            pre_q,lift_q,ik_lift=lift_ik_targets(model,qadr,bid,lo,hi,q,args)
            choices=[]
            for entry in options:
                choices.append({k:v for k,v in entry.items() if k!='q'})
            write_json(out/'grasp_candidates.json',
                       dict(yaw_deg=yaw,world_center_m=xyz.tolist(),
                            world_quat_wxyz=quat.tolist(),axis_world=axis.tolist(),
                            selected=choices[0],candidates=choices))
            # Always render diagnostic evidence before preflight can skip motion.
            try:
                collision_visual=render_table_collision_diagnostic(
                    model,qadr,bq,q,xyz,quat,lo,hi,floor,tool_geoms,
                    args,out,bench,bid)
            except Exception as render_exc:
                collision_visual=dict(status='DIAGNOSTIC_RENDER_FAILED',
                                      error=repr(render_exc),video=None)
                (out/'diagnostic_render_error.txt').write_text(repr(render_exc))
            preview=approach_preview(model,data,qadr,bq,bid,lo,hi,q,xyz,quat,
                                     args,out,bench,floor,tool_geoms,pre_q=pre_q)
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
            # Check the ACTUAL dynamic approach and subsequent world-Z lift
            # trajectories with jaw open/closed respectively.
            trajectory_clearances=[]
            for phase,first,last,angle in (
                    ('approach',pre_q,q,hi[5]),
                    ('lift',q,lift_q,lo[5])):
                for step in range(21):
                    alpha=step/20.
                    posture=(1-alpha)*first+alpha*last
                    mm,geom=tool_table_gap(model,data,qadr,bq,posture,xyz,quat,
                                            hi,floor,tool_geoms,jaw_angle=angle)
                    trajectory_clearances.append(dict(phase=phase,
                        sample=step,gap_mm=mm,nearest_geom=geom))
            with (out/'actual_motion_table_clearance.csv').open('w',newline='') as fp:
                writer=csv.DictWriter(fp,fieldnames=trajectory_clearances[0].keys())
                writer.writeheader();writer.writerows(trajectory_clearances)
            real_motion_min_gap=min(row['gap_mm'] for row in trajectory_clearances)
            preflight=(best['predicts_paired'] and
                       real_motion_min_gap>=args.min_table_clearance_mm and
                       table_gap>=args.min_table_clearance_mm and
                       preview.get('table_preflight_ok',False) and
                       max(ik_lift['pregrasp_ik_error_mm'],
                           ik_lift['lift_ik_error_mm'])<=6.)
            physics=physical_close(model,qadr,aids,bq,bd,q,xyz,quat,lo,hi,
                                   fixed,moving,bottle,args,out,bid,pre_q,lift_q) if preflight else {
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
                         actual_motion_min_table_gap_mm=real_motion_min_gap,
                         joint_angles_deg=dict(zip(JOINT_NAMES,np.rad2deg(q).tolist())),
                         geometry=best['geometry'],
                         geometric_score=best['score'],
                         fixed_inner_surface_gap_mm=best['fixed_inner_surface_gap_mm'],
                         lift_ik_diagnostics=ik_lift,
                         requested_lift_mm=args.lift_mm,
                         predicted_bilateral=best['predicts_paired'],
                         dynamic=physics,approach_preview=preview,
                         collision_diagnostic=collision_visual,
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
               actual_motion_min_table_gap_mm=r.get('actual_motion_min_table_gap_mm'),
               predicted_bilateral=r.get('predicted_bilateral'),
               bilateral_hold_steps=r.get('dynamic',{}).get('bilateral_hold_steps'),
               max_xy_displacement_mm=r.get('dynamic',{}).get('max_xy_displacement_mm'),
               deep_penetration=r.get('dynamic',{}).get('deep_penetration_over_3mm'),
               geometric_score=r.get('geometric_score'),
               fixed_inner_surface_gap_mm=r.get('fixed_inner_surface_gap_mm'),
               lift_measured_mm=r.get('dynamic',{}).get('measured_lift_mm'),
               lift_success=r.get('dynamic',{}).get('lift_success'),
               video=r.get('dynamic',{}).get('video'),
               approach_preview=r.get('approach_preview',{}).get('video'),
               collision_diagnostic=r.get('collision_diagnostic',{}).get('video'),
               error=r.get('error')) for r in outputs]
    with (args.out/'comparison.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys());wr.writeheader();wr.writerows(rows)
    write_json(args.out/'summary.json',dict(
        schema='phase7a_fixedjaw_lift20_v2',simulation_only=True,
        target_world_xy_m=[args.target_x,args.target_y],
        horizontal_bottle_center_height_m=args.table_z+physical.MUHI_RADIUS+.0005,
        bottle_length_m=physical.MUHI_LENGTH,bottle_radius_m=physical.MUHI_RADIUS,
        yaw_degrees=list(args.yaw_deg),joint_sign_constraint='URDF limits only',
        finger_long_axis_local=faxis.tolist(),
        permitted_perpendicular_error_deg=45.0,
        fixed_inner_jaw_margin_target_mm=[2,3],
        fixed_jaw_inner_face_preflight='center between jaws, inner-facing nearest witness and axial offset <= 25mm',
        requested_lift_mm=args.lift_mm,
        j6_torque_limit_nm=args.j6_torque,
        table_preflight_min_clearance_mm=args.min_table_clearance_mm,
        dynamic_tool_table_contact_enabled=True,
        table_preflight_j6_samples=[0.0,0.5,0.85,1.0],
        act_dataset_ready=False,
        preview_is_kinematic_only=True,
        validation_note='Fixed-jaw 2-3mm preflight; synchronized 3-view dynamic approach, torque-limited J6 close, and contact-gated 20mm world-Z lift; actual motion table gaps scanned. MuJoCo execution not yet verified.',
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
    p.add_argument('--max-seeds',type=int,default=6)
    p.add_argument('--fixed-jaw-gap-mm',type=float,default=2.5)
    p.add_argument('--lift-mm',type=float,default=20.0)
    p.add_argument('--j6-torque-limit-nm',type=float,default=0.45)
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
    # Retain Phase6C's j6-torque argument while setting the conservative
    # Phase7A lift actuator cap explicitly.
    a.j6_torque=a.j6_torque_limit_nm
    if a.max_seeds<1 or any(y<0 or y>=180 for y in a.yaw_deg):
        p.error('yaw must be in [0,180) and max-seeds>=1; 180-fold symmetry audit is separate')
    run(a)


if __name__=='__main__':main()
