#!/usr/bin/env python3
"""Phase B: one-pass open J6 self-collision/actuation ablation.

Two ISOLATED 0.40s simulations at the same J1-J5 pose:
- original fixed-finger split BOX collision + moving-jaw collision
- same, but only the gripper_link / moving-jaw BODY pair is excluded.

Neither benchmark has the Muhi/table: this isolates jaw-to-jaw mechanics.
The existing real-to-sim/table bench, Muhi, camera mount and five-view
videos are NEVER edited. This is NOT a grasp or Lift test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np

import real_to_sim as r2s
import gripper_geometry_phase_a_audit as base
import gripper_stl_convex_overlay_phase_a as overlay
import gripper_fixed_surface_proxy_phase_a as proxy

TCP_NAME="phase_b_open_j6_tcp"
DURATION=.40
TORQUE_NM=-.015
PITCH_FINE_MM=3.
PITCH_COARSE_MM=6.


def original_mesh_to_boxes(urdf,home):
    spec=mujoco.MjSpec.from_file(str(urdf))
    spec.compiler.discardvisual=False
    grip=spec.body("gripper_link")
    if grip is None:
        raise RuntimeError("gripper_link missing")
    grip.add_site(name=TCP_NAME,pos=base.TCP_LOCAL,size=[.002]*3)
    model=spec.compile()
    data=proxy.get_pose(model,home,-10.)
    fid,_=base.owned_mesh_geom(model,"gripper_link",base.FINGER_MESH)
    tip=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,TCP_NAME)
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"gripper_link")
    origin=np.asarray(data.site_xpos[tip]).copy()
    R=np.asarray(data.xmat[bid]).reshape(3,3)
    v,f=overlay.mesh_in_gripper(model,data,fid,origin,R)
    boxes=proxy.triangles_to_surface_boxes(
        v,f,PITCH_FINE_MM,PITCH_COARSE_MM,2400)
    return boxes


def compile_variant(urdf,boxes,excluded):
    spec=mujoco.MjSpec.from_file(str(urdf))
    spec.compiler.discardvisual=False
    grip=spec.body("gripper_link")
    grip.add_site(name=TCP_NAME,pos=base.TCP_LOCAL,size=[.002]*3)
    if excluded:
        # Critical: only suppress the pair of jaw bodies. Unlike disabling
        # geom.contype, this does NOT remove their contact with objects.
        spec.add_exclude(
            name="phase_b_jaw_jaw_ablation",
            bodyname1="gripper_link",
            bodyname2="moving_jaw_so101_v1_link")
    pre=spec.compile()
    fid,_=base.owned_mesh_geom(pre,"gripper_link",base.FINGER_MESH)
    bits=(int(pre.geom_contype[fid]),int(pre.geom_conaffinity[fid]))
    if bits[0]==0 or bits[1]==0:
        raise RuntimeError("Original fixed-finger collision bits unavailable")
    targets=[g for g in spec.geoms if
             str(getattr(g,"meshname","") or "").startswith(base.FINGER_MESH)]
    if len(targets)!=1:
        raise RuntimeError(f"Expected one fixed STL geom, found {len(targets)}")
    targets[0].contype=0
    targets[0].conaffinity=0
    for i,b in enumerate(boxes):
        g=grip.add_geom()
        g.name=f"{proxy.PROXY_PREFIX}{i:04d}"
        g.type=mujoco.mjtGeom.mjGEOM_BOX
        g.pos=(np.asarray(base.TCP_LOCAL)+b["center_tcp_mm"]/1000.).tolist()
        g.size=(b["half_mm"]/1000.).tolist()
        g.contype,g.conaffinity=bits
        g.density=0
        g.rgba=[.1,.8,.6,.15]
        g.group=4
    model=spec.compile()
    fids=[i for i in range(model.ngeom) if (
        mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i)
        or "").startswith(proxy.PROXY_PREFIX)]
    mg,_=base.owned_mesh_geom(
        model,"moving_jaw_so101_v1_link",base.MOVING_MESH)
    if len(fids)!=len(boxes):
        raise RuntimeError("Missing compiled proxy BOX geom")
    return model,fids,mg


def one_run(model,fixed,moving,home,excluded):
    data=proxy.get_pose(model,home,100.)
    data.qvel[:]=0.
    j6,j6_limits=base.joint_addr(model,base.JOINT)
    jid=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,base.JOINT))
    j6dof=int(model.jnt_dofadr[jid])
    # Stabilization assumptions identical to the existing bench.
    model.dof_armature[j6dof]=max(
        float(model.dof_armature[j6dof]),.002)
    model.dof_damping[j6dof]=max(
        float(model.dof_damping[j6dof]),.06)
    tip=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,TCP_NAME)
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"gripper_link")
    tcp=np.asarray(data.site_xpos[tip]).copy()
    R=np.asarray(data.xmat[bid]).reshape(3,3).copy()
    witness=np.zeros(6)
    records=[]
    for gid in fixed:
        distance=float(mujoco.mj_geomDistance(
            model,data,int(gid),moving,.3,witness))*1000.
        p=np.asarray(witness[:3]).copy()
        center=np.asarray(data.geom_xpos[gid]).copy()
        records.append({
            "mode":"jaw_pair_excluded" if excluded else "original_contacts",
            "box_name":mujoco.mj_id2name(
                model,mujoco.mjtObj.mjOBJ_GEOM,int(gid)),
            "signed_gap_mm":distance,
            "box_center_depth_mm":float((center-tcp)@R[:,2]*1000.),
            "fixed_witness_depth_mm":float((p-tcp)@R[:,2]*1000.),
            "fixed_witness_x_mm":float((p-tcp)@R[:,0]*1000.),
            "fixed_witness_y_mm":float((p-tcp)@R[:,1]*1000.),
        })
    records.sort(key=lambda x:x["signed_gap_mm"])
    start_angle=float(data.qpos[j6])
    # The source model has the same J1--J5 home angles in both runs.
    qlist=[base.joint_addr(model,name)[0] for name in r2s.JOINTS]
    dlist=[]
    for name in r2s.JOINTS:
        jj=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name))
        dlist.append(int(model.jnt_dofadr[jj]))
    initial_arm=data.qpos[qlist].copy()
    rows=[]
    nsteps=int(DURATION/model.opt.timestep)
    fixed_set=set(fixed)
    for step in range(nsteps):
        data.qpos[qlist]=initial_arm
        data.qvel[dlist]=0.
        data.qfrc_applied[:]=0.
        data.qfrc_applied[j6dof]=TORQUE_NM
        mujoco.mj_step(model,data)
        ncontact=0
        peak_force=0.
        for k,con in enumerate(data.contact[:data.ncon]):
            if ({int(con.geom1),int(con.geom2)} & fixed_set and
                    moving in (int(con.geom1),int(con.geom2))):
                ncontact+=1
                force=np.zeros(6)
                mujoco.mj_contactForce(model,data,k,force)
                peak_force=max(peak_force,float(force[0]))
        angle=float(data.qpos[j6])
        rows.append({
            "mode":"jaw_pair_excluded" if excluded else "original_contacts",
            "time_s":float(data.time),
            "j6_angle_deg":math.degrees(angle),
            "j6_closure_deg":math.degrees(start_angle-angle),
            "j6_velocity_deg_s":math.degrees(float(data.qvel[j6dof])),
            "commanded_torque_nm":TORQUE_NM,
            "jaw_jaw_solver_contacts":ncontact,
            "peak_jaw_contact_force_n":peak_force,
        })
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("J6 simulation non-finite")
        if (angle>j6_limits[1]+math.radians(6.) or
                angle<j6_limits[0]-math.radians(6.)):
            raise RuntimeError("J6 joint limit exceeded with safety margin")
    return rows,records


def plot_svg(runs,path):
    """Japanese-axis SVG chart; y=J6 angle deg, x=elapsed time s."""
    W,H=1100,570
    left,right,top,bottom=115,1015,148,440
    allrows=[r for block in runs for r in block]
    ys=[r["j6_angle_deg"] for r in allrows]
    low=min(ys)-3
    high=max(ys)+3
    if high-low<10:
        center=(high+low)/2
        low,high=center-5,center+5
    x=lambda t:left+(right-left)*t/DURATION
    y=lambda v:bottom-(v-low)/(high-low)*(bottom-top)
    chunks=[
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="35" y="40" font-size="26" font-weight="bold" fill="#273f53">J6が閉じない原因：爪同士の接触あり／なし比較</text>',
        '<text x="35" y="78" font-size="18" fill="#425e75">縦軸：第6関節J6の実際の角度［°］（小さいほど閉じる）</text>',
        '<text x="35" y="106" font-size="18" fill="#425e75">横軸：−0.015 N·mの閉方向トルクを与えてからの経過時間［秒］</text>',
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#f5f7f9"/>',
    ]
    for k in range(5):
        v=low+(high-low)*k/4
        chunks.append(f'<line x1="{left}" x2="{right}" y1="{y(v):.2f}" y2="{y(v):.2f}" stroke="#dde4e9"/>')
        chunks.append(f'<text x="{left-11}" y="{y(v)+6:.2f}" text-anchor="end" font-size="16" fill="#4b6276">{v:.1f}</text>')
        t=DURATION*k/4
        chunks.append(f'<text x="{x(t)}" y="{bottom+28}" text-anchor="middle" font-size="17" fill="#4b6276">{t:.2f}</text>')
    for rows,color in zip(runs,("#2876b6","#d5752c")):
        pts=" ".join(f"{x(r['time_s']):.1f},{y(r['j6_angle_deg']):.1f}" for r in rows)
        chunks.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
    chunks.extend([
        '<line x1="170" y1="495" x2="215" y2="495" stroke="#2876b6" stroke-width="4"/><text x="224" y="501" font-size="18" fill="#354d60">青：爪同士の接触あり（従来）</text>',
        '<line x1="620" y1="495" x2="665" y2="495" stroke="#d5752c" stroke-width="4"/><text x="675" y="501" font-size="18" fill="#354d60">橙：爪同士だけ接触除外</text>',
        '<text x="35" y="545" font-size="15" fill="#516e80">注意：物体なしのJ6単独実験。凸形状の距離と実際の接触力は別。把持成功の証拠ではありません。</text>',
        '</svg>'])
    path.write_text("\n".join(chunks),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--home-deg",nargs=5,type=float,
                        default=[-30.,-30.,45.,-20.,-90.])
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Output path already exists")
    args.out.mkdir(parents=True)
    info={"status":"INITIALIZING","source_urdf_modified":False,
          "physical_grasp_proven":False,"j6_start_deg":100.,
          "torque_nm":TORQUE_NM,"duration_s":DURATION,
          "object_present":False,"table_present":False,
          "original_table_bench_modified":False}
    try:
        boxes=original_mesh_to_boxes(args.urdf.resolve(strict=True),args.home_deg)
        times=[];collisions=[];modes=[]
        for excluded in (False,True):
            model,fixed,moving=compile_variant(
                args.urdf.resolve(),boxes,excluded)
            trace,records=one_run(
                model,fixed,moving,args.home_deg,excluded)
            times.append(trace)
            collisions.extend(records)
            modes.append({
                "mode":"jaw_pair_excluded" if excluded else "original_contacts",
                "min_geom_distance_mm":records[0]["signed_gap_mm"],
                "worst_box":records[0]["box_name"],
                "worst_box_center_depth_mm":
                    records[0]["box_center_depth_mm"],
                "worst_witness_depth_mm":
                    records[0]["fixed_witness_depth_mm"],
                "j6_start_deg":100.,
                "j6_final_deg":trace[-1]["j6_angle_deg"],
                "j6_closed_deg":trace[-1]["j6_closure_deg"],
                "max_jaw_self_solver_contacts":
                    max(r["jaw_jaw_solver_contacts"] for r in trace),
                "max_jaw_normal_force_n":
                    max(r["peak_jaw_contact_force_n"] for r in trace),
            })
        with (args.out/"open_j6_torque_trace.csv").open(
                "w",newline="",encoding="utf-8") as file:
            writer=csv.DictWriter(file,fieldnames=list(times[0][0]))
            writer.writeheader()
            for run in times: writer.writerows(run)
        with (args.out/"open_j6_collision_boxes.csv").open(
                "w",newline="",encoding="utf-8") as file:
            writer=csv.DictWriter(file,fieldnames=list(collisions[0]))
            writer.writeheader()
            writer.writerows(collisions)
        plot_svg(times,args.out/"open_j6_ablation.svg")
        difference=modes[1]["j6_closed_deg"]-modes[0]["j6_closed_deg"]
        info.update({
            "status":"DIAGNOSTIC_RECORDED",
            "fixed_proxy_box_count":len(boxes),
            "modes":modes,
            "closure_improvement_deg_when_masked":difference,
            "interpretation":
                ("A larger masked closure suggests jaw-jaw solver contact"
                 " suppresses J6 motion" if difference>2. else
                 "No decisive evidence jaw-jaw contact alone limits closure"),
            "caution":"Isolated robot-only bench: original object/table/cameras/"
                      "five-view pipeline are unchanged but not exercised here",
        })
    except Exception as exc:
        import traceback
        info["status"]="DIAGNOSTIC_FAILED"
        info["error"]=f"{type(exc).__name__}: {exc}"
        info["traceback"]=traceback.format_exc(limit=7)
    finally:
        (args.out/"summary.json").write_text(
            json.dumps(info,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
