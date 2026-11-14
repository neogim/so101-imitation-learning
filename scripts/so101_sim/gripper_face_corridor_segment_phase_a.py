#!/usr/bin/env python3
"""Phase A: compare face-center segment with full-mesh convex hulls & STL crossings.

Diagnostic, no edits to URDF, no object, physics or grasp claims. For each
candidate facing-face pair, compute exact line-interval inside each convex
hull, and line/triangle crossings of the compiled STL surface. An STL
crossing does not establish solid occupancy if mesh is non-watertight.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

import real_to_sim as r2s
import gripper_geometry_phase_a_audit as base
import gripper_inner_face_phase_a as face_a
import gripper_stl_convex_overlay_phase_a as overlay

DEPTH_BANDS=((8.,20.),(20.,30.))
POSE_DEG=-10.


def hull_segment_interval(hull,start,stop):
    """Return exact closed t interval in [0,1] lying within convex hull."""
    start=np.asarray(start,dtype=float)
    direction=np.asarray(stop,dtype=float)-start
    low,high=0.,1.
    for eq in np.asarray(hull.equations):
        v=float(np.dot(eq[:3],start)+eq[3])
        slope=float(np.dot(eq[:3],direction))
        if abs(slope)<1e-11:
            if v>1e-7:
                return None
        else:
            boundary=-v/slope
            if slope>0:
                high=min(high,boundary)
            else:
                low=max(low,boundary)
            if high<low-1e-9:
                return None
    low,high=max(0.,low),min(1.,high)
    if high<low:
        return None
    return (float(low),float(high))


def triangle_crossing_t(vertices,faces,start,stop):
    """Moller-Trumbore segment/triangle crossing t, with duplicate removal."""
    triangles=vertices[faces]
    p=np.asarray(start)
    d=np.asarray(stop)-p
    v0=triangles[:,0,:]
    e1=triangles[:,1,:]-v0
    e2=triangles[:,2,:]-v0
    h=np.cross(np.broadcast_to(d,e2.shape),e2)
    a=np.einsum("ij,ij->i",e1,h)
    good=np.abs(a)>1e-10
    inv=np.divide(1.0,a,out=np.zeros_like(a),where=good)
    rel=p-v0
    u=inv*np.einsum("ij,ij->i",rel,h)
    q=np.cross(rel,e1)
    v=inv*(q@d)
    t=inv*np.einsum("ij,ij->i",e2,q)
    valid=good&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(t>=0)&(t<=1)
    ordered=np.sort(t[valid])
    unique=[]
    # Shared STL triangle edges appear as duplicate crossings.
    for value in ordered:
        if not unique or value-unique[-1]>1e-5:
            unique.append(float(value))
    return unique


def plot_svg(rows,path):
    """Compact Japanese SVG. Distance x-axis is mm, not the t fraction."""
    W,H=1140,535
    x0,x1=245,1050
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<text x="35" y="41" font-size="25" font-weight="bold" fill="#26394b">Phase A：把持候補の線分上で凸包が占める範囲</text>',
       '<text x="35" y="77" font-size="18" fill="#445e70">横軸：固定爪候補面の中心から可動爪候補面の中心へ進む距離［mm］</text>',
       '<text x="35" y="107" font-size="16" fill="#607789">J6＝−10°（閉）、2つの候補深さ帯。色の帯は凸包の内部、縦線はSTL面との交差。</text>']
    colors=[("#2875b5","#dd7f33"),("#2875b5","#dd7f33")]
    for idx,r in enumerate(rows):
        y=185+idx*160
        L=float(r["segment_length_mm"])
        X=lambda dist:x0+(dist/max(L,1e-9))*(x1-x0)
        parts.extend([
          f'<text x="35" y="{y-36}" font-size="19" font-weight="bold" fill="#314a5e">爪先から{r["depth_start_mm"]:g}〜{r["depth_end_mm"]:g}mm</text>',
          f'<rect x="{x0}" y="{y-17}" width="{x1-x0}" height="73" fill="#f5f7f9" rx="5"/>'])
        for j,which in enumerate(("fixed","moving")):
            yy=y+j*34
            iv=r[f"{which}_hull_t_interval"]
            label="固定爪の凸包" if which=="fixed" else "可動爪の凸包"
            color="#2875b5" if which=="fixed" else "#dd7f33"
            parts.append(f'<text x="{x0-17}" y="{yy+13}" font-size="16" text-anchor="end" fill="#3b5264">{label}</text>')
            if iv is not None:
                a,b=iv
                parts.append(f'<rect x="{X(a*L):.2f}" y="{yy}" width="{max(1.,(b-a)/max(L,1e-9)*L*(x1-x0)):.2f}" height="20" rx="3" fill="{color}" opacity=".78"/>')
            for t in r[f"{which}_stl_cross_t"]:
                parts.append(f'<line x1="{X(t*L):.2f}" y1="{yy-4}" x2="{X(t*L):.2f}" y2="{yy+24}" stroke="#202d38" stroke-width="2"/>')
        for k in range(5):
            d=L*k/4
            parts.append(f'<text x="{X(d):.1f}" y="{y+81}" font-size="15" text-anchor="middle" fill="#4a5e6d">{d:.1f}</text>')
        parts.append(f'<text x="{x0}" y="{y+103}" font-size="15" fill="#4b6375">全区間長：{L:.2f}mm　／　青：固定凸包　橙：可動凸包　黒い縦線：STL面交差</text>')
    parts.extend([
      '<text x="35" y="511" font-size="15" fill="#546d7d">注意：凸包内部≠実STLの実体。STL面との交差も実体内外の厳密判定ではない（非閉鎖メッシュ等）。</text>',
      '</svg>'])
    path.write_text("\n".join(parts),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--home-deg",nargs=5,type=float,
                    default=[-30.,-30.,45.,-20.,-90.])
    args=ap.parse_args()
    if args.out.exists(): ap.error("Output path already exists")
    args.out.mkdir(parents=True)
    summary={"status":"INITIALIZING","pose_j6_deg":POSE_DEG,
             "original_urdf_modified":False,"object_present":False,
             "physical_grasp_proven":False,
             "method":"analytical hull line intervals plus STL triangle crossings"}
    rows=[]
    try:
        spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
        spec.compiler.discardvisual=False
        grip=spec.body("gripper_link")
        if grip is None: raise RuntimeError("Missing gripper_link")
        grip.add_site(name="phase_a_corridor_tcp",
                      pos=base.TCP_LOCAL,size=[.002]*3)
        model=spec.compile()
        data=mujoco.MjData(model)
        qids=[]
        for name,deg in zip(r2s.JOINTS,args.home_deg):
            adr,lims=base.joint_addr(model,name)
            if not lims[0]<=math.radians(deg)<=lims[1]:
                raise ValueError(f"Joint {name} outside limits")
            qids.append(adr)
        j6,lims=base.joint_addr(model,base.JOINT)
        if not lims[0]<=math.radians(POSE_DEG)<=lims[1]:
            raise ValueError("J6 outside limits")
        data.qpos[qids]=np.radians(args.home_deg)
        data.qpos[j6]=math.radians(POSE_DEG)
        mujoco.mj_forward(model,data)
        fg,fname=base.owned_mesh_geom(model,"gripper_link",base.FINGER_MESH)
        mg,mname=base.owned_mesh_geom(
            model,"moving_jaw_so101_v1_link",base.MOVING_MESH)
        siteid=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_SITE,"phase_a_corridor_tcp"))
        bid=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_BODY,"gripper_link"))
        tcp=np.array(data.site_xpos[siteid])
        R=np.array(data.xmat[bid]).reshape(3,3)
        mesh=[]
        for gid in (fg,mg):
            verts,faces=overlay.mesh_in_gripper(model,data,gid,tcp,R)
            mesh.append((verts,faces,ConvexHull(verts)))
        fc,fn=face_a.mesh_triangles_world(model,data,fg)
        mc,mn=face_a.mesh_triangles_world(model,data,mg)
        fd=(fc-tcp)@R[:,2]*1000
        md=(mc-tcp)@R[:,2]*1000
        for low,high in DEPTH_BANDS:
            f,nf,fcnt=face_a.choose_subset(fc,fn,fd,low,high)
            m,nm,mcnt=face_a.choose_subset(mc,mn,md,low,high)
            pair=face_a.face_pair(f,nf,m,nm)
            if pair["pair_count"]==0:
                raise RuntimeError(f"No facing face pair at depth {low:g}-{high:g} mm")
            # These are world coordinates in meters, transform as the meshes did.
            fp=(np.asarray(pair["fixed_face_center_world_m"])-tcp)@R*1000.
            mp=(np.asarray(pair["moving_face_center_world_m"])-tcp)@R*1000.
            length=float(np.linalg.norm(mp-fp))
            row={"j6_angle_deg":POSE_DEG,
                "depth_start_mm":low,"depth_end_mm":high,
                "segment_length_mm":length,
                "face_normal_angle_deg":pair["face_angle_deg"],
                "fixed_center_gripper_mm":fp.tolist(),
                "moving_center_gripper_mm":mp.tolist()}
            for which,(v,faces,hull) in zip(("fixed","moving"),mesh):
                interval=hull_segment_interval(hull,fp,mp)
                intersections=triangle_crossing_t(v,faces,fp,mp)
                row.update({
                    f"{which}_hull_t_interval":
                        list(interval) if interval is not None else None,
                    f"{which}_hull_length_mm":
                        float((interval[1]-interval[0])*length)
                        if interval is not None else 0.,
                    f"{which}_stl_cross_t":intersections,
                    f"{which}_stl_cross_count":len(intersections),
                })
            rows.append(row)
        with (args.out/"face_corridor_segment.csv").open(
                "w",newline="",encoding="utf-8") as out:
            w=csv.DictWriter(out,fieldnames=list(rows[0]))
            w.writeheader()
            for row in rows:
                w.writerow({k:json.dumps(v) if isinstance(v,list) else v
                           for k,v in row.items()})
        plot_svg(rows,args.out/"face_corridor_segment.svg")
        summary.update({
            "status":"DIAGNOSTIC_RECORDED",
            "fixed_mesh":fname,"moving_mesh":mname,
            "rows":rows,
            "interpretation_limits":[
                "Convex hull interval is computed from full compiled mesh",
                "STL surface intersection is not STL volume occupancy",
                "Chosen face-pair depends on triangle winding and sampling",
                "No physical grasp, Muhi, actuator, or true aperture check"
            ],
            "next_decision":"Only split collision hull if grasp corridor obstruction "
                            "is established and localized; else move to grasp validation"
        })
    except Exception as exc:
        import traceback
        summary["status"]="DIAGNOSTIC_FAILED"
        summary["error"]=f"{type(exc).__name__}: {exc}"
        summary["traceback"]=traceback.format_exc(limit=5)
    finally:
        (args.out/"summary.json").write_text(
            json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8")
        print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
