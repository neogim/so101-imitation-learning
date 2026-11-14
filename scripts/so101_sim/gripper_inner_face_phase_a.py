#!/usr/bin/env python3
"""Phase A: inspect opposing triangular inner surfaces of the SO-101 jaws.

Diagnostic ONLY. No object, no motors, no hardware control, no grasp claims.
Measures candidate facing TRIANGLE CENTROIDS in two TCP depth bands.
This is NOT a surface-to-surface clearance, contact check or usable aperture.
Source URDF and existing five-view movie generator are NOT modified.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import mujoco
import real_to_sim as r2s
import gripper_geometry_phase_a_audit as base

ANGLES=(100.,55.,20.,-10.)
DEPTH_BANDS=((8.,20.),(20.,30.))
MAX_FACES=650
MIN_FACING_COS=0.65

def mesh_triangles_world(model,data,geom_id):
    mid=int(model.geom_dataid[geom_id])
    v0=int(model.mesh_vertadr[mid])
    nv=int(model.mesh_vertnum[mid])
    f0=int(model.mesh_faceadr[mid])
    nf=int(model.mesh_facenum[mid])
    verts=np.asarray(model.mesh_vert[v0:v0+nv],dtype=np.float64)
    faces=np.asarray(model.mesh_face[f0:f0+nf],dtype=int).copy()
    if faces.size==0 or verts.size==0:
        raise RuntimeError("compiled mesh contains no triangles or vertices")
    # MuJoCo normally uses local per-mesh vertex indices.
    if faces.min()<0 or faces.max()>=nv:
        if faces.min()>=v0 and faces.max()<v0+nv:
            faces-=v0
        else:
            raise RuntimeError("mesh_face indices do not reference mesh vertices")
    R=np.asarray(data.geom_xmat[geom_id]).reshape(3,3)
    world=verts@R.T+np.asarray(data.geom_xpos[geom_id])
    tri=world[faces]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area=np.linalg.norm(cross,axis=1)
    good=area>1e-11
    centroids=tri[good].mean(axis=1)
    normals=cross[good]/area[good,None]
    return centroids,normals

def choose_subset(c,n,depths,low,high):
    mask=(depths>=low)&(depths<high)
    c=c[mask]
    n=n[mask]
    if len(c)>MAX_FACES:
        ids=np.linspace(0,len(c)-1,MAX_FACES,dtype=int)
        c,n=c[ids],n[ids]
    return c,n,int(mask.sum())

def face_pair(f,fn,m,mn):
    """Most-local plausible facing triangle pair, with winding caveat."""
    if not len(f) or not len(m):
        return dict(pair_distance_mm=None,face_angle_deg=None,
                    fixed_normal_toward_moving_cos=None,
                    moving_normal_toward_fixed_cos=None,
                    pair_count=0,
                    fixed_face_center_world_m=None,
                    moving_face_center_world_m=None,
                    fixed_outward_normal_world=None,
                    moving_outward_normal_world=None)
    diff=m[None,:,:]-f[:,None,:]
    distances=np.linalg.norm(diff,axis=-1)
    valid=distances>1e-7
    unit=diff/np.maximum(distances[:,:,None],1e-7)
    cf=np.einsum("ik,ijk->ij",fn,unit)
    cm=-np.einsum("jk,ijk->ij",mn,unit)
    facing=valid&(cf>=MIN_FACING_COS)&(cm>=MIN_FACING_COS)
    count=int(np.count_nonzero(facing))
    if not count:
        return dict(pair_distance_mm=None,face_angle_deg=None,
                    fixed_normal_toward_moving_cos=None,
                    moving_normal_toward_fixed_cos=None,
                    pair_count=0,
                    fixed_face_center_world_m=None,
                    moving_face_center_world_m=None,
                    fixed_outward_normal_world=None,
                    moving_outward_normal_world=None)
    dist=np.where(facing,distances,np.inf)
    i,j=np.unravel_index(np.argmin(dist),dist.shape)
    # 0deg means parallel normals; 180deg means opposing normals.
    facing_angle=math.degrees(math.acos(
        float(np.clip(np.dot(fn[i],mn[j]),-1,1))))
    return dict(pair_distance_mm=float(distances[i,j]*1000),
                face_angle_deg=facing_angle,
                fixed_normal_toward_moving_cos=float(cf[i,j]),
                moving_normal_toward_fixed_cos=float(cm[i,j]),
                pair_count=count,
                fixed_face_center_world_m=f[i].tolist(),
                moving_face_center_world_m=m[j].tolist(),
                fixed_outward_normal_world=fn[i].tolist(),
                moving_outward_normal_world=mn[j].tolist())

def svg_plot(rows,out):
    # No Matplotlib/font dependency; Japanese labels directly embedded in SVG.
    import xml.sax.saxutils as xml
    w,h=1070,510
    left,right,top,bottom=180,1005,128,370
    values=[r["pair_distance_mm"] for r in rows
            if r["pair_distance_mm"] is not None]
    vmax=max(20.,min(200.,(max(values)*1.1 if values else 20.)))
    X=lambda a:left+(100-a)/110*(right-left)
    Y=lambda d:bottom-d/vmax*(bottom-top)
    colors={(8.,20.):"#d47525",(20.,30.):"#1e8c7b"}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           '<text x="35" y="37" font-family="sans-serif" font-size="24" font-weight="bold" fill="#263c51">Phase A：向かい合う爪の面の候補（診断用）</text>',
           '<text x="35" y="69" font-family="sans-serif" font-size="16" fill="#53687a">縦軸＝向かい合う三角形の中心点間距離［mm］（面間の実際の隙間ではない）</text>',
           '<text x="35" y="93" font-family="sans-serif" font-size="15" fill="#53687a">両方の面法線が互いを向く候補のみ。三角形の向き／サンプリングに依存</text>']
    for i in range(5):
        val=vmax*i/4
        yy=Y(val)
        parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#e4e9ee"/>',
                  f'<text x="{left-15}" y="{yy+5:.1f}" text-anchor="end" font-size="15" fill="#4a5d6d">{val:.1f}</text>']
    for angle in ANGLES:
        xx=X(angle)
        parts += [f'<text x="{xx:.1f}" y="{bottom+28}" text-anchor="middle" font-size="16" fill="#4a5d6d">{angle:g}°</text>']
    for band in DEPTH_BANDS:
        series=[r for r in rows if (r["depth_start_mm"],r["depth_end_mm"])==band]
        pts=[]
        for r in series:
            val=r["pair_distance_mm"]
            if val is None:
                continue
            xx,yy=X(r["j6_angle_deg"]),Y(val)
            pts.append(f"{xx:.1f},{yy:.1f}")
            parts += [f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{colors[band]}"/>']
        if len(pts)>1:
            parts += [f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[band]}" stroke-width="3"/>']
    parts += [
        f'<text x="{(left+right)/2:.1f}" y="430" text-anchor="middle" font-size="17" fill="#2c495c">横軸：J6角度［°］（左＝開100°　右＝閉−10°）</text>',
        '<line x1="230" y1="457" x2="273" y2="457" stroke="#d47525" stroke-width="5"/>',
        '<text x="285" y="463" font-size="17" fill="#3a5368">爪先から8–20mm</text>',
        '<line x1="590" y1="457" x2="633" y2="457" stroke="#1e8c7b" stroke-width="5"/>',
        '<text x="645" y="463" font-size="17" fill="#3a5368">爪先から20–30mm</text>',
        '<text x="35" y="495" font-size="15" fill="#566d80">線・点がない箇所＝法線条件を満たす候補が見つからなかった（隙間0mmではない）。</text>',
        '</svg>']
    out.write_text("\n".join(parts),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--home-deg",nargs=5,type=float,
                    default=[-30.,-30.,45.,-20.,-90.])
    args=ap.parse_args()
    if args.out.exists():
        ap.error("--out already exists")
    args.out.mkdir(parents=True)
    report={"status":"INITIALIZING","phase":"A inner face triangulation",
            "original_urdf_unchanged":True,"no_object":True,
            "no_physics_grasp_proven":True,"collision_clearance_measured":False}
    rows=[]
    try:
        spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
        spec.compiler.discardvisual=False
        grip=spec.body("gripper_link")
        if grip is None:
            raise RuntimeError("Missing gripper_link")
        grip.add_site(name="phase_a_face_tcp",pos=base.TCP_LOCAL,size=[.002]*3)
        model=spec.compile()
        data=mujoco.MjData(model)
        joints=[]
        for name,value in zip(r2s.JOINTS,args.home_deg):
            adr,limits=base.joint_addr(model,name)
            if not limits[0]<=math.radians(value)<=limits[1]:
                raise RuntimeError(f"Home angle for {name} outside joint limit")
            joints.append(adr)
        j6,lim=base.joint_addr(model,base.JOINT)
        fg,fmesh=base.owned_mesh_geom(model,"gripper_link",base.FINGER_MESH)
        mg,mmesh=base.owned_mesh_geom(
            model,"moving_jaw_so101_v1_link",base.MOVING_MESH)
        tcp=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_SITE,"phase_a_face_tcp"))
        gb=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_BODY,"gripper_link"))
        for angle in ANGLES:
            data.qpos[joints]=np.radians(args.home_deg)
            data.qpos[j6]=math.radians(angle)
            mujoco.mj_forward(model,data)
            pos=np.asarray(data.site_xpos[tcp])
            R=np.asarray(data.xmat[gb]).reshape(3,3)
            fc,fn=mesh_triangles_world(model,data,fg)
            mc,mn=mesh_triangles_world(model,data,mg)
            fdepth=(fc-pos)@R[:,2]*1000.
            mdepth=(mc-pos)@R[:,2]*1000.
            for low,high in DEPTH_BANDS:
                f,ff,fcount=choose_subset(fc,fn,fdepth,low,high)
                m,mn_,mcount=choose_subset(mc,mn,mdepth,low,high)
                pair=face_pair(f,ff,m,mn_)
                rows.append({
                    "j6_angle_deg":float(angle),
                    "depth_start_mm":low,
                    "depth_end_mm":high,
                    "fixed_faces_in_band":fcount,
                    "moving_faces_in_band":mcount,
                    "fixed_faces_sampled":len(f),
                    "moving_faces_sampled":len(m),
                    **pair,
                })
        with (args.out/"inner_face_candidates.csv").open(
                "w",newline="",encoding="utf-8") as output:
            writer=csv.DictWriter(output,fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k:(json.dumps(v) if isinstance(v,list)
                                   else v) for k,v in row.items()})
        svg_plot(rows,args.out/"inner_face_candidates.svg")
        report.update({
            "status":"DIAGNOSTIC_RECORDED",
            "original_collision_mesh_remains":True,
            "fixed_mesh":fmesh,
            "moving_mesh":mmesh,
            "angles_deg":list(ANGLES),
            "depth_bands_mm":[list(z) for z in DEPTH_BANDS],
            "candidate_pairs":[
                {k:v for k,v in row.items()
                 if k in ("j6_angle_deg","depth_start_mm","depth_end_mm",
                          "pair_distance_mm","pair_count")}
                for row in rows],
            "normal_direction_caveat":
                "Face winding from compiled STL is assumed outward. "
                "Reversed/inconsistent winding may exclude valid patches.",
            "distance_caveat":
                "triangle centroid to triangle centroid, not exact opening, "
                "signed gap, or convex-hull collision clearance",
            "next_step":
                "Review inward face locations and normal orientation with "
                "mesh/collision overlay before defining contact pads."
        })
    except Exception as e:
        report["status"]="DIAGNOSTIC_FAILED"
        report["error"]=f"{type(e).__name__}: {e}"
    finally:
        (args.out/"summary.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
