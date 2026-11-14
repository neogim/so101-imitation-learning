#!/usr/bin/env python3
"""SO-101 Phase A: compiled STL triangles vs convex collision hull, in 3D.

No dynamics, no object, no edits to URDF, no grasp or reaching. Uses the
same compiled mesh assets/poses for both the STL surface and its convex
envelope. In MuJoCo ordinary mesh collisions use their convex hull.
A scipy-computed hull is a visualization/diagnostic approximation; the
MuJoCo collision result remains authoritative.
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
from scipy.spatial import QhullError

import real_to_sim as r2s
import gripper_geometry_phase_a_audit as base
import gripper_inner_face_phase_a as face_a

TCP_SITE = "phase_a_overlay_tcp"
ANGLES_DEG = (100., -10.)
DEPTH_BANDS = ((8.,20.), (20.,30.))
MAX_PLOT_TRIANGLES = 1700


def mesh_in_gripper(model, data, geom_id, tcp_world, gripper_rotation):
    """Return STL triangles and vertices, in gripper-link axes, mm from TCP."""
    mid=int(model.geom_dataid[geom_id])
    vs=int(model.mesh_vertadr[mid])
    nv=int(model.mesh_vertnum[mid])
    fs=int(model.mesh_faceadr[mid])
    nf=int(model.mesh_facenum[mid])
    vertices=np.array(model.mesh_vert[vs:vs+nv],dtype=np.float64)
    faces=np.array(model.mesh_face[fs:fs+nf],dtype=np.int64)
    if nv<4 or nf<4:
        raise RuntimeError(f"Mesh {mid} is empty or degenerate")
    if faces.min()<0 or faces.max()>=nv:
        if faces.min()>=vs and faces.max()<vs+nv:
            faces-=vs
        else:
            raise RuntimeError("Compiled mesh faces have unexpected indices")
    geom_R=np.asarray(data.geom_xmat[geom_id]).reshape(3,3)
    geom_xyz=np.asarray(data.geom_xpos[geom_id])
    points_world=vertices@geom_R.T+geom_xyz
    xyz=(points_world-tcp_world)@gripper_rotation*1000.
    return xyz,faces


def select_stl_triangles(vertices, faces):
    """Only show the region in the gripper near the intended grasp depth."""
    triangles=vertices[faces]
    depth=triangles[:,:,2].mean(axis=1)
    valid=triangles[(depth>=0.)&(depth<=70.)]
    if len(valid)>MAX_PLOT_TRIANGLES:
        idx=np.linspace(0,len(valid)-1,MAX_PLOT_TRIANGLES,dtype=np.int64)
        valid=valid[idx]
    return valid


def hull_edges_clipped(hull, points, low=0., high=70.):
    """Clip hull wireframe to displayed depth, preserving full-mesh hull."""
    edges=set()
    for triangle in hull.simplices:
        for k in range(3):
            i,j=sorted((int(triangle[k]),int(triangle[(k+1)%3])))
            edges.add((i,j))
    segments=[]
    for i,j in sorted(edges):
        p,q=points[i].copy(),points[j].copy()
        a,b=float(p[2]),float(q[2])
        if max(a,b)<low or min(a,b)>high:
            continue
        if abs(b-a)>1e-10:
            lo=max(0.,min(1.,(low-a)/(b-a)))
            hi=max(0.,min(1.,(high-a)/(b-a)))
            if b<a:
                lo,hi=1.-hi,1.-lo
            # Robust alternative: clip parameter t by interval intersection.
            t0=min((low-a)/(b-a),(high-a)/(b-a))
            t1=max((low-a)/(b-a),(high-a)/(b-a))
            t0,t1=max(0.,t0),min(1.,t1)
            if t1<t0:
                continue
            start=p+(q-p)*t0
            stop=p+(q-p)*t1
        else:
            start,stop=p,q
        if np.linalg.norm(stop-start)>1e-7:
            segments.append(np.stack([start,stop]))
    return segments


def inside_hull(hull,point,tolerance_mm=0.05):
    # Hull equations: a*x + b <= 0 for points inside.
    values=hull.equations[:,:3]@point+hull.equations[:,3]
    normal_lengths=np.linalg.norm(hull.equations[:,:3],axis=1)
    return bool(np.all(values<=tolerance_mm*normal_lengths))


def make_pair_records(model,data,fg,mg,tcpxyz,R,fh,mh,angle):
    fw,fn=face_a.mesh_triangles_world(model,data,fg)
    mw,mn=face_a.mesh_triangles_world(model,data,mg)
    fdepth=(fw-tcpxyz)@R[:,2]*1000.
    mdepth=(mw-tcpxyz)@R[:,2]*1000.
    rows=[]
    for low,high in DEPTH_BANDS:
        f,ff,nf=face_a.choose_subset(fw,fn,fdepth,low,high)
        m,mm,nm=face_a.choose_subset(mw,mn,mdepth,low,high)
        pair=face_a.face_pair(f,ff,m,mm)
        result={"j6_angle_deg":angle,"depth_low_mm":low,
                "depth_high_mm":high,"fixed_triangles_in_depth":nf,
                "moving_triangles_in_depth":nm,"facing_pairs":pair["pair_count"],
                "center_distance_mm":pair["pair_distance_mm"],
                "face_normal_angle_deg":pair["face_angle_deg"],
                "midpoint_inside_fixed_hull":None,
                "midpoint_inside_moving_hull":None,
                "fixed_center_gripper_mm":None,
                "moving_center_gripper_mm":None}
        if pair.get("fixed_face_center_world_m") is not None:
            fxyz=(np.array(pair["fixed_face_center_world_m"])-tcpxyz)@R*1000.
            mxyz=(np.array(pair["moving_face_center_world_m"])-tcpxyz)@R*1000.
            midpoint=(fxyz+mxyz)/2
            result.update({
                "midpoint_inside_fixed_hull":inside_hull(fh,midpoint),
                "midpoint_inside_moving_hull":inside_hull(mh,midpoint),
                "fixed_center_gripper_mm":fxyz.tolist(),
                "moving_center_gripper_mm":mxyz.tolist()})
        rows.append(result)
    return rows


def plot_overlay(shapes, rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    from matplotlib.lines import Line2D

    palette=(("#2376b7","#104c7a"),("#e18a38","#a24b18"))
    points=[]
    for pose in shapes.values():
        for item in pose:
            vv=item["vertices"]
            points.append(vv[(vv[:,2]>=0)&(vv[:,2]<=70)])
    allpoints=np.concatenate([p for p in points if len(p)],axis=0)
    xmin,xmax=np.min(allpoints[:,:2],axis=0),np.max(allpoints[:,:2],axis=0)
    lo=np.array([xmin[0]-5,xmin[1]-5,0.])
    hi=np.array([xmax[0]+5,xmax[1]+5,70.])
    # avoid clipping excessively large gripper root. Geometry is not resized.
    views=((30.,-65.),(22.,32.))
    fig=plt.figure(figsize=(16.5,11.0),facecolor="white")
    for row,angle in enumerate(ANGLES_DEG):
        for col,(elev,azim) in enumerate(views):
            ax=fig.add_subplot(2,2,row*2+col+1,projection="3d")
            for i,item in enumerate(shapes[angle]):
                vertex_color,hull_color=palette[i]
                tris=select_stl_triangles(item["vertices"],item["faces"])
                mesh=Poly3DCollection(tris,facecolor=vertex_color,
                    edgecolor="none",linewidth=0,alpha=0.24)
                ax.add_collection3d(mesh)
                edges=hull_edges_clipped(item["hull"],item["vertices"])
                if edges:
                    ax.add_collection3d(Line3DCollection(
                        edges,colors=hull_color,linewidths=0.75,alpha=0.73))
            if angle==-10.:
                for r in rows:
                    if (r["j6_angle_deg"]!=angle or
                        r["fixed_center_gripper_mm"] is None):
                        continue
                    f=np.asarray(r["fixed_center_gripper_mm"])
                    m=np.asarray(r["moving_center_gripper_mm"])
                    ax.plot([f[0],m[0]],[f[1],m[1]],[f[2],m[2]],
                            color="#d92f52",linestyle="--",
                            linewidth=2,marker="o",markersize=4)
            ax.view_init(elev=elev,azim=azim)
            ax.set_xlim(lo[0],hi[0])
            ax.set_ylim(lo[1],hi[1])
            ax.set_zlim(0.,70.)
            ax.set_box_aspect(np.maximum(hi-lo,1.))
            ax.set_xlabel("X (mm)",labelpad=4)
            ax.set_ylabel("Y (mm)",labelpad=4)
            ax.set_zlabel("Depth from TCP (mm)",labelpad=3)
            ax.set_title(
                f"J6 = {angle:g} deg ({'OPEN' if angle>0 else 'CLOSED'})"
                f"  |  viewpoint {'A' if col==0 else 'B'}",
                fontsize=12,pad=12)
    fig.suptitle(
        "SO-101: STL triangles (pale surfaces) + their convex hull "
        "(dark wireframe) in SAME 3D pose",fontsize=17,y=.995)
    fig.text(.5,.95,
        "Blue = fixed jaw  |  Orange = moving jaw  |  "
        "Red dashed = selected facing-face centers at closed J6",
        fontsize=12,ha="center")
    fig.text(.5,.02,
        "Depth from TCP: 0–70 mm. Full-mesh convex hull, clipped for "
        "display. Mesh hulls model collision; this is NOT a contact proof.",
        fontsize=11,ha="center")
    fig.subplots_adjust(left=.03,right=.98,bottom=.05,top=.92,
                        wspace=.06,hspace=.22)
    fig.savefig(out,dpi=145)
    plt.close(fig)


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
    summary={"status":"INITIALIZING","purpose":"Phase A STL vs convex hull 3D comparison",
             "source_urdf_modified":False,"physical_grasp_proven":False,
             "object_present":False,"simulation_only":True,
             "angles_deg":list(ANGLES_DEG)}
    records=[]
    try:
        spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
        spec.compiler.discardvisual=False
        grip=spec.body("gripper_link")
        if grip is None:
            raise RuntimeError("gripper_link not found in source URDF")
        grip.add_site(name=TCP_SITE,pos=base.TCP_LOCAL,size=[.002]*3)
        model=spec.compile()
        data=mujoco.MjData(model)
        fq,fname=base.owned_mesh_geom(
            model,"gripper_link",base.FINGER_MESH)
        mq,mname=base.owned_mesh_geom(
            model,"moving_jaw_so101_v1_link",base.MOVING_MESH)
        tcpid=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,TCP_SITE))
        gripid=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_BODY,"gripper_link"))
        j6,j6limit=base.joint_addr(model,base.JOINT)
        qid=[]
        for name,deg in zip(r2s.JOINTS,args.home_deg):
            addr,limits=base.joint_addr(model,name)
            if not limits[0]<=math.radians(deg)<=limits[1]:
                raise ValueError(f"Home angle outside joint range: {name}")
            qid.append(addr)
        if not all(j6limit[0]<=math.radians(a)<=j6limit[1]
                   for a in ANGLES_DEG):
            raise RuntimeError("J6 sample angles exceed model limits")
        shapes={}
        for angle in ANGLES_DEG:
            data.qpos[qid]=np.radians(args.home_deg)
            data.qpos[j6]=math.radians(angle)
            mujoco.mj_forward(model,data)
            tcp=np.array(data.site_xpos[tcpid])
            R=np.array(data.xmat[gripid]).reshape(3,3)
            pair=[]
            for geom in (fq,mq):
                vertices,faces=mesh_in_gripper(model,data,geom,tcp,R)
                try:
                    hull=ConvexHull(vertices)
                except QhullError as exc:
                    raise RuntimeError(f"Convex hull failed for geom {geom}: {exc}")
                pair.append({"vertices":vertices,"faces":faces,"hull":hull})
            shapes[angle]=pair
            witness=np.zeros(6)
            dist=mujoco.mj_geomDistance(model,data,fq,mq,.3,witness)
            cand=make_pair_records(model,data,fq,mq,tcp,R,
                                   pair[0]["hull"],pair[1]["hull"],angle)
            records.extend(cand)
            summary[f"j6_{angle:g}_whole_mesh_collision_signed_gap_mm"]=float(dist*1000)
        plot_overlay(shapes,records,args.out/"stl_convex_hull_3d_overlay.png")
        with (args.out/"face_corridor_hull_checks.csv").open(
                "w",encoding="utf-8",newline="") as fp:
            writer=csv.DictWriter(fp,fieldnames=list(records[0]))
            writer.writeheader()
            for row in records:
                writer.writerow({key:json.dumps(value) if isinstance(value,list)
                                 else value for key,value in row.items()})
        summary.update({
            "status":"DIAGNOSTIC_RECORDED",
            "fixed_mesh":fname,"moving_mesh":mname,
            "visualization":"stl_convex_hull_3d_overlay.png",
            "face_candidate_rows":len(records),
            "mesh_collision_uses_convex_hull":True,
            "hull_method":"scipy.spatial.ConvexHull of MuJoCo compiled mesh vertices",
            "pose_coordinate_frame":"gripper_link orientation, origin at phase_a_overlay_tcp; millimeters",
            "midpoint_check_caveat":
                "Only candidate face-center midpoint vs each convex hull; "
                "neither a continuous aperture clearance nor grasp proof",
            "stl_rendering_caveat":
                "Sampled STL triangle surfaces (up to 1700 per geom) "
                "and full convex-hull wireframes cropped at TCP depth 0–70 mm",
            "future_step":"Visually assess jaw grasp corridor; only then "
                          "consider convex-part decomposition or contact pads.",
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
