#!/usr/bin/env python3
"""Phase A: reversible, simulation-only fixed-jaw collision proxy experiment.

The original URDF and visual STL are unchanged. In an in-memory MjSpec,
disable contact on the original fixed-jaw mesh (otherwise its full convex
hull would still obstruct the pocket). Replace contact with small surface
boxes: 2mm cells near the target depth 8-20mm, 5mm cells elsewhere.
This is a THIN SURFACE approximation, not validated rigid geometry.
Compare analytic occupancy of facing-face candidate segments, plus a
MuJoCo signed-distance sanity check. Do not use for production or grasp
success claims before validation.
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
import gripper_inner_face_phase_a as faces
import gripper_stl_convex_overlay_phase_a as overlay
from gripper_face_corridor_segment_phase_a import hull_segment_interval

BANDS=((8.,20.),(20.,30.))
J6_ANGLES=(100.,-10.)
SITE="phase_a_proxy_tcp"
PROXY_PREFIX="phase_a_fixed_surface_box_"


def triangles_to_surface_boxes(vertices,triangles,pitch_fine,pitch_coarse,max_boxes):
    """Deterministic thin-shell voxels sampled from STL surface, in TCP-mm frame.

    The coarse/fine regions are separated by z=8 and z=20mm. No cell can
    extend across these boundaries by more than half of its own pitch.
    Surface cells are merged along x within each (y,z,pitch) row.
    """
    tri=vertices[triangles]
    mid=(tri[:,0]+tri[:,1]+tri[:,2])/3
    pts=np.concatenate([tri.reshape(-1,3),mid,
                        (tri[:,0]+tri[:,1])/2,
                        (tri[:,1]+tri[:,2])/2,
                        (tri[:,2]+tri[:,0])/2],axis=0)
    # Finger solid elsewhere should not disappear from collision: include
    # every sampled fixed-jaw surface, not only the grasp-candidate region.
    rows=[]
    for mask,pitch in (
        ((pts[:,2]>=8.)&(pts[:,2]<20.),pitch_fine),
        ((pts[:,2]<8.)|(pts[:,2]>=20.),pitch_coarse)):
        selected=pts[mask]
        if len(selected)==0:
            continue
        # Align grid to TCP coordinate origin, with no per-angle dependence.
        voxels=np.unique(np.floor(selected/pitch).astype(np.int64),axis=0)
        # Merge adjacent x voxels, preserving each y,z slice. This avoids
        # dozens of contacts along continuous sections of a finger.
        last=None
        zone="fine" if pitch==pitch_fine else "coarse"
        for ix,iy,iz in sorted(map(tuple,voxels),key=lambda q:(q[1],q[2],q[0])):
            if last and last["pitch"]==pitch and last["iy"]==iy and \
               last["iz"]==iz and ix==last["ixmax"]+1:
                last["ixmax"]=ix
            else:
                last={"pitch":float(pitch),"zone":zone,
                      "ixmin":ix,"ixmax":ix,"iy":iy,"iz":iz}
                rows.append(last)
    if not rows or len(rows)>max_boxes:
        raise RuntimeError(
            f"Proxy box count {len(rows)} outside [1,{max_boxes}]; "
            "change pitch or --max-boxes explicitly, never silently decimate")
    boxes=[]
    for r in rows:
        p=r["pitch"]
        center=np.array([(r["ixmin"]+r["ixmax"]+1)*.5*p,
                         (r["iy"]+.5)*p,(r["iz"]+.5)*p])
        half=np.array([(.5*(r["ixmax"]-r["ixmin"]+1))*p,
                       p*.5,p*.5])
        # The coarse cells must not bridge across the finely modeled
        # 8--20mm pocket. Cut boxes at the boundaries before applying
        # the slight numerical overlap.
        bottom,top=center[2]-half[2],center[2]+half[2]
        if r["zone"]=="coarse":
            if center[2]<8.:
                top=min(top,8.)
            elif center[2]>=20.:
                bottom=max(bottom,20.)
        else:
            bottom=max(bottom,8.)
            top=min(top,20.)
        if top<=bottom:
            continue
        center[2]=(bottom+top)*.5
        half[2]=(top-bottom)*.5
        # Only 0.5 percent overlap to avoid gaps along box edges.
        half*=1.005
        boxes.append({"center_tcp_mm":center,
                      "half_mm":half,"pitch_mm":p})
    return boxes


def box_interval(start,stop,center,half):
    """Liang-Barsky style analytic segment AABB intersection in t units."""
    direction=stop-start
    low,high=0.,1.
    for axis in range(3):
        lo=center[axis]-half[axis]
        hi=center[axis]+half[axis]
        if abs(direction[axis])<1e-11:
            if start[axis]<lo or start[axis]>hi:
                return None
        else:
            t1=(lo-start[axis])/direction[axis]
            t2=(hi-start[axis])/direction[axis]
            low=max(low,min(t1,t2))
            high=min(high,max(t1,t2))
            if high<low:
                return None
    return (float(max(low,0.)),float(min(high,1.)))


def interval_union(items):
    items=sorted((max(0.,a),min(1.,b)) for a,b in items if a is not None and b>=a)
    result=[]
    for a,b in items:
        if result and a<=result[-1][1]+1e-8:
            result[-1]=(result[-1][0],max(result[-1][1],b))
        else:
            result.append((a,b))
    return result


def overlap_length(intervals_a,intervals_b,L):
    return float(L*sum(max(0.,min(b,d)-max(a,c))
        for a,b in intervals_a for c,d in intervals_b))


def diagram_svg(rows,out):
    """Japanese SVG: all axes and units visible within figure itself."""
    width,height=1210,655
    x0,x1=290,1100
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="32" y="40" font-size="25" font-weight="bold" fill="#223a50">固定爪の衝突形状：元の凸包と局所細分化モデルの比較</text>',
           '<text x="32" y="73" font-size="18" fill="#405e72">横軸：固定爪の把持面候補中心から可動爪候補中心への距離［mm］</text>',
           '<text x="32" y="101" font-size="16" fill="#526e81">縦方向の3行は「形状の種類」であり、位置や力を示す数値軸ではありません。J6＝−10°（閉）</text>']
    for idx,r in enumerate(rows):
        yy=148+idx*235
        L=r["segment_length_mm"]
        X=lambda t:x0+(x1-x0)*t
        parts.append(f'<text x="32" y="{yy}" font-size="20" font-weight="bold" fill="#2c4960">爪先から{r["depth_low_mm"]:g}〜{r["depth_high_mm"]:g}mm</text>')
        parts.append(f'<text x="32" y="{yy+26}" font-size="16" fill="#536c7a">候補面中心間 {L:.2f}mm</text>')
        specs=[
         ("固定爪：元の凸包",r["original_fixed_intervals"],"#2b6eaa"),
         ("固定爪：分割モデル",r["proxy_fixed_intervals"],"#28a184"),
         ("可動爪：元の凸包",r["moving_hull_intervals"],"#dd8442")]
        for j,(title,intervals,col) in enumerate(specs):
            y=yy+39+j*38
            parts.append(f'<text x="{x0-14}" y="{y+18}" font-size="16" text-anchor="end" fill="#364f61">{title}</text>')
            parts.append(f'<rect x="{x0}" y="{y}" width="{x1-x0}" height="23" rx="4" fill="#f0f3f6"/>')
            for a,b in intervals:
                parts.append(f'<rect x="{X(a):.2f}" y="{y}" width="{max(1.,(b-a)*(x1-x0)):.2f}" height="23" rx="3" fill="{col}"/>')
        parts.append(f'<text x="{x0}" y="{yy+171}" font-size="16" fill="#3c586b">元凸包との重なり {r["original_overlap_mm"]:.2f}mm → 分割モデルでは {r["proxy_overlap_mm"]:.2f}mm</text>')
        for k in range(5):
            parts.append(f'<text x="{X(k/4):.1f}" y="{yy+150}" text-anchor="middle" font-size="15" fill="#526a7a">{(L*k/4):.1f}</text>')
    parts.append('<text x="32" y="626" font-size="16" fill="#526a7e">色帯＝各形状の衝突判定上の占有区間。元STLは表示形状として残し、衝突だけ分割BOXで近似。</text>')
    parts.append('<text x="32" y="650" font-size="14" fill="#526a7e">注意：分割BOXは近似モデル。重なりが減っても正しい開口幅・把持成功・接触力はまだ検証していません。</text></svg>')
    out.write_text("\n".join(parts),encoding="utf-8")



def plot_proxy_3d(original,ref,fixed,moving,boxes,tcp,R,out):
    """Two-panel 3D schematic from actual STL and experimental box proxies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    names={f.name for f in font_manager.fontManager.ttflist}
    for name in ("Hiragino Sans","Yu Gothic","Noto Sans CJK JP",
                 "IPAexGothic","Meiryo"):
        if name in names:
            plt.rcParams["font.family"]=name
            break
    fixed_v,fixed_faces=overlay.mesh_in_gripper(original,ref,fixed,tcp,R)
    moving_v,moving_faces=overlay.mesh_in_gripper(original,ref,moving,tcp,R)
    # Use the exact same 3D scale and camera for before vs after.
    selected=[v[(v[:,2]>=0)&(v[:,2]<38)] for v in (fixed_v,moving_v)]
    pool=np.concatenate([v for v in selected if len(v)],axis=0)
    mn=pool.min(axis=0)-5
    mx=pool.max(axis=0)+5
    mn[2],mx[2]=0.,38.
    fig=plt.figure(figsize=(15,7.6),facecolor="white")
    edges=overlay.hull_edges_clipped(ConvexHull(fixed_v),fixed_v,0.,38.)
    for col in range(2):
        ax=fig.add_subplot(1,2,col+1,projection="3d")
        for verts,faces,color in ((fixed_v,fixed_faces,"#448dc8"),
                                  (moving_v,moving_faces,"#e3a16d")):
            tri=verts[faces]
            z=tri[:,:,2].mean(axis=1)
            tri=tri[(z>=0)&(z<=38)]
            if len(tri)>1300:
                tri=tri[np.linspace(0,len(tri)-1,1300,dtype=int)]
            ax.add_collection3d(Poly3DCollection(
                tri,facecolor=color,alpha=.2,linewidths=0))
        if col==0:
            ax.add_collection3d(Line3DCollection(
                edges,colors="#205d92",linewidths=.9,alpha=.86))
        else:
            # Only the 8-20mm high-resolution part is highlighted;
            # all other boxes are still installed for collision.
            import itertools
            lines=[]
            for b in boxes:
                if b["pitch_mm"]!=min(x["pitch_mm"] for x in boxes):
                    continue
                c,h=b["center_tcp_mm"],b["half_mm"]
                if not (7.<=c[2]<=21.):
                    continue
                corners=[c+np.array([sx,sy,sz])*h
                     for sx,sy,sz in itertools.product((-1,1),repeat=3)]
                for i in range(8):
                    for j in range(i+1,8):
                        if sum(abs(corners[i]-corners[j])>1e-8)==1:
                            lines.append([corners[i],corners[j]])
                if len(lines)>3500:
                    break
            if lines:
                ax.add_collection3d(Line3DCollection(
                    lines,colors="#158768",linewidths=.6,alpha=.65))
        ax.set_xlim(mn[0],mx[0])
        ax.set_ylim(mn[1],mx[1])
        ax.set_zlim(mn[2],mx[2])
        ax.set_box_aspect(np.maximum(mx-mn,1.))
        ax.view_init(elev=24,azim=-65)
        ax.set_xlabel("X位置［mm］")
        ax.set_ylabel("Y位置［mm］")
        ax.set_zlabel("TCPから根元方向の深さ［mm］")
        ax.set_title("変更前：全メッシュ凸包" if col==0 else
                     "変更後：8〜20mm局所細分化BOX",pad=15)
    fig.suptitle("固定爪のSTL表示は同じ／衝突形状だけを比較（J6＝−10°）",
                 fontsize=17,y=.99)
    fig.text(.5,.03,
        "青の薄い面＝固定爪STL／橙の薄い面＝可動爪STL／"
        "濃青の線＝元の凸包／緑の線＝8〜20mmのBOX。実物の把持成功は未検証。",
        ha="center",fontsize=11)
    fig.subplots_adjust(top=.89,bottom=.12,wspace=.05)
    fig.savefig(out,dpi=135)
    plt.close(fig)

def get_pose(model,home,angle):
    data=mujoco.MjData(model)
    for name,value in zip(r2s.JOINTS,home):
        jid=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name))
        adr=int(model.jnt_qposadr[jid])
        data.qpos[adr]=math.radians(value)
    jid=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"gripper"))
    data.qpos[int(model.jnt_qposadr[jid])]=math.radians(angle)
    mujoco.mj_forward(model,data)
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--home-deg",nargs=5,type=float,
                    default=[-30.,-30.,45.,-20.,-90.])
    ap.add_argument("--fine-mm",type=float,default=3.)
    ap.add_argument("--coarse-mm",type=float,default=6.)
    ap.add_argument("--max-boxes",type=int,default=2400)
    args=ap.parse_args()
    if args.out.exists(): ap.error("--out already exists")
    if not (1.5<=args.fine_mm<=5. and 3.<=args.coarse_mm<=10.):
        ap.error("Invalid fine/coarse pitch in mm")
    if not 1<=args.max_boxes<=3000:
        ap.error("max-boxes must be between 1 and 3000")
    args.out.mkdir(parents=True)
    summary={"status":"INITIALIZING",
             "experimental_only":True,
             "source_urdf_modified":False,
             "source_stl_modified":False,
             "fixed_jaw_original_visual_preserved":True,
             "physical_grasp_proven":False,
             "contact_geometry_validated":False,
             "candidate_depth_mm":[8,20],
             "fine_pitch_mm":args.fine_mm,
             "coarse_pitch_mm":args.coarse_mm}
    try:
        # Unmodified model defines triangle assets and the original collision shape.
        source=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
        source.compiler.discardvisual=False
        grip=source.body("gripper_link")
        if grip is None: raise RuntimeError("No gripper_link in URDF")
        grip.add_site(name=SITE,pos=base.TCP_LOCAL,size=[.002]*3)
        original=source.compile()
        fixed,fname=base.owned_mesh_geom(original,"gripper_link",base.FINGER_MESH)
        moving,mname=base.owned_mesh_geom(
            original,"moving_jaw_so101_v1_link",base.MOVING_MESH)
        ref=get_pose(original,args.home_deg,-10.)
        tcpid=int(mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_SITE,SITE))
        gripid=int(mujoco.mj_name2id(original,mujoco.mjtObj.mjOBJ_BODY,"gripper_link"))
        tcp=np.array(ref.site_xpos[tcpid])
        R=np.array(ref.xmat[gripid]).reshape(3,3)
        vertices,meshfaces=overlay.mesh_in_gripper(original,ref,fixed,tcp,R)
        boxes=triangles_to_surface_boxes(vertices,meshfaces,args.fine_mm,
                                        args.coarse_mm,args.max_boxes)
        # Disable the original whole-mesh collision (visual mesh stays).
        # Never disable the moving jaw or unrelated gripper geoms.
        matches=[g for g in source.geoms
                 if str(getattr(g,"meshname","") or "").startswith(
                     base.FINGER_MESH)]
        if len(matches)!=1:
            raise RuntimeError(
                f"Expected exactly one fixed finger mesh spec geom, got {len(matches)}")
        specgeom=matches[0]
        original_contype=int(original.geom_contype[fixed])
        original_conaffinity=int(original.geom_conaffinity[fixed])
        if not original_contype or not original_conaffinity:
            raise RuntimeError("Original fixed finger mesh has no collision bits")
        specgeom.contype=0
        specgeom.conaffinity=0
        # Use direct MjSpec additions; the local origin for these positions
        # is gripper_link, while box centers were measured from TCP.
        for i,box in enumerate(boxes):
            pos=np.asarray(base.TCP_LOCAL)+box["center_tcp_mm"]/1000.
            half=box["half_mm"]/1000.
            geom=grip.add_geom()
            geom.name=f"{PROXY_PREFIX}{i:04d}"
            geom.type=mujoco.mjtGeom.mjGEOM_BOX
            geom.pos=pos.tolist()
            geom.size=half.tolist()
            geom.contype=original_contype
            geom.conaffinity=original_conaffinity
            geom.density=0
            geom.rgba=[.12,.83,.60,.22] if box["pitch_mm"]==args.fine_mm \
                       else [.22,.62,.83,.12]
            geom.group=4
        modified=source.compile()
        assert modified.ngeom>original.ngeom
        fixed_new, _=base.owned_mesh_geom(
            modified,"gripper_link",base.FINGER_MESH)
        if int(modified.geom_contype[fixed_new])!=0 or \
           int(modified.geom_conaffinity[fixed_new])!=0:
            raise RuntimeError("Original fixed mesh collision was not disabled")
        proxy_ids=[i for i in range(modified.ngeom)
                   if (mujoco.mj_id2name(modified,mujoco.mjtObj.mjOBJ_GEOM,i)
                       or "").startswith(PROXY_PREFIX)]
        if len(proxy_ids)!=len(boxes):
            raise RuntimeError("Compiled proxy geom count mismatch")
        # Validate source STL remains attached to gripper_link for visuals.
        mref=get_pose(modified,args.home_deg,-10.)
        moving_new,_=base.owned_mesh_geom(
            modified,"moving_jaw_so101_v1_link",base.MOVING_MESH)
        rows=[]
        fworld,fn=faces.mesh_triangles_world(original,ref,fixed)
        mworld,mn=faces.mesh_triangles_world(original,ref,moving)
        fdepth=(fworld-tcp)@R[:,2]*1000.
        mdepth=(mworld-tcp)@R[:,2]*1000.
        fixedhull=ConvexHull(vertices)
        mwv,_=overlay.mesh_in_gripper(original,ref,moving,tcp,R)
        movinghull=ConvexHull(mwv)
        for low,high in BANDS:
            f,ff,_=faces.choose_subset(fworld,fn,fdepth,low,high)
            m,mm,_=faces.choose_subset(mworld,mn,mdepth,low,high)
            pair=faces.face_pair(f,ff,m,mm)
            if not pair["pair_count"]:
                raise RuntimeError(f"No facing surface pair at {low:g}-{high:g}mm")
            p0=(np.asarray(pair["fixed_face_center_world_m"])-tcp)@R*1000.
            p1=(np.asarray(pair["moving_face_center_world_m"])-tcp)@R*1000.
            L=float(np.linalg.norm(p1-p0))
            oh=hull_segment_interval(fixedhull,p0,p1)
            mh=hull_segment_interval(movinghull,p0,p1)
            orig=[oh] if oh is not None else []
            moving_intervals=[mh] if mh is not None else []
            intervals=interval_union([iv for b in boxes
                       if (iv:=box_interval(p0,p1,b["center_tcp_mm"],b["half_mm"]))
                       is not None])
            rows.append({"depth_low_mm":low,"depth_high_mm":high,
                  "j6_angle_deg":-10.,"segment_length_mm":L,
                  "original_fixed_intervals":orig,
                  "proxy_fixed_intervals":intervals,
                  "moving_hull_intervals":moving_intervals,
                  "original_overlap_mm":overlap_length(orig,moving_intervals,L),
                  "proxy_overlap_mm":overlap_length(intervals,moving_intervals,L),
                  "original_fixed_covered_mm":overlap_length(orig,[(0,1)],L),
                  "proxy_fixed_covered_mm":overlap_length(intervals,[(0,1)],L),
                  "fixed_center_tcp_mm":p0.tolist(),
                  "moving_center_tcp_mm":p1.tolist()})
        with (args.out/"collision_proxy_compare.csv").open(
                "w",newline="",encoding="utf-8") as fh:
            writer=csv.DictWriter(fh,fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v) if isinstance(v,list) else v
                                for k,v in row.items()})
        diagram_svg(rows,args.out/"collision_proxy_before_after.svg")
        plot_proxy_3d(original,ref,fixed,moving,boxes,tcp,R,
                      args.out/"fixed_jaw_proxy_3d_before_after.png")
        # Signed distance across all proxy boxes to the moving mesh.
        # A minimum gap is a sanity signal, not proof of true aperture.
        witness=np.zeros(6)
        sourcegap=float(mujoco.mj_geomDistance(
            original,ref,fixed,moving,.3,witness)*1000.)
        distances=[float(mujoco.mj_geomDistance(
            modified,mref,i,moving_new,.3,witness)*1000.) for i in proxy_ids]
        summary.update({
            "status":"COMPARISON_RECORDED",
            "proxy_boxes":len(proxy_ids),
            "visualization":"fixed_jaw_proxy_3d_before_after.png",
            "proxy_fine_boxes":sum(b["pitch_mm"]==args.fine_mm for b in boxes),
            "original_signed_mesh_gap_mm":sourcegap,
            "proxy_min_signed_gap_mm":min(distances),
            "rows":rows,
            "warning":
                "Surface-box collision model is experimental, not a production "
                "physics mesh. Need object grasp/contact tests and pose sweeps.",
            "source_urdf_modified":False,
            "original_finger_mesh_collision_disabled_in_modified_model":True,
            "original_fixed_mesh_visual_preserved":True,
        })
        # Preserve a concrete, inspectable record of generated proxies.
        with (args.out/"proxy_boxes_gripper_local.csv").open(
                "w",newline="",encoding="utf-8") as fh:
            fields=["index","pitch_mm","center_x_mm","center_y_mm","center_z_mm",
                    "half_x_mm","half_y_mm","half_z_mm"]
            writer=csv.DictWriter(fh,fieldnames=fields)
            writer.writeheader()
            for i,b in enumerate(boxes):
                center=b["center_tcp_mm"]+np.asarray(base.TCP_LOCAL)*1000.
                half=b["half_mm"]
                writer.writerow({"index":i,"pitch_mm":b["pitch_mm"],
                    **{f"center_{j}_mm":center[k] for k,j in enumerate("xyz")},
                    **{f"half_{j}_mm":half[k] for k,j in enumerate("xyz")}})
        # Save the modified model as MJCF for reproducible offline inspection.
        # Visual STL assets remain external references; no source files changed.
        try:
            (args.out/"collision_proxy_in_memory.xml").write_text(
                source.to_xml(),encoding="utf-8")
        except Exception as exc:
            summary["mjcf_export_note"]=f"Not exported: {type(exc).__name__}: {exc}"
    except Exception as exc:
        import traceback
        summary["status"]="DIAGNOSTIC_FAILED"
        summary["error"]=f"{type(exc).__name__}: {exc}"
        summary["traceback"]=traceback.format_exc(limit=6)
    finally:
        (args.out/"summary.json").write_text(
            json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
