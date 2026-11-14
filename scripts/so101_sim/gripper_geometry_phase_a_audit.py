#!/usr/bin/env python3
"""SO-101 Phase A: URDF vs compiled MuJoCo gripper geometry audit.

No Muhi, no grasp search, no servo dynamics, no hardware. Sweep J6 while
J1..J5 are held at a user-selected view pose. Report mesh identity,
relative positions, closest-pair collision witnesses, and 5-view video.
This audit cannot by itself establish usable inner finger contact patches.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

import real_to_sim as r2s
import wrist_camera_sim as wrist

TCP_LOCAL = (-.0079, -.000218121, -.0981274)
SITE = "phase_a_gripper_tcp"
FINGER_MESH = "wrist_roll_follower_so101_v1"
MOVING_MESH = "moving_jaw_so101_v1"
JOINT = "gripper"


def joint_addr(model, name):
    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if jid < 0:
        raise RuntimeError(f"Missing joint: {name}")
    return (int(model.jnt_qposadr[jid]),
            tuple(float(x) for x in model.jnt_range[jid]))


def owned_mesh_geom(model, body_name, mesh_stem):
    bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if bid < 0:
        raise RuntimeError(f"Missing body: {body_name}")
    found=[]
    for i in range(model.ngeom):
        if (int(model.geom_bodyid[i]) != bid or
            int(model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_MESH)):
            continue
        mid = int(model.geom_dataid[i])
        meshname = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_MESH,mid) or ""
        if meshname.startswith(mesh_stem):
            found.append((i,meshname))
    if len(found) != 1:
        raise RuntimeError(
            f"Expected exactly one {body_name}/{mesh_stem} geom; got {found}")
    return found[0]


def mesh_points_world(model,data,geom_id):
    """Sample compiled mesh vertices, NOT certified collision-surface points.

    MuJoCo's compiled mesh reference frame is placed at the geom frame;
    use runtime geom pose to bring samples to world coordinates.
    """
    mesh_id=int(model.geom_dataid[geom_id])
    start=int(model.mesh_vertadr[mesh_id])
    count=int(model.mesh_vertnum[mesh_id])
    local=np.asarray(model.mesh_vert[start:start+count])
    R=np.asarray(data.geom_xmat[geom_id]).reshape(3,3)
    return local@R.T+np.asarray(data.geom_xpos[geom_id])


def sample_distance_mm(a,b):
    """Nearest VERTEX-SAMPLE distance; nonnegative and not contact gap."""
    if not len(a) or not len(b):
        return None
    try:
        from scipy.spatial import cKDTree
        distances,_=cKDTree(b).query(a,k=1)
        return float(np.min(distances)*1000)
    except ImportError:
        # Deterministic subset, fixed time/memory even without scipy.
        a=a[np.linspace(0,len(a)-1,min(700,len(a)),dtype=int)]
        b=b[np.linspace(0,len(b)-1,min(700,len(b)),dtype=int)]
        minimum=float("inf")
        for chunk in np.array_split(a,min(10,len(a))):
            if not len(chunk):
                continue
            delta=chunk[:,None,:]-b[None,:,:]
            minimum=min(minimum,float(np.min(np.sum(delta*delta,axis=2))))
        return math.sqrt(minimum)*1000


def write_csv(path,rows):
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--home-deg",type=float,nargs=5,
                    default=[-30.,-30.,45.,-20.,-90.],
                    help="J1..J5 held static; no object or collision-based reaching")
    ap.add_argument("--fps",type=int,default=15)
    ap.add_argument("--video",action="store_true",
                    help="Create 5-view 1440x780 combined MP4 plus individual MP4s")
    args=ap.parse_args()
    if args.out.exists(): ap.error("Output path exists")
    if not 1<=args.fps<=60: ap.error("fps outside 1..60")
    if not all(math.isfinite(v) for v in args.home_deg):
        ap.error("Nonfinite home angles")

    args.out.mkdir(parents=True,exist_ok=False)
    summary={"purpose":"Phase A URDF / MuJoCo gripper geometry",
             "simulation_only":True,
             "source_urdf_modified":False,
             "object_present":False,
             "physical_grasp_proven":False,
             "internal_contact_face_proven":False,
             "video_layout":"5 views: top side A/B; bottom wrist TCP/shelf/retracted, 1440x780",
             "status":"INITIALIZING"}
    rows=[]
    tip_rows=[]
    # Relative to the true fixed gripper_frame_link TCP origin, measured
    # along +Z of the parent gripper_link, i.e. toward the palm.
    # This is a VERTEX sample diagnostic, not a contact/collision proxy.
    depth_zones=((0.,8.),(8.,20.),(20.,30.),(30.,45.),(45.,70.))
    frame_dir=args.out/"_video_frames"
    names=("combined","side","side_b","wrist","shelf","retracted")
    output_names=("gripper_geometry_combined.mp4","gripper_geometry_side.mp4",
                  "gripper_geometry_side_b.mp4","gripper_geometry_wrist.mp4",
                  "gripper_geometry_wrist_shelf.mp4",
                  "gripper_geometry_wrist_retracted.mp4")
    renderer=None
    try:
        spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
        spec.compiler.discardvisual=False
        offset=tuple(TCP_LOCAL[i]-wrist.DEFAULT_MOUNT_POS[i] for i in range(3))
        wrist.install(spec,bracket_profile="ruler_15_35_35",
                      camera_aim_at=offset,camera_retract_mm=5.)
        grip=spec.body("gripper_link")
        if grip is None: raise RuntimeError("No gripper_link in URDF")
        grip.add_site(name=SITE,pos=TCP_LOCAL,size=[.002]*3,
                      rgba=[.15,.9,.6,1])
        model=spec.compile()
        data=mujoco.MjData(model)
        model.vis.map.znear=min(float(model.vis.map.znear),.001)
        qids=[]
        for name,deg in zip(r2s.JOINTS,args.home_deg):
            q,limits=joint_addr(model,name)
            rad=math.radians(deg)
            if not limits[0]<=rad<=limits[1]:
                raise ValueError(f"{name}: {deg} degrees outside URDF limits")
            qids.append(q)
        j6,limits=joint_addr(model,JOINT)
        fixed,fmesh=owned_mesh_geom(
            model,"gripper_link",FINGER_MESH)
        moving,mmesh=owned_mesh_geom(
            model,"moving_jaw_so101_v1_link",MOVING_MESH)
        tcpid=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_SITE,SITE))
        gripbid=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_BODY,"gripper_link"))
        jointid=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_JOINT,JOINT))
        # gripper_frame_link is connected to gripper_link by a FIXED
        # URDF joint; MuJoCo may fuse it away when compiling the model.
        # The pre-installed TCP site on gripper_link retains its position.
        framebid=int(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_BODY,"gripper_frame_link"))
        summary["gripper_frame_link_body_id"]=(
            framebid if framebid>=0 else None)
        summary["gripper_frame_link_merged"]=bool(framebid<0)
        summary["TCP_measurement_source"]="phase_a_gripper_tcp site on gripper_link"
        summary["tip_zone_depths_mm"]=[list(v) for v in depth_zones]
        summary["tip_zone_method"] = (
            "nearest pair of sampled STL mesh VERTICES within matched "
            "TCP-relative depth bands; nonnegative, NOT inner-face clearance "
            "or collision-solver gap; face normals/convex hull ignored")
        def mesh_geometry_report(gid):
            mesh_id=int(model.geom_dataid[gid])
            vert_start=int(model.mesh_vertadr[mesh_id])
            vert_count=int(model.mesh_vertnum[mesh_id])
            verts=np.asarray(model.mesh_vert[
                vert_start:vert_start+vert_count])
            bbox_min=verts.min(axis=0).tolist() if len(verts) else None
            bbox_max=verts.max(axis=0).tolist() if len(verts) else None
            return {
                "mesh_id":mesh_id,"compiled_vertices":vert_count,
                "compiled_mesh_vertex_bbox_min_m":bbox_min,
                "compiled_mesh_vertex_bbox_max_m":bbox_max,
                "collision_uses_convex_hull":True,
                "geom_contype":int(model.geom_contype[gid]),
                "geom_conaffinity":int(model.geom_conaffinity[gid]),
            }
        summary.update({
            "urdf":str(args.urdf),
            "J6_joint_axis_local":model.jnt_axis[jointid].tolist(),
            "J6_joint_position_parent_frame_m":model.jnt_pos[jointid].tolist(),
            "J6_limits_deg":list(np.degrees(limits)),
            "TCP_gripper_link_xyz_m":list(TCP_LOCAL),
            "TCP_frame_orientation_note":
                "gripper_frame_joint has RPY(0,pi,0); this site records position only",
            "fixed_finger":{"geom_id":fixed,"mesh_name":fmesh,
                            "body":"gripper_link",
                            **mesh_geometry_report(fixed)},
            "moving_finger":{"geom_id":moving,"mesh_name":mmesh,
                             "body":"moving_jaw_so101_v1_link",
                             **mesh_geometry_report(moving)},
            "wrist_mount":"ruler_15_35_35 photo/ruler estimate",
            "wrist_mount_has_collision_proxies":True,
            "original_video_contract_preserved":True,
        })
        model.opt.timestep=.002
        mujoco.mj_resetData(model,data)
        data.qpos[qids]=np.radians(args.home_deg)
        mujoco.mj_forward(model,data)
        if args.video:
            for name in names:(frame_dir/name).mkdir(parents=True,exist_ok=True)
            model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),480)
            model.vis.global_.offheight=max(int(model.vis.global_.offheight),360)
            renderer=mujoco.Renderer(model,width=480,height=360)
        option=mujoco.MjvOption()
        option.geomgroup[3]=0
        hull_option=mujoco.MjvOption()
        hull_option.geomgroup[3]=0
        hull_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]=1
        summary["hull_visualization"]="Separate 5-view PNG at J6 open/mid/closed; convex hull flag enabled"
        side=mujoco.MjvCamera()
        side.type=mujoco.mjtCamera.mjCAMERA_FREE
        side.distance=.36
        side.elevation=-19
        side.azimuth=70
        angles=np.linspace(limits[1],limits[0],23)
        for k,angle in enumerate(angles):
            data.qpos[qids]=np.radians(args.home_deg)
            data.qpos[j6]=float(angle)
            data.qvel[:]=0
            mujoco.mj_forward(model,data)
            tcp=np.asarray(data.site_xpos[tcpid]).copy()
            R=np.asarray(data.xmat[gripbid]).reshape(3,3)
            witness=np.zeros(6)
            gap=float(mujoco.mj_geomDistance(
                model,data,fixed,moving,.3,witness))
            pfix=witness[:3].copy()
            pmov=witness[3:].copy()
            d=(pfix-tcp)@R
            fw=mesh_points_world(model,data,fixed)
            mw=mesh_points_world(model,data,moving)
            fdepth=(fw-tcp)@R[:,2]*1000.
            mdepth=(mw-tcp)@R[:,2]*1000.
            for zlow,zhigh in depth_zones:
                fm=fw[(fdepth>=zlow)&(fdepth<zhigh)]
                mm=mw[(mdepth>=zlow)&(mdepth<zhigh)]
                tip_rows.append({
                    "index":k,"j6_angle_deg":math.degrees(angle),
                    "depth_from_tcp_min_mm":zlow,
                    "depth_from_tcp_max_mm":zhigh,
                    "fixed_sample_vertices":len(fm),
                    "moving_sample_vertices":len(mm),
                    "nearest_vertex_sample_distance_mm":
                        sample_distance_mm(fm,mm),
                    "method":"vertex-sample distance; not contact gap",
                })
            rows.append({
                "index":k,"j6_angle_deg":math.degrees(angle),
                "fixed_moving_signed_gap_mm":gap*1000,
                "fixed_witness_tcp_distance_mm":float(
                    np.linalg.norm(pfix-tcp)*1000),
                "fixed_witness_depth_toward_palm_mm":float(d[2]*1000),
                "fixed_witness_gripper_x_mm":float(d[0]*1000),
                "fixed_witness_gripper_y_mm":float(d[1]*1000),
                "fixed_witness_world_x_m":float(pfix[0]),
                "fixed_witness_world_y_m":float(pfix[1]),
                "fixed_witness_world_z_m":float(pfix[2]),
                "moving_witness_world_x_m":float(pmov[0]),
                "moving_witness_world_y_m":float(pmov[1]),
                "moving_witness_world_z_m":float(pmov[2]),
                "TCP_world_x_m":float(tcp[0]),
                "TCP_world_y_m":float(tcp[1]),
                "TCP_world_z_m":float(tcp[2]),
            })
            if renderer is None: continue
            from PIL import Image,ImageDraw
            side.lookat[:]=data.xpos[gripbid]
            side.azimuth=70
            renderer.update_scene(data,camera=side,scene_option=option)
            a=Image.fromarray(renderer.render().copy())
            side.azimuth=-70
            renderer.update_scene(data,camera=side,scene_option=option)
            b=Image.fromarray(renderer.render().copy())
            images=[a,b]
            for cam in (wrist.CAMERA,wrist.SHELF_DIAGNOSTIC_CAMERA,
                        wrist.RETRACTED_DIAGNOSTIC_CAMERA):
                renderer.update_scene(data,camera=cam,scene_option=option)
                images.append(Image.fromarray(renderer.render().copy()))
            canvas=Image.new("RGB",(1440,780),(22,26,32))
            for img,pos in zip(images,((240,30),(720,30),(0,420),
                                      (480,420),(960,420))):
                canvas.paste(img,pos)
            draw=ImageDraw.Draw(canvas)
            labels=[f"SIDE A | J6={math.degrees(angle):.1f} deg",
                    "SIDE B","WRIST | TCP","WRIST | SHELF",
                    "WRIST | RETRACTED 5mm"]
            for label,pos in zip(labels,((245,10),(725,10),(7,395),
                                         (487,395),(967,395))):
                draw.text(pos,label,fill="white")
            outputs=[canvas,*images]
            for name,img in zip(names,outputs):
                img.save(frame_dir/name/f"frame_{k:04d}.png")
            if k in (0,11,22):
                canvas.save(args.out/f"gripper_fiveview_{k:02d}.png")
                hull_canvas=Image.new("RGB",(1440,780),(22,26,32))
                side.azimuth=70
                renderer.update_scene(
                    data,camera=side,scene_option=hull_option)
                hull_a=Image.fromarray(renderer.render().copy())
                side.azimuth=-70
                renderer.update_scene(
                    data,camera=side,scene_option=hull_option)
                hull_b=Image.fromarray(renderer.render().copy())
                hull_images=[hull_a,hull_b]
                for cam in (wrist.CAMERA,wrist.SHELF_DIAGNOSTIC_CAMERA,
                            wrist.RETRACTED_DIAGNOSTIC_CAMERA):
                    renderer.update_scene(
                        data,camera=cam,scene_option=hull_option)
                    hull_images.append(
                        Image.fromarray(renderer.render().copy()))
                for img,pos in zip(hull_images,
                    ((240,30),(720,30),(0,420),(480,420),(960,420))):
                    hull_canvas.paste(img,pos)
                hd=ImageDraw.Draw(hull_canvas)
                hd.text((245,8),
                    f"CONVEX HULL VIEW | J6={math.degrees(angle):.1f} deg",
                    fill="white")
                hd.text((8,393),
                    "WRIST TCP / SHELF / RETRACTED: convex hull enabled",
                    fill="white")
                hull_canvas.save(
                    args.out/f"gripper_collision_hulls_{k:02d}.png")
        write_csv(args.out/"j6_collision_geometry_sweep.csv",rows)
        write_csv(args.out/"j6_tip_depth_sample_distances.csv",tip_rows)
        # Show why the global minimum (near the base) is not the
        # relevant opening width at 8..30 mm grasp depths.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(11,5.5))
            for low,high in depth_zones:
                series=[r for r in tip_rows if
                    r["depth_from_tcp_min_mm"]==low]
                yy=[r["nearest_vertex_sample_distance_mm"] for r in series]
                ax.plot([r["j6_angle_deg"] for r in series],yy,
                    marker=".",label=f"Depth {low:g}-{high:g} mm")
            ax.set_xlabel("J6 angle (degrees; open 100 / closed -10)")
            ax.set_ylabel("Nearest sampled mesh-vertex distance (mm)")
            ax.set_title("SO-101 grasp-depth vertex distance (NOT physical clearance)")
            ax.legend()
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(args.out/"j6_tip_depth_samples.png",dpi=160)
            plt.close(fig)
            summary["tip_depth_plot"]="j6_tip_depth_samples.png"
        except Exception as plot_error:
            summary["tip_depth_plot_error"]=repr(plot_error)
        summary["frames_or_angles_measured"]=len(rows)
        summary["j6_min_fixed_moving_gap_mm"]=min(
            r["fixed_moving_signed_gap_mm"] for r in rows)
        summary["j6_max_fixed_moving_gap_mm"]=max(
            r["fixed_moving_signed_gap_mm"] for r in rows)
        if args.video:
            exe=shutil.which("ffmpeg")
            if not exe: raise RuntimeError("ffmpeg not found; numeric CSV saved")
            summary["video_files"]=[]
            for name,filename in zip(names,output_names):
                cmd=[exe,"-hide_banner","-loglevel","error","-y",
                     "-framerate",str(args.fps),"-i",
                     str(frame_dir/name/"frame_%04d.png"),
                     "-c:v","libx264","-pix_fmt","yuv420p",
                     "-movflags","+faststart",str(args.out/filename)]
                subprocess.run(cmd,check=True,capture_output=True,text=True)
                summary["video_files"].append(filename)
            shutil.rmtree(frame_dir)
        summary["status"]="URDF_GEOMETRY_AUDIT_RECORDED"
        summary["next_required_checks"]=[
            "Check mesh visual/collision mismatch and convex hull behaviour",
            "Inspect inside grasp-face at intended depth, not just nearest gap",
            "Verify opposing normals and opening width with target object",
            "Check whether sampled vertices cover the inner grasp face; "
            "vertex distances are not signed contact gaps or opening widths",
            "Only then generate valid 6D Grasp Pose candidates"]
    except Exception as exc:
        summary["status"]="AUDIT_INCOMPLETE"
        summary["error"]=f"{type(exc).__name__}: {exc}"
    finally:
        if renderer is not None: renderer.close()
        (args.out/"summary.json").write_text(
            json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8")
        print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
