#!/usr/bin/env python3
"""Phase7O: gated, read-only real episode FK and optional MuJoCo physics replay.

Never drives hardware.  The SO-101 motor-deg -> URDF joint axes, bottle pose,
and normalized J6 -> URDF angle are NOT identified by the source dataset.
A physics run requires an explicitly approved mapping and bottle world pose.
"""
from __future__ import annotations
import argparse, csv, json, math, shutil, subprocess
from pathlib import Path
import numpy as np
import mujoco

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
def load_episode(path):
    out=[]
    with Path(path).open(newline="",encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            out.append(dict(t=float(row["timestamp"]),state=np.asarray(json.loads(row["observation.state"]),float),
                            action=np.asarray(json.loads(row["action"]),float),frame=int(row["frame_index"])))
    if not out: raise ValueError("Empty episode CSV")
    if any(v["state"].shape!=(6,) or v["action"].shape!=(6,) for v in out):
        raise ValueError("Expected 6 joint state/action values")
    return out

def joint_map(arr,mapping):
    signs=np.asarray(mapping["sign"],float)
    offsets=np.asarray(mapping["offset_deg"],float)
    if signs.shape!=(5,) or offsets.shape!=(5,) or not all(abs(x)==1 for x in signs):
        raise ValueError("mapping requires sign[5] in ±1 and offset_deg[5]")
    return np.deg2rad(arr[:5]*signs+offsets)

def clip_preflight(model,qadr,q):
    failures=[]
    for i,n in enumerate(JOINTS):
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
        if j<0: raise ValueError("Missing URDF joint "+n)
        lo,hi=model.jnt_range[j]
        if not(lo<=q[i]<=hi):
            failures.append(dict(joint=n,angle_deg=float(np.rad2deg(q[i])),
                                 urdf_limits_deg=np.rad2deg([lo,hi]).tolist()))
    return failures

def render_fk(urdf,episode,mapping,out,fps=8):
    # NO bottle geometry, NO IK, NO physics and NO J6 angle assumption.
    spec=mujoco.MjSpec.from_file(str(urdf.resolve()))
    model=spec.compile()
    d=mujoco.MjData(model)
    qadr=np.asarray([model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in JOINTS],int)
    snapshots=[(int(round(t/.05)),t) for t in (14.5,15.0,15.5,16.0,17.0,18.0)]
    rows=[]
    frames=[]
    renderer=None
    try:
        try:renderer=mujoco.Renderer(model,height=480,width=640)
        except Exception as e:(out/"render_error.txt").write_text(repr(e))
        camera=mujoco.MjvCamera()
        camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:]=[.12,0,.10];camera.distance=.57
        camera.azimuth=90;camera.elevation=-25
        for idx,t in snapshots:
            idx=min(range(len(episode)),key=lambda k:abs(episode[k]["t"]-t))
            record=episode[idx]
            q=joint_map(record["state"],mapping)
            errors=clip_preflight(model,qadr,q)
            row=dict(time_s=record["t"],frame=record["frame"],
                     joint_urdf_deg=np.rad2deg(q).tolist(),limit_errors=errors)
            rows.append(row)
            if errors or renderer is None:continue
            d.qpos[qadr]=q
            mujoco.mj_forward(model,d)
            renderer.update_scene(d,camera=camera)
            frames.append((row,renderer.render().copy()))
    finally:
        if renderer is not None:renderer.close()
    (out/"fk_preflight.json").write_text(json.dumps(dict(
        status="FK_ONLY_UNCALIBRATED_MAPPING",
        mapping=mapping,frames=rows,
        note="No evidence that rendered camera angle or joint zero is calibrated; no physics run."
    ),ensure_ascii=False,indent=2))
    if frames:
        from PIL import Image,ImageDraw
        tiles=[]
        for row,frame in frames:
            tile=Image.new("RGB",(640,510),"#263238")
            tile.paste(Image.fromarray(frame),(0,30))
            ImageDraw.Draw(tile).text((10,8),f"FK ONLY | recorded t={row['time_s']:.2f}s | frame={row['frame']}",fill="white")
            tiles.append(tile)
        contact=Image.new("RGB",(640*min(3,len(tiles)),510*math.ceil(len(tiles)/3)),"#263238")
        for i,tile in enumerate(tiles):contact.paste(tile,((i%3)*640,(i//3)*510))
        contact.save(out/"fk_contact_sheet.png")
    return dict(n_frames=len(rows),num_urdf_limit_failures=sum(bool(x["limit_errors"]) for x in rows))

def physics_replay(urdf,episode,mapping,bottle,out,dt=.002,fps=8):
    import phase6c_upright_reach_grasp as phase6c
    import physical_workcell_sim as physical
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from PIL import Image,ImageDraw
    # This is a strict, explicit experiment gate. There is no implicit default bottle pose.
    if mapping.get("visual_fit_verified") is not True or bottle.get("pose_verified") is not True:
        return dict(status="BLOCKED_UNVERIFIED_FK_OR_BOTTLE_POSE")
    if mapping.get("j6_normalization_verified") is not True:
        return dict(status="BLOCKED_UNVERIFIED_J6_MAPPING")
    if mapping.get("j6_closes_at_lower_norm") is not True:
        return dict(status="BLOCKED_J6_DIRECTION_NOT_CONFIRMED")
    xyz=np.asarray(bottle["xyz_m"],float)
    quat=np.asarray(bottle["quat_wxyz"],float)
    if xyz.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(np.r_[xyz,quat]).all():
        raise ValueError("Invalid bottle pose")
    quat=quat/np.linalg.norm(quat)
    table_z=float(bottle["table_z_m"])
    args=SimpleNamespace(urdf=urdf,out=out,jaw_collision_mode="axial_cross_convex_patches",
        jaw_collision_slabs=12,jaw_collision_cross_bands=3,support_plane_z_m=table_z,
        jaw_table_contact=True,dt=dt,arm_kp=18.,arm_kd=1.8,arm_torque=3.,
        j6_kp=2.,j6_kd=.08,j6_torque=.4)
    model,qadr,vadr,aids,lo,hi,fixed,moving,bodies,floor=phase6c.model_build(args)
    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    bq=int(model.jnt_qposadr[jid])
    d=mujoco.MjData(model)
    # Replay the observed closure (~15.0-16.0s) and subsequent lift (~17-18s).
    start,end=14.8,18.3
    records=[r for r in episode if start<=r["t"]<=end]
    if not records:raise ValueError("No records inside replay window")
    q0=joint_map(records[0]["state"],mapping)
    fails=clip_preflight(model,qadr,q0)
    if fails:return dict(status="BLOCKED_URDF_JOINT_LIMITS",joint_errors=fails)
    # Map J6 percentage to URDF qpos only after explicit independent validation.
    def gripper_angle(norm):
        # Episode015 visual review: normalized J6 decreases when jaws close.
        # Angle endpoints still require independent verification before physics.
        return lo[5]+(hi[5]-lo[5])*float(norm)/100.
    d.qpos[qadr[:5]]=q0;d.qpos[qadr[5]]=gripper_angle(records[0]["state"][5])
    d.qpos[bq:bq+3]=xyz;d.qpos[bq+3:bq+7]=quat
    mujoco.mj_forward(model,d)
    from physical_grasp_j6_bench import geometry_gaps
    fixed_gap=1000.*geometry_gaps(model,d,bodies,fixed)
    moving_gap=1000.*geometry_gaps(model,d,bodies,moving)
    metrics=dict(status="PHYSICS_NOT_RUN",initial_fixed_gap_mm=fixed_gap,
                 initial_moving_gap_mm=moving_gap)
    (out/"physical_preflight.json").write_text(json.dumps(metrics,indent=2))
    if min(fixed_gap,moving_gap)<-0.25:
        metrics["status"]="BLOCKED_INITIAL_INTERPENETRATION_GT_0_25MM"
        return metrics
    frames=[];rows=[]
    renderer=None
    try:
        try:renderer=mujoco.Renderer(model,height=420,width=640)
        except Exception as exc:(out/"render_error.txt").write_text(str(exc))
        camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:]=xyz;camera.distance=.57;camera.elevation=-23
        stride=max(1,int(round(1/(dt*fps))))
        steps=math.ceil((end-start)/dt)
        for k in range(steps):
            t=start+k*dt
            idx=min(len(records)-1,max(0,int(round((t-records[0]["t"])/.05))))
            r=records[idx]
            qtarget=joint_map(r["action"],mapping)
            failures=clip_preflight(model,qadr,qtarget)
            if failures:
                metrics["status"]="BLOCKED_TARGET_OUTSIDE_URDF_LIMITS"
                metrics["time_s"]=t;metrics["joint_errors"]=failures
                break
            d.ctrl[aids]=np.r_[qtarget,gripper_angle(r["action"][5])]
            mujoco.mj_step(model,d)
            if not np.isfinite(d.qpos).all():
                metrics["status"]="ABORT_NONFINITE_STATE";break
            rows.append(dict(time_s=d.time+start,
                fixed_jaw_centroid_z_m=float(np.mean(d.geom_xpos[list(fixed),2])),
                bottle_x_m=float(d.qpos[bq]),bottle_y_m=float(d.qpos[bq+1]),
                j2_target_deg=float(np.rad2deg(qtarget[1])),
                j2_actual_deg=float(np.rad2deg(d.qpos[qadr[1]])),
                j3_target_deg=float(np.rad2deg(qtarget[2])),
                j3_actual_deg=float(np.rad2deg(d.qpos[qadr[2]])),
                j6_norm_command=float(r["action"][5]),
                j6_actual_urdf_deg=float(np.rad2deg(d.qpos[qadr[5]]))))
            if renderer is not None and k%stride==0:
                panels=[]
                for az,el,name in ((90,-15,"SIDE"),(270,-15,"OPPOSITE"),(120,-80,"OVERHEAD")):
                    camera.azimuth=az;camera.elevation=el
                    renderer.update_scene(d,camera=camera)
                    tile=Image.new("RGB",(640,445),"#263238")
                    tile.paste(Image.fromarray(renderer.render().copy()),(0,25))
                    ImageDraw.Draw(tile).text((7,7),f"PHYSICS mj_step {name} | t={t:.2f}s",fill="white")
                    panels.append(tile)
                canvas=Image.new("RGB",(1920,445))
                for j,tile in enumerate(panels):canvas.paste(tile,(j*640,0))
                frames.append(canvas)
        else:metrics["status"]="PHYSICS_DIAGNOSTIC_COMPLETED_NOT_VALIDATED_GRASP"
    finally:
        if renderer is not None:renderer.close()
    if rows:
        with (out/"physical_joint_trace.csv").open("w",newline="") as fh:
            writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if frames and shutil.which("ffmpeg"):
        fd=out/"rendered_frames";fd.mkdir(exist_ok=True)
        for i,frame in enumerate(frames):frame.save(fd/f"{i:05d}.png")
        subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(fps),
            "-i",str(fd/"%05d.png"),"-c:v","libx264","-pix_fmt","yuv420p",str(out/"physics_replay.mp4")],check=True)
        shutil.rmtree(fd)
    metrics["steps"]=len(rows)
    return metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episode-csv",type=Path,required=True)
    p.add_argument("--urdf",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mapping",type=Path,required=True)
    p.add_argument("--bottle-pose",type=Path)
    p.add_argument("--run-physics",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    episode=load_episode(args.episode_csv)
    mapping=json.loads(args.mapping.read_text())
    summary=dict(reference_episode=15,source_csv=str(args.episode_csv),
                 mapping_verified=bool(mapping.get("visual_fit_verified",False)))
    summary["fk"]=render_fk(args.urdf,episode,mapping,args.out)
    if args.run_physics:
        if args.bottle_pose is None:
            summary["physics"]={"status":"BLOCKED_BOTTLE_POSE_NOT_PROVIDED"}
        else:
            bottle=json.loads(args.bottle_pose.read_text())
            summary["physics"]=physics_replay(args.urdf,episode,mapping,bottle,args.out)
    else:
        summary["physics"]={"status":"NOT_REQUESTED_FK_STAGE_ONLY"}
    (args.out/"phase7o_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
