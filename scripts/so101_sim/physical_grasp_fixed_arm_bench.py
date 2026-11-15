#!/usr/bin/env python3
"""Isolated MuJoCo SO-101: J1-J5 deleted, dynamic J6 / Muhi / table.

This is a deliberately DIFFERENT setup from the previous moving-pregrasp
equality-constraint trial. Intended to isolate solver contact from arm dynamics.
No physical robot control. Original URDF/STL and photographed wrist rig retained.
"""
import argparse,csv,json,math,shutil,subprocess
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
import physical_grasp_j6_bench as search
import physical_grasp_table_bench as prior
import physical_workcell_sim as physical
import real_to_sim as r2s
import wrist_camera_sim as wrist
import compare_id1_eye_to_hand as preview

TABLE="fixed_arm_bench_table"

def freeze_arm(spec,j5_deg):
    frozen={}
    for name in r2s.JOINTS:
        joint=spec.joint(name)
        if joint is None: raise ValueError(f"Missing arm joint {name}")
        if np.linalg.norm(np.asarray(joint.pos,float))>1e-7:
            raise ValueError(f"Cannot freeze {name}: joint pivot is not at body origin")
        angle=math.radians(j5_deg) if name=="wrist_roll" else 0.
        if angle:
            axis=np.asarray(joint.axis,float);axis/=np.linalg.norm(axis)
            local=np.r_[math.cos(angle/2),axis*math.sin(angle/2)]
            body=joint.parent
            if body is None:raise ValueError(f"No joint owner: {name}")
            q=np.zeros(4)
            mujoco.mju_mulQuat(q,np.asarray(body.quat),local)
            body.quat[:]=q
        frozen[name]=math.degrees(angle)
        spec.delete(joint) # Official MjSpec method: remove DOF, not a spring.
    return frozen

def contact_force(model,data,objects,fingers):
    total=0.;min_dist=0.
    for i,c in enumerate(data.contact[:data.ncon]):
        a,b=int(c.geom1),int(c.geom2)
        if not((a in objects and b in fingers) or (b in objects and a in fingers)):continue
        f=np.zeros(6);mujoco.mj_contactForce(model,data,i,f)
        total+=max(0.,float(f[0]));min_dist=min(min_dist,float(c.dist))
    return total,1000.*min_dist

def encode(directory,filename,fps):
    exe=shutil.which("ffmpeg")
    if not exe:return "ffmpeg_missing_pngs_saved"
    try:
        subprocess.run([exe,"-hide_banner","-loglevel","error","-y",
            "-framerate",str(fps),"-i",str(directory/"frame_%04d.png"),
            "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(filename)],check=True)
        return "created"
    except subprocess.CalledProcessError:return "ffmpeg_failed_pngs_saved"

def run(args):
    if args.out.exists():raise FileExistsError(args.out)
    if not(0<args.duration_s<=10 and args.render_every>=1 and args.fps>=1):
        raise ValueError("Invalid duration/render rate")
    spec=mujoco.MjSpec.from_file(str(args.urdf.resolve(strict=True)))
    # URDF defaults to fusestatic=True. Once J1-J5 are removed,
    # gripper_link becomes a static child and can be merged into its parent,
    # making named-body jaw lookup fail. Preserve body names/geometries
    # in the isolated benchmark, including temporary proxy compilations.
    spec.compiler.fusestatic=False
    spec.compiler.discardvisual=False
    physical.install_dynamic_muhi(spec)
    target=tuple(prior.GRIPPER_FRAME_LOCAL_POS[i]-wrist.DEFAULT_MOUNT_POS[i] for i in range(3))
    wrist.install(spec,bracket_profile="ruler_15_35_35",camera_aim_at=target,camera_retract_mm=5.)
    # The existing fixed-jaw proxy generator compiles the spec temporarily
    # and locates prior.TIP_SITE. Install it BEFORE generating proxies.
    gripper_body=spec.body("gripper_link")
    if gripper_body is None:
        raise ValueError("Missing gripper_link: cannot define TCP site")
    gripper_body.add_site(name=prior.TIP_SITE,
        pos=list(prior.GRIPPER_FRAME_LOCAL_POS),
        size=[.002,.002,.002],rgba=[.1,.9,.8,1.])
    table_body=spec.worldbody.add_body(name="fixed_arm_table_mocap",mocap=True,pos=[0,0,-10])
    table_body.add_geom(name=TABLE,type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[.08,.08,prior.TABLE_HALF_THICKNESS_M],rgba=[.47,.51,.56,1.],
        friction=[.9,.01,.001],contype=1,conaffinity=1)
    fixed_meta=prior.install_fixed_surface_proxy_bench(spec,fine_mm=2.,coarse_mm=4.,max_boxes=3500)
    moving_meta=prior.install_moving_surface_proxy_bench(spec,pitch_mm=2.,max_boxes=3500)
    servo=spec.add_actuator(name="fixed_arm_j6_position",target="gripper")
    servo.trntype=mujoco.mjtTrn.mjTRN_JOINT
    servo.dyntype=mujoco.mjtDyn.mjDYN_NONE
    servo.gaintype=mujoco.mjtGain.mjGAIN_FIXED
    servo.biastype=mujoco.mjtBias.mjBIAS_AFFINE
    servo.gainprm[0]=1.5
    servo.biasprm[1]=-1.5;servo.biasprm[2]=-.12
    servo.ctrllimited=False;servo.forcelimited=True
    servo.forcerange[:]=[-.08,.08]
    frozen=freeze_arm(spec,args.j5_deg)
    # A mocap base gives the *arm* a prescribed vertical lift trajectory.
    # J1-J5 remain absent; J6 and the Muhi freejoint remain physically
    # simulated. This is a contact-retention test, NOT servo-driven IK.
    base_spec=spec.body("base_link")
    if base_spec is None:
        raise ValueError("Missing SO-101 base_link in URDF")
    base_spec.mocap=True
    model=spec.compile()
    count=sum(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)>=0 for n in r2s.JOINTS)
    if count:raise AssertionError(f"{count} arm joints survived freeze")
    base_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"base_link")
    if base_id<0:raise RuntimeError("Compiled mocap base body is missing")
    base_mocap_id=int(model.body_mocapid[base_id])
    if base_mocap_id<0:
        raise RuntimeError("base_link must remain a mocap body for physical Lift")
    # Assert the source link names remain addressable after compilation.
    # Without fusestatic=False the URDF compiler may merge gripper_link.
    for name in ("gripper_link", "moving_jaw_so101_v1_link"):
        if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)<0:
            raise RuntimeError(f"Compiled body missing despite fusestatic=False: {name}")
    preview.apply_white_follower_preview(model)
    model.opt.timestep=.002
    model.vis.headlight.ambient[:]=[.7,.7,.7]
    model.vis.headlight.diffuse[:]=[.9,.9,.9]
    jawq,jawd,limits=search.joint_addr(model,"gripper")
    model.dof_armature[jawd]=max(float(model.dof_armature[jawd]),.002)
    model.dof_damping[jawd]=max(float(model.dof_damping[jawd]),.06)
    data=mujoco.MjData(model);mujoco.mj_resetData(model,data)
    initial_base_mocap=np.asarray(data.mocap_pos[base_mocap_id],dtype=float).copy()
    opened=float(limits[1])-math.radians(1)
    closed=float(limits[0])
    servo_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,"fixed_arm_j6_position")
    data.qpos[jawq]=opened;data.ctrl[servo_id]=opened
    mujoco.mj_forward(model,data)
    fixed=search.owned_collision_geoms(model,search.FINGER_BODY)
    moving=search.owned_collision_geoms(model,search.MOVING_BODY)
    bottle=tuple(physical.geom_id(model,n) for n in physical.DYNAMIC_GEOMS)
    table=physical.geom_id(model,TABLE)
    freejoint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,physical.DYNAMIC_JOINT)
    freeq=int(model.jnt_qposadr[freejoint]);freed=int(model.jnt_dofadr[freejoint])
    tbid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"fixed_arm_table_mocap")
    mocapid=int(model.body_mocapid[tbid])
    if mocapid<0:raise RuntimeError("Table mocap id missing")
    pose,search_diag=search.select_bench_pose(
        model,data,freeq,opened,closed,jawq,fixed,moving,bottle,
        # Deliberately leave 1-2 mm at the fixed finger, while J6's
        # swept collision path must contact the bottle first.
        min_fixed_gap_m=.001,max_fixed_gap_m=.002,prefer_small_moving_gap=True,
        candidate_local_z_range_m=(prior.GRIPPER_FRAME_LOCAL_POS[2]+.008,
                                   prior.GRIPPER_FRAME_LOCAL_POS[2]+.020))
    args.out.mkdir(parents=True,exist_ok=False)
    summary={"status":"SETUP_NOT_FOUND","mode":"TRUE_FIXED_J1_J5_DYNAMIC_J6",
        "arm_joint_count":count,"frozen_angles_deg":frozen,
        "same_pregrasp_pose_as_previous_trial":False,
        "lift_method":"kinematic vertical motion of mocap base; Muhi freejoint never overwritten",
        "lift_goal_mm":float(args.lift_mm),
        "lift_ramp_s":float(args.lift_ramp_s),
        "lift_min_consecutive_bilateral_steps":int(args.bilateral_hold_steps),
        "is_dynamic_j1_j5_joint_lift":False,
        "renderer_resolution_fix":"native wrist 480x390 and side 720x390; no image stretch",
        "target_initial_fixed_jaw_gap_mm":[1.,2.],
        "contact_sequence_intent":"J6 moving jaw touches and translates Muhi into stationary fixed jaw; no J1 collision approach",
        "source_urdf_stl_unmodified":True,
        "wrist_rig":"ruler_15_35_35",
        "collision_proxy_metadata":{"fixed":fixed_meta,"moving":moving_meta},
        "setup_search":search_diag,
        "warnings":["Separate fixed pose, not exact reproduction of previous 2.154s contact",
                    "Base mocap lift is prescribed; does not validate J1-J5 actuators or IK",
                    "Muhi translation and rotation are exclusively MuJoCo freejoint physics",
                    "50g Muhi properties and proxy geometry remain provisional"]}
    def save():
        (args.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps({k:summary.get(k) for k in ("status","arm_joint_count","physics_steps",
                        "peak_fixed_force_n","peak_moving_force_n","bilateral_frames",
                        "lift_started","max_arm_command_lift_mm","max_object_rise_mm",
                        "lift_success","video_status")},indent=2))
        print("Output:",args.out)
    if pose is None:save();return
    center,quat=pose[:2]
    if abs(float(quat[0])-1)>1e-5:raise ValueError("Tabletop requires upright Muhi")
    tabletop=float(center[2])-prior.OBJECT_HALF_LENGTH_M-.0005
    data.mocap_pos[mocapid]=[float(center[0]),float(center[1]),
                              tabletop-prior.TABLE_HALF_THICKNESS_M]
    data.qpos[freeq:freeq+3]=center
    data.qpos[freeq+3:freeq+7]=quat
    data.qpos[jawq]=opened;data.qvel[:]=0.
    mujoco.mj_forward(model,data)
    excluded={0,tbid,int(model.geom_bodyid[bottle[0]])}
    robot=[i for i in range(model.ngeom)
           if int(model.geom_bodyid[i]) not in excluded and
              (model.geom_contype[i] or model.geom_conaffinity[i])]
    robot_gap=min((search.signed_pair_gap(model,data,g,table) for g in robot),default=float("inf"))
    fg=search.geometry_gaps(model,data,bottle,fixed)
    mg=search.geometry_gaps(model,data,bottle,moving)
    summary.update(initial_fixed_gap_mm=fg*1000,initial_moving_gap_mm=mg*1000,
                   initial_robot_table_gap_mm=robot_gap*1000)
    if min(fg,mg)<.0005 or robot_gap<.001:
        summary["status"]="INITIAL_GEOMETRY_NOT_CLEAR";save();return
    cameras=(wrist.CAMERA,wrist.SHELF_DIAGNOSTIC_CAMERA,wrist.RETRACTED_DIAGNOSTIC_CAMERA)
    for n in cameras:
        if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,n)<0:
            raise ValueError("Missing mandatory wrist camera "+n)
    sides=[]
    for az in (45.,135.):
        cam=mujoco.MjvCamera()
        cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]=np.asarray(center);cam.distance=.34
        cam.azimuth=az;cam.elevation=-25.
        sides.append(cam)
    # Preserve final panel pixels: previous 480x360 renders were stretched
    # to 480x390 wrist and 720x390 side, visibly degrading proportions.
    # Two render targets match the final aspect ratios with no resampling.
    model.vis.global_.offwidth=max(int(model.vis.global_.offwidth),720)
    model.vis.global_.offheight=max(int(model.vis.global_.offheight),390)
    wrist_renderer=mujoco.Renderer(model,width=480,height=390)
    side_renderer=mujoco.Renderer(model,width=720,height=390)
    all_dir=args.out/"combined_frames";all_dir.mkdir()
    wrist_dirs=[]
    for name in ("wrist_camera","wrist_shelf_camera","wrist_retracted_camera"):
        folder=args.out/(name+"_frames");folder.mkdir();wrist_dirs.append(folder)
    rows=[];images=0
    bilateral_streak=0
    lift_started_at=None
    lift_object_z_initial=None
    lift_reference_base_z=None
    lift_motion_mm=0.
    try:
        steps=int(math.ceil(args.duration_s/model.opt.timestep))
        for step in range(steps):
            t=float(data.time)
            target=max(closed,opened-math.radians(args.j6_speed_deg_s)*t)
            data.ctrl[servo_id]=target
            if lift_started_at is not None:
                u=np.clip((t-lift_started_at)/args.lift_ramp_s,0.,1.)
                # Smooth position/velocity at both ends of the ramp.
                smooth=u*u*(3.-2.*u)
                lift_motion_mm=float(args.lift_mm*smooth)
                data.mocap_pos[base_mocap_id,:]=initial_base_mocap
                data.mocap_pos[base_mocap_id,2]+=lift_motion_mm/1000.
            mujoco.mj_step(model,data)
            # Read actual solver contacts BEFORE any mj_forward.
            force_f,depth_f=contact_force(model,data,set(bottle),set(fixed))
            force_m,depth_m=contact_force(model,data,set(bottle),set(moving))
            table_force,table_depth=contact_force(model,data,set(bottle),{table})
            bilateral=(force_f>=args.bilateral_force_min_n and
                       force_m>=args.bilateral_force_min_n)
            bilateral_streak=bilateral_streak+1 if bilateral else 0
            if (lift_started_at is None and
                bilateral_streak>=args.bilateral_hold_steps):
                # Establish the baseline after sustained bilateral contact,
                # prior to the FIRST mocap displacement next timestep.
                lift_started_at=float(data.time)
                lift_object_z_initial=float(data.qpos[freeq+2])
                lift_reference_base_z=float(data.mocap_pos[base_mocap_id,2])
            object_z=float(data.qpos[freeq+2])
            object_rise_mm=(1000.*(object_z-lift_object_z_initial)
                            if lift_object_z_initial is not None else 0.)
            # Approximation for upright Muhi, not a full oriented-geometry gap.
            bottom_gap_mm=1000.*(
                object_z-prior.OBJECT_HALF_LENGTH_M-tabletop)
            rows.append({"step":step,"time_s":float(data.time),
              "phase":("contact_closure" if lift_started_at is None
                       else "lift_ramp" if t<lift_started_at+args.lift_ramp_s
                       else "lift_hold"),
              "arm_lift_command_mm":lift_motion_mm,
              "arm_base_mocap_z_m":float(data.mocap_pos[base_mocap_id,2]),
              "object_rise_from_bilateral_mm":object_rise_mm,
              "object_bottom_table_gap_approx_mm":bottom_gap_mm,
              "object_table_contact_force_n":table_force,
              "bilateral_contact":bool(bilateral),
              "bilateral_streak_steps":bilateral_streak,
              "j6_deg":math.degrees(float(data.qpos[jawq])),
              "j6_target_deg":math.degrees(target),
              "fixed_force_n":force_f,"moving_force_n":force_m,
              "fixed_penetration_mm":depth_f,"moving_penetration_mm":depth_m,
              "object_z_m":object_z,
              "object_speed_m_s":float(np.linalg.norm(data.qvel[freed:freed+3]))})
            if step%args.render_every==0 or step==steps-1:
                mujoco.mj_forward(model,data)
                views=[]
                for cam in cameras:
                    wrist_renderer.update_scene(data,camera=cam)
                    views.append(Image.fromarray(wrist_renderer.render().copy()))
                for cam in sides:
                    side_renderer.update_scene(data,camera=cam)
                    views.append(Image.fromarray(side_renderer.render().copy()))
                panel=Image.new("RGB",(1440,780),(224,228,232))
                for i in range(3):panel.paste(views[i],(480*i,0))
                for i in range(2):panel.paste(views[3+i],(720*i,390))
                draw=ImageDraw.Draw(panel)
                draw.text((8,8),
                    f"J6 FIRST / Lift | t={data.time:.3f}s / arm +{lift_motion_mm:.2f}mm",
                    fill=(15,15,15))
                draw.text((8,400),
                    f"J6={math.degrees(data.qpos[jawq]):.2f}deg | fixed={force_f:.2f}N "
                    f"moving={force_m:.2f}N | object +{object_rise_mm:.2f}mm",
                    fill=(15,15,15))
                panel.save(all_dir/f"frame_{images:04d}.png")
                for i,folder in enumerate(wrist_dirs):views[i].save(folder/f"frame_{images:04d}.png")
                images+=1
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                summary["status"]="NUMERICAL_DIVERGENCE";break
    finally:
        wrist_renderer.close()
        side_renderer.close()
    if rows:
        with (args.out/"grasp_trace.csv").open("x",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    # Success requires genuine object lift and sustained contacts while
    # physically detached from the table (not just arm-base motion).
    final_hold=(rows[-max(1,int(.3/model.opt.timestep)):] if rows else [])
    stable_hold=bool(final_hold and all(
        r["bilateral_contact"] and r["object_table_contact_force_n"]<.01
        for r in final_hold))
    observed_rise_mm=(rows[-1]["object_rise_from_bilateral_mm"] if rows else 0.)
    lift_complete=bool(
        lift_started_at is not None and lift_motion_mm>=args.lift_mm-.01)
    lift_success=bool(lift_complete and observed_rise_mm>=args.lift_mm*.8
                      and stable_hold
                      and rows[-1]["object_bottom_table_gap_approx_mm"]>=1.)
    summary.update(
      lift_started=lift_started_at is not None,
      lift_start_time_s=lift_started_at,
      lift_initial_object_z_m=lift_object_z_initial,
      max_arm_command_lift_mm=max((r["arm_lift_command_mm"] for r in rows),default=0),
      max_object_rise_mm=max((r["object_rise_from_bilateral_mm"] for r in rows),default=0),
      final_object_rise_mm=observed_rise_mm,
      final_object_table_gap_approx_mm=(rows[-1]["object_bottom_table_gap_approx_mm"] if rows else None),
      final_bilateral_contact=bool(rows[-1]["bilateral_contact"]) if rows else False,
      lift_success=lift_success,
      lift_success_criteria=">=80% of commanded 5mm real object rise, >=1mm approximate tabletop gap, bilateral contact and no table contact for final 0.3s",
      physics_steps=len(rows),rendered_frames=images,
      peak_fixed_force_n=max((r["fixed_force_n"] for r in rows),default=0),
      peak_moving_force_n=max((r["moving_force_n"] for r in rows),default=0),
      bilateral_frames=sum(r["fixed_force_n"]>.02 and r["moving_force_n"]>.02 for r in rows),
      max_object_speed_m_s=max((r["object_speed_m_s"] for r in rows),default=0))
    if summary["status"]!="NUMERICAL_DIVERGENCE":
        summary["status"]=("PHYSICAL_5MM_LIFT_OBSERVED" if lift_success
                           else "LIFT_NOT_CONFIRMED" if lift_started_at is not None
                           else "NO_SUSTAINED_BILATERAL_CONTACT")
    if images:
        summary["video_status"]=encode(all_dir,args.out/"table_contact_views.mp4",args.fps)
        for folder,name in zip(wrist_dirs,("wrist_camera","wrist_shelf_camera","wrist_retracted_camera")):
            summary[name+"_video_status"]=encode(folder,args.out/(name+".mp4"),args.fps)
    save()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf",type=Path,default=r2s.DEFAULT_URDF)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--j5-deg",type=float,default=-90.)
    p.add_argument("--duration-s",type=float,default=5.5)
    p.add_argument("--lift-mm",type=float,default=5.)
    p.add_argument("--lift-ramp-s",type=float,default=1.)
    p.add_argument("--bilateral-hold-steps",type=int,default=150)
    p.add_argument("--bilateral-force-min-n",type=float,default=.02)
    p.add_argument("--j6-speed-deg-s",type=float,default=35.)
    p.add_argument("--render-every",type=int,default=25)
    p.add_argument("--fps",type=int,default=20)
    a=p.parse_args()
    if not(0<a.lift_mm<=10 and .2<=a.lift_ramp_s<=3.
           and 10<=a.bilateral_hold_steps<=1000 and
           0<a.bilateral_force_min_n<=2. and
           a.duration_s>a.lift_ramp_s+.3):
        p.error("Invalid lift parameters")
    run(a)
if __name__=="__main__":main()
