#!/usr/bin/env python3
"""SO-101 ORIGINAL URDF forward-kinematics / joint mount audit (NO IK).

Loads RECORDED follower observation.state from LeRobot parquet; never
uses generated IK trajectories or inferred safe mechanical joint limits.
Renders one pose and +/-10 degree J2/J3/J4 perturbations, measuring each
world joint pivot and rotation axis. Never moves a physical robot.
Reference side camera placement is not calibrated to the MuJoCo world;
photo-image alignment is qualitative until camera extrinsics are known.
"""
from __future__ import annotations
import argparse,csv,json,os,sys
from pathlib import Path
import numpy as np

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex",
        "wrist_roll","gripper")
BODIES=("base_link","shoulder_link","upper_arm_link","lower_arm_link",
        "wrist_link","gripper_link")
PIVOTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")


def choose_recorded_frames(root,episode):
    import pandas as pd
    files=sorted((root/"data").glob("chunk-*/*.parquet"))
    if not files:raise FileNotFoundError("No LeRobot v3 parquet at "+str(root/"data"))
    for file in files:
        frame=pd.read_parquet(file)
        if "episode_index" not in frame or "observation.state" not in frame:
            raise ValueError("Dataset lacks episode_index / observation.state")
        ep=frame.loc[frame["episode_index"]==episode].copy()
        if ep.empty:continue
        selected=np.unique(np.linspace(0,len(ep)-1,min(3,len(ep)),dtype=int))
        out=[]
        for i in selected:
            row=ep.iloc[int(i)]
            q=np.asarray(row["observation.state"],dtype=float).ravel()
            if len(q)<6 or not np.isfinite(q[:6]).all():
                raise ValueError("Invalid LeRobot observation.state")
            if np.max(np.abs(q[:5]))>180:
                raise ValueError("Dataset joint values do not look like degrees")
            out.append(dict(episode=int(episode),
              frame_index=int(row["frame_index"]) if "frame_index" in row else int(i),
              source_file=str(file),state_deg=q[:5].tolist(),
              gripper_percent=float(q[5]),
              dataset_row_index=int(ep.index[int(i)])))
        return out
    raise ValueError("Episode not present: "+str(episode))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--urdf",required=True,type=Path)
    p.add_argument("--dataset-root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--delta-deg",type=float,default=10.)
    a=p.parse_args()
    if a.delta_deg<=0 or a.delta_deg>20:raise ValueError("delta-deg must be (0,20]")
    a.out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MUJOCO_GL","egl")
    import mujoco
    from PIL import Image,ImageDraw
    from scipy.spatial.transform import Rotation
    xml=a.urdf.resolve(strict=True)
    model=mujoco.MjSpec.from_file(str(xml)).compile()
    data=mujoco.MjData(model)
    jid={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)
         for n in JOINTS}
    bid={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n)
         for n in BODIES}
    if min(*jid.values(),*bid.values())<0:
        raise ValueError("Missing SO-101 joint/body names in URDF")
    adr={n:int(model.jnt_qposadr[jid[n]]) for n in JOINTS}
    source=choose_recorded_frames(a.dataset_root,a.episode)
    observations=[];sweeps=[]
    def set_fk(deg):
        mujoco.mj_resetData(model,data)
        for n,x in zip(JOINTS[:5],deg):
            data.qpos[adr[n]]=np.deg2rad(x)
        # Leave J6 at model default: not interpreting LeRobot percent as radians.
        mujoco.mj_forward(model,data)
    def snap(deg):
        set_fk(deg)
        joints={}
        for name in PIVOTS:
            j=jid[name]
            joints[name]=dict(pivot_m=np.asarray(data.xanchor[j]).tolist(),
                    axis_world=np.asarray(data.xaxis[j]).tolist(),
                    urdf_limit_deg=np.rad2deg(model.jnt_range[j]).tolist())
        body={name:np.asarray(data.xpos[id]).tolist() for name,id in bid.items()}
        return dict(joints=joints,bodies=body)
    # Render independent world views, not a calibrated copy of the side camera.
    renderer=None
    try:
        model.vis.global_.offwidth=max(640,model.vis.global_.offwidth)
        model.vis.global_.offheight=max(440,model.vis.global_.offheight)
        renderer=mujoco.Renderer(model,width=640,height=440)
    except Exception as exc:
        (a.out/"render_warning.txt").write_text(repr(exc))
    def render(deg,path,caption):
        if renderer is None:return
        set_fk(deg)
        cams=[("SIDE",90.,-8.),("OPPOSITE",270.,-8.),("TOP",90.,-89.)]
        frames=[]
        for title,azi,elev in cams:
            cam=mujoco.MjvCamera()
            cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:]=[.12,0,.08]
            cam.distance=.67
            cam.azimuth=azi
            cam.elevation=elev
            renderer.update_scene(data,camera=cam)
            im=Image.fromarray(renderer.render().copy())
            panel=Image.new("RGB",(640,475),(25,30,35))
            panel.paste(im,(0,35))
            ImageDraw.Draw(panel).text((9,10),title+" | FK ONLY | "+caption,fill="white")
            frames.append(panel)
        sheet=Image.new("RGB",(1920,475))
        for k,im in enumerate(frames):sheet.paste(im,(640*k,0))
        sheet.save(path)
    try:
        for i,record in enumerate(source):
            base=record["state_deg"]
            base_fk=snap(base)
            label=f"EP{a.episode} frame {record['frame_index']}"
            render(base,a.out/f"recorded_pose_{i:02d}_three_views.png",label)
            observations.append({**record,"fk":base_fk})
            for n in ("shoulder_lift","elbow_flex","wrist_flex"):
                axis_index=JOINTS.index(n)
                for sign in (-1,1):
                    shifted=np.asarray(base).copy()
                    shifted[axis_index]+=sign*a.delta_deg
                    jid_n=jid[n]
                    allowed=np.rad2deg(model.jnt_range[jid_n])
                    if shifted[axis_index]<allowed[0] or shifted[axis_index]>allowed[1]:
                        sweeps.append(dict(reference_index=i,joint=n,sign=sign,
                             status="OUTSIDE_ORIGINAL_URDF_LIMIT"))
                        continue
                    check=snap(shifted)
                    refpivot=np.asarray(base_fk["joints"][n]["pivot_m"])
                    shiftedpivot=np.asarray(check["joints"][n]["pivot_m"])
                    # joint's own pivot must not move when only that joint moves.
                    pivot_drift=float(np.linalg.norm(shiftedpivot-refpivot)*1000)
                    tip=np.asarray(check["bodies"]["gripper_link"])
                    original_tip=np.asarray(base_fk["bodies"]["gripper_link"])
                    delta_tip=float(np.linalg.norm(tip-original_tip)*1000)
                    item=dict(reference_index=i,joint=n,sign=sign,
                        delta_command_deg=sign*a.delta_deg,
                        status="FK_COMPUTED",
                        joint_pivot_drift_mm=pivot_drift,
                        gripper_body_displacement_mm=delta_tip,
                        parent_and_child_not_verified_against_real=True,
                        reference_joint_axis_world=base_fk["joints"][n]["axis_world"])
                    sweeps.append(item)
                    render(shifted,a.out/f"recorded_{i:02d}_{n}_{'plus' if sign>0 else 'minus'}_{a.delta_deg:g}deg.png",
                        label+f" {n} {sign*a.delta_deg:+g}deg")
        # Dynamic-looking geometry is NOT evidence of an incorrect joint origin:
        # compare the same recorded joint configuration to actual camera video.
        summary=dict(schema="so101_original_urdf_fk_audit_v1",
            urdf=str(xml),source="recorded LeRobot follower observations",
            input_episode=a.episode,reference_poses=len(source),
            tested_joint_names=["shoulder_lift","elbow_flex","wrist_flex"],
            delta_deg=a.delta_deg,no_ik=True,no_mj_step=True,no_hardware_control=True,
            render_camera_extrinsics_calibrated=False,
            mesh_origin_and_joint_axis_truth="UNVERIFIED_WITH_REAL_IMAGE",
            note="Pivot invariance under self joint rotation is a consistency check, not proof of correct CAD mounting. Need real video comparisons.",
            observations=observations,sweeps=sweeps)
        (a.out/"fk_audit.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
        with (a.out/"joint_sweep.csv").open("w",newline="") as f:
            fields=["reference_index","joint","sign","status","delta_command_deg",
                    "joint_pivot_drift_mm","gripper_body_displacement_mm"]
            w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
            w.writeheader();w.writerows(sweeps)
        print("AUDIT_DIR",a.out)
        print("REFERENCE_POSES",len(source),"SWEEP_RESULTS",len(sweeps))
    finally:
        if renderer is not None:renderer.close()


if __name__=="__main__":main()
