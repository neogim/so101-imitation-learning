#!/usr/bin/env python3
"""Compare existing XYZ-only IK with position+orientation IK, offline only.
URDF joint sign/zero unverified. No collision or hardware I/O.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import placo
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from generate_offset_ik_offline import joint_limits
from export_cartesian_action_sidecar import JOINT_NAMES

def metrics(v):
    a=np.asarray(v,float)
    return {"mean":float(a.mean()),"p95":float(np.percentile(a,95)),"max":float(a.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--candidates",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    p.add_argument("--orientation-weight",type=float,default=25.,help="mm/rad")
    p.add_argument("--joint-weight",type=float,default=.03,help="mm/degree, same as XYZ-only baseline")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.orientation_weight<0 or a.joint_weight<0:raise ValueError("Nonnegative weights required")
    with a.predictions.open(newline="") as f:
        prediction=list(csv.DictReader(f))
    candidates=json.loads(a.candidates.read_text())
    ps={int(r["frame_index"]):r for r in prediction}
    cs={int(r["frame_index"]):r for r in candidates}
    if len(ps)!=len(prediction) or len(cs)!=len(candidates) or set(ps)!=set(cs) or len(ps)<3:
        raise ValueError("Frame mismatch or duplicates")
    keys=sorted(ps)
    if any(np.diff(keys)!=1):raise ValueError("Nonconsecutive frames")
    episodes={int(r["episode_index"]) for r in prediction}
    if len(episodes)!=1:raise ValueError("Multiple episodes")
    lo,hi=joint_limits(a.urdf)
    robot=placo.RobotWrapper(str(a.urdf.resolve()))
    def fk(q):
        for n,v in zip(JOINT_NAMES,q):robot.set_joint(n,float(v))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"),float).copy()
    records=[]
    for frame in keys:
        r=ps[frame];c=cs[frame]
        state=np.array([float(r[f"state_{i}"]) for i in range(5)])
        delta=np.array([float(r[f"pred_d{x}_m"]) for x in "xyz"])
        rv=np.array([float(r[f"pred_dr{x}_rad"]) for x in "xyz"])
        old_deg=np.array(c["candidate_motor_deg"],float)
        if not np.isfinite(np.r_[state,delta,rv,old_deg]).all():raise ValueError(f"Nonfinite frame {frame}")
        qstate=np.deg2rad(state)
        seed=np.clip(qstate,lo+1e-7,hi-1e-7)
        base=fk(qstate)
        target_xyz=base[:3,3]+delta
        target_R=Rotation.from_rotvec(rv).as_matrix()@base[:3,:3]
        def error(q):
            t=fk(q)
            pos=float(np.linalg.norm((t[:3,3]-target_xyz)*1000))
            ang=float(np.rad2deg(Rotation.from_matrix(target_R@t[:3,:3].T).magnitude()))
            return pos,ang
        def objective(q):
            t=fk(q)
            xyz=(t[:3,3]-target_xyz)*1000
            rot=Rotation.from_matrix(target_R@t[:3,:3].T).as_rotvec()
            return np.r_[xyz,rot*a.orientation_weight,np.rad2deg(q-seed)*a.joint_weight]
        sol=least_squares(objective,seed,bounds=(lo,hi),max_nfev=150)
        oldpos,oldang=error(np.deg2rad(old_deg))
        newpos,newang=error(sol.x)
        records.append({"frame":frame,"old_position_mm":oldpos,"old_orientation_deg":oldang,
                        "new_position_mm":newpos,"new_orientation_deg":newang,
                        "old_joint_deg":old_deg.tolist(),"new_joint_deg":np.rad2deg(sol.x).tolist(),
                        "optimizer_success":bool(sol.success),
                        "state_outside_urdf":bool(np.any(qstate<lo)|np.any(qstate>hi)),
                        "new_joint_limits_ok":bool(np.all(sol.x>=lo-1e-8)&np.all(sol.x<=hi+1e-8))})
    def aggregate(prefix):
        q=np.array([r[f"{prefix}_joint_deg"] for r in records])
        # Consecutive frame indexes; rate assumes known recording FPS=20.
        v=np.diff(q,axis=0)*20.
        acc=np.diff(v,axis=0)*20.
        return {"position_error_mm":metrics([r[f"{prefix}_position_mm"] for r in records]),
                "orientation_error_deg":metrics([r[f"{prefix}_orientation_deg"] for r in records]),
                "max_joint_jump_deg":float(np.abs(np.diff(q,axis=0)).max()),
                "max_joint_velocity_deg_s":float(np.abs(v).max()),
                "max_joint_acceleration_deg_s2":float(np.abs(acc).max())}
    report={"status":"OFFLINE_ONLY_NOT_FOR_SERVO","episode":next(iter(episodes)),"frames":len(keys),
            "fps_assumption":20,"orientation_weight_mm_per_rad":a.orientation_weight,
            "joint_weight_mm_per_deg":a.joint_weight,
            "xyz_only":aggregate("old"),"position_plus_orientation":aggregate("new"),
            "optimizer_success_count":sum(r["optimizer_success"] for r in records),
            "new_joint_limit_violation_count":sum(not r["new_joint_limits_ok"] for r in records),
            "state_outside_urdf_count":sum(r["state_outside_urdf"] for r in records),
            "collision_check":"NOT_PERFORMED","swept_path_check":"NOT_PERFORMED",
            "warning":"URDF mapping unverified; old/new solves independent per recorded state, not closed-loop trajectories. Position vs orientation tradeoff does not certify safety."}
    a.output.mkdir(parents=True)
    (a.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (a.output/"per_frame.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
