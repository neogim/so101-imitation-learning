#!/usr/bin/env python3
"""Audit predicted Cartesian orientation vs XYZ-only IK on the same 100 frames.
Read-only: no device commands. Rotation delta is left-multiplied in base frame.
URDF collision warning is NOT a collision test.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import placo
from scipy.spatial.transform import Rotation
from generate_offset_ik_offline import joint_limits
from export_cartesian_action_sidecar import JOINT_NAMES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--candidates",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(f"Output exists: {a.output}")
    with a.predictions.open(newline="") as f:
        rows={int(r["frame_index"]):r for r in csv.DictReader(f)}
    cand=json.loads(a.candidates.read_text())
    if len(rows)!=len(cand) or len(rows)<3:raise ValueError("Prediction/candidate count mismatch")
    if len({int(r["frame_index"]) for r in cand})!=len(cand):raise ValueError("Duplicate candidates")
    keys=sorted(rows)
    if sorted(int(r["frame_index"]) for r in cand)!=keys or any(np.diff(keys)!=1):
        raise ValueError("Frame IDs mismatch or are nonconsecutive")
    cand={int(r["frame_index"]):r for r in cand}
    lower,upper=joint_limits(a.urdf)
    robot=placo.RobotWrapper(str(a.urdf.resolve()))
    def fk(q_deg):
        for name,angle in zip(JOINT_NAMES,np.deg2rad(q_deg)):
            robot.set_joint(name,float(angle))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"),float).copy()

    result=[]
    for frame in keys:
        r=rows[frame]; c=cand[frame]
        state=np.array([float(r[f"state_{i}"]) for i in range(5)])
        q=np.array(c["candidate_motor_deg"],float)
        pred=np.array([float(r[f"pred_dr{x}_rad"]) for x in "xyz"])
        delta_xyz=np.array([float(r[f"pred_d{x}_m"]) for x in "xyz"])
        if not np.isfinite(np.r_[state,q,pred,delta_xyz]).all():raise ValueError(f"Nonfinite: {frame}")
        t0=fk(state); t=fk(q)
        target_R=Rotation.from_rotvec(pred).as_matrix()@t0[:3,:3]
        orientation_deg=float(np.rad2deg(Rotation.from_matrix(target_R@t[:3,:3].T).magnitude()))
        target_xyz=t0[:3,3]+delta_xyz
        pos_mm=float(np.linalg.norm(t[:3,3]-target_xyz)*1000)
        qrad=np.deg2rad(q)
        within=bool(np.all(qrad>=lower-1e-8)&np.all(qrad<=upper+1e-8))
        result.append(dict(frame=frame,orientation_error_deg=orientation_deg,
                           position_error_mm=pos_mm,joint_limits_ok=within,
                           initial_state_outside_urdf=bool(np.any(np.deg2rad(state)<lower)|np.any(np.deg2rad(state)>upper)),
                           optimizer_success=bool(c["optimizer_success"])))
    angle=np.array([r["orientation_error_deg"] for r in result])
    position=np.array([r["position_error_mm"] for r in result])
    def stat(x):
        return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))
    report=dict(status="OFFLINE_ONLY_UNCALIBRATED_MAPPING",frames=len(result),
                orientation_error_deg=stat(angle),position_error_mm=stat(position),
                worst_orientation_frame=int(result[int(np.argmax(angle))]["frame"]),
                ik_success_count=int(sum(r["optimizer_success"] for r in result)),
                joint_limit_violation_count=int(sum(not r["joint_limits_ok"] for r in result)),
                initial_state_outside_urdf_count=int(sum(r["initial_state_outside_urdf"] for r in result)),
                collision_check="NOT_PERFORMED",swept_path_check="NOT_PERFORMED",
                calibration="LeRobot-to-URDF sign/zero hypothesis; NOT VERIFIED",
                caveat="Predicted orientation is not constrained by prior XYZ-only IK. Separate per-frame seeded IK is not a physical rollout; do not send servo commands.")
    a.output.mkdir(parents=True)
    (a.output/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (a.output/"per_frame.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
