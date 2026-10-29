#!/usr/bin/env python3
"""Convert offline ACT Cartesian prediction CSV to bounded XYZ-priority IK candidates.
DOES NOT communicate with hardware. Mapping is a hypothesis, not calibration.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import placo
from batch_cartesian_fk_ik_hypotheses_offline import mapping,to_urdf_rad,to_motor_deg
from export_cartesian_action_sidecar import JOINT_NAMES,rotation_log
from generate_offset_ik_offline import joint_limits

NAMES=("dx_m","dy_m","dz_m","drx_rad","dry_rad","drz_rad","gripper")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions",type=Path,default=Path("outputs/so101-cartesian-act-offline-pred-v1/predictions.csv"))
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    p.add_argument("--output",type=Path,default=Path("outputs/so101-cartesian-ik-smoke-v1"))
    p.add_argument("--mapping",choices=["j2+_j3+","j2-_j3+","j2+_j3-","j2-_j3-"],default="j2-_j3-")
    p.add_argument("--j2-offset-deg",type=float,default=-3.)
    p.add_argument("--j3-offset-deg",type=float,default=1.)
    p.add_argument("--max-frames",type=int,default=8)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(f"Output already exists: {a.output}")
    if not 1<=a.max_frames<=100:raise ValueError("Invalid max-frames")
    with a.predictions.open(newline="") as f:rows=list(csv.DictReader(f))[:a.max_frames]
    if not rows:raise ValueError("No predictions")
    signs,offsets=mapping(a.mapping,a.j2_offset_deg,a.j3_offset_deg)
    robot=placo.RobotWrapper(str(a.urdf))
    if not set(JOINT_NAMES).issubset(set(robot.joint_names())):raise ValueError("URDF joint mismatch")
    lower,upper=joint_limits(a.urdf)
    def fk(q):
        for name,value in zip(JOINT_NAMES,q,strict=True):robot.set_joint(name,float(value))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"),dtype=float).copy()
    results=[]
    for r in rows:
        state=np.array([float(r[f"state_{i}"]) for i in range(6)])
        pred=np.array([float(r[f"pred_{n}"]) for n in NAMES])
        if not np.isfinite(np.r_[state,pred]).all():raise ValueError("Nonfinite row")
        q0=to_urdf_rad(state[:5],signs,offsets)
        was_clipped=bool(np.any(q0<lower) or np.any(q0>upper))
        seed=np.clip(q0,lower+1e-7,upper-1e-7)
        base=fk(q0)
        xyz_goal=base[:3,3]+pred[:3]
        # Delta rotvec uses R_goal = exp([delta]x) R_state in base frame.
        # Position is prioritized over orientation: SO-101 has only five arm DOF.
        def residual(q):
            T=fk(q)
            xyz=(T[:3,3]-xyz_goal)*1000.
            prior=.03*np.rad2deg(q-seed)
            return np.r_[xyz,prior]
        sol=least_squares(residual,seed,bounds=(lower,upper),max_nfev=100)
        T=fk(sol.x)
        xyz_err=float(np.linalg.norm(T[:3,3]-xyz_goal)*1000.)
        motor=to_motor_deg(sol.x,signs,offsets)
        results.append(dict(episode_index=int(r["episode_index"]),frame_index=int(r["frame_index"]),
          optimizer_success=bool(sol.success),input_state_clipped_for_seed=was_clipped,
          target_tcp_xyz_m=xyz_goal.tolist(),achieved_tcp_xyz_m=T[:3,3].tolist(),
          position_error_mm=xyz_err,angle_change_max_deg=float(np.max(np.abs(motor-state[:5]))),
          candidate_motor_deg=motor.tolist(),gripper_prediction=float(pred[6]),
          orientation_prediction_unconstrained_rad=pred[3:6].tolist(),
          NOT_SAFE_FOR_SERVO=True))
    a.output.mkdir(parents=True)
    (a.output/"ik_candidates.json").write_text(json.dumps(results,indent=2)+"\n")
    summary=dict(status="OFFLINE_ONLY_NOT_HARDWARE_APPROVED",frames=len(results),mapping=a.mapping,
      offsets_deg=[a.j2_offset_deg,a.j3_offset_deg],
      mean_position_error_mm=float(np.mean([x["position_error_mm"] for x in results])),
      max_position_error_mm=float(max(x["position_error_mm"] for x in results)),
      optimizer_success_count=sum(x["optimizer_success"] for x in results),
      note="XYZ-only bounded IK; no collision, path safety, physical calibration or orientation constraint. DO NOT send candidates to motors.")
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
