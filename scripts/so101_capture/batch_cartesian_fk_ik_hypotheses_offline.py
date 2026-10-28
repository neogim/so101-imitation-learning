#!/usr/bin/env python3
"""Batch SO-101 offline Cartesian FK/IK experiments, no servo access.

Source episodes and URDF are read-only. Joint mapping is a HYPOTHESIS, not a
physical calibration. Cartesian sidecars are not directly trainable ACT data.
Requires numpy, pyarrow, placo, scipy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from export_cartesian_action_sidecar import JOINT_NAMES, read_rows, target_from_fk
from generate_offset_ik_offline import joint_limits

HYPOTHESES = {
    "j2+_j3+": (1, 1),
    "j2-_j3+": (-1, 1),
    "j2+_j3-": (1, -1),
    "j2-_j3-": (-1, -1),
}


def mapping(label, j2_offset_deg=0., j3_offset_deg=0.):
    if label not in HYPOTHESES:
        raise ValueError(f"Unknown mapping: {label}")
    signs = np.array([1., *HYPOTHESES[label], 1., 1.], dtype=float)
    offsets = np.array([0., j2_offset_deg, j3_offset_deg, 0., 0.], dtype=float)
    if not np.isfinite(offsets).all():
        raise ValueError("Offsets must be finite")
    return signs, offsets


def to_urdf_rad(motor_deg, signs, offsets):
    x = np.asarray(motor_deg, dtype=float)
    if x.shape != (5,) or not np.isfinite(x).all():
        raise ValueError("Expected five finite motor angles")
    return np.deg2rad(x * signs + offsets)


def to_motor_deg(urdf_rad, signs, offsets):
    x = np.asarray(urdf_rad, dtype=float)
    if x.shape != (5,) or not np.isfinite(x).all():
        raise ValueError("Expected five finite URDF angles")
    return (np.rad2deg(x) - offsets) / signs


def load_episode(root, episode):
    rows = [r for r in read_rows(root) if int(r["episode_index"]) == episode]
    rows.sort(key=lambda r: int(r["frame_index"]))
    if not rows:
        raise ValueError(f"Episode {episode} not found")
    if [int(r["frame_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Episode frame indices not contiguous from zero")
    if any(float(b["timestamp"]) <= float(a["timestamp"]) for a,b in zip(rows,rows[1:])):
        raise ValueError("Timestamps not strictly increasing")
    return rows


def run(args):
    from scipy.optimize import least_squares
    import placo

    source = args.dataset.resolve()
    output = args.output.resolve()
    urdf = args.urdf.resolve()
    if not (source / "meta" / "info.json").is_file():
        raise ValueError("Source is not a LeRobot dataset")
    if not urdf.is_file():
        raise ValueError("URDF not found")
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Output must be a new directory outside the dataset")
    if args.ik_stride < 1 or args.ik_max_frames < 1 or args.ik_offset_mm <= 0:
        raise ValueError("Invalid IK parameters")
    if args.max_position_error_mm <= 0 or args.max_motor_deviation_deg <= 0:
        raise ValueError("Invalid screening parameters")
    if not np.isfinite([args.ik_offset_mm,args.max_position_error_mm,args.max_motor_deviation_deg]).all():
        raise ValueError("Nonfinite parameters")

    rows = load_episode(source,args.episode)
    robot = placo.RobotWrapper(str(urdf))
    names = set(robot.joint_names())
    if not set(JOINT_NAMES).issubset(names):
        raise ValueError("URDF joint mismatch")
    lower,upper = joint_limits(urdf)

    def fk(q_rad):
        for joint, q in zip(JOINT_NAMES,q_rad,strict=True):
            robot.set_joint(joint,float(q))
        robot.update_kinematics()
        pose = np.asarray(robot.get_T_world_frame("gripper_frame_link"),dtype=float).copy()
        if pose.shape != (4,4) or not np.isfinite(pose).all():
            raise ValueError("Invalid FK pose")
        return pose

    cases = list(HYPOTHESES) if args.mapping == "all" else [args.mapping]
    all_summary = {}
    all_sidecars = []
    for label in cases:
        signs,offsets = mapping(label,args.j2_offset_deg,args.j3_offset_deg)
        within_urdf = 0
        fk_positions = []
        for r in rows:
            state=np.asarray(r["observation.state"],dtype=float)
            action=np.asarray(r["action"],dtype=float)
            if state.shape != (6,) or action.shape != (6,):
                raise ValueError("Invalid six-component episode state/action")
            q_state=to_urdf_rad(state[:5],signs,offsets)
            q_action=to_urdf_rad(action[:5],signs,offsets)
            valid=bool(np.all(q_state>=lower) and np.all(q_state<=upper)
                       and np.all(q_action>=lower) and np.all(q_action<=upper))
            within_urdf+=int(valid)
            pose=fk(q_state)
            fk_positions.append(pose[:3,3].tolist())
            def fk_degrees(motor):
                return fk(to_urdf_rad(motor,signs,offsets))
            delta=target_from_fk(state,action,fk_degrees)
            all_sidecars.append(dict(mapping=label,episode_index=args.episode,
                                     frame_index=int(r["frame_index"]),
                                     timestamp=float(r["timestamp"]),
                                     state_and_action_inside_urdf_limits=valid,
                                     tcp_xyz_m=pose[:3,3].tolist(),
                                     cartesian_action=delta.tolist()))

        # Limited and explicit counterfactual IK: XYZ-only, for exploratory
        # numerical sensitivity, not a valid robot-executable trajectory.
        sampled=rows[::args.ik_stride][:args.ik_max_frames]
        ik_rows=[]
        for r in sampled:
            original=np.asarray(r["action"][:5],dtype=float)
            q=to_urdf_rad(original,signs,offsets)
            base=fk(q)
            target=base[:3,3]+np.array([0,0,args.ik_offset_mm/1000.])
            seed=np.clip(q,lower+1e-8,upper-1e-8)
            def residual(candidate):
                xyz_residual=(fk(candidate)[:3,3]-target)*1000.
                # Small prior biases toward the recorded joint angles.
                return np.r_[xyz_residual,.05*np.rad2deg(candidate-seed)]
            sol=least_squares(residual,seed,bounds=(lower,upper),max_nfev=100)
            candidate=to_motor_deg(sol.x,signs,offsets)
            error=float(np.linalg.norm((fk(sol.x)[:3,3]-target)*1000.))
            deviation=float(np.max(np.abs(candidate-original)))
            accepted=bool(sol.success and error<=args.max_position_error_mm
                          and deviation<=args.max_motor_deviation_deg)
            ik_rows.append(dict(frame_index=int(r["frame_index"]),success=bool(sol.success),
                                passes_numerical_screen=accepted,
                                position_error_mm=error,max_motor_deviation_deg=deviation,
                                candidate_motor_deg=candidate.tolist(),
                                recorded_action_motor_deg=original.tolist()))
        all_summary[label]=dict(
            signs=dict(zip(JOINT_NAMES,signs.astype(int).tolist())),
            offsets_deg=dict(zip(JOINT_NAMES,offsets.tolist())),
            frames=len(rows),frames_state_and_action_inside_urdf_limits=within_urdf,
            ik_sampled_frames=len(ik_rows),
            ik_numerically_accepted=sum(x["passes_numerical_screen"] for x in ik_rows),
            ik_samples=ik_rows,
        )

    # Make output only after every computation succeeds.
    output.mkdir(parents=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(all_sidecars),output/"cartesian_hypotheses.parquet")
    report=dict(
        classification="OFFLINE_HYPOTHESES_NOT_PHYSICAL_CALIBRATION",
        dataset=str(source),urdf=str(urdf),episode=args.episode,
        source_frame_count=len(rows),ik_offset_mm=args.ik_offset_mm,
        source_untouched=True,hardware_access=False,
        mapping_results=all_summary,
        limitations=[
            "No hypothesis proves physical sign or zero offset.",
            "TCP and frame pose depend on assumed URDF joint mapping.",
            "Cartesian sidecar is not a trainable ACT dataset.",
            "IK is XYZ-only, not orientation constrained, and has no collision or swept-path checks.",
            "No IK result is approved for servo commands.",
            "Recorded camera images are not valid for counterfactual IK waypoints.",
        ])
    (output/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:{x:y for x,y in v.items() if x!="ik_samples"}
                      for k,v in all_summary.items()},ensure_ascii=False,indent=2))
    print("OFFLINE ONLY. Saved",output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--mapping",choices=["all",*HYPOTHESES],default="all")
    p.add_argument("--j2-offset-deg",type=float,default=0.)
    p.add_argument("--j3-offset-deg",type=float,default=0.)
    p.add_argument("--ik-stride",type=int,default=20)
    p.add_argument("--ik-max-frames",type=int,default=33)
    p.add_argument("--ik-offset-mm",type=float,default=3.)
    p.add_argument("--max-position-error-mm",type=float,default=2.)
    p.add_argument("--max-motor-deviation-deg",type=float,default=15.)
    run(p.parse_args())

if __name__=="__main__":
    main()
