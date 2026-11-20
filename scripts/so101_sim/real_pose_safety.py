#!/usr/bin/env python3
"""Conservative SO-101 real-to-sim candidate audit.

Sources are deliberately distinct:
- follower calibration JSON: encoder calibration; NOT mechanical safe limits.
- LeRobot observation.state parquet: ACTUAL recorded follower joint samples.
- MuJoCo collision geoms: geometric self-collision independent of contact masks.

A recorded neighboring posture is evidence of *observed* reachability, not
proof of path feasibility, no collision, or joint zero/sign correctness.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import mujoco

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
DEFAULT_MAX_J234_AXIS_DEG=30.0
DEFAULT_MAX_J234_L2_DEG=45.0


def load_follower_calibration(path):
    raw=json.loads(Path(path).read_text())
    for name in (*JOINTS,"gripper"):
        c=raw[name]
        if not 0 <= int(c["range_min"]) < int(c["range_max"]) <= 4095:
            raise ValueError("Invalid encoder range for "+name)
        if "homing_offset" not in c or "drive_mode" not in c:
            raise ValueError("Incomplete follower calibration for "+name)
    return raw


def calibration_report(calibration):
    rows=[]
    for name,c in calibration.items():
        lo,hi=int(c["range_min"]),int(c["range_max"])
        rows.append(dict(joint=name,encoder_min=lo,encoder_max=hi,
            encoder_midpoint=(lo+hi)/2.,
            observed_width_deg=(hi-lo)*360./4096.,
            homing_offset=int(c["homing_offset"]),
            drive_mode=int(c["drive_mode"]),
            warning="Observed encoder range is not automatically a mechanical safe range."))
    return rows


def recorded_joint_samples(dataset_root, max_samples=5000):
    """Read real LeRobot v3 observations, never sim IK trajectories.

    LeRobot SO-101 follower observation.state J1..J5 is in controller
    calibrated degrees; gripper is a percentage and is ignored here.
    """
    root=Path(dataset_root)
    files=sorted((root/"data").glob("chunk-*/*.parquet"))
    if not files:
        files=sorted(root.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError("No LeRobot episode .parquet under "+str(root))
    import pandas as pd
    samples=[]
    total=0
    for file in files:
        df=pd.read_parquet(file,columns=["observation.state"])
        if df.empty:continue
        vals=np.stack(df["observation.state"].map(lambda x: np.asarray(x,dtype=float)).values)
        if vals.ndim!=2 or vals.shape[1]<5:
            raise ValueError("observation.state is not a 5/6 joint vector")
        if not np.isfinite(vals[:,:5]).all():
            raise ValueError("Nonfinite real follower observation.state")
        total+=len(vals)
        # Limit data size while keeping every episode represented.
        stride=max(1,int(np.ceil(len(vals)/max(1,int(max_samples/max(1,len(files)))))))
        samples.extend(vals[::stride,:5])
    if not samples:raise ValueError("No real joint samples found")
    data=np.asarray(samples,dtype=float)
    if len(data)>max_samples:
        data=data[np.linspace(0,len(data)-1,max_samples,dtype=int)]
    if np.max(np.abs(data))>360.:
        raise ValueError("Recorded state is not plausible degrees; check dataset encoding")
    return data,dict(files=len(files),recorded_rows=total,retained=len(data),
        source="LeRobot observation.state (recorded hardware; not simulated poses)",
        warning="Pose neighborhood, not mechanical reachability proof")


def recorded_j234_proximity(q,real_samples,max_axis_deg=DEFAULT_MAX_J234_AXIS_DEG,
                            max_l2_deg=DEFAULT_MAX_J234_L2_DEG):
    deg=np.rad2deg(np.asarray(q,dtype=float)[:5])
    deltas=real_samples[:,1:4]-deg[None,1:4]
    l2=np.linalg.norm(deltas,axis=1)
    idx=int(np.argmin(l2))
    worst=float(np.max(np.abs(deltas[idx])))
    dist=float(l2[idx])
    return dict(valid=bool(worst<=max_axis_deg and dist<=max_l2_deg),
        candidate_j234_deg=deg[1:4].tolist(),
        nearest_recorded_j234_deg=real_samples[idx,1:4].tolist(),
        nearest_recorded_j12345_deg=real_samples[idx,:5].tolist(),
        max_absolute_j234_difference_deg=worst,
        joint234_euclidean_difference_deg=dist,
        threshold_per_joint_deg=max_axis_deg,threshold_l2_deg=max_l2_deg,
        caution="Observed-pose neighborhood does NOT certify safe motion or J3 sign.")


def _is_adjacent(model,a,b):
    if a==b:return True
    pa=int(model.body_parentid[a]);pb=int(model.body_parentid[b])
    return pa==b or pb==a


def nonadjacent_self_collisions(model,data,excluded_body_pairs=(),
                               penetration_tolerance_mm=1.0):
    """Audit non-adjacent collisions irrespective of disabled contact masks.

    Adjacent bodies naturally overlap around pivot joints. Exclusions are
    explicit and reported; geometric false positives require visual review.
    """
    ignored={tuple(sorted((str(a),str(b)))) for a,b in excluded_body_pairs}
    gnames=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ""
            for i in range(model.ngeom)]
    ids=[]
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g])==0:continue
        # URDF visual geom is typically group 2, collision geom group 0.
        # Patch geoms are included regardless of group.
        if int(model.geom_group[g])==2 and "collision_patch" not in gnames[g]:
            continue
        ids.append(g)
    collisions=[]
    checked=0
    for k,a in enumerate(ids):
        ba=int(model.geom_bodyid[a])
        for b in ids[k+1:]:
            bb=int(model.geom_bodyid[b])
            if _is_adjacent(model,ba,bb):continue
            na=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,ba) or str(ba)
            nb=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bb) or str(bb)
            if tuple(sorted((na,nb))) in ignored:continue
            checked+=1
            dist=float(mujoco.mj_geomDistance(model,data,int(a),int(b),.1,
                     np.zeros(6,dtype=float)))
            if dist < -penetration_tolerance_mm/1000.:
                collisions.append(dict(body_a=na,body_b=nb,
                   geom_a=gnames[a],geom_b=gnames[b],
                   penetration_mm=-dist*1000.))
    collisions.sort(key=lambda item:-item["penetration_mm"])
    return dict(valid=bool(checked>0 and not collisions),
        checked_geom_pairs=checked,penetrating_pairs=len(collisions),
        worst_collisions=collisions[:20],
        ignored_adjacent_bodies=True,
        explicit_exclusions=[list(p) for p in sorted(ignored)],
        warning="Convex mesh collision may report false positives; verify named link pairs.")


def audit_candidate(model,data,qadr,q,real_samples,
                    excluded_body_pairs=(),max_axis_deg=30.,
                    max_l2_deg=45.,penetration_tolerance_mm=1.):
    q=np.asarray(q,dtype=float)
    for adr,val in zip(qadr[:5],q[:5]):
        data.qpos[int(adr)]=float(val)
    mujoco.mj_forward(model,data)
    range_errors={}
    for j,name in enumerate(JOINTS):
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        a,b=np.asarray(model.jnt_range[jid],dtype=float)
        if q[j]<a-1e-6 or q[j]>b+1e-6:
            range_errors[name]=dict(actual_deg=float(np.rad2deg(q[j])),
                        lower_deg=float(np.rad2deg(a)),
                        upper_deg=float(np.rad2deg(b)))
    empirical=recorded_j234_proximity(q,real_samples,max_axis_deg,max_l2_deg)
    collisions=nonadjacent_self_collisions(model,data,excluded_body_pairs,
                                            penetration_tolerance_mm)
    return dict(valid=bool(not range_errors and empirical["valid"] and collisions["valid"]),
        urdf_joint_limit_errors=range_errors,
        observed_real_pose_proximity=empirical,
        nonadjacent_self_collision=collisions,
        confidence="Preflight screening ONLY: sign/zero and real motion must still be confirmed")
