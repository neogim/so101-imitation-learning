"""Simulation-only grasp-face alignment planner for SO-101 Muhi.

After world-frame Reach, align the *inside* of the fixed finger with Muhi,
not merely its fingertip. Validate J6's swept path against the bottle.
Physical contact force is verified separately by the caller during replay.
No hardware commands, no Muhi teleportation, no changes to source URDF.
"""
from __future__ import annotations

import math
import mujoco
import numpy as np


def plan_fixed_fingertip_approach(
    model, data, initial_pose, qpos_ids, object_geoms,
    fixed_geoms, other_robot_geoms, table_geom, tcp_site_id,
    limits, *, target_gap_mm=-0.02, tip_zone_mm=30.,
    clearance_mm=4., max_steps=500,
    object_center_world=None, grip_body_id=None,
    moving_geoms=(), jaw_qpos_id=None, jaw_limits=None,
    target_depth_mm=18., depth_tolerance_mm=7.,
):
    """Safely search for fixed INSIDE face contact and two-jaw capture.

    Positive depth means toward the palm from TCP along gripper_link +Z.
    URDF TCP is at gripper_link local z=-98.13 mm. The depth is a
    model-derived coordinate, not pixel-right, and is NOT calibrated.
    For final success require depth and J6 closed-sweep intersection.
    """
    qs=np.asarray(qpos_ids,dtype=int)
    pose=np.asarray(initial_pose,dtype=float).copy()
    home=pose.copy()
    path=[pose.copy()]
    records=[]
    object_geoms=tuple(map(int,object_geoms))
    fixed_geoms=tuple(map(int,fixed_geoms))
    other_robot_geoms=tuple(map(int,other_robot_geoms))
    moving_geoms=tuple(map(int,moving_geoms))
    if (object_center_world is None or grip_body_id is None or
        jaw_qpos_id is None or jaw_limits is None or
        not object_geoms or not fixed_geoms or not moving_geoms):
        raise ValueError("Grasp face/capture analysis requires all robot and object geometry")
    center=np.asarray(object_center_world,dtype=float).copy()
    jaw_open=float(jaw_limits[1])
    jaw_closed=float(jaw_limits[0])
    if not (10<=target_depth_mm<=30 and 2<=depth_tolerance_mm<=12):
        raise ValueError("Invalid diagnostic grasp depth")

    def gap_between(gs,os,witness=False):
        best=(float("inf"),None,None,None)
        for g in gs:
            for o in os:
                cp=np.zeros(6)
                dist=float(mujoco.mj_geomDistance(
                    model,data,g,o,.25,cp))
                if dist<best[0]:
                    best=(dist,g,o,cp.copy())
        return best

    def measure(q):
        data.qpos[qs]=q
        data.qpos[jaw_qpos_id]=jaw_open
        mujoco.mj_forward(model,data)
        tcp=np.asarray(data.site_xpos[tcp_site_id]).copy()
        # The positive local-Z direction is back from the TCP tip toward
        # the gripper body. R is body-to-world, columns are local axes.
        R=np.asarray(data.xmat[grip_body_id]).reshape(3,3)
        relative=center-tcp
        depth=float(np.dot(relative,R[:,2])*1000.)
        side=float(np.dot(relative,R[:,0])*1000.)
        fixed,gid,oid,w=gap_between(fixed_geoms,object_geoms)
        moving,_,_,_=gap_between(moving_geoms,object_geoms)
        witness_tip=float(np.linalg.norm(w[:3]-tcp)*1000.)
        # Require contact on the inside-depth zone, not the first
        # millimeters of the fingertip.
        witness_depth=float(np.dot(w[:3]-tcp,R[:,2])*1000.)
        other=gap_between(other_robot_geoms,object_geoms)[0]
        table=gap_between(
            tuple(sorted(set(fixed_geoms+other_robot_geoms))),
            (int(table_geom),))[0]
        return {
            "fixed_gap_mm":fixed*1000.,
            "fixed_geom_id":gid,
            "muhi_geom_id":oid,
            "witness_world_m":w[:3].tolist(),
            "witness_tcp_distance_mm":witness_tip,
            "fixed_contact_depth_mm":witness_depth,
            "grasp_depth_mm":depth,
            "grasp_side_mm":side,
            "moving_open_gap_mm":moving*1000.,
            "other_robot_gap_mm":other*1000.,
            "robot_table_gap_mm":table*1000.,
            "tcp_world_m":tcp.tolist(),
        }

    def swept_capture(q):
        """Test whether the opened moving jaw would cross the bottle.

        An intersection is a *geometric* feasibility check, not a grasp.
        Preserve qpos after virtual J6 sweep. Never move Muhi.
        """
        saved=float(data.qpos[jaw_qpos_id])
        min_sweep=float("inf")
        crossing=None
        try:
            data.qpos[qs]=q
            for angle in np.linspace(jaw_open,jaw_closed,17)[1:]:
                data.qpos[jaw_qpos_id]=float(angle)
                mujoco.mj_forward(model,data)
                g=gap_between(moving_geoms,object_geoms)[0]*1000.
                min_sweep=min(min_sweep,g)
                if crossing is None and g<-.10:
                    crossing=float(math.degrees(angle))
        finally:
            data.qpos[jaw_qpos_id]=saved
            mujoco.mj_forward(model,data)
        return min_sweep,crossing

    def errors(m):
        return (abs(m["grasp_depth_mm"]-target_depth_mm),
                max(0.,m["fixed_gap_mm"]-target_gap_mm))

    current=measure(pose)
    start=current.copy()
    reason=None
    tested=0
    capture_rejected=0
    if (current["other_robot_gap_mm"]<clearance_mm or
        current["robot_table_gap_mm"]<clearance_mm):
        reason="START_OTHER_GEOMETRY_UNSAFE"
    else:
        for _ in range(max_steps):
            records.append({"waypoint":len(path)-1,**current})
            depth_err,gap_err=errors(current)
            depth_ok=depth_err<=depth_tolerance_mm
            contact_ok=(
                current["fixed_gap_mm"]<=target_gap_mm and
                current["fixed_gap_mm"]>=-.05 and
                current["witness_tcp_distance_mm"]<=tip_zone_mm and
                8.<=current["fixed_contact_depth_mm"]<=30. and
                current["moving_open_gap_mm"]>=.5)
            if depth_ok and contact_ok:
                minimum,crossing=swept_capture(pose)
                tested+=1
                if crossing is not None:
                    reason="GRASP_FACE_GEOMETRY_AND_J6_SWEEP_REACHED"
                    break
                capture_rejected+=1
            choice=None
            for idx in range(5):  # Include J5: grasp-face orientation matters
                for sign in (-1.,1.):
                    for step in (.012,.004,.001,.0002):
                        proposed=pose.copy()
                        proposed[idx]+=sign*step
                        if not limits[idx][0]<=proposed[idx]<=limits[idx][1]:
                            continue
                        if idx==3 and abs(proposed[3]-home[3])>math.radians(12):
                            continue
                        if idx==4 and abs(proposed[4]-home[4])>math.radians(25):
                            continue
                        m=measure(proposed)
                        if (m["other_robot_gap_mm"]<clearance_mm or
                            m["robot_table_gap_mm"]<clearance_mm or
                            m["fixed_gap_mm"]<-.05 or
                            m["moving_open_gap_mm"]<.5):
                            continue
                        # Favor moving Muhi deeper into the open jaws,
                        # before the finger-to-object gap is reduced.
                        new_de,new_gap=errors(m)
                        old_witness_penalty=max(
                            0.,8.-current["fixed_contact_depth_mm"])
                        new_witness_penalty=max(
                            0.,8.-m["fixed_contact_depth_mm"])
                        score=(1.5*(depth_err-new_de)+.35*(gap_err-new_gap)
                               +.5*(old_witness_penalty-new_witness_penalty)
                               -.03*abs(step)/.012)
                        if score>0.002 and (choice is None or score>choice[0]):
                            choice=(score,proposed,m)
            if choice is None:
                reason=("GRASP_DEPTH_OR_CAPTURE_ALIGNMENT_STALLED"
                        if depth_err>depth_tolerance_mm or capture_rejected
                        else "NO_SAFE_FIXED_FACE_GAP_REDUCTION")
                break
            pose=choice[1]
            current=choice[2]
            path.append(pose.copy())
        else:
            reason="GRASP_FACE_ALIGNMENT_ITERATION_LIMIT"

    final=measure(pose)
    min_sweep,crossing=swept_capture(pose)
    success=reason=="GRASP_FACE_GEOMETRY_AND_J6_SWEEP_REACHED"
    report={
        "status":reason,
        "success":success,
        "fixed_jaw_start_gap_mm":start["fixed_gap_mm"],
        "fixed_jaw_final_gap_mm":final["fixed_gap_mm"],
        "fixed_jaw_witness_tcp_distance_mm":
            final["witness_tcp_distance_mm"],
        "closest_fixed_geom_id":final["fixed_geom_id"],
        "closest_muhi_geom_id":final["muhi_geom_id"],
        "contact_witness_world_m":final["witness_world_m"],
        "grasp_depth_mm":final["grasp_depth_mm"],
        "fixed_contact_depth_mm":final["fixed_contact_depth_mm"],
        "min_accepted_fixed_contact_depth_mm":8.,
        "grasp_side_mm":final["grasp_side_mm"],
        "target_grasp_depth_mm":target_depth_mm,
        "grasp_depth_tolerance_mm":depth_tolerance_mm,
        "moving_jaw_open_gap_mm":final["moving_open_gap_mm"],
        "moving_jaw_sweep_min_gap_mm":min_sweep,
        "moving_jaw_crossing_angle_deg":crossing,
        "sweep_checks":tested,
        "failed_capture_checks":capture_rejected,
        "other_robot_gap_mm":final["other_robot_gap_mm"],
        "robot_table_gap_mm":final["robot_table_gap_mm"],
        "requested_gap_mm":target_gap_mm,
        "fingertip_zone_limit_mm":tip_zone_mm,
        "additional_joint_motion_deg":np.degrees(pose-home).tolist(),
        "accepted_waypoints":len(path)-1,
        "physical_contact_force_confirmed":False,
        "meaning":"fixed inner grasp depth + closing-jaw sweep are geometric only",
    }
    data.qpos[qs]=pose
    data.qpos[jaw_qpos_id]=jaw_open
    mujoco.mj_forward(model,data)
    return path,records,report
