"""Simulation-only fixed-fingertip pre-contact planner for SO-101 Muhi.

Plans from a completed pregrasp pose, using the actual source-URDF
collision meshes. The bottle is fixed in WORLD coordinates during planning.
A candidate only counts as a *fingertip* contact if its collision witness
is near the TCP marker. This is not force control; actual forces are
evaluated during MuJoCo replay by the calling script.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


def plan_fixed_fingertip_approach(
    model, data, initial_pose, qpos_ids, object_geoms,
    fixed_geoms, other_robot_geoms, table_geom, tcp_site_id,
    limits, *, target_gap_mm=-0.02, tip_zone_mm=30.0,
    clearance_mm=4.0, max_steps=500,
):
    """Coordinate-descent motion on J1-J4; keep J5 and J6 fixed.

    Stop when fixed-jaw gap <= target AND witness belongs to the tip
    zone. Any other robot link stays >= clearance from bottle and table.
    A very small negative gap (default -0.02 mm) deliberately allows the
    MuJoCo solver to develop contact force. This remains a kinematic
    simulation-only diagnostic, not safe physical force control.
    """
    qs=np.asarray(qpos_ids,dtype=int)
    pose=np.asarray(initial_pose,dtype=float).copy()
    qhome=pose.copy()
    path=[pose.copy()]
    records=[]
    min_other_mm=float("inf")
    min_table_mm=float("inf")

    def measure(q):
        data.qpos[qs]=q
        mujoco.mj_forward(model,data)
        tcp=np.asarray(data.site_xpos[tcp_site_id]).copy()
        best=(float("inf"),None,None,None,float("inf"))
        for fixed in fixed_geoms:
            for obj in object_geoms:
                witness=np.zeros(6)
                gap=float(mujoco.mj_geomDistance(
                    model,data,int(fixed),int(obj),.25,witness))
                tip_distance=float(np.linalg.norm(witness[:3]-tcp))
                if gap<best[0]:
                    best=(gap,int(fixed),int(obj),witness[:3].copy(),
                          tip_distance)
        other=min((float(mujoco.mj_geomDistance(
            model,data,int(g),int(obj),.25,np.zeros(6)))
                   for g in other_robot_geoms for obj in object_geoms),
                  default=float("inf"))
        table=min((float(mujoco.mj_geomDistance(
            model,data,int(g),int(table_geom),.25,np.zeros(6)))
                   for g in set((*fixed_geoms,*other_robot_geoms))),
                  default=float("inf"))
        return {
            "fixed_gap_mm":best[0]*1000,
            "fixed_geom_id":best[1],
            "muhi_geom_id":best[2],
            "witness_world_m":best[3].tolist() if best[3] is not None else None,
            "witness_tcp_distance_mm":best[4]*1000,
            "other_robot_gap_mm":other*1000,
            "robot_table_gap_mm":table*1000,
            "tcp_world_m":tcp.tolist(),
        }

    def acceptable(m):
        return (m["other_robot_gap_mm"]>=clearance_mm and
                m["robot_table_gap_mm"]>=clearance_mm and
                m["fixed_gap_mm"]>=-0.05 and
                m["witness_tcp_distance_mm"]<=tip_zone_mm)

    current=measure(pose)
    start=current.copy()
    reason=None
    if current["other_robot_gap_mm"]<clearance_mm or (
            current["robot_table_gap_mm"]<clearance_mm):
        reason="START_OTHER_GEOMETRY_UNSAFE"
    else:
        for i in range(max_steps):
            min_other_mm=min(min_other_mm,current["other_robot_gap_mm"])
            min_table_mm=min(min_table_mm,current["robot_table_gap_mm"])
            records.append({"waypoint":len(path)-1,**current})
            if (current["fixed_gap_mm"]<=target_gap_mm and
                    current["witness_tcp_distance_mm"]<=tip_zone_mm):
                reason="GEOMETRIC_FINGERTIP_PRECONTACT_REACHED"
                break
            choice=None
            # Larger angles for far-away movement, then fine approach.
            # J4 is penalized to avoid flinging the wrist, but can move.
            for idx in (0,1,2,3):
                for sign in (-1.0,1.0):
                    for step in (.012,.004,.001,.0002):
                        proposed=pose.copy()
                        proposed[idx]+=sign*step
                        if not (limits[idx][0]<=proposed[idx]<=limits[idx][1]):
                            continue
                        if idx==3 and abs(proposed[3]-qhome[3])>math.radians(12):
                            continue
                        m=measure(proposed)
                        if (m["other_robot_gap_mm"]<clearance_mm or
                            m["robot_table_gap_mm"]<clearance_mm or
                            m["fixed_gap_mm"]<-.05):
                            continue
                        # Reject root/palm contact, not merely bad TCP.
                        # Early approach may have root witness, but require
                        # the final witness to lie at fingertip.
                        tip_cost=max(0.,m["witness_tcp_distance_mm"]-tip_zone_mm)
                        prev_tip_cost=max(
                            0.,current["witness_tcp_distance_mm"]-tip_zone_mm)
                        score=(current["fixed_gap_mm"]-m["fixed_gap_mm"]
                               +0.5*(prev_tip_cost-tip_cost)
                               - (0.6 if idx==3 else 0.)*abs(step)/.012)
                        if score>0.002 and (choice is None or score>choice[0]):
                            choice=(score,proposed,m)
            if choice is None:
                reason="NO_SAFE_FINGERTIP_GAP_REDUCTION"
                break
            pose=choice[1]
            current=choice[2]
            path.append(pose.copy())
        else:
            reason="FINGERTIP_APPROACH_ITERATION_LIMIT"
    final=measure(pose)
    report={
        "status":reason,
        "success":reason=="GEOMETRIC_FINGERTIP_PRECONTACT_REACHED",
        "fixed_jaw_start_gap_mm":start["fixed_gap_mm"],
        "fixed_jaw_final_gap_mm":final["fixed_gap_mm"],
        "fixed_jaw_witness_tcp_distance_mm":
            final["witness_tcp_distance_mm"],
        "closest_fixed_geom_id":final["fixed_geom_id"],
        "closest_muhi_geom_id":final["muhi_geom_id"],
        "contact_witness_world_m":final["witness_world_m"],
        "other_robot_gap_mm":final["other_robot_gap_mm"],
        "robot_table_gap_mm":final["robot_table_gap_mm"],
        "requested_gap_mm":target_gap_mm,
        "fingertip_zone_limit_mm":tip_zone_mm,
        "additional_joint_motion_deg":
            np.degrees(pose-qhome).tolist(),
        "accepted_waypoints":len(path)-1,
        "physical_contact_force_confirmed":False,
        "meaning":"small bounded geometric overlap; physical force confirmation required",
    }
    data.qpos[qs]=pose
    mujoco.mj_forward(model,data)
    return path,records,report
