"""Collision-aware kinematic joint-step limiter for SO-101 Real-to-Sim.

This is a conservative safety filter, NOT a physics solver or an optimal motion
planner. It checks continuous interpolation samples for MOVING robot geometry
against saved tabletop geoms, independently of the contact-mask settings.
Original source URDF and unclipped reference trajectories are untouched.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


MOVING_BODY_NAMES = frozenset((
    "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link",
    "gripper_link", "moving_jaw_so101_v1_link", "r2s_wrist_camera_rig",
))


def interpolate_joint_segment(start, end, max_delta_rad=.01):
    start, end = np.asarray(start, float), np.asarray(end, float)
    if start.shape != end.shape or start.ndim != 1:
        raise ValueError("start and end must be matching 1D joint vectors")
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError("non-finite joint values")
    if not math.isfinite(max_delta_rad) or max_delta_rad <= 0:
        raise ValueError("invalid interpolation resolution")
    count = max(1, math.ceil(float(np.max(np.abs(end-start))) / max_delta_rad))
    return [start+(end-start)*(i/count) for i in range(1, count+1)]


class TableCollisionGuard:
    def __init__(self, model, data, qids, *, clearance_mm=3., max_delta_rad=.01,
                 include_cup=False, exclude_moving_bodies=()):
        self.model, self.data = model, data
        self.qids = np.asarray(qids, dtype=int)
        self.clearance_m = float(clearance_mm)/1000.
        self.max_delta_rad = float(max_delta_rad)
        if not 0 <= self.clearance_m <= .05 or not math.isfinite(self.clearance_m):
            raise ValueError("clearance_mm must be 0..50")
        self.excluded = frozenset(exclude_moving_bodies)
        self.table_ids = []
        self.robot_ids = []
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            body = int(model.geom_bodyid[gid])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if (name == "r2s_preview_table" or name.startswith("r2s_table_front_")
                    or (include_cup and name.startswith("r2s_cup_"))):
                self.table_ids.append(gid)
            is_moving = (body_name in MOVING_BODY_NAMES or
                         body_name.startswith("r2s_wrist_camera_"))
            # Never use duplicate, non-collidable visual-only geometry as a
            # physical exclusion boundary (notably wrist visual duplicates).
            is_collision_proxy = bool(model.geom_contype[gid] or model.geom_conaffinity[gid])
            if is_moving and body_name not in self.excluded and is_collision_proxy:
                self.robot_ids.append(gid)
        if not self.table_ids or not self.robot_ids:
            raise ValueError("No collidable table or moving robot geoms")
        self.table_ids = tuple(self.table_ids)
        self.robot_ids = tuple(self.robot_ids)

    def inspect(self, q):
        """Return minimum signed geom pair distance and the responsible pair."""
        q = np.asarray(q, dtype=float)
        if q.shape != self.qids.shape or not np.isfinite(q).all():
            raise ValueError("joint vector shape / value mismatch")
        prior = self.data.qpos[self.qids].copy()
        try:
            self.data.qpos[self.qids] = q
            mujoco.mj_forward(self.model, self.data)
            worst = float("inf")
            pair = None
            for robot_id in self.robot_ids:
                for table_id in self.table_ids:
                    gap = float(mujoco.mj_geomDistance(
                        self.model, self.data, int(robot_id), int(table_id),
                        .05, np.zeros(6)))
                    if gap < worst:
                        worst = gap
                        pair = (
                            (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_id)
                             or f"robot_geom_{robot_id}") + " [body=" +
                            (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                int(self.model.geom_bodyid[robot_id])) or
                             str(int(self.model.geom_bodyid[robot_id]))) + "]",
                            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, table_id)
                            or f"table_geom_{table_id}",
                        )
            return {"minimum_distance_mm":worst*1000., "nearest_pair":pair,
                    "clearance_satisfied":bool(worst >= self.clearance_m)}
        finally:
            self.data.qpos[self.qids] = prior
            mujoco.mj_forward(self.model, self.data)

    def nearest_pairs(self, q, *, limit=12):
        """Rank real moving robot collision geoms vs tabletop; diagnostic only.

        Preserve the current MjData state. Include geom IDs, body labels,
        type and world-space centers so the correct STL/proxy can be found.
        """
        q=np.asarray(q,dtype=float)
        if q.shape != self.qids.shape or not np.isfinite(q).all():
            raise ValueError("joint vector shape / value mismatch")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        old=self.data.qpos[self.qids].copy()
        try:
            self.data.qpos[self.qids]=q
            mujoco.mj_forward(self.model,self.data)
            results=[]
            for ri in self.robot_ids:
                for ti in self.table_ids:
                    witness=np.zeros(6)
                    gap=float(mujoco.mj_geomDistance(
                        self.model,self.data,int(ri),int(ti),.1,witness))
                    body=int(self.model.geom_bodyid[ri])
                    results.append({
                        "distance_mm":round(gap*1000,5),
                        "robot_geom_id":int(ri),
                        "robot_geom_name":mujoco.mj_id2name(
                            self.model,mujoco.mjtObj.mjOBJ_GEOM,int(ri)),
                        "robot_body":mujoco.mj_id2name(
                            self.model,mujoco.mjtObj.mjOBJ_BODY,body),
                        "robot_geom_type":int(self.model.geom_type[ri]),
                        "robot_geom_center_world_m":
                            self.data.geom_xpos[ri].round(6).tolist(),
                        "table_geom_id":int(ti),
                        "table_geom_name":mujoco.mj_id2name(
                            self.model,mujoco.mjtObj.mjOBJ_GEOM,int(ti)),
                        "table_geom_center_world_m":
                            self.data.geom_xpos[ti].round(6).tolist(),
                        "witness_points_world_m":witness.round(6).tolist(),
                    })
            return sorted(results,key=lambda x:x["distance_mm"])[:limit]
        finally:
            self.data.qpos[self.qids]=old
            mujoco.mj_forward(self.model,self.data)

    def filter_step(self, current, proposed):
        """Backtrack toward current; block if no collision-free positive step."""
        current, proposed = np.asarray(current,float), np.asarray(proposed,float)
        start = self.inspect(current)
        if not start["clearance_satisfied"]:
            return current.copy(), {"status":"unsafe_start", **start}
        if np.allclose(current, proposed, atol=1e-12):
            return current.copy(), {"status":"no_motion", **start}
        for scale in (1., .5, .25, .125, .0625, .03125):
            target = current + scale*(proposed-current)
            checks = [self.inspect(q) for q in interpolate_joint_segment(
                current, target, self.max_delta_rad)]
            if all(item["clearance_satisfied"] for item in checks):
                return target, {"status":"accepted" if scale == 1 else "shortened",
                                "scale":scale,"minimum_distance_mm":min(
                                    item["minimum_distance_mm"] for item in checks),
                                "nearest_pair":min(checks,key=lambda x:x["minimum_distance_mm"])["nearest_pair"],
                                "clearance_satisfied":True}
        return current.copy(), {"status":"blocked","scale":0.,**start}
