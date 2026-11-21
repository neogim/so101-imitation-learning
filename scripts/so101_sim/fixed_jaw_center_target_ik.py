#!/usr/bin/env python3
"""Contact-point IK for a horizontal Muhi.

Target: a FIXED world point on the central cylindrical SIDE, not the
nearest point of the full bottle (which can jump to its end caps).
The candidate fixed-jaw surface point is converted into the jaw body's
local frame before optimization. No physics or actuator motion here.
"""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.optimize import least_squares


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Degenerate grasp direction")
    return vector / norm


def _witness_anchors(model, data, fixed, bottle, goal, moving, limit=8):
    """Select surface points on the FIXED jaw facing the MOVING jaw.

    A fixed-vs-bottle witness can land on the outside of the fixed jaw.
    Instead collect fixed-vs-moving collision witnesses on the open aperture.
    """
    fc = np.mean(np.asarray(data.geom_xpos)[list(fixed)], axis=0)
    mc = np.mean(np.asarray(data.geom_xpos)[list(moving)], axis=0)
    direction = _unit(mc-fc)
    candidates = []
    for gid in fixed:
        body = int(model.geom_bodyid[int(gid)])
        rot = np.asarray(data.xmat[body]).reshape(3,3)
        origin = np.asarray(data.xpos[body])
        for mid in moving:
            witness = np.zeros(6)
            distance = mujoco.mj_geomDistance(
                model, data, int(gid), int(mid), .5, witness)
            if not np.isfinite(distance) or not np.isfinite(witness).all():
                continue
            point = witness[:3].copy()
            if float(np.dot(witness[3:]-point,direction)) <= .001:
                continue
            local = rot.T @ (point-origin)
            candidates.append((float(np.linalg.norm(point-goal)),
                               body,local,int(gid)))
    candidates.sort(key=lambda x:x[0])
    chosen=[]
    for item in candidates:
        if any(item[1]==prev[1] and
               np.linalg.norm(item[2]-prev[2]) < .002 for prev in chosen):
            continue
        chosen.append(item)
        if len(chosen)>=limit: break
    return chosen


def solve(model, data, qadr, bq, seed, aligned, lo, hi,
          xyz, quat, object_axis, fixed, moving, bottle, radius,
          max_j1_deg=65., max_j23_deg=45., max_j4_deg=35.,
          max_j5_deg=35., max_nfev=160, finger_body=None,
          finger_axis_local=None, orientation_tolerance_deg=15.,
          fixed_jaw_floor_gap_mm=None, floor_target_mm=.5):
    """Solve explicit 3D position AND task-oriented end-effector orientation.

    Physics eligibility, real-recorded envelope, collision and full path gates
    MUST still be checked by the caller; this is purely kinematic.
    """
    axis = _unit(object_axis)
    vertical = np.array([0., 0., 1.])
    upright = abs(float(np.dot(axis, vertical))) >= .8
    lateral = _unit(np.cross(axis, [1., 0., 0.] if upright else vertical))
    if finger_body is None or finger_axis_local is None:
        raise ValueError("Explicit finger body and measured finger STL axis are required")
    finger_axis_local = _unit(finger_axis_local)
    widths = np.deg2rad([max_j1_deg, max_j23_deg, max_j23_deg,
                        max_j4_deg, max_j5_deg])
    lower = np.maximum(lo[:5] + .0002, seed - widths)
    upper = np.minimum(hi[:5] - .0002, seed + widths)
    if np.any(upper <= lower):
        raise ValueError("Empty bounded 5-DOF IK range")

    data.qpos[qadr[:5]] = aligned
    data.qpos[bq:bq+3] = xyz
    data.qpos[bq+3:bq+7] = quat
    mujoco.mj_forward(model, data)
    fc = np.mean(np.asarray(data.geom_xpos)[list(fixed)], axis=0)
    # Choose the side already facing the fixed jaw first, then test its opposite.
    signs = [1., -1.] if np.dot(fc - xyz, lateral) >= 0 else [-1., 1.]
    trials = []
    for sign in signs:
        # Each side must start from the same recorded/aligned posture.
        data.qpos[qadr[:5]] = aligned
        data.qpos[qadr[5]] = hi[5]
        data.qpos[bq:bq+3] = xyz
        data.qpos[bq+3:bq+7] = quat
        mujoco.mj_forward(model, data)
        normal = sign * lateral
        goal = np.asarray(xyz) + (radius + .0015) * normal
        for _, body, local, gid in _witness_anchors(
                model, data, fixed, bottle, goal, moving):
            def point_at(q):
                data.qpos[qadr[:5]] = q
                mujoco.mj_forward(model, data)
                return (np.asarray(data.xpos[body]) +
                        np.asarray(data.xmat[body]).reshape(3, 3) @ local)

            def pose_metrics(q):
                point_at(q)
                fc_now = np.mean(np.asarray(data.geom_xpos)[list(fixed)], axis=0)
                mc_now = np.mean(np.asarray(data.geom_xpos)[list(moving)], axis=0)
                vec = mc_now - fc_now
                sep = float(np.linalg.norm(vec))
                aperture = _unit(vec)
                facing = float(np.dot(aperture, -normal))
                along = float(np.dot(np.asarray(xyz)-fc_now, aperture))
                finger_rot = np.asarray(data.xmat[int(finger_body)]).reshape(3, 3)
                finger_world = _unit(finger_rot @ finger_axis_local)
                # Upright Muhi: finger runs approximately parallel to table.
                # Horizontal Muhi: finger runs approximately perpendicular.
                axial_vertical = float(np.dot(finger_world, vertical))
                if upright:
                    # A horizontal jaw-tip axis can point either horizontal way.
                    elevation_error = float(np.degrees(np.arcsin(
                        np.clip(abs(axial_vertical), 0., 1.))))
                else:
                    # For a horizontal object the jaw-tip MUST point down,
                    # not merely be parallel to the table normal.
                    elevation_error = float(np.degrees(np.arccos(
                        np.clip(-axial_vertical, -1., 1.))))
                finger_object_dot = abs(float(np.dot(finger_world, axis)))
                aperture_object_dot = abs(float(np.dot(aperture, axis)))
                finger_object_error = float(np.degrees(np.arcsin(np.clip(
                    finger_object_dot, 0., 1.))))
                aperture_object_error = float(np.degrees(np.arcsin(np.clip(
                    aperture_object_dot, 0., 1.))))
                return (facing, along, sep, elevation_error,
                        finger_object_error, aperture_object_error,
                        finger_world.copy(), aperture.copy())

            def residual(q):
                point = point_at(q)
                (facing, along, sep, elev, finger_cross, aperture_cross,
                 finger_world, aperture) = pose_metrics(q)
                # Strongly prioritize the ACTUAL center-side point (x,y,z).
                # Only penalize angular errors once they exceed a good range;
                # never trade 40 mm of position for a few degrees of attitude.
                terms = [(point-goal)/.001,
                         22.*max(0.,.85-facing),
                         400.*max(0.,.002-along),
                         400.*max(0.,along-sep+.002),
                         .5*max(0.,elev-8.),
                         .4*max(0.,finger_cross-8.),
                         .4*max(0.,aperture_cross-8.)]
                if fixed_jaw_floor_gap_mm is not None and not upright:
                    gap_mm = float(fixed_jaw_floor_gap_mm())
                    terms.extend([1.5*max(0.,gap_mm-1.),
                                  10.*max(0.,-gap_mm)])
                terms.append(np.rad2deg(q-seed)*.001)
                return np.r_[tuple(terms)]

            fit = least_squares(
                residual, np.clip(aligned, lower + 1e-5, upper - 1e-5),
                bounds=(lower, upper), max_nfev=max_nfev,
                ftol=1e-5, xtol=1e-5, gtol=1e-5)
            actual = point_at(fit.x)
            error_mm = float(np.linalg.norm(actual - goal) * 1000.)
            (facing, along, separation, elev, finger_cross,
             aperture_cross, finger_world, aperture_world) = pose_metrics(fit.x)
            inside_valid = bool(facing>=.85 and .002<=along<=separation-.002)
            orientation_valid = bool(
                elev<=orientation_tolerance_deg and
                finger_cross<=orientation_tolerance_deg and
                aperture_cross<=orientation_tolerance_deg)
            floor_gap_mm = (None if fixed_jaw_floor_gap_mm is None
                            else float(fixed_jaw_floor_gap_mm()))
            floor_valid = bool(floor_gap_mm is None or
                               (0. <= floor_gap_mm <= 1.))
            rank = (0 if inside_valid and orientation_valid and floor_valid
                    and error_mm<=2. else 1,
                    25.*error_mm + 100.*max(0.,.85-facing) +
                    1000.*max(0.,.002-along) +
                    1000.*max(0.,along-separation+.002) +
                    3.*elev + 2.*finger_cross + 2.*aperture_cross +
                    (0. if floor_gap_mm is None else
                     5.*max(0.,floor_gap_mm-1.) +
                     30.*max(0.,-floor_gap_mm)))
            trials.append(dict(q=fit.x.copy(), error_mm=error_mm,
                inward_face_valid=inside_valid,
                orientation_valid=orientation_valid,
                fixed_jaw_floor_gap_mm=floor_gap_mm,
                fixed_jaw_floor_near_contact_valid=floor_valid,
                orientation_mode="upright_horizontal_finger" if upright else "horizontal_vertical_finger",
                finger_table_elevation_error_deg=elev,
                finger_to_muhi_axis_perpendicular_error_deg=finger_cross,
                aperture_to_muhi_axis_perpendicular_error_deg=aperture_cross,
                finger_world_axis=finger_world.tolist(),
                aperture_world_axis=aperture_world.tolist(),
                jaw_aperture_alignment_cos=facing,
                object_between_jaws_along_mm=along*1000.,
                jaw_centroid_separation_mm=separation*1000.,
                rank=list(rank),
                sign=int(sign), target_world_m=goal.tolist(),
                fixed_surface_world_m=actual.tolist(),
                fixed_surface_body=int(body), fixed_geom=int(gid),
                fixed_local_m=local.tolist(),
                nfev=int(fit.nfev), solver_success=bool(fit.success)))
    if not trials:
        raise RuntimeError("No fixed-jaw collision witness anchors")
    trials.sort(key=lambda t: tuple(t["rank"]))
    chosen = trials[0]
    return chosen["q"], {k: v for k, v in chosen.items() if k != "q"}, [
        {k: v for k, v in trial.items() if k != "q"} for trial in trials]
