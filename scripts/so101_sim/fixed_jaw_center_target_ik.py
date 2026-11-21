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


def _witness_anchors(model, data, fixed, bottle, goal, moving, limit=4):
    """Candidate actual collision-surface material points, in body coordinates.

    Each witness is frozen to its geom body's frame for the entire IK.
    This avoids differentiating through a discontinuous closest-geom switch.
    """
    fixed_center = np.mean(np.asarray(data.geom_xpos)[list(fixed)], axis=0)
    moving_center = np.mean(np.asarray(data.geom_xpos)[list(moving)], axis=0)
    inward = _unit(moving_center - fixed_center)
    candidates = []
    for gid in fixed:
        body = int(model.geom_bodyid[int(gid)])
        rot = np.asarray(data.xmat[body]).reshape(3, 3)
        body_origin = np.asarray(data.xpos[body])
        for bid in bottle:
            witness = np.zeros(6, dtype=float)
            dist = mujoco.mj_geomDistance(
                model, data, int(gid), int(bid), 0.5, witness)
            if not np.isfinite(dist):
                continue
            point = witness[:3].copy()
            if not np.isfinite(point).all():
                continue
            # Prefer the face pointing from fixed to moving jaw.
            inward_score = float(np.dot(goal - point, inward))
            local = rot.T @ (point - body_origin)
            priority = float(np.linalg.norm(point - goal))
            if inward_score < 0:
                priority += 0.1
            candidates.append((priority, body, local, int(gid)))
    candidates.sort(key=lambda x: x[0])
    chosen = []
    # Different collision patches can provide the same point. Deduplicate.
    for cand in candidates:
        if any(cand[1] == other[1] and np.linalg.norm(cand[2]-other[2]) < .003
               for other in chosen):
            continue
        chosen.append(cand)
        if len(chosen) >= limit:
            break
    return chosen


def solve(model, data, qadr, bq, seed, aligned, lo, hi,
          xyz, quat, object_axis, fixed, moving, bottle, radius,
          max_j1_deg=65., max_j23_deg=45., max_j4_deg=35.,
          max_j5_deg=35., max_nfev=160):
    """Return candidate with smallest explicit central-side 3D position error.

    Physics eligibility, real-recorded envelope, collision and full path gates
    MUST still be checked by the caller; this is purely kinematic.
    """
    axis = _unit(object_axis)
    lateral = _unit(np.cross(axis, [0., 0., 1.]))
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
        normal = sign * lateral
        goal = np.asarray(xyz) + (radius + .0015) * normal
        for _, body, local, gid in _witness_anchors(
                model, data, fixed, bottle, goal, moving):
            def point_at(q):
                data.qpos[qadr[:5]] = q
                mujoco.mj_forward(model, data)
                return (np.asarray(data.xpos[body]) +
                        np.asarray(data.xmat[body]).reshape(3, 3) @ local)

            def residual(q):
                point = point_at(q)
                # Strong direct Cartesian error; weak regularization only.
                return np.r_[(point - goal) / .002,
                             np.rad2deg(q - seed) * .003]

            fit = least_squares(
                residual, np.clip(aligned, lower + 1e-5, upper - 1e-5),
                bounds=(lower, upper), max_nfev=max_nfev,
                ftol=1e-5, xtol=1e-5, gtol=1e-5)
            actual = point_at(fit.x)
            error_mm = float(np.linalg.norm(actual - goal) * 1000.)
            trials.append(dict(q=fit.x.copy(), error_mm=error_mm,
                sign=int(sign), target_world_m=goal.tolist(),
                fixed_surface_world_m=actual.tolist(),
                fixed_surface_body=int(body), fixed_geom=int(gid),
                fixed_local_m=local.tolist(),
                nfev=int(fit.nfev), solver_success=bool(fit.success)))
    if not trials:
        raise RuntimeError("No fixed-jaw collision witness anchors")
    trials.sort(key=lambda t: t["error_mm"])
    chosen = trials[0]
    return chosen["q"], {k: v for k, v in chosen.items() if k != "q"}, [
        {k: v for k, v in trial.items() if k != "q"} for trial in trials]
