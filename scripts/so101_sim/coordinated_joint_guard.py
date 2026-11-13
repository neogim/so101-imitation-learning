"""Local multi-joint action shielding for either SAC or DLS.

This does not train SAC, solve global motion planning, or recover from an
already colliding pose. It explores small, coordinated changes to J2/J3/J4
and only adopts actions that pass the existing swept tabletop collision guard.
All candidates are checked as a *path from the current safe q*, not as
isolated end poses. On a dead end it stops, never teleports.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def joint_candidates(current, desired, low, high, *,
                     joint_indices=(1,2,3), step_rad=.04):
    """Generate coordinated alternative *single-step* joint targets.

    Baseline desired first; then bounded ± changes in shoulder, elbow, wrist
    (one-, two- and three-axis compensation). Does not hard-code a J3 sign.
    """
    q=np.asarray(current,float)
    goal=np.asarray(desired,float)
    low=np.asarray(low,float)
    high=np.asarray(high,float)
    if not (q.shape==goal.shape==low.shape==high.shape and q.ndim==1):
        raise ValueError("All joint vectors must be 1D and equally shaped")
    if not all(np.isfinite(a).all() for a in (q,goal,low,high)):
        raise ValueError("Nonfinite joint values")
    if np.any(q<low-1e-8) or np.any(q>high+1e-8):
        raise ValueError("Current q is outside joint limits")
    if step_rad<=0 or not math.isfinite(step_rad):
        raise ValueError("step_rad must be positive")
    indices=tuple(int(i) for i in joint_indices)
    if len(set(indices))!=len(indices) or any(i<0 or i>=len(q) for i in indices):
        raise ValueError("Invalid joint indices")
    base=np.clip(q+np.clip(goal-q,-step_rad,step_rad),low,high)
    result=[base]
    for n in range(1,len(indices)+1):
        for subset in itertools.combinations(indices,n):
            for signs in itertools.product((-1.,1.),repeat=n):
                candidate=base.copy()
                for axis,sign in zip(subset,signs):
                    candidate[axis]+=sign*step_rad
                candidate=np.clip(candidate,low,high)
                candidate=np.clip(candidate,q-step_rad,q+step_rad)
                if not any(np.allclose(candidate,x,atol=1e-10,rtol=0)
                           for x in result):
                    result.append(candidate)
    return result


def coordinate_step(guard,current,proposed,low,high,goal_cost, *,
                    step_rad=.04,action_weight=.05):
    """Select a swept-safe J2/J3/J4 candidate balancing goal vs policy.

    goal_cost(q): dimensionless nonnegative measure of task performance.
    No endpoint-only collision checks are sufficient. If the *starting*
    pose is unsafe, return it with unsafe_start (no pretend recovery).
    """
    q=np.asarray(current,float)
    proposal=np.asarray(proposed,float)
    start=guard.inspect(q)
    if not start["clearance_satisfied"]:
        return q.copy(),{"status":"unsafe_start",**start}
    if not np.isfinite(action_weight) or action_weight<0:
        raise ValueError("Invalid action_weight")
    best=None
    best_score=float("inf")
    best_info=None
    candidates=joint_candidates(q,proposal,low,high,step_rad=step_rad)
    # Include hold position as a viable fallback, but consider it only if
    # no nontrivial motion is feasible: do not turn every hard pose into a
    # silently chosen no-op.
    for idx,candidate in enumerate(candidates):
        if np.allclose(candidate,q,atol=1e-10,rtol=0):
            continue
        accepted,info=guard.filter_step(q,candidate)
        if info["status"] not in ("accepted","shortened"):
            continue
        if np.allclose(accepted,q,atol=1e-10,rtol=0):
            continue
        task=float(goal_cost(accepted))
        if not math.isfinite(task) or task<0:
            raise ValueError("Invalid goal cost")
        score=task+float(action_weight)*float(
            np.sum(((accepted-proposal)/step_rad)**2))
        if score<best_score:
            best_score=score
            best=accepted
            best_info={**info,"status":("accepted" if idx==0 and
                         info["status"]=="accepted" else "coordinated"),
                       "candidate_index":idx,"candidate_count":len(candidates),
                       "objective":score,
                       "original_proposal_delta_rad":(proposal-q).tolist(),
                       "selected_delta_rad":(accepted-q).tolist()}
    if best is None:
        return q.copy(),{"status":"blocked","candidate_count":len(candidates),
                        "scale":0.0,**start}
    return best,best_info
