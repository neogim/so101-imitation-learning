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

def find_static_safe_seed(guard, source, low, high, *, max_delta_rad=.12,
                          grid_offsets=(-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.),
                          indices=(1,2,3), tcp_position=None,
                          tcp_at=None, position_weight=1.):
    """Search a nearby *static* table-clear pose for simulation reinitialization.

    This is NOT a collision-free transition from the unsafe recorded state.
    Candidate coordinates include shoulder/elbow/wrist (J2/J3/J4). Other joints
    remain unchanged. URDF clipping is explicit. For identical clearance,
    prefer the least joint adjustment plus modest TCP displacement.
    """
    q0=np.asarray(source,float)
    low=np.asarray(low,float)
    high=np.asarray(high,float)
    if q0.ndim!=1 or not(q0.shape==low.shape==high.shape):
        raise ValueError("Invalid q shapes")
    if not all(np.isfinite(x).all() for x in (q0,low,high)):
        raise ValueError("Invalid q values")
    if not(0<max_delta_rad<=.5):
        raise ValueError("Invalid search radius")
    indices=tuple(indices)
    if len(set(indices))!=len(indices) or not indices:
        raise ValueError("Invalid indices")
    if any(i<0 or i>=len(q0) for i in indices):
        raise ValueError("Invalid search index")
    base=np.clip(q0,low,high)
    # Evaluate all static poses before scoring; no force or dynamics.
    checks=0
    safe=0
    best=None
    best_cost=float("inf")
    best_report=None
    for factors in itertools.product(grid_offsets,repeat=len(indices)):
        candidate=base.copy()
        for index,f in zip(indices,factors):
            candidate[index]+=float(f)*max_delta_rad
        candidate=np.clip(candidate,low,high)
        result=guard.inspect(candidate)
        checks+=1
        if not result["clearance_satisfied"]:
            continue
        safe+=1
        delta=candidate-base
        cost=float(np.dot(delta,delta))
        if tcp_position is not None:
            if tcp_at is None:
                raise ValueError("tcp_at callback required with tcp_position")
            difference=np.asarray(tcp_at(candidate),float)-np.asarray(tcp_position,float)
            cost+=float(position_weight)*float(np.dot(difference,difference))
        if cost<best_cost:
            best=candidate.copy()
            best_cost=cost
            best_report=result
    return best,{
        "status":"FOUND_STATIC_SAFE_SIM_INITIALIZATION" if best is not None
                 else "NO_STATIC_SAFE_SEED_IN_GRID",
        "evaluated_configurations":checks,
        "safe_configurations":safe,
        "original_q_rad":q0.tolist(),
        "urdf_clipped_q_rad":base.tolist(),
        "new_start_q_rad":best.tolist() if best is not None else None,
        "joint_adjustment_from_clipped_deg":
            np.degrees(best-base).tolist() if best is not None else None,
        "objective":best_cost if best is not None else None,
        "clearance":best_report,
        "warning":"Static simulation reset ONLY. The motion from the old "
                   "penetrating or out-of-limit home to this new seed was "
                   "NOT planned or validated. This is a new comparison baseline.",
    }
