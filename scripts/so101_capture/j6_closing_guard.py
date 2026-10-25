"""J6-only position lead guard. Motor load/current never change a Goal here.

Smaller normalized positions close the gripper. A held Goal may still exert force;
this guard limits additional closing travel, not gripping force.
"""
import math


def limit_j6_closing(*, measured, previous_goal, requested, proposed_goal, max_lead):
    """Return (goal, reason) after the existing slew limiter has run."""
    if not all(math.isfinite(x) for x in
               (measured, previous_goal, requested, proposed_goal, max_lead)) or not 0 < max_lead < 8:
        raise ValueError('J6 closing guard requires finite positions and 0 < lead < tracking limit 8')
    if requested >= previous_goal or proposed_goal >= previous_goal:
        return proposed_goal, 'opening_or_hold'
    bounded = max(proposed_goal, measured - max_lead)
    goal = min(previous_goal, bounded)  # Never open automatically to correct an old Goal.
    return goal, 'closing_lead_limited' if goal > proposed_goal else 'closing_within_lead'
