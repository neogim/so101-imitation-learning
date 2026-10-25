"""Offline-testable, single-writer gate for a future same-connection handoff.

This module has no serial-port imports or hardware entry point. The caller must
run its own fresh register checks and feed one verified snapshot to ``arm_act``.
"""

from __future__ import annotations

import math

from act_action_guard import NAMES, ActionGuard, GuardConfig


class HandoffFault(RuntimeError):
    pass


class HandoffComplete(RuntimeError):
    """A first-motion trial limit was reached before another Goal write."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class HandoffGate:
    """Only this object may call the supplied Follower Goal writer.

    All methods must run on the same 20 Hz control thread. No queued Goal
    commands or second controller may bypass ``send``.
    """

    def __init__(self, writer, limits):
        self.writer = writer
        self.limits = tuple(tuple(limits[name]) for name in NAMES)
        self.owner = "TELEOP"
        self.last_sent = None
        self.guard = None
        self.fault_reason = None

    def send_teleop(self, goal):
        if self.owner != "TELEOP":
            raise HandoffFault("Leader Goal blocked: owner is not TELEOP")
        if set(goal) != set(NAMES) or not all(math.isfinite(goal[n]) for n in NAMES):
            return self.fault("invalid Leader Goal")
        try:
            sent = self.writer(dict(goal))
            if set(sent) != set(NAMES):
                raise HandoffFault("Goal writer returned incomplete Action")
            self.last_sent = dict(sent)
            return sent
        except BaseException as exc:
            self.fault(f"Leader Goal write failed: {exc}")
            raise

    def begin_handoff(self):
        if self.owner != "TELEOP" or self.last_sent is None:
            raise HandoffFault("handoff requires active TELEOP and a sent Goal")
        # From this point, neither source can write until all checks pass.
        self.owner = "CHECK"

    def verify_handoff(self, *, goal, present, torque, status, calibration_ok,
                       sample_age_s, communication_ok, queued_goals=0):
        if self.owner != "CHECK":
            raise HandoffFault("handoff verification requires exclusive CHECK state")
        try:
            if not calibration_ok or not communication_ok or queued_goals:
                raise HandoffFault("calibration, communication or pending Goal invalid")
            if not 0 <= sample_age_s <= 0.10:
                raise HandoffFault("stale handoff snapshot")
            if any(set(row) != set(NAMES) for row in (goal, present, torque, status)):
                raise HandoffFault("incomplete six-axis snapshot")
            if any(torque[n] != 1 or status[n] != 0 for n in NAMES):
                raise HandoffFault("torque or status invalid")
            for i, name in enumerate(NAMES):
                lo, hi = self.limits[i]
                if not all(math.isfinite(v) for v in (goal[name], present[name])):
                    raise HandoffFault(f"{name}: nonfinite Goal/Present")
                if not lo <= goal[name] <= hi or not lo <= present[name] <= hi:
                    raise HandoffFault(f"{name}: outside calibrated range")
                if abs(goal[name] - present[name]) > 8:
                    raise HandoffFault(f"{name}: tracking error >8")
                if abs(goal[name] - self.last_sent[name]) > 0.5:
                    raise HandoffFault(f"{name}: Goal readback differs from last teleop send")
            return True
        except BaseException as exc:
            self.fault(f"handoff check failed: {exc}")
            raise

    def arm_act(self, *, speed_per_s=None, max_displacement=None, **snapshot):
        self.verify_handoff(**snapshot)
        speeds = (30.0,) * 6 if speed_per_s is None else tuple(speed_per_s)
        guard = ActionGuard(GuardConfig(limits=self.limits,
                                       speed_per_s=speeds,
                                       tracking_error_limit=8.0))
        guard.reset_for_new_episode([snapshot['goal'][n] for n in NAMES])
        self.guard = guard
        self.act_origin = tuple(snapshot['goal'][n] for n in NAMES)
        self.max_displacement = max_displacement
        self.total_displacement = [0.0] * 5
        self.owner = "ACT"

    def send_act(self, predicted, present, *, sample_age_s, communication_ok=True):
        if self.owner != "ACT":
            raise HandoffFault("ACT Goal blocked: ACT does not own writer")
        if set(present) != set(NAMES):
            self.fault("incomplete ACT Present_Position")
        previous_goal = self.guard.last_goal.copy()
        proposed, reasons = self.guard.propose(predicted, [present[n] for n in NAMES],
                                               sample_age_s=sample_age_s,
                                               communication_ok=communication_ok)
        if proposed is None:
            self.fault("ACT guard: " + ", ".join(reasons))
            raise HandoffFault(self.fault_reason)
        if self.max_displacement is not None and any(
            abs(proposed[i] - self.act_origin[i]) > self.max_displacement + 1e-9
            for i in range(5)
        ):
            self.stop()
            raise HandoffComplete("cumulative_displacement_limit")
        if self.max_displacement is not None and any(
            self.total_displacement[i] + abs(proposed[i]-previous_goal[i])
            > self.max_displacement + 1e-9 for i in range(5)
        ):
            self.stop()
            raise HandoffComplete("cumulative_displacement_limit")
        if self.max_displacement is not None and abs(proposed[5]-self.act_origin[5]) > 1e-9:
            self.fault("first-motion gripper Goal must remain fixed")
        try:
            sent = self.writer(dict(zip(NAMES, map(float, proposed))))
            if set(sent) != set(NAMES) or not all(math.isfinite(sent[n]) for n in NAMES):
                raise HandoffFault("ACT Goal writer returned incomplete Action")
            if self.max_displacement is not None and (
                any(abs(sent[NAMES[i]]-self.act_origin[i]) > self.max_displacement + 1e-9
                    for i in range(5)) or abs(sent[NAMES[5]]-self.act_origin[5]) > 1e-9
            ):
                raise HandoffFault("first-motion sent Goal exceeded displacement or gripper hold limit")
            if self.max_displacement is not None and any(
                abs(sent[n]-proposed[i]) > 1e-9 for i, n in enumerate(NAMES)
            ):
                raise HandoffFault("first-motion Goal writer changed guarded Action")
            if self.max_displacement is not None:
                for i in range(5):
                    self.total_displacement[i] += abs(sent[NAMES[i]]-previous_goal[i])
            self.last_sent = dict(sent)
            self.guard.last_goal = self.guard._vector([sent[n] for n in NAMES])
            return sent, reasons
        except BaseException as exc:
            self.fault(f"ACT Goal write failed: {exc}")
            raise

    def fault(self, reason):
        self.fault_reason = self.fault_reason or reason
        self.owner = "FAULT"
        # Deliberately no Goal or torque command during fault cleanup.
        raise HandoffFault(self.fault_reason)

    def stop(self):
        """Normal trial timeout: block both writers without a final Goal."""
        self.owner = "STOPPED"
