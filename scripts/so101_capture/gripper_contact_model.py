"""SIMULATION ONLY. Not wired into HardwareIO/Session; no motor APIs.

Hypothesis: decreasing normalized position closes gripper. All constants below
are synthetic-test parameters, not validated force/contact limits.
"""
import math


class ContactStop(RuntimeError):
    pass


class GripperContactModel:
    def __init__(self):
        self.latched = None
        self.since = None
        self.previous_time = None
        self.open_count = 0
        self.opening = False

    def step(self, *, now, measured, previous_goal, requested, communication_ok=True):
        if not communication_ok or not all(math.isfinite(x) for x in (now, measured, previous_goal, requested)):
            raise ContactStop('Invalid communication/data; no command')
        if not all(0 <= x <= 100 for x in (measured, previous_goal, requested)):
            raise ContactStop('Out of range; no command')
        dt = .05 if self.previous_time is None else now-self.previous_time
        if not 0 < dt <= .075:
            raise ContactStop('Stale/nonmonotonic sample; no command')
        self.previous_time = now
        error = measured-previous_goal
        if abs(error) > 8:
            raise ContactStop('Gripper hard error limit 8 unchanged; no command')
        step = min(dt, .05)*30
        proposed = previous_goal+max(-step,min(step,requested-previous_goal))
        # Freeze BEFORE issuing another closing command that would increase a
        # material position error. False contact is intentionally conservative.
        if self.latched is None and requested < previous_goal and (error >= 3 or measured-proposed >= 3):
            self.latched, self.since = previous_goal, now
        if self.latched is not None:
            # Never interpret position-only contact as proven normal grasp.
            if now-self.since >= .5:
                raise ContactStop('Unverified contact timeout; human intervention required')
            if requested > max(self.latched, previous_goal)+1:
                self.open_count += 1
            else:
                self.open_count = 0
            if self.open_count >= 2:
                self.opening = True
            if self.opening:
                # Opening is explicit and slew limited; never automatically reclose.
                return max(previous_goal, proposed)
            return self.latched
        return proposed
