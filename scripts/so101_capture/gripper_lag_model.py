"""SIMULATION ONLY: tentative lag-aware goal limiter, NOT imported by recorder.
Position error is not force. Slow free motion and an obstruction remain ambiguous.
"""
import math
from gripper_contact_model import ContactStop


class LagAwareContactModel:
    def __init__(self):
        self.previous_time = None
        self.previous_position = None
        self.latched = None
        self.since = None
        self.suspect_count = 0
        self.open_count = 0
        self.opening = False
        self.failed = False

    def stop(self, message):
        self.failed = True
        raise ContactStop(message)

    def step(self, *, now, measured, previous_goal, requested, communication_ok=True):
        if self.failed:
            raise ContactStop('Fault latched; no automatic recovery')
        if not communication_ok or not all(math.isfinite(v) for v in (now,measured,previous_goal,requested)):
            self.stop('Communication/data failure')
        if not all(0 <= v <= 100 for v in (measured,previous_goal,requested)):
            self.stop('Position range failure')
        dt=.05 if self.previous_time is None else now-self.previous_time
        if not 0 < dt <= .075:
            self.stop('Stale sample')
        velocity=None if self.previous_position is None else (measured-self.previous_position)/dt
        self.previous_time,self.previous_position=now,measured
        error=measured-previous_goal
        if abs(error)>8:
            self.stop('Hard error limit 8')
        step=min(dt,.05)*30
        proposed=previous_goal+max(-step,min(step,requested-previous_goal))
        closing=requested<previous_goal
        # Limit how far a NEW closing goal can lead measured position, without
        # raising an already-issued goal (which would automatically open).
        if closing:
            proposed=max(proposed,min(previous_goal,measured-3))
        suspect=closing and error>=3-1e-6 and velocity is not None and velocity>=-5
        self.suspect_count=self.suspect_count+1 if suspect else 0
        if self.latched is None and self.suspect_count>=2:
            self.latched,self.since=previous_goal,now
        if self.latched is not None:
            if now-self.since>=.5:
                self.stop('Unverified contact timeout')
            self.open_count=self.open_count+1 if requested>max(self.latched,previous_goal)+1 else 0
            if self.open_count>=2:self.opening=True
            return max(previous_goal,proposed) if self.opening else self.latched
        return proposed
