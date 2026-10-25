"""Pure, offline-testable proposed-action guard. No motor imports or I/O."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


NAMES = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
         "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")
UNITS = ("degrees",) * 5 + ("normalized_0_100",)


@dataclass(frozen=True)
class GuardConfig:
    limits: tuple[tuple[float, float], ...]
    speed_per_s: tuple[float, ...]
    gripper_closing_lead: float | None = None
    period_s: float = 0.05
    max_sample_age_s: float = 0.10
    tracking_error_limit: float = 8.0

    def __post_init__(self):
        if len(self.limits) != 6 or len(self.speed_per_s) != 6:
            raise ValueError("Six joint limits and speeds are required")
        if any(not all(map(math.isfinite, pair)) or pair[0] >= pair[1] for pair in self.limits):
            raise ValueError("Invalid joint limits")
        if any(not math.isfinite(x) or x <= 0 for x in self.speed_per_s):
            raise ValueError("Invalid joint speed")
        if not (0 < self.period_s <= 0.05 and 0 < self.max_sample_age_s <= 0.10):
            raise ValueError("Invalid period or freshness limit")
        if self.tracking_error_limit != 8.0:
            raise ValueError("Existing tracking error threshold must remain 8")
        if self.gripper_closing_lead is not None and (
            not math.isfinite(self.gripper_closing_lead) or self.gripper_closing_lead < 0
        ):
            raise ValueError("Invalid closing lead")
        if self.limits[5][0] < 0 or self.limits[5][1] > 100:
            raise ValueError("Gripper limits must stay within normalized 0..100")


class ActionGuard:
    def __init__(self, config: GuardConfig):
        self.config = config
        self.last_goal: np.ndarray | None = None
        self.latched_reason: str | None = None

    def reset_for_new_episode(self, measured):
        state = self._vector(measured)
        for i, (lower, upper) in enumerate(self.config.limits):
            if not lower <= state[i] <= upper:
                raise ValueError(f"Initial {NAMES[i]} outside configured limits")
        self.last_goal = state.copy()
        self.latched_reason = None

    @staticmethod
    def _vector(values):
        result = np.asarray(values, dtype=np.float64).reshape(-1)
        if result.shape != (6,) or not np.isfinite(result).all():
            raise ValueError("Six finite joint values are required")
        return result

    def stop(self, reason):
        self.latched_reason = self.latched_reason or reason
        return None, [self.latched_reason]

    def propose(self, predicted, measured, *, sample_age_s=0.0, communication_ok=True):
        """Never sends an Action. Caller must halt on a None result; no auto recovery."""
        if self.latched_reason is not None:
            return None, [self.latched_reason]
        if not communication_ok:
            return self.stop("communication_fault")
        if not math.isfinite(sample_age_s) or sample_age_s < 0 or sample_age_s > self.config.max_sample_age_s:
            return self.stop("stale_measurement")
        try:
            requested, actual = self._vector(predicted), self._vector(measured)
        except ValueError:
            return self.stop("nonfinite_or_wrong_shape")
        if self.last_goal is None:
            return self.stop("initial_goal_missing")
        errors = np.abs(actual - self.last_goal)
        offenders = [f"tracking_error:{NAMES[i]}:{errors[i]:.6f}:{UNITS[i]}"
                     for i in range(6) if errors[i] > self.config.tracking_error_limit]
        if offenders:
            return self.stop(";".join(offenders))
        reasons = []
        bounded = requested.copy()
        for i, (lower, upper) in enumerate(self.config.limits):
            clipped = np.clip(bounded[i], lower, upper)
            if clipped != bounded[i]:
                reasons.append(f"range:{NAMES[i]}")
            bounded[i] = clipped
        # Range clamp and optional measured-position closing lead are independent.
        if self.config.gripper_closing_lead is not None and bounded[5] < self.last_goal[5]:
            floor = min(self.last_goal[5], actual[5] - self.config.gripper_closing_lead)
            if bounded[5] < floor:
                bounded[5] = floor
                reasons.append("gripper_closing_lead")
        max_delta = np.asarray(self.config.speed_per_s) * self.config.period_s
        limited = self.last_goal + np.clip(bounded - self.last_goal, -max_delta, max_delta)
        for i in range(6):
            if abs(limited[i] - bounded[i]) > 1e-8:
                reasons.append(f"speed:{NAMES[i]}")
        self.last_goal = limited.copy()
        # Position lag is an observation, never proof of safe grasp force.
        if abs(actual[5] - limited[5]) >= 6:
            reasons.append("contact_or_lag_warning")
        return limited, reasons
