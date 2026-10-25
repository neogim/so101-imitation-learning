"""Offline, advisory-only contact estimate from j6_telemetry.json rows.

No servo writes. Raw Present_Current units are not amperes; decoded Present_Load
is not fingertip force. Uncalibrated, stale or incomplete samples are invalid.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ContactCalibration:
    free_current_raw: float
    free_load_raw: float
    current_rise_raw: float
    load_change_raw: float
    min_position_lag: float
    max_age_s: float


def estimate_contact(row, *, now, calibration=None):
    if calibration is None:
        return {'valid': False, 'contact_suspected': False, 'reason': 'uncalibrated'}
    try:
        values = [now, row['completed_monotonic'], row['Present_Current'],
                  row['Present_Load'], row['measured_position'], row['previous_sent_goal'],
                  row['leader_requested_this_cycle']]
        limits = [calibration.free_current_raw, calibration.free_load_raw,
                  calibration.current_rise_raw, calibration.load_change_raw,
                  calibration.min_position_lag, calibration.max_age_s]
        if not all(math.isfinite(x) for x in values + limits) or any(x <= 0 for x in limits[2:]):
            raise ValueError('invalid telemetry or calibration')
        age = now - row['completed_monotonic']
        if age < 0 or age > calibration.max_age_s:
            return {'valid': False, 'contact_suspected': False, 'reason': 'stale'}
        closing = row['leader_requested_this_cycle'] < row['previous_sent_goal']
        evidence = (row['Present_Current'] - calibration.free_current_raw >= calibration.current_rise_raw
                    and abs(row['Present_Load'] - calibration.free_load_raw) >= calibration.load_change_raw
                    and row['measured_position'] - row['previous_sent_goal'] >= calibration.min_position_lag)
        return {'valid': True, 'contact_suspected': bool(closing and evidence),
                'reason': 'candidate_contact' if closing and evidence else 'insufficient_evidence'}
    except (KeyError, TypeError, ValueError):
        return {'valid': False, 'contact_suspected': False, 'reason': 'missing_or_invalid'}
