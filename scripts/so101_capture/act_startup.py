"""ACT startup state machine. No CLI or hardware connection is provided here.

The recording startup order is retained, but SO101Follower.configure() is not
called: its torque_disabled() context manager re-enables torque in ``finally``
even when a configuration write fails. A future, separately approved hardware
runner may inject the LeRobot-backed adapter below. Importing this module does
not access a servo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time


MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
REGISTERS = ("Goal_Position", "Present_Position", "Torque_Enable", "Status")


class StartupFault(RuntimeError):
    pass


@dataclass
class StartupResult:
    checkpoints: list[dict]
    ready: bool = False
    latched_fault: str | None = None


class ACTStartup:
    """Single-use fail-closed startup using an injected, six-axis device.

    The device owns serial access. Its methods must never retry an ambiguous
    Goal write, and ``close()`` must not send a Goal or change torque on fault.
    """

    def __init__(self, device, *, diagnostics_path=None, max_position_change_raw=5,
                 max_goal_error_raw=5, settle_after_goal_s=0.02, sleep=time.sleep):
        self.device = device
        self.diagnostics_path = Path(diagnostics_path) if diagnostics_path is not None else None
        self.max_position_change_raw = max_position_change_raw
        self.max_goal_error_raw = max_goal_error_raw
        self.settle_after_goal_s = settle_after_goal_s
        self.sleep = sleep
        self.result = StartupResult([])
        self._started = False
        self._aligned = False
        self._log_ready = False
        self.stage = "created"
        self.fault_readback = None

    def _persist(self):
        if not self._log_ready:
            return
        validated = [row for row in self.result.checkpoints if row.get("validation") == "ok"]
        payload = dict(schema_version=1, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                       stage=self.stage, **asdict(self.result),
                       last_confirmed_raw=validated[-1]["raw"] if validated else None,
                       fault_readback=self.fault_readback,
                       no_new_action_after_fault=True,
                       warning="Software stop sends no new Goal. Torque may remain on and holding or motion is not guaranteed.")
        tmp = self.diagnostics_path.with_name(self.diagnostics_path.name + ".tmp")
        with tmp.open("w") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.diagnostics_path)

    def _fail(self, reason):
        self.result.latched_fault = self.result.latched_fault or str(reason)
        self.result.ready = False
        if self.result.checkpoints and self.result.checkpoints[-1].get("validation") == "pending":
            self.result.checkpoints[-1]["validation"] = "failed"
        self._persist()
        raise StartupFault(self.result.latched_fault)

    def _snapshot(self, phase, *, expected_torque, baseline=None, goal_target=None,
                  allow_stale_goal=False):
        # Each register is read serially; this is not an atomic hardware snapshot.
        self.stage = phase
        read_started = time.perf_counter()
        raw = self.device.read_raw()
        row = {"phase": phase, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
               "read_started_monotonic": read_started,
               "raw": raw,
               "raw_goal_present_gap": None,
               "validation": "pending"}
        self.result.checkpoints.append(row)
        self._persist()
        for register in REGISTERS:
            values = raw.get(register)
            if not isinstance(values, dict) or set(values) != set(MOTORS):
                self._fail(f"{phase}: incomplete {register}")
            if any(not isinstance(v, int) for v in values.values()):
                self._fail(f"{phase}: noninteger {register}")
        row["raw_goal_present_gap"] = {
            name: raw["Goal_Position"][name]-raw["Present_Position"][name]
            for name in MOTORS
        }
        limits = self.device.calibration_bounds_raw()
        if set(limits) != set(MOTORS) or not self.device.calibration_matches():
            self._fail(f"{phase}: calibration mismatch")
        for name in MOTORS:
            position = raw["Present_Position"][name]
            low, high = limits[name]
            if not all(math.isfinite(v) for v in (low, high)) or not low <= position <= high:
                self._fail(f"{phase}: {name} outside calibrated raw range")
            # Before synchronization, disagreement is expected; a stale Goal
            # outside calibration is still unsafe if torque turns on during
            # the write, as happened in the earlier single-axis trial.
            if not allow_stale_goal and not low <= raw["Goal_Position"][name] <= high:
                self._fail(f"{phase}: {name} stored Goal outside calibrated raw range")
            if raw["Torque_Enable"][name] != expected_torque or raw["Status"][name] != 0:
                self._fail(f"{phase}: {name} torque/status unexpected")
            if baseline is not None and abs(position - baseline["Present_Position"][name]) > self.max_position_change_raw:
                self._fail(f"{phase}: {name} moved unexpectedly")
            if goal_target is not None and abs(raw["Goal_Position"][name] - goal_target[name]) > self.max_goal_error_raw:
                self._fail(f"{phase}: {name} Goal readback mismatch")
            if goal_target is not None and abs(position - raw["Goal_Position"][name]) > self.max_goal_error_raw:
                self._fail(f"{phase}: {name} Goal/Present mismatch after synchronization")
        row["validation"] = "ok"
        self._persist()
        return raw

    def run(self, *, operator_approved=False, arm_supported=False, workspace_clear=False):
        return self._run(operator_approved=operator_approved, arm_supported=arm_supported,
                         workspace_clear=workspace_clear, allow_hardware=False)

    def run_hardware(self, *, operator_approved=False, arm_supported=False, workspace_clear=False):
        """Separately armed entry point; caller must also gate physical startup."""
        return self._run(operator_approved=operator_approved, arm_supported=arm_supported,
                         workspace_clear=workspace_clear, allow_hardware=True)

    def run_hardware_goal_sync_only(self, *, operator_approved=False, arm_supported=False,
                                    workspace_clear=False, power_stop_ready=False):
        """One broadcast Goal write; no Leader, configure, torque command or Alignment."""
        if not power_stop_ready:
            raise StartupFault("Independent servo-power cutoff required for Goal-only trial")
        return self._run(operator_approved=operator_approved, arm_supported=arm_supported,
                         workspace_clear=workspace_clear, allow_hardware=True,
                         goal_sync_only=True)

    def _run(self, *, operator_approved, arm_supported, workspace_clear, allow_hardware,
             goal_sync_only=False):
        if isinstance(self.device, LeRobotStartupDevice) and not allow_hardware:
            raise StartupFault("Hardware startup is disabled until an independently approved staged runner exists")
        if isinstance(self.device, LeRobotStartupDevice) and self.device.follower_only and not goal_sync_only:
            raise StartupFault("Follower-only adapter cannot configure or align")
        if self._started:
            self._fail("startup cannot be retried or automatically rearmed")
        if self.diagnostics_path is not None:
            if not self.diagnostics_path.parent.is_dir():
                raise FileNotFoundError(self.diagnostics_path.parent)
            # Exclusive create prevents a concurrent trial from replacing an
            # existing diagnostic file before any motor-related operation.
            with self.diagnostics_path.open("x"):
                pass
            self._log_ready = True
        self._started = True
        self.stage = "authorization"
        self._persist()
        try:
            if not (operator_approved and arm_supported and workspace_clear):
                self._fail("motor-activation approval, arm support and clear workspace required")
            # The read-only preflight used a separate serial connection. Check
            # again here before any configuration or Goal-related write.
            self._snapshot("before_initialization", expected_torque=0,
                           allow_stale_goal=goal_sync_only)
            if goal_sync_only:
                # No torque-disable write: that itself could change actuator
                # state. Both reads must show torque already disabled.
                before = self._snapshot("before_goal_write", expected_torque=0,
                                        allow_stale_goal=True)
                target = before["Present_Position"].copy()
                if time.perf_counter()-self.result.checkpoints[-1]["read_started_monotonic"] > .10:
                    self._fail("Present sample older than 100 ms before Goal write")
                self.stage = "broadcast_goal_write"
                self.device.write_present_as_goals(target)
                for phase, delay in (("after_goal_write", 0),
                                     ("after_goal_20ms", .02),
                                     ("after_goal_100ms", .08),
                                     ("after_goal_500ms", .40)):
                    if delay:
                        self.sleep(delay)
                    self._snapshot(phase, expected_torque=0,
                                   baseline=before, goal_target=target)
                self.stage = "goal_sync_only_complete"
                self._persist()
                return self.result  # ready=False: never arms Alignment or Action
            # The recording route checks both calibrations before configuration.
            self.stage = "initial_alignment_check"
            self.device.check_initial_alignment()
            self.stage = "configure_leader"
            self.device.configure_leader()
            self.stage = "disable_follower_torque"
            self.device.disable_follower_torque()
            before = self._snapshot("before_goal_write", expected_torque=0)
            target = before["Present_Position"].copy()
            # Same six-axis LeRobot sync_write as record_episode.start_hardware().
            self.stage = "broadcast_goal_write"
            self.device.write_present_as_goals(target)
            after = self._snapshot("after_goal_write", expected_torque=0,
                                   baseline=before, goal_target=target)
            # The incident's torque change was seen several milliseconds after
            # the first readback. Observe again after a short no-write interval.
            self.sleep(self.settle_after_goal_s)
            self._snapshot("before_configure", expected_torque=0,
                           baseline=before, goal_target=target)
            # No SO101Follower.configure(): its finally block enables torque on error.
            self.stage = "configure_follower_without_enable"
            self.device.configure_follower_without_enable()
            self._snapshot("after_configure_before_enable", expected_torque=0,
                           baseline=before, goal_target=target)
            if not self.device.communication_healthy():
                self._fail("communication fault before torque enable")
            # This is an explicit motor-activation boundary, never implicit cleanup.
            self.stage = "explicit_torque_enable"
            self.device.enable_follower_torque()
            self._snapshot("after_configure", expected_torque=1,
                           baseline=before, goal_target=target)
            self._snapshot("before_alignment", expected_torque=1,
                           baseline=before, goal_target=target)
            self.stage = "recording_alignment"
            self.device.align_with_recording_routine()
            self._aligned = True
            self.check_before_action()
            self.result.ready = True
            self.stage = "ready_for_act_action"
            self._persist()
            return self.result
        except BaseException as exc:
            self.result.latched_fault = self.result.latched_fault or f"{type(exc).__name__}: {exc}"
            self.result.ready = False
            failed_stage = self.stage
            # One diagnostic read only; its success never clears the latch or
            # resumes control. If it fails, current torque state is UNKNOWN.
            try:
                self.fault_readback = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
                                           raw=self.device.read_raw())
            except BaseException as read_exc:
                self.fault_readback = dict(raw=None,
                    error=f"{type(read_exc).__name__}: {read_exc}", current_state="unknown")
            self.stage = "fault_stopped_from_" + failed_stage
            self._persist()
            # No Goal, torque, or calibration write in fault cleanup.
            raise

    def check_before_action(self):
        if not self._aligned:
            self._fail("alignment not complete; Action forbidden")
        if self.result.latched_fault:
            self._fail("startup fault latched; Action forbidden")
        try:
            self._snapshot("before_first_act_action", expected_torque=1)
            state = self.device.read_calibrated_state()
            goal = self.device.read_calibrated_goal()
            if (set(state) != set(MOTORS) or set(goal) != set(MOTORS)
                    or any(not math.isfinite(v) for v in (*state.values(), *goal.values()))):
                self._fail("ACT state/Goal missing or nonfinite")
            for name in MOTORS:
                if abs(state[name] - goal[name]) > 8:
                    self._fail(f"{name}: tracking error >8 before first ACT Action")
            if not self.device.communication_healthy():
                self._fail("communication fault latched; Action forbidden")
            return state, goal
        except BaseException as exc:
            self.result.latched_fault = self.result.latched_fault or f"{type(exc).__name__}: {exc}"
            self.result.ready = False
            self._persist()
            raise


def make_act_guard(limits, initial_state, initial_goal):
    """Reuse the existing 30 units/s, tracking-error-8 guard; no Action is sent."""
    from act_action_guard import ActionGuard, GuardConfig

    names = tuple(name + ".pos" for name in MOTORS)
    bounds = tuple(tuple(limits[name]) for name in names)
    guard = ActionGuard(GuardConfig(limits=bounds, speed_per_s=(30.0,) * 6,
                                    gripper_closing_lead=None, tracking_error_limit=8.0))
    if set(initial_state) != set(MOTORS) or set(initial_goal) != set(MOTORS):
        raise ValueError("Six measured and Goal positions are required")
    if any(not math.isfinite(v) for v in (*initial_state.values(), *initial_goal.values())):
        raise ValueError("Finite measured and Goal positions are required")
    if any(abs(initial_state[name] - initial_goal[name]) > 8 for name in MOTORS):
        raise ValueError("Initial Goal/Present tracking error >8")
    # The first slew limit must start from the *actual previous Goal*, not
    # from measured position when those differ within the allowed threshold.
    guard.reset_for_new_episode([initial_goal[name] for name in MOTORS])
    return guard


class LeRobotStartupDevice:
    """Adapter for an already-open, separately authorized Leader/Follower pair.

    This class has no executable entry point and never opens a serial port.
    The caller must hold the arm, verify the workspace and explicitly authorize
    motor activation before calling ACTStartup.run().
    """

    def __init__(self, follower, leader, *, alignment_timeout=60, alignment_tolerance=2.0,
                 follower_only=False):
        self.follower, self.leader = follower, leader
        self.alignment_timeout = alignment_timeout
        self.alignment_tolerance = alignment_tolerance
        self.target = None
        self.communication_latched = False
        self.follower_only = follower_only

    def _read(self, bus, register, *, normalize=False):
        try:
            return bus.sync_read(register, normalize=normalize, num_retry=0)
        except Exception:
            self.communication_latched = True
            raise

    def read_raw(self):
        return {name: self._read(self.follower.bus, name) for name in REGISTERS}

    def calibration_matches(self):
        try:
            return self.follower.is_calibrated and (
                self.follower_only or self.leader.is_calibrated)
        except Exception:
            self.communication_latched = True
            raise

    def calibration_bounds_raw(self):
        return {name: (self.follower.bus.calibration[name].range_min,
                       self.follower.bus.calibration[name].range_max) for name in MOTORS}

    def check_initial_alignment(self):
        from record_episode import validate_calibrations

        validate_calibrations((self.leader, self.follower))
        current = self._read(self.follower.bus, "Present_Position", normalize=True)
        self.target = self._read(self.leader.bus, "Present_Position", normalize=True)
        for name in MOTORS:
            if not math.isfinite(current[name]) or not math.isfinite(self.target[name]):
                raise StartupFault(f"{name}: nonfinite initial position")
            limit = 100 if name == "gripper" else 120
            if abs(current[name] - self.target[name]) > limit:
                raise StartupFault(f"{name}: initial pose mismatch")

    def configure_leader(self):
        self.leader.configure()

    def disable_follower_torque(self):
        self.follower.bus.disable_torque(num_retry=0)

    def write_present_as_goals(self, target):
        # No retry or status ACK; a failure is terminal for this startup.
        self.follower.bus.sync_write("Goal_Position", target, normalize=False, num_retry=0)

    def configure_follower_without_enable(self):
        # Mirrors SO101Follower.configure() in local LeRobot v0.6.2, omitting
        # torque_disabled()'s unconditional enable in its finally block.
        from lerobot.motors.feetech import OperatingMode

        bus = self.follower.bus
        bus.disable_torque(num_retry=0)
        original_write = bus.write

        def checked_write(*args, **kwargs):
            result = original_write(*args, **kwargs)
            # Startup is slow by design. A change after any config write stops
            # the remaining writes rather than waiting for configure() to end.
            for register in ("Torque_Enable", "Status"):
                values = self._read(bus, register)
                if set(values) != set(MOTORS) or any(v != 0 for v in values.values()):
                    raise StartupFault(f"{register} changed during Follower configuration")
            return result

        bus.write = checked_write
        try:
            bus.configure_motors()
            for motor in bus.motors:
                bus.write("Operating_Mode", motor, OperatingMode.POSITION.value, num_retry=0)
                bus.write("P_Coefficient", motor, self.follower.config.position_p_coefficient, num_retry=0)
                bus.write("I_Coefficient", motor, self.follower.config.position_i_coefficient, num_retry=0)
                bus.write("D_Coefficient", motor, self.follower.config.position_d_coefficient, num_retry=0)
                if motor == "gripper":
                    bus.write("Max_Torque_Limit", motor, 500, num_retry=0)
                    bus.write("Protection_Current", motor, 250, num_retry=0)
                    bus.write("Overload_Torque", motor, 25, num_retry=0)
        finally:
            # Restore the method; this finally block sends no servo command.
            bus.write = original_write

    def enable_follower_torque(self):
        self.follower.bus.enable_torque(num_retry=0)

    def align_with_recording_routine(self):
        from record_episode import synchronize_positions

        synchronize_positions(self.follower, self.leader, self.target,
                              timeout=self.alignment_timeout,
                              tolerance=self.alignment_tolerance)

    def read_calibrated_state(self):
        return self._read(self.follower.bus, "Present_Position", normalize=True)

    def read_calibrated_goal(self):
        return self._read(self.follower.bus, "Goal_Position", normalize=True)

    def communication_healthy(self):
        return not self.communication_latched
