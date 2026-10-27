"""Single-process Leader-to-ACT handoff; synthetic tests are separate.

Hardware startup uses the recorder's alignment with staged register checks and
WILL MOVE the robot. No hardware path is entered without separate CLI arming.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from act_action_guard import NAMES
from teleop_act_handoff import HandoffComplete, HandoffFault, HandoffGate


PERIOD = 0.05
JOINTS = tuple(n.removesuffix('.pos') for n in NAMES)
TRIAL_SECONDS = 1.0
TRIAL_SPEED_PER_S = 3.0  # degrees/s for five rotary joints; gripper Goal is fixed
TRIAL_MAX_DISPLACEMENT = 1.0  # degrees from the handoff Goal, per rotary joint
TRIAL_MAX_PREDICTION_GAP = 5.0  # degrees from current Goal and Present
TRIAL_MAX_INITIAL_GAP = 2.0  # five rotary joints, degrees; start check only
TRIAL_MAX_INITIAL_GRIPPER_GAP = 1.5  # normalized_0_100; initial handoff check only
TRIAL_MAX_GRIPPER_MOVEMENT = 1.0  # normalized_0_100; preserve in-trial movement limit
CONTINUOUS_DEFAULT_SECONDS = 10.0
CONTINUOUS_DEFAULT_SPEED = 10.0  # degrees/s, capped at the existing 30 degrees/s
CONTINUOUS_DEFAULT_GRIPPER_SPEED = 20.0  # normalized_0_100/s


def keyed(values):
    if set(values) != set(JOINTS):
        raise ValueError('Expected all six SO-101 joints')
    return {n + '.pos': float(values[n]) for n in JOINTS}


def save_prediction_input_images(session, output):
    from PIL import Image
    for frame_id, rgb in session.prediction_input_images:
        stem = f'act_input_{frame_id:03d}' if isinstance(frame_id, int) else f'act_input_{frame_id}'
        images = rgb if isinstance(rgb, dict) else {'iphone': rgb}
        for name, frame in images.items():
            suffix = '' if not isinstance(rgb, dict) else f'_{name}'
            source = Image.fromarray(frame)
            source.save(output/f'{stem}{suffix}_{source.width}x{source.height}.png')
            source.resize((224,168), Image.Resampling.BILINEAR).save(
                output/f'{stem}{suffix}_224x168.png')


def finalize_handoff_run(args, io, session, cycle_rows, hashes, primary_error=None):
    """Preserve the first fault while reporting cleanup and report-write faults."""
    cleanup_errors = []
    try:
        io.close()  # no Goal or torque write
    except BaseException as exc:
        cleanup_errors.append(f'close: {type(exc).__name__}: {exc}')
    try:
        (args.output/'startup_checkpoints.json').write_text(
            json.dumps(io.startup_checkpoints, indent=2, default=str)+'\n')
    except BaseException as exc:
        cleanup_errors.append(f'startup_checkpoints: {type(exc).__name__}: {exc}')
    image_artifacts_error = None
    if session is not None and session.check_image is not None:
        try:
            from PIL import Image
            images = session.check_image if isinstance(session.check_image, dict) else {
                'iphone': session.check_image}
            for name, frame in images.items():
                suffix = '' if not isinstance(session.check_image, dict) else f'_{name}'
                source = Image.fromarray(frame)
                source.save(args.output/f'observed_rgb{suffix}_{source.width}x{source.height}.png')
                source.resize((224,168), Image.Resampling.BILINEAR).save(
                    args.output/f'model_rgb{suffix}_224x168.png')
        except BaseException as exc:
            image_artifacts_error = f'{type(exc).__name__}: {exc}'
            cleanup_errors.append(f'image: {image_artifacts_error}')
    if session is not None:
        try:
            save_prediction_input_images(session, args.output)
        except BaseException as exc:
            image_artifacts_error = f'{type(exc).__name__}: {exc}'
            cleanup_errors.append(f'prediction_images: {image_artifacts_error}')
    for path, digest in hashes.items():
        try:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                cleanup_errors.append(f'calibration changed unexpectedly: {path}')
        except BaseException as exc:
            cleanup_errors.append(f'calibration_hash: {path}: {type(exc).__name__}: {exc}')
    checkpoint = args.checkpoint
    if checkpoint is None:
        try:
            from observe_act_read_only import CHECKPOINT as checkpoint
        except BaseException as exc:
            cleanup_errors.append(f'default_checkpoint: {type(exc).__name__}: {exc}')
    report = {'mode':'hardware','observe_only':args.observe_only,
              'checkpoint':None if checkpoint is None else str(Path(checkpoint).resolve()),
              'first_motion_trial':args.first_motion_trial,
              'continuous_act':getattr(args, 'continuous_act', False),
              'first_motion_limits':None if not args.first_motion_trial else dict(
                  seconds=TRIAL_SECONDS, speed_per_s=TRIAL_SPEED_PER_S,
                  max_displacement=TRIAL_MAX_DISPLACEMENT,
                  max_prediction_gap=TRIAL_MAX_PREDICTION_GAP,
                  max_initial_gap_rotary_deg=TRIAL_MAX_INITIAL_GAP,
                  max_initial_gap_gripper_normalized=TRIAL_MAX_INITIAL_GRIPPER_GAP,
                  gripper='hold handoff Goal'),
              'startup_path':'record_episode.start_hardware guarded_teleop_start',
              'events':[] if session is None else session.events,
              'last_owner':None if session is None else session.gate.owner,
              'fault_state':None if session is None else session.phase,
              'original_exception':None if primary_error is None else dict(
                  type=type(primary_error).__name__, message=str(primary_error),
                  mode_before=None if not cycle_rows else cycle_rows[-1]['mode_before']),
              'cleanup_errors':cleanup_errors,
              'post_init_goal_write_attempts':io.goal_writes,
              'post_handoff_goal_write_attempts':None if session is None or session.handoff_write_baseline is None else io.goal_writes-session.handoff_write_baseline,
              'first_motion_completion':None if session is None else session.trial_completion,
              'continuous_act_completion':None if session is None else session.continuous_completion,
              'cycle_timings':cycle_rows,
              'cycles':len(cycle_rows),
              'cycle_work_max_ms':max((r['work_ms'] for r in cycle_rows),default=None),
              'cycle_over_50ms':sum(r['work_ms']>50 for r in cycle_rows),
              'cycle_interval_over_50ms':sum(r['interval_ms'] is not None and r['interval_ms']>50 for r in cycle_rows),
              'cycle_interval_over_60ms':sum(r['interval_ms'] is not None and r['interval_ms']>60 for r in cycle_rows),
              'image_artifacts_error':image_artifacts_error,
              'warning':'Software stop sends no new Goal or torque command; servos may retain holding torque. Support arm and use independent servo-power cutoff if needed.'}
    try:
        (args.output/'handoff_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    except BaseException as exc:
        message = f'REPORT SAVE FAILED: {type(exc).__name__}: {exc}'
        cleanup_errors.append(message)
        print(message, file=sys.stderr, flush=True)
        if primary_error is not None:
            print(f'ORIGINAL FAULT: {type(primary_error).__name__}: {primary_error}',
                  file=sys.stderr, flush=True)
    return cleanup_errors


def raise_run_fault(primary_error, cleanup_errors):
    if primary_error is not None:
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_errors:
        raise RuntimeError('Handoff cleanup/report fault: ' + '; '.join(cleanup_errors))


class ReadOnlyPreflight:
    """Reuse the existing Follower-only raw-register reader; never writes."""

    def __init__(self):
        from probe_goal_broadcast import HardwareIO
        self.probe = HardwareIO()

    def open(self):
        return self.probe.open()

    def limits(self):
        return self.probe.limits()

    def calibration_matches(self):
        return self.probe.calibration_matches()

    def read_raw(self):
        return self.probe.read_raw()

    def read_calibrated_positions(self):
        bus = self.probe.robot.bus
        return {register: keyed(bus.sync_read(register, num_retry=0))
                for register in ('Goal_Position', 'Present_Position')}

    def close_without_writes(self):
        self.probe.close_without_writes()


def read_only_startup_preflight(reader, *, goal_sync_only=False):
    """Reject anomalous six-axis registers before recorder startup can write."""
    from probe_goal_broadcast import _check_snapshot

    report = dict(result='pending', note='Read-only; no Goal or torque write')
    try:
        report['metadata'] = reader.open()
        if not reader.calibration_matches():
            raise HandoffFault('Follower calibration mismatch before startup')
        limits = reader.limits()
        report['limits'] = limits
        first = reader.read_raw()
        report['first_raw'] = first
        _check_snapshot(first, limits, phase='startup_preflight_first')
        second = reader.read_raw()
        report['second_raw'] = second
        _check_snapshot(second, limits, phase='startup_preflight_second', baseline=first)
        for name in JOINTS:
            lo, hi = limits[name]
            if not goal_sync_only and not lo <= second['Goal_Position'][name] <= hi:
                raise HandoffFault(f'{name}: stored Goal outside calibrated raw range')
        report['raw_goal_present_gap'] = {
            name: second['Goal_Position'][name]-second['Present_Position'][name]
            for name in JOINTS
        }
        if not goal_sync_only:
            calibrated = reader.read_calibrated_positions()
            report['calibrated_goal_present'] = calibrated
            report['goal_present_gap'] = {
                name: calibrated['Goal_Position'][name]-calibrated['Present_Position'][name]
                for name in NAMES
            }
        report.update(result='ok',
                      note=('Goal-sync-only: stale Goal allowed before a separately armed single broadcast; no torque/configure/alignment.'
                            if goal_sync_only else
                            'Pre-initialization: Goal/Present gap recorded, not used as a tracking fault. Stored Goal must still be in calibrated range. State can change after serial close.'))
        return report
    except BaseException as exc:
        report.update(result='refused', fault=f'{type(exc).__name__}: {exc}')
        fault = HandoffFault(report['fault'])
        fault.diagnostics = report
        raise fault from exc
    finally:
        reader.close_without_writes()


class ACTPredictor:
    """Selected ACT checkpoint; select_action consumes five queued Actions/chunk."""

    def __init__(self, device='mps', checkpoint=None,
                 image_keys=('observation.images.iphone',), image_shape=(480, 640, 3)):
        from observe_act_read_only import load_policy
        self.image_keys = tuple(image_keys)
        self.image_shape = tuple(image_shape)
        self.policy, self.pre, self.post = load_policy(
            device, checkpoint=checkpoint, image_keys=self.image_keys)
        self.policy.config.n_action_steps = 5
        if self.policy.config.chunk_size != 50 or self.policy.config.temporal_ensemble_coeff is not None:
            raise ValueError('Unexpected ACT chunk contract')
        self.policy.reset()
        self.device = device

    def predict(self, state, image):
        import torch
        from PIL import Image
        from lerobot.policies.utils import prepare_observation_for_inference

        images = image if isinstance(image, dict) else {'iphone': image}
        if {f'observation.images.{n}' for n in images} != set(self.image_keys):
            raise ValueError('Live camera keys differ from checkpoint features')
        if any(frame.shape != self.image_shape or frame.dtype != np.uint8
               for frame in images.values()):
            raise ValueError(f'OBS images must be uint8 RGB {self.image_shape}')
        if set(state) != set(NAMES) or not all(math.isfinite(state[n]) for n in NAMES):
            raise ValueError('Invalid Follower state for ACT')
        small = {f'observation.images.{name}': np.array(
            Image.fromarray(frame).resize((224, 168), Image.Resampling.BILINEAR), copy=True)
            for name, frame in images.items()}
        obs = prepare_observation_for_inference({
            'observation.state': np.asarray([state[n] for n in NAMES], dtype=np.float32),
            **small,
        }, torch.device(self.device))
        refreshed = len(self.policy._action_queue) == 0
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(obs))).detach().cpu().numpy().reshape(-1)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError('Invalid six-axis ACT Action')
        return action, refreshed

    def reset(self):
        self.policy.reset()


class HardwareAdapter:
    def __init__(self, camera_cfg=None):
        self.follower = self.leader = self.camera = None
        self.camera_cfg = camera_cfg
        self.goal_writes = 0
        self.startup_checkpoints = []

    def open(self):
        from record_episode import make_hardware, start_hardware
        if self.camera_cfg:
            from multi_camera import CameraRig
            self.camera = CameraRig(self.camera_cfg)
        else:
            from obs_camera import OBSCamera
            self.camera = OBSCamera()  # legacy one-camera checkpoint
        self.camera.connect()
        self.follower, self.leader = make_hardware(self.camera)
        # Same connection, calibration checks, one-shot Goal readback,
        # configure handling and slow Alignment as the successful teleop trial.
        start_hardware(self.follower, self.leader, guarded_teleop_start=True,
                       startup_checkpoints=self.startup_checkpoints)

    def open_goal_sync_only(self):
        from record_episode import make_hardware
        from act_startup import ACTStartup, LeRobotStartupDevice
        self.follower, self.leader = make_hardware(None)
        self.follower.bus.connect()  # no Leader connection or camera
        device = LeRobotStartupDevice(self.follower, self.leader, follower_only=True)
        startup = ACTStartup(device, diagnostics_path=self.startup_diagnostics_path)
        return startup.run_hardware_goal_sync_only(
            operator_approved=True, arm_supported=True, workspace_clear=True,
            power_stop_ready=True)

    def limits(self):
        from capture_session import calibration_limits
        limits = calibration_limits(self.follower)
        lo, hi = limits['gripper.pos']
        limits['gripper.pos'] = (max(0, lo), min(100, hi))
        return limits

    def request(self):
        return keyed(self.leader.bus.sync_read('Present_Position', num_retry=0))

    def snapshot(self):
        bus = self.follower.bus
        start = time.perf_counter()
        present = keyed(bus.sync_read('Present_Position', num_retry=0))
        goal = keyed(bus.sync_read('Goal_Position', num_retry=0))
        torque = keyed(bus.sync_read('Torque_Enable', normalize=False, num_retry=0))
        status = keyed(bus.sync_read('Status', normalize=False, num_retry=0))
        return dict(present=present, goal=goal, torque=torque, status=status,
                    sampled_at=start, sample_age_s=time.perf_counter()-start)

    def image(self):
        if self.camera_cfg is not None:
            return self.camera.read_latest()
        return self.camera.read_latest(max_age_ms=100)

    def image_sample(self):
        # Copy image and capture timestamp under the same lock. read_latest()
        # alone does not return the timestamp paired with its frame.
        camera = self.camera
        if self.camera_cfg is not None:
            images, stamps = camera.sample_all()
            return images, min(stamps.values()), {
                name: {**camera.cameras[name].identity,
                       'sampled_at_monotonic': stamps[name]}
                for name in images}
        if not camera.is_connected or camera.thread is None or not camera.thread.is_alive():
            raise RuntimeError('OBS camera read thread is not running')
        with camera.frame_lock:
            frame, captured_at = camera.latest_frame, camera.latest_timestamp
            image = None if frame is None else frame.copy()
        if image is None or captured_at is None:
            raise RuntimeError('OBS camera has no captured frame')
        age_s = time.perf_counter() - captured_at
        if not 0 <= age_s <= .10:
            raise TimeoutError(f'OBS image stale: {age_s*1000:.1f} ms')
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError('OBS image is not RGB uint8 640x480')
        return image, captured_at, dict(index=camera.config.index_or_path,
                                         backend=int(camera.config.backend), device_name=None)

    def write(self, goal):
        # The only post-initialization Follower Goal writer in this process.
        if self.follower.config.max_relative_target is not None:
            raise RuntimeError('Follower send_action may clip Goal; ACT write refused')
        self.goal_writes += 1  # count attempted writes, including ambiguous failures
        return self.follower.send_action(goal)

    def calibration_ok(self):
        from record_episode import validate_calibrations
        validate_calibrations((self.leader, self.follower))
        return True

    def close(self):
        # No torque-off or Goal during cleanup; holding force may remain.
        errors = []
        if self.camera is not None and self.camera.is_connected:
            try:self.camera.disconnect()
            except Exception as exc:errors.append(f'camera: {exc}')
        for device in (self.follower, self.leader):
            if device is not None and device.bus.is_connected:
                try:device.bus.disconnect(disable_torque=False)
                except Exception as exc:errors.append(f'serial: {exc}')
        if errors:raise RuntimeError('; '.join(errors))


class HandoffController:
    def __init__(self, io, predictor, *, observe_only=True, observe_frames=40,
                 first_motion_trial=False, continuous_act=False,
                 act_seconds=CONTINUOUS_DEFAULT_SECONDS,
                 act_speed_deg_s=CONTINUOUS_DEFAULT_SPEED,
                 act_gripper_speed_s=CONTINUOUS_DEFAULT_GRIPPER_SPEED):
        if not 1 <= observe_frames <= 200:
            raise ValueError('Observation-only duration must be 1..200 frames')
        if first_motion_trial and observe_only:
            raise ValueError('First-motion trial requires the separately armed ACT mode')
        if continuous_act and (observe_only or first_motion_trial):
            raise ValueError('Continuous ACT requires its own armed mode')
        if not math.isfinite(act_seconds) or not 0 < act_seconds <= 120:
            raise ValueError('ACT duration must be within 0..120 seconds')
        if not math.isfinite(act_speed_deg_s) or not 0 < act_speed_deg_s <= 30:
            raise ValueError('ACT rotary speed must be within 0..30 degrees/s')
        if not math.isfinite(act_gripper_speed_s) or not 0 < act_gripper_speed_s <= 30:
            raise ValueError('ACT gripper speed must be within 0..30 normalized units/s')
        self.io, self.predictor = io, predictor
        self.observe_only = observe_only
        self.first_motion_trial = first_motion_trial
        self.continuous_act = continuous_act
        self.act_seconds = act_seconds
        self.act_speed_deg_s = act_speed_deg_s
        self.act_gripper_speed_s = act_gripper_speed_s
        self.observe_frames = observe_frames
        self.observed_frames = 0
        self.calibration_verified_at_handoff = False
        self.gate = HandoffGate(io.write, io.limits())
        self.phase = 'TELEOP'
        self.last_tick = None
        self.chunk_offset = 0
        self.events = []
        self.handoff_write_baseline = None
        self.check_image = None
        self.act_started_at = None
        self.act_origin_present = None
        self.prepared_action = None
        self.prepared_sampled_at = None
        self.prediction_input_images = []
        self.trial_completion = None
        self.continuous_completion = None
        self.last_act_snapshot = None

    def _write_attempt_count(self):
        return getattr(self.io, 'goal_writes', len(getattr(self.io, 'writes', ())))

    def _record(self, event, **data):
        self.events.append(dict(event=event, at_monotonic=time.perf_counter(), **data))

    def _image_sample(self):
        if hasattr(self.io, 'image_sample'):
            return self.io.image_sample()
        return self.io.image(), time.perf_counter(), dict(index=None, backend=None, device_name=None)

    def _check_snapshot(self, snap):
        if not math.isfinite(snap['sample_age_s']) or not 0 <= snap['sample_age_s'] <= .10:
            raise HandoffFault('stale six-axis register snapshot')
        return snap

    def _trial_prediction(self, prediction, snap):
        values = np.asarray(prediction, dtype=float).reshape(-1)
        if values.shape != (6,) or not np.isfinite(values).all():
            self.gate.fault('invalid first-motion prediction')
        gaps = {name: dict(predicted=float(values[i]), goal=snap['goal'][name],
                           present=snap['present'][name],
                           goal_gap=abs(values[i]-snap['goal'][name]),
                           present_gap=abs(values[i]-snap['present'][name]))
                for i, name in enumerate(NAMES[:5])}
        self._record('first_motion_action_check', joints=gaps,
                     gripper_predicted=float(values[5]),
                     gripper_held_goal=snap['goal'][NAMES[5]])
        for i, name in enumerate(NAMES[:5]):
            if (abs(values[i]-snap['goal'][name]) > TRIAL_MAX_PREDICTION_GAP
                    or abs(values[i]-snap['present'][name]) > TRIAL_MAX_PREDICTION_GAP):
                self.gate.fault(f'first-motion prediction too far: {name}')
        values = values.copy()
        values[5] = snap['goal'][NAMES[5]] if self.gate.owner == 'CHECK' else self.gate.act_origin[5]
        return values

    def _check_trial_initial_gap(self, snap):
        gaps = {n: abs(snap['goal'][n]-snap['present'][n]) for n in NAMES}
        if any(gaps[n] > TRIAL_MAX_INITIAL_GAP for n in NAMES[:5]) or (
            gaps[NAMES[5]] > TRIAL_MAX_INITIAL_GRIPPER_GAP
        ):
            self.gate.fault(f'first-motion initial Goal/Present gap exceeds rotary {TRIAL_MAX_INITIAL_GAP:g} deg '
                            'or gripper 1.5 normalized_0_100: ' + str(gaps))

    def _complete_first_motion(self, reason, snap):
        self.gate.stop()  # revoke ACT and Leader Goal writing before reporting
        self.phase = 'STOPPED'
        self.trial_completion = dict(reason=reason,
                                     limits=dict(seconds=TRIAL_SECONDS,
                                                 speed_per_s=TRIAL_SPEED_PER_S,
                                                 max_displacement=TRIAL_MAX_DISPLACEMENT),
                                     final_sent_goal=None if self.gate.last_sent is None else self.gate.last_sent.copy(),
                                     last_recorded_present=snap['present'].copy(),
                                     last_recorded_goal=snap['goal'].copy(),
                                     goal_write_attempts_since_handoff=(
                                         self._write_attempt_count()-self.handoff_write_baseline),
                                     total_displacement=self.gate.total_displacement.copy(),
                                     elapsed_s=time.perf_counter()-self.act_started_at)
        self._record('first_motion_complete', **self.trial_completion)

    def _complete_continuous(self, reason, snap):
        self.gate.stop()
        self.phase = 'STOPPED'
        self.continuous_completion = dict(
            reason=reason, duration_limit_s=self.act_seconds,
            speed_deg_s=self.act_speed_deg_s,
            gripper_speed_normalized_s=self.act_gripper_speed_s,
            last_sent_goal=None if self.gate.last_sent is None else self.gate.last_sent.copy(),
            last_recorded_goal=None if snap is None else snap['goal'].copy(),
            last_recorded_present=None if snap is None else snap['present'].copy(),
            goal_write_attempts_since_handoff=(0 if self.handoff_write_baseline is None else
                                               self._write_attempt_count()-self.handoff_write_baseline),
            elapsed_s=(0.0 if self.act_started_at is None else
                       time.perf_counter()-self.act_started_at))
        self._record('continuous_act_complete', **self.continuous_completion)

    def tick(self, key=''):
        now = time.perf_counter()
        if self.phase == 'FAULT':
            raise HandoffFault('fault latched; no automatic restart')
        if self.continuous_act and self.phase in ('TELEOP', 'CHECK', 'ACT') and key == 'q':
            self._complete_continuous('operator_stop', self.last_act_snapshot)
            return
        try:
            if self.last_tick is not None and now-self.last_tick > .25:
                self.gate.fault('control gap >250 ms')
            self.last_tick = now
            if self.phase == 'TELEOP':
                snap = self._check_snapshot(self.io.snapshot())
                if any(snap['torque'][n] != 1 or snap['status'][n] != 0 for n in NAMES):
                    self.gate.fault('teleop torque/status invalid')
                if any(abs(snap['goal'][n]-snap['present'][n]) > 8 for n in NAMES):
                    self.gate.fault('teleop Goal/Present tracking error >8')
                if self.gate.last_sent is not None and any(
                    abs(snap['present'][n]-self.gate.last_sent[n]) > 8 for n in NAMES):
                    self.gate.fault('teleop tracking error >8')
                if key == 'x':
                    self.gate.begin_handoff()  # blocks both writers immediately
                    self.phase = 'CHECK'
                    self.handoff_write_baseline = self._write_attempt_count()
                    self._record('leader_stopped', queued_goals=0,
                                 post_init_goal_write_attempts=self.handoff_write_baseline)
                    return
                request = self.io.request()
                if set(request) != set(NAMES):
                    self.gate.fault('incomplete Leader request')
                prior = self.gate.last_sent or snap['goal']
                command = {n: prior[n]+max(-1.5,min(1.5,request[n]-prior[n])) for n in NAMES}
                self.gate.send_teleop(command)
                return
            snap = self._check_snapshot(self.io.snapshot())
            if self.phase == 'CHECK':
                check = dict(goal=snap['goal'], present=snap['present'],
                             torque=snap['torque'], status=snap['status'],
                             calibration_ok=self.io.calibration_ok(),
                             sample_age_s=snap['sample_age_s'],
                             communication_ok=True, queued_goals=0)
                self.gate.verify_handoff(**check)
                # Warm the model on the live observation while BOTH writers are blocked.
                self.check_image = self.io.image()
                _, refreshed = self.predictor.predict(snap['present'], self.check_image)
                self.predictor.reset()  # first sent Action must use a fresh observation
                self._record('warmup_prediction', refreshed=refreshed)
                # Inference may be slow on first call. Re-read all registers;
                # never arm using a snapshot made stale by model warm-up.
                snap = self._check_snapshot(self.io.snapshot())
                check.update(goal=snap['goal'], present=snap['present'],
                             torque=snap['torque'], status=snap['status'],
                             calibration_ok=self.io.calibration_ok(),
                             sample_age_s=snap['sample_age_s'])
                self.gate.verify_handoff(**check)
                if self.observe_only:
                    # Startup and both CHECK snapshots verified the full motor
                    # calibration. Do not repeat 36 serial register reads twice
                    # per observation cycle; live Goal/Present/Torque/Status
                    # and communication checks continue below.
                    self.calibration_verified_at_handoff = True
                    self.phase = 'OBSERVE_ACT'
                    self.last_tick = time.perf_counter()  # CHECK warm-up may exceed 250 ms
                    self._record('observe_only_started', snapshot=snap,
                                 frames_planned=self.observe_frames)
                    return
                # No new Goal sync or configure; readbacks must match last teleop Goal.
                if self.first_motion_trial:
                    self._check_trial_initial_gap(snap)
                    # Reset discarded the warm-up queue. Prepare the first
                    # real chunk while BOTH writers are still blocked.
                    prepared_sample = snap['sampled_at']
                    prepared_image, prepared_image_at, camera_info = self._image_sample()
                    self.prediction_input_images.append(('prepared', prepared_image.copy()))
                    prepared, refreshed = self.predictor.predict(snap['present'], prepared_image)
                    snap = self._check_snapshot(self.io.snapshot())
                    check.update(goal=snap['goal'], present=snap['present'],
                                 torque=snap['torque'], status=snap['status'],
                                 calibration_ok=self.io.calibration_ok(),
                                 sample_age_s=snap['sample_age_s'])
                    self.gate.verify_handoff(**check)
                    self._check_trial_initial_gap(snap)
                    self.prepared_action = self._trial_prediction(prepared, snap)
                    self.prepared_sampled_at = prepared_sample
                    if time.perf_counter()-prepared_sample > .10:
                        self.gate.fault('prepared first-motion Action has stale observation')
                    print('FIRST MOTION PRECHECK: rotary prediction within 5 deg of Goal/Present; '
                          'gripper Goal held. No ACT Goal sent yet.', flush=True)
                    self._record('first_motion_chunk_prepared', refreshed=refreshed,
                                 input_age_ms=(time.perf_counter()-prepared_sample)*1000,
                                 image_captured_at_monotonic=prepared_image_at,
                                 image_captured_at_unix_s=time.time()-(time.perf_counter()-prepared_image_at),
                                 camera=camera_info)
                    self.gate.arm_act(**check,
                                      speed_per_s=(TRIAL_SPEED_PER_S,) * 6,
                                      max_displacement=TRIAL_MAX_DISPLACEMENT)
                    self.act_origin_present = snap['present'].copy()
                else:
                    self.gate.arm_act(**check, speed_per_s=(self.act_speed_deg_s,) * 5 +
                                      (self.act_gripper_speed_s,))
                self.phase = 'ACT'
                self.chunk_offset = 0
                self._record('act_armed', snapshot=snap)
                self.act_started_at = time.perf_counter()
                self.last_tick = time.perf_counter()  # warm-up was a no-follow CHECK interval
                return  # first ACT send is a later cycle, after warm-up
            if self.phase == 'OBSERVE_ACT':
                if not self.calibration_verified_at_handoff:
                    self.gate.fault('observation started without verified handoff calibration')
                check = dict(goal=snap['goal'], present=snap['present'],
                             torque=snap['torque'], status=snap['status'],
                             calibration_ok=self.calibration_verified_at_handoff,
                             sample_age_s=snap['sample_age_s'],
                             communication_ok=True, queued_goals=0)
                self.gate.verify_handoff(**check)
                image, image_at, camera_info = self._image_sample()
                inference_started = time.perf_counter()
                prediction, refreshed = self.predictor.predict(snap['present'], image)
                inference_finished = time.perf_counter()
                if refreshed:
                    self.prediction_input_images.append((self.observed_frames, image.copy()))
                # No Action is sent in this mode. The input can age while MPS
                # runs; read fresh servo state afterward instead of treating
                # inference time as the age of the next control observation.
                post = self._check_snapshot(self.io.snapshot())
                check.update(goal=post['goal'], present=post['present'],
                             torque=post['torque'], status=post['status'],
                             calibration_ok=self.calibration_verified_at_handoff,
                             sample_age_s=post['sample_age_s'])
                self.gate.verify_handoff(**check)
                self._record('observe_act', frame=self.observed_frames,
                             chunk_offset=self.observed_frames%5, refreshed=refreshed,
                             image_captured_at_monotonic=image_at,
                             image_captured_at_unix_s=time.time()-(time.perf_counter()-image_at),
                             image_age_at_inference_ms=(inference_started-image_at)*1000,
                             camera=camera_info,
                             present=snap['present'], goal=snap['goal'],
                             prediction_goal_abs_error={n:abs(float(prediction[i])-snap['goal'][n])
                                                        for i,n in enumerate(NAMES)},
                             inference_ms=(inference_finished-inference_started)*1000,
                             input_age_after_inference_ms=(inference_finished-snap['sampled_at'])*1000,
                             post_inference_sample_age_ms=post['sample_age_s']*1000,
                             predicted_action=dict(zip(NAMES,map(float,prediction))))
                self.observed_frames += 1
                if self._write_attempt_count() != self.handoff_write_baseline:
                    self.gate.fault('Goal write attempted after Leader stop in observation-only mode')
                if self.observed_frames >= self.observe_frames:
                    self._record('observe_only_complete', snapshot=post,
                                 post_handoff_goal_write_attempts=0)
                    self.phase = 'OBSERVED'
                return
            if self.phase == 'OBSERVED':
                return
            if self.phase == 'ACT':
                self.last_act_snapshot = snap
                if self.continuous_act and now-self.act_started_at >= self.act_seconds:
                    self._complete_continuous('time_limit', snap)
                    return
                if self.first_motion_trial and now-self.act_started_at >= TRIAL_SECONDS:
                    self._complete_first_motion('time_limit', snap)
                    return
                if any(snap['torque'][n] != 1 or snap['status'][n] != 0 for n in NAMES):
                    self.gate.fault('ACT torque/status invalid')
                if (self.first_motion_trial or self.continuous_act) and any(
                    abs(snap['goal'][n]-self.gate.last_sent[n]) > .5 for n in NAMES
                ):
                    self.gate.fault('ACT Goal readback differs from last sent Goal')
                if self.first_motion_trial and any(
                    abs(snap['present'][n]-self.act_origin_present[n]) > TRIAL_MAX_DISPLACEMENT
                    for n in NAMES[:5]
                ):
                    self.gate.fault('first-motion measured displacement limit exceeded')
                if self.first_motion_trial and abs(
                    snap['present'][NAMES[5]]-self.act_origin_present[NAMES[5]]
                ) > TRIAL_MAX_GRIPPER_MOVEMENT:
                    self.gate.fault('first-motion gripper moved while Goal was held')
                # Every fifth call refreshes a 50-Action chunk; only its first
                # five are consumed. Warm-up queue was discarded in CHECK.
                if self.first_motion_trial and self.prepared_action is not None:
                    prediction, refreshed = self.prepared_action, True
                    prediction_sampled_at = self.prepared_sampled_at
                else:
                    prediction_sampled_at = snap['sampled_at']
                    prediction, refreshed = self.predictor.predict(snap['present'], self.io.image())
                if self.continuous_act:
                    # Inference may take >100 ms. Do not send against the
                    # pre-inference Follower snapshot: refresh telemetry first.
                    # This does NOT waive the 100 ms snapshot freshness gate.
                    snap = self._check_snapshot(self.io.snapshot())
                    self.last_act_snapshot = snap
                    if any(snap['torque'][n] != 1 or snap['status'][n] != 0 for n in NAMES):
                        self.gate.fault('ACT torque/status invalid after inference')
                    if any(abs(snap['goal'][n]-self.gate.last_sent[n]) > .5 for n in NAMES):
                        self.gate.fault('ACT Goal readback differs after inference')
                    # Guard sees the age of the refreshed position measurement.
                    prediction_sampled_at = snap['sampled_at']
                if self.first_motion_trial:
                    prediction = self._trial_prediction(prediction, snap)
                age = time.perf_counter()-prediction_sampled_at
                if self.continuous_act and time.perf_counter()-self.act_started_at >= self.act_seconds:
                    self._complete_continuous('time_limit', snap)
                    return
                if self.first_motion_trial and time.perf_counter()-self.act_started_at >= TRIAL_SECONDS:
                    self._complete_first_motion('time_limit', snap)
                    return
                try:
                    sent, reasons = self.gate.send_act(prediction, snap['present'], sample_age_s=age)
                except HandoffComplete as complete:
                    self._complete_first_motion(complete.reason, snap)
                    return
                self.prepared_action = None
                self._record('act_sent', chunk_offset=self.chunk_offset,
                             refreshed=refreshed, sent=sent, reasons=reasons,
                             pre_send_present=snap['present'].copy(),
                             pre_send_goal=snap['goal'].copy(),
                             total_displacement=(self.gate.total_displacement.copy()
                                                 if self.first_motion_trial else None))
                self.chunk_offset = (self.chunk_offset+1)%5
        except BaseException as exc:
            self.phase = 'FAULT'
            if self.gate.owner != 'FAULT':
                try:self.gate.fault(f'{type(exc).__name__}: {exc}')
                except HandoffFault:pass
            self._record('fault', reason=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('hardware',), required=True)
    parser.add_argument('--hardware-init-approved', action='store_true')
    parser.add_argument('--goal-sync-only', action='store_true',
                        help='Separate one-broadcast trial; never configures, enables torque or aligns')
    parser.add_argument('--hardware-goal-sync-approved', action='store_true')
    parser.add_argument('--ack-arm-supported', action='store_true')
    parser.add_argument('--ack-clear-workspace', action='store_true')
    parser.add_argument('--ack-power-stop-ready', action='store_true')
    parser.add_argument('--hardware-act-approved', action='store_true')
    parser.add_argument('--first-motion-trial', action='store_true',
                        help='Required for ACT Goal writes: fixed 1 s, 3 deg/s, 1 deg per-axis test')
    parser.add_argument('--continuous-act', action='store_true',
                        help='Run bounded ACT Pick & Place after Leader handoff')
    parser.add_argument('--act-seconds', type=float, default=CONTINUOUS_DEFAULT_SECONDS)
    parser.add_argument('--act-speed-deg-s', type=float, default=CONTINUOUS_DEFAULT_SPEED)
    parser.add_argument('--act-gripper-speed-s', type=float, default=CONTINUOUS_DEFAULT_GRIPPER_SPEED)
    parser.add_argument('--observe-only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--observe-seconds', type=float, default=2.0,
                        help='Observation-only ACT inference duration after x, 0.05..10 s')
    parser.add_argument('--device', choices=('cpu','mps'), default='mps')
    parser.add_argument('--checkpoint', type=Path,
                        help='Local ACT pretrained_model directory; default is pinned 55-demo model')
    parser.add_argument('--camera-config', type=Path,
                        help='Independent OBS inputs; omit for legacy iphone virtual camera')
    parser.add_argument('--enable-cameras', help='Comma-separated logical names overriding enabled flags')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.enable_cameras and not args.camera_config:
        parser.error('--enable-cameras requires --camera-config')
    camera_cfg = None
    image_keys = ('observation.images.iphone',)
    image_shape = (480, 640, 3)
    if args.camera_config:
        from multi_camera import capture_image_shape, image_keys as configured_image_keys, load_config
        camera_cfg = load_config(args.camera_config, args.enable_cameras)
        image_keys = configured_image_keys(camera_cfg)
        image_shape = capture_image_shape(camera_cfg)
    if args.output.exists():
        parser.error('Output already exists')
    if not math.isfinite(args.observe_seconds) or not .05 <= args.observe_seconds <= 10:
        parser.error('--observe-seconds must be 0.05..10')
    if args.first_motion_trial and args.observe_only:
        parser.error('--first-motion-trial requires --no-observe-only')
    if args.continuous_act and (args.observe_only or args.first_motion_trial):
        parser.error('--continuous-act requires --no-observe-only and excludes --first-motion-trial')
    if args.continuous_act and args.checkpoint is None:
        parser.error('--continuous-act requires an explicit local --checkpoint')
    if (not math.isfinite(args.act_seconds) or not 0 < args.act_seconds <= 120 or
            not math.isfinite(args.act_speed_deg_s) or not 0 < args.act_speed_deg_s <= 30 or
            not math.isfinite(args.act_gripper_speed_s) or not 0 < args.act_gripper_speed_s <= 30):
        parser.error('ACT duration must be 0..120 s and speeds 0..30 units/s')
    if args.goal_sync_only:
        if args.hardware_init_approved or args.hardware_act_approved or args.first_motion_trial or args.continuous_act or not args.observe_only:
            parser.error('Goal-sync-only cannot be combined with initialization or ACT writes')
        if not all((args.hardware_goal_sync_approved, args.ack_arm_supported,
                    args.ack_clear_workspace, args.ack_power_stop_ready)):
            parser.error('Goal-sync-only requires separate approval and all physical safety acknowledgments')
        typed = input('Goal-sync-only can activate torque. Arm supported and power cutoff ready? Type YES: ')
        if typed.strip() != 'YES':
            parser.error('Goal-sync-only trial not armed')
        args.output.mkdir(parents=True, exist_ok=False)
        io = HardwareAdapter(camera_cfg)
        io.startup_diagnostics_path = args.output/'startup_checkpoints.json'
        try:
            try:
                preflight = read_only_startup_preflight(ReadOnlyPreflight(), goal_sync_only=True)
                (args.output/'preflight.json').write_text(json.dumps(preflight,indent=2,default=str)+'\n')
            except BaseException as exc:
                report = getattr(exc,'diagnostics',dict(result='refused',fault=repr(exc)))
                (args.output/'preflight.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
                raise
            result = io.open_goal_sync_only()
            print(f'Goal-sync-only complete: ready_for_action={result.ready}. No configure/Alignment/ACT.',flush=True)
        finally:
            io.close()  # no Goal or torque write in cleanup
        return
    if not args.hardware_init_approved:
        parser.error('Hardware initialization and Alignment require separate approval')
    if not args.observe_only and not args.hardware_act_approved:
        parser.error('ACT Goal writes require separate explicit approval')
    if not args.observe_only and (not (args.first_motion_trial or args.continuous_act) or not all((
            args.ack_arm_supported, args.ack_clear_workspace, args.ack_power_stop_ready))):
        parser.error('ACT Goal writes require --first-motion-trial or --continuous-act and all three physical acknowledgments')
    typed_init = input('Startup may enable torque and move the arm. Arm supported? Type YES: ')
    if typed_init.strip() != 'YES':
        parser.error('Hardware startup not armed')
    if not args.observe_only:
        typed = input('ACT Goal writes require separate authorization. Type YES: ')
        if typed.strip() != 'YES':
            parser.error('ACT Goal writes not armed')
    for name in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name]='1'
    from record_episode import hardware_preflight, calibration_paths
    from capture_session import Keyboard
    hardware_preflight(args.output)
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in calibration_paths()}
    predictor = ACTPredictor(args.device, args.checkpoint, image_keys=image_keys,
                             image_shape=image_shape)
    # Verify checkpoint camera keys before any motor initialization.
    io = HardwareAdapter(camera_cfg)
    io.startup_diagnostics_path = args.output/'startup_checkpoints.json'
    session = None
    cycle_rows = []
    primary_error = None
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        io.open()  # successful teleop startup path; includes Goal write and Alignment
        session = HandoffController(io,predictor,observe_only=args.observe_only,
                                   observe_frames=math.ceil(args.observe_seconds/PERIOD),
                                   first_motion_trial=args.first_motion_trial,
                                   continuous_act=args.continuous_act,
                                   act_seconds=args.act_seconds,
                                   act_speed_deg_s=args.act_speed_deg_s,
                                   act_gripper_speed_s=args.act_gripper_speed_s)
        print('Teleop active. x=stop Leader and run handoff check; Ctrl-C=FAULT stop. '
              + ('First-motion trial: max 1 s, 3 deg/s, 1 deg, gripper held.'
                 if args.first_motion_trial else
                 f'Continuous ACT: max {args.act_seconds:g} s; q=planned stop.'
                 if args.continuous_act else 'Observation only; ACT Goal writes disabled.'), flush=True)
        with Keyboard() as keyboard:
            last_cycle_start = None
            while True:
                start=time.perf_counter()
                key=keyboard.poll()
                before=session.phase
                try:session.tick(key)
                finally:
                    cycle_rows.append(dict(cycle=len(cycle_rows),
                        mode_before=before,mode_after=session.phase,key=key,
                        interval_ms=None if last_cycle_start is None else (start-last_cycle_start)*1000,
                        work_ms=(time.perf_counter()-start)*1000,
                        post_init_goal_write_attempts=io.goal_writes))
                last_cycle_start=start
                if session.phase in ('OBSERVED', 'STOPPED'):break
                time.sleep(max(0,PERIOD-(time.perf_counter()-start)))
        if session.trial_completion is not None:
            print('FIRST_MOTION_COMPLETE', flush=True)
            print('reason: ' + session.trial_completion['reason'], flush=True)
            print('ACT Goal writes: ' + str(session.trial_completion['goal_write_attempts_since_handoff']),
                  flush=True)
        if session.continuous_completion is not None:
            print('CONTINUOUS_ACT_COMPLETE', flush=True)
            print('reason: ' + session.continuous_completion['reason'], flush=True)
            print('ACT Goal writes: ' + str(session.continuous_completion['goal_write_attempts_since_handoff']),
                  flush=True)
    except BaseException as exc:
        primary_error = exc
        if session is not None:
            try:
                session._record('stopped', reason=repr(exc))
            except BaseException as record_exc:
                print(f'FAULT EVENT SAVE FAILED: {type(record_exc).__name__}: {record_exc}',
                      file=sys.stderr, flush=True)
    finally:
        try:
            cleanup_errors = finalize_handoff_run(args, io, session, cycle_rows, hashes, primary_error)
        except BaseException as finalize_exc:
            message = f'REPORT FINALIZATION FAILED: {type(finalize_exc).__name__}: {finalize_exc}'
            print(message, file=sys.stderr, flush=True)
            cleanup_errors = [message]
    raise_run_fault(primary_error, cleanup_errors)


if __name__ == '__main__':main()
