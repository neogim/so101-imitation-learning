"""Offline-first multi-episode control. Only the main thread may touch motors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty

from communication_guard import GuardedIO
from j6_closing_guard import limit_j6_closing

from record_episode import (NAMES, RECORD_FPS, SyntheticCamera, validate_destination,
                            hardware_preflight, make_hardware, start_hardware, close_hardware)

TASK = 'Pick up the Muhi bottle and place it into the target container.'
PERIOD = 1 / RECORD_FPS


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


class DatasetSink:
    """One owner of dataset mutations; bounded nonblocking submissions from control."""
    def __init__(self, dataset, root, capacity=64):
        self.dataset, self.root = dataset, root
        self.queue = queue.Queue(capacity)
        self.error = None
        self.saved = 0
        self.thread = threading.Thread(target=self._run, name='dataset-writer', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Dataset writer failed; inspect writer_error.txt') from self.error

    def submit(self, kind, value=None):
        self.check()
        try:
            self.queue.put_nowait((kind, value))
        except queue.Full:
            raise RuntimeError('Dataset queue full; refusing to drop frames') from None

    def _run(self):
        import traceback
        try:
            while True:
                kind, value = self.queue.get()
                if kind == 'frame':
                    self.dataset.add_frame(value)
                elif kind == 'save':
                    self.dataset.save_episode(parallel_encoding=False)
                    self.saved += 1
                elif kind == 'labels':
                    atomic_json(self.root / 'episode_outcomes.json', value)
                elif kind == 'close':
                    # Also preserves frames queued before an interruption/full queue.
                    if self.dataset.has_pending_frames():
                        self.dataset.save_episode(parallel_encoding=False)
                        self.saved += 1
                    break
                else:
                    raise ValueError(kind)
        except BaseException as exc:
            self.error = exc
            (self.root / 'writer_error.txt').write_text(traceback.format_exc())
        finally:
            try:
                self.dataset.finalize()
            except BaseException as exc:
                if self.error is None:
                    self.error = exc
                with (self.root / 'writer_error.txt').open('a') as f:
                    f.write(traceback.format_exc())

    def close(self):
        # Control has stopped and motor hold/disconnection already occurred.
        while self.thread.is_alive():
            try:
                self.queue.put(('close', None), timeout=.1)
                break
            except queue.Full:
                pass
        self.thread.join()
        self.check()


class Keyboard:
    """cbreak preserves Ctrl-C signal handling; polling never waits for a line."""
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('Hardware control requires an interactive terminal')
        import codecs
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        self.polls = 0
        self.bytes_read = 0
        self.entered_at = time.time()
        tty.setcbreak(self.fd)
        return self

    def poll(self):
        self.polls += 1
        if select.select([self.fd], [], [], 0)[0]:
            # Only this thread reads stdin. One byte per cycle bounds key handling.
            data = os.read(self.fd, 1)
            self.bytes_read += len(data)
            value = self.decoder.decode(data)
            return value
        return ''

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class HardwareIO:
    def __init__(self, robot, leader, camera):
        self.robot, self.leader, self.camera = robot, leader, camera

    def communication_settings(self, register, role):
        device = self.robot if role == 'follower' else self.leader
        port = device.bus.port_handler
        baud = port.getBaudRate()
        length = 2 if register in ('Present_Position', 'Present_Load', 'Present_Current') else 1
        count = 1 if register in ('Present_Load', 'Present_Current') else 6
        # Installed Feetech SDK's patched setPacketTimeout formula, unchanged.
        packet_ms = (1000. / baud)*10*((6+length)*count+3)+50
        return dict(baudrate=baud, pyserial_timeout_s=port.ser.timeout,
                    calculated_packet_timeout_ms=packet_ms,
                    packet_timeout_snapshot_ms=port.packet_timeout,
                    configured_num_read_retries=getattr(device.config,'num_read_retries',None),
                    effective_num_read_retries=0, nominal_hz=20)

    def state(self):
        data = self.robot.bus.sync_read('Present_Position', num_retry=0)
        return {k + '.pos': v for k, v in data.items()}

    def request(self):
        return {k+'.pos': v for k,v in self.leader.bus.sync_read('Present_Position', num_retry=0).items()}

    def j6_register(self, register):
        if register not in ('Present_Load', 'Present_Current'):
            raise ValueError('Unsupported J6 diagnostic register')
        return self.robot.bus.read(register, 'gripper', normalize=False, num_retry=0)

    def send(self, value):
        return self.robot.send_action(value)

    def image(self):
        return self.camera.read_latest()

    def images(self):
        if hasattr(self.camera, 'sample_all'):
            return self.camera.sample_all()
        image = self.camera.read_latest()
        with self.camera.frame_lock:
            stamp = self.camera.latest_timestamp
        return {'iphone': image}, {'iphone': stamp}

    def torque_ok(self):
        values = self.robot.bus.sync_read('Torque_Enable', normalize=False, num_retry=0)
        return len(values) == 6 and all(v == 1 for v in values.values())


class SimulationIO:
    def __init__(self, image_names=('iphone',), image_shape=(480, 640, 3)):
        self.positions = dict.fromkeys(NAMES, 0.)
        self.camera = SyntheticCamera()
        self.image_names = tuple(image_names)
        self.image_shape = tuple(image_shape)
        self.sent = []
        self.requests = 0
        self.torque = True

    def state(self):
        return self.positions.copy()

    def request(self):
        self.requests += 1
        return self.positions.copy()

    def send(self, values):
        self.positions = values.copy()
        self.sent.append(values.copy())
        return values.copy()

    def image(self):
        return self.camera.read_latest()

    def images(self):
        frame, stamp = self.image(), time.perf_counter()
        if frame.shape != self.image_shape:
            from PIL import Image
            import numpy as np
            frame = np.asarray(Image.fromarray(frame).resize(
                (self.image_shape[1], self.image_shape[0]), Image.Resampling.BILINEAR))
        return ({name: frame.copy() for name in self.image_names},
                {name: stamp for name in self.image_names})

    def torque_ok(self):
        return self.torque

    def j6_register(self, register):
        return {'Present_Load': 0, 'Present_Current': 0}[register]


def calibration_limits(follower):
    """Convert existing calibrated endpoints using the installed bus implementation."""
    limits = {}
    for name, motor in follower.bus.motors.items():
        c = follower.calibration[name]
        ends = [follower.bus._normalize({motor.id: x})[motor.id]
                for x in (c.range_min, c.range_max)]
        limits[name + '.pos'] = sorted(ends)
    return limits


class Session:
    def __init__(self, io, sink, *, seconds=40, num_episodes=1, task=TASK,
                 min_seconds=1, countdown=3, limits=None, slew=30, display=None,
                 image_names=('iphone',), image_shape=(480, 640, 3), j6_telemetry_hz=0,
                 j6_max_closing_lead=None, rotary_slew=None, gripper_slew=None):
        self.io, self.sink = GuardedIO(io), sink
        self.display = display
        self.display_snapshot = None
        self.display_error = None
        self.seconds, self.num_episodes, self.task = seconds, num_episodes, task
        self.image_names = tuple(image_names)
        self.image_shape = tuple(image_shape)
        if not math.isfinite(j6_telemetry_hz) or not 0 <= j6_telemetry_hz <= RECORD_FPS:
            raise ValueError('J6 telemetry frequency must be 0..20 Hz')
        self.j6_telemetry_hz = j6_telemetry_hz
        if j6_max_closing_lead is not None and not (math.isfinite(j6_max_closing_lead)
                                                    and 0 < j6_max_closing_lead < 8):
            raise ValueError('J6 closing lead must be >0 and < tracking limit 8')
        self.j6_max_closing_lead = j6_max_closing_lead
        self.j6_next_sample = None
        self.j6_diagnostics = []
        self.j6_guard_events = []
        if not self.image_names or len(set(self.image_names)) != len(self.image_names):
            raise ValueError('Expected unique enabled camera names')
        self.min_seconds, self.countdown, self.slew = min_seconds, countdown, slew
        self.rotary_slew = slew if rotary_slew is None else rotary_slew
        self.gripper_slew = slew if gripper_slew is None else gripper_slew
        if not all(math.isfinite(x) and x > 0 for x in (self.rotary_slew, self.gripper_slew)):
            raise ValueError('Rotary and gripper slew must be finite and positive')
        self.limits = limits or {n: (0, 100) if n == 'gripper.pos' else (-180, 180) for n in NAMES}
        self.mode = 'SAFE_RESET'
        self.hold = None
        self.last_sent = None
        self.safe_since = None
        self.deadline = None
        self.started = None
        self.frames = 0
        self.outcomes = []
        self.events = []
        self.timings = []
        self.frame_timings = []
        self.note = None
        self.done = False
        self.last_tick = None
        self.countdown_target = None
        self.countdown_remaining = None
        self.last_requested = None
        self.diagnostics = []
        self.key_events = []
        self.key_decision = None

    def checked(self, values):
        if set(values) != set(NAMES):
            raise ValueError('Unexpected joint keys')
        for n in NAMES:
            x = values[n]
            lo, hi = self.limits[n]
            if not math.isfinite(x) or not lo - .2 <= x <= hi + .2:
                raise ValueError(f'Joint outside calibrated range: {n}={x} [{lo}, {hi}]')
        return values

    def transition(self, mode, now):
        self.events.append(dict(time=now, previous=self.mode, mode=mode))
        self.mode = mode
        print(f'[{mode}] episodes={len(self.outcomes)}/{self.num_episodes}', flush=True)

    def enter_hold(self, state, now):
        # Explicit entry action only. All following SAFE_RESET cycles are read-only.
        self.hold = dict(self.checked(state))
        self.last_sent = self.checked(self.io.send(self.hold))
        if self.display_snapshot is not None:self.display_snapshot['sent']=self.last_sent['gripper.pos']
        self.safe_since = now
        self.deadline = None
        self.transition('SAFE_RESET', now)

    def labels(self):
        # Snapshot avoids concurrent mutation by keyboard input and writer thread.
        self.sink.submit('labels', json.loads(json.dumps(self.outcomes)))

    def finish(self, reason, now, *, resume_teleop=True):
        if self.frames:
            self.outcomes.append(dict(episode_index=len(self.outcomes),
                duration_s=now-self.started, nominal_duration_s=self.frames/RECORD_FPS,
                frames=self.frames, end_reason=reason, pick='unknown', place='unknown',
                overall='unknown', adopt=False, notes=''))
            self.frames = 0
            # If save submission fails, sink.close still saves buffered frames.
            self.sink.submit('save')
            self.labels()
        if resume_teleop:
            self.transition('TELEOP_RESET', now)

    def resume(self, state, now, record):
        if self.safe_since is None or now-self.safe_since < .5:
            self.key_decision = 'REFUSED: hold has not been monitored for 0.5 seconds'
            print(self.key_decision, flush=True)
            return
        if record and (len(self.outcomes) >= self.num_episodes or self.sink.saved < len(self.outcomes)):
            self.key_decision = 'REFUSED: episode limit reached or previous episode still saving'
            print(self.key_decision, flush=True)
            return
        target = self.checked(self.io.request())
        if self.display_snapshot is not None:self.display_snapshot['requested']=target['gripper.pos']
        if max(abs(target[n]-state[n]) for n in NAMES) > 3:
            self.key_decision = 'REFUSED: Leader/Follower pose difference >3'
            print(self.key_decision + '. Match poses, then retry. Do not push Follower.', flush=True)
            return
        self.last_sent = state.copy()
        if record:
            self.deadline = now+self.countdown
            self.countdown_target = target.copy()
            self.countdown_remaining = None
            self.transition('COUNTDOWN', now)
            self.key_decision = 'ACCEPTED: COUNTDOWN'
        else:
            self.transition('TELEOP_RESET', now)
            self.key_decision = 'ACCEPTED: TELEOP_RESET'

    def tick(self, now, key=''):
        start = time.perf_counter()
        cycle = getattr(self, 'cycle_number', 0)
        self.cycle_number = cycle+1
        self.io.context = dict(mode=self.mode, cycle=cycle, cycle_start_monotonic=now,
                               episode_elapsed_s=None if self.started is None else now-self.started)
        entry = dict(self.io.context, previous_interval_s=None if self.last_tick is None else now-self.last_tick)
        self.cycle_timing = entry
        prior = self.display_snapshot or {}
        self.display_snapshot = dict(mode=self.mode,cycle=cycle,measured=prior.get('measured'),
            measured_at=prior.get('measured_at'),measured_utc=prior.get('measured_utc'),
            previous_sent=None if self.last_sent is None else self.last_sent['gripper.pos'],
            error=None,requested=None,sent=None,event='normal',communication_fault=False,measurement_this_cycle=False)
        self.key_decision = None
        key_event = None
        if key:
            key_label = '<note-character>' if self.note is not None and key not in ('\n','\r','\x1b') else repr(key)
            key_event = dict(time_monotonic=now, cycle=cycle, mode_before=self.mode,
                             key=key_label, decision=None, mode_after=None)
            self.key_events.append(key_event)
            prefix='\n' if self.display is not None and getattr(self.display,'terminal',False) else ''
            print(f'{prefix}KEY RECEIVED: {key_label} in {self.mode} (cycle {cycle})', flush=True)
        try:
            self._tick(now, key)
            entry['ok'] = True
        except BaseException as exc:
            entry.update(ok=False, exception=repr(exc))
            self.display_snapshot.update(event=type(exc).__name__+': '+str(exc),communication_fault=self.io.latched)
            raise
        finally:
            if key_event is not None:
                key_event['mode_after'] = self.mode
                if self.key_decision is None:
                    if key == 'e' and self.mode == 'RECORDING':
                        self.key_decision = f'REFUSED: minimum recording duration {self.min_seconds:g}s not reached'
                    elif key in ('r','t') and self.mode == 'COUNTDOWN':
                        self.key_decision = 'IGNORED: countdown already active'
                    elif self.mode != key_event['mode_before']:
                        self.key_decision = 'ACCEPTED: mode changed'
                    else:
                        self.key_decision = f'IGNORED: {repr(key)} has no action in {self.mode}'
                key_event['decision'] = self.key_decision
                print(f"KEY RESULT: {key_event['decision']}; mode={self.mode}", flush=True)
            entry['work_s'] = time.perf_counter()-start
            self.io.cycles.append(entry)
            self.cycle_timing = None
            self.display_snapshot['mode'] = self.mode
            if self.display is not None:
                try:self.display.submit(self.display_snapshot)
                except Exception as exc:
                    self.display_error=repr(exc)  # A failed viewer must not interrupt control/storage.
                    self.display=None

    def _tick(self, now, key=''):
        if self.mode in ('STOPPING', 'STOPPED'):
            raise RuntimeError('Session stopped; restart requires explicit setup')
        if self.io.latched:
            raise ConnectionError('Communication fault latched; restart requires explicit setup')
        self.sink.check()
        if self.last_tick is not None and now-self.last_tick > .25:
            raise RuntimeError('Control gap >250 ms; no catch-up movement')
        self.last_tick = now
        state = self.checked(self.io.state())
        observed_at = time.perf_counter()
        previous = self.display_snapshot['previous_sent']
        self.display_snapshot.update(measured=state['gripper.pos'],measured_at=observed_at,
            measured_utc=time.time(),measurement_this_cycle=True,error=None if previous is None else abs(state['gripper.pos']-previous))
        if not self.io.torque_ok():
            raise RuntimeError('Follower torque is not enabled; hold cannot be guaranteed')
        if self.hold is None:
            self.enter_hold(state, now)
        if self.mode in ('SAFE_RESET', 'COUNTDOWN'):
            if max(abs(state[n]-self.hold[n]) for n in NAMES) > 2:
                raise RuntimeError('Hold drift >2; hold cannot be guaranteed')
        if self.note is not None:
            self.key_decision = 'NOTE EDIT: key consumed by note entry'
            if key in ('\n', '\r'):
                self.outcomes[-1]['notes'] = self.note
                self.note = None
                self.labels()
            elif key == '\x1b':
                self.note = None
            elif key in ('\x7f', '\b'):
                self.note = self.note[:-1]
                print('\rNote: ' + self.note + '\x1b[K', end='', flush=True)
            elif key.isprintable():
                self.note = (self.note + key)[:1000]
                print('\rNote: ' + self.note + '\x1b[K', end='', flush=True)
            return
        if self.mode == 'RECORDING' and ((key == 'e' and now-self.started >= self.min_seconds)
                                        or now-self.started >= self.seconds):
            reason = 'time_limit' if now-self.started >= self.seconds else 'manual'
            if key == 'e':self.key_decision = 'ACCEPTED: episode ended'
            self.finish(reason, now)
            key = ''
        if key == 'h' and self.mode == 'TELEOP_RESET':
            self.enter_hold(state, now)
            self.key_decision = 'ACCEPTED: SAFE_RESET'
        elif self.mode == 'SAFE_RESET':
            if key == 'q':
                self.done = True
                self.key_decision = 'ACCEPTED: quit'
                return
            if key in ('r', 't'):
                self.resume(state, now, record=key == 'r')
            elif self.outcomes:
                row = self.outcomes[-1]
                mapping = {'p': ('pick', 'success'), 'P': ('pick', 'failure'),
                           'l': ('place', 'success'), 'L': ('place', 'failure'),
                           's': ('overall', 'success'), 'f': ('overall', 'failure'),
                           'u': ('overall', 'unknown')}
                if key in mapping:
                    field, value = mapping[key]
                    row[field] = value
                    row['adopt'] = False
                    self.labels()
                    print(json.dumps(row, ensure_ascii=False))
                    self.key_decision = 'ACCEPTED: episode label updated'
                elif key == 'a':
                    eligible = all(row[k] == 'success' for k in ('pick', 'place', 'overall')) and row['end_reason'] != 'interrupted'
                    row['adopt'] = not row['adopt'] if eligible else False
                    self.labels()
                    print(f"adopt={row['adopt']} (requires Pick/Place/overall success)")
                    self.key_decision = 'ACCEPTED: adopt toggled' if eligible else 'REFUSED: Pick, Place and overall must be successful'
                elif key == 'n':
                    self.note = ''
                    self.key_decision = 'ACCEPTED: note entry'
                    print('Note: type then Enter to save; Escape cancels. Hold monitoring continues.')
        if self.mode == 'COUNTDOWN':
            target = self.checked(self.io.request())
            if self.display_snapshot is not None:self.display_snapshot['requested']=target['gripper.pos']
            if (max(abs(target[n]-state[n]) for n in NAMES) > 3
                or max(abs(target[n]-self.countdown_target[n]) for n in NAMES) > 3):
                self.deadline = None
                self.transition('SAFE_RESET', now)
                print('COUNTDOWN CANCELLED: Leader moved. Match poses and press r again.')
                return
            remaining = max(0, math.ceil(self.deadline-now))
            if remaining != self.countdown_remaining:
                print(f'COUNTDOWN: {remaining} seconds remaining. Keep Leader still; follow OFF.', flush=True)
                self.countdown_remaining = remaining
            if now < self.deadline:
                return
            self.started, self.frames = now, 0
            self.transition('RECORDING', now)
            print('RECORDING STARTED NOW: follow ON, recording ON. e=end episode.', flush=True)
        if self.mode not in ('RECORDING', 'TELEOP_RESET'):
            return
        joints = {n: dict(measured=state[n], previous_sent=self.last_sent[n],
                    signed_error=state[n]-self.last_sent[n],
                    abs_error=abs(state[n]-self.last_sent[n]), threshold=8,
                    unit='normalized_0_100' if n == 'gripper.pos' else 'degrees') for n in NAMES}
        offenders = [n for n in NAMES if joints[n]['abs_error'] > 8]
        if offenders:
            self.diagnostics.append(dict(kind='tracking_error', time_monotonic=now,
                mode=self.mode, episode_index=len(self.outcomes), queued_frame_count=self.frames,
                episode_elapsed_s=None if self.started is None else now-self.started,
                offending_joints=offenders, joints=joints, slew=self.slew,
                rotary_slew=self.rotary_slew, gripper_slew=self.gripper_slew,
                last_leader_requested=self.last_requested,
                current_leader_requested=None,
                comparison='current measurement minus previous sent goal; before new Leader read/send'))
            raise RuntimeError('Tracking error >8; stopping follow: ' + ', '.join(
                f"{n} error={joints[n]['abs_error']:.6f} {joints[n]['unit']}" for n in offenders))
        j6_row = None
        if self.j6_telemetry_hz and (self.j6_next_sample is None or now >= self.j6_next_sample):
            j6_row = dict(cycle=self.cycle_number-1, mode=self.mode,
                       episode_index=len(self.outcomes), episode_frame_index=self.frames,
                       time_monotonic=now, time_utc=time.time(),
                       unit='normalized_0_100', measured_position=state['gripper.pos'],
                       previous_sent_goal=self.last_sent['gripper.pos'],
                       tracking_error=state['gripper.pos']-self.last_sent['gripper.pos'],
                       leader_requested_previous_cycle=None if self.last_requested is None
                       else self.last_requested['gripper.pos'])
            self.j6_diagnostics.append(j6_row)
            self.j6_next_sample = now + 1 / self.j6_telemetry_hz
            try:
                j6_row.update(self.io.j6_telemetry())
                reads = list(self.io.history)[-2:]
                j6_row['reads'] = {r['register']: dict(started_monotonic=r['callback_started_monotonic'],
                    finished_monotonic=r['callback_finished_monotonic'],
                    duration_s=r['callback_duration_s']) for r in reads}
                j6_row['completed_monotonic'] = time.perf_counter()
            except BaseException as exc:
                j6_row['error'] = f'{type(exc).__name__}: {exc}'
                raise
        requested = self.checked(self.io.request())
        if j6_row is not None:
            j6_row['leader_requested_this_cycle'] = requested['gripper.pos']
        self.display_snapshot['requested']=requested['gripper.pos']
        self.last_requested = requested.copy()
        command = {}
        for n in NAMES:
            step = (self.gripper_slew if n == 'gripper.pos' else self.rotary_slew) * PERIOD
            command[n] = self.last_sent[n]+max(-step, min(step, requested[n]-self.last_sent[n]))
        guard_event = None
        if self.j6_max_closing_lead is not None:
            original = command['gripper.pos']
            command['gripper.pos'], reason = limit_j6_closing(
                measured=state['gripper.pos'], previous_goal=self.last_sent['gripper.pos'],
                requested=requested['gripper.pos'], proposed_goal=original,
                max_lead=self.j6_max_closing_lead)
            guard_event = dict(cycle=self.cycle_number-1, mode=self.mode,
                time_monotonic=now, episode_index=len(self.outcomes), frame_index=self.frames,
                measured=state['gripper.pos'], previous_goal=self.last_sent['gripper.pos'],
                leader_requested=requested['gripper.pos'], original_goal=original,
                guarded_goal=command['gripper.pos'], reason=reason,
                unit='normalized_0_100')
            self.j6_guard_events.append(guard_event)
        # A multi-camera dropout must be detected before issuing this cycle's Goal.
        # Keep the legacy single-camera acquisition order unchanged.
        images = image_times = None
        if self.mode == 'RECORDING' and self.image_names != ('iphone',):
            self.cycle_timing['image_check_started_monotonic'] = time.perf_counter()
            try:
                images, image_times = self.io.images()
                self._validate_images(images, image_times)
            finally:
                self.cycle_timing['image_check_finished_monotonic'] = time.perf_counter()
                try:
                    self.cycle_timing['cameras'] = self.io.camera_timing()
                except Exception as exc:
                    self.cycle_timing['camera_timing_error'] = f'{type(exc).__name__}: {exc}'
        sent = self.checked(self.io.send(command))
        if guard_event is not None:
            guard_event['sent_goal'] = sent['gripper.pos']
        if j6_row is not None:
            j6_row['sent_goal_this_cycle'] = sent['gripper.pos']
        sent_at = time.perf_counter()
        self.last_sent = sent.copy()
        self.display_snapshot['sent']=sent['gripper.pos']
        if self.mode == 'RECORDING':
            import numpy as np
            if images is None:
                images, image_times = self.io.images()
                self._validate_images(images, image_times)
            frame = {k: np.asarray([v[n] for n in NAMES], dtype=np.float32) for k, v in
                     [('observation.state', state), ('action', sent), ('action_requested', requested)]}
            frame.update({f'observation.images.{name}': image.copy() for name, image in images.items()})
            frame['task'] = self.task
            self.sink.submit('frame', frame)
            self.frame_timings.append(dict(episode_index=len(self.outcomes), frame_index=self.frames,
                observed_at=observed_at, sent_at=sent_at,
                image_captured_at=image_times, image_enqueued_at=time.perf_counter()))
            self.frames += 1

    def _validate_images(self, images, image_times):
        import numpy as np
        if set(images) != set(self.image_names) or set(image_times) != set(self.image_names):
            raise ValueError('Enabled camera keys differ from Dataset features')
        for name, image in images.items():
            if image.shape != self.image_shape or image.dtype != np.uint8:
                raise ValueError(f'{name}: invalid RGB contract: {image.shape}, {image.dtype}')

    def stop(self, now):
        """Stop follow first. No torque-off and no retry loop on communication failure."""
        if self.mode == 'STOPPED':
            return []
        self.transition('STOPPING', now)
        errors = []
        try:
            if self.io.latched:
                raise ConnectionError('Communication fault latched: cleanup performs NO reads or Goal_Position writes')
            self.io.context = dict(mode='STOPPING', cycle=getattr(self,'cycle_number',0))
            target = dict(self.checked(self.io.state()))
            # Do not automatically open/reset the gripper on a fault: this can drop an object.
            # Retaining the previous goal does NOT guarantee a safe grip force.
            if self.last_sent is not None:
                target['gripper.pos'] = self.last_sent['gripper.pos']
            self.io.send(target)
        except Exception as exc:
            errors.append(f'Hold unavailable: {exc}')
        try:
            if self.frames:
                self.finish('interrupted', now, resume_teleop=False)
        except Exception as exc:
            errors.append(f'Partial episode: {exc}')
        self.transition('STOPPED', now)
        print('CONTROL STOPPED: servo may continue toward/hold its last goal. Physical stopping and holding NOT guaranteed. No torque-off.', flush=True)
        return errors


def create_dataset(root, simulated, image_names=('iphone',), image_shape=(480, 640, 3)):
    # Keep the validated cv2-before-PyAV import order.
    import cv2
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {k: dict(dtype='float32', shape=(6,), names=NAMES)
                for k in ('observation.state', 'action', 'action_requested')}
    for name in image_names:
        features[f'observation.images.{name}'] = dict(
            dtype='video', shape=image_shape, names=['height','width','channels'])
    return LeRobotDataset.create('local/so101_session', fps=20, root=root,
        robot_type='synthetic_so101' if simulated else 'so101_follower', features=features,
        use_videos=True, video_backend='pyav', rgb_encoder=RGBEncoderConfig(vcodec='h264'),
        image_writer_threads=2, image_writer_processes=0)


class ScriptedKeys:
    """Integration scenario: alternating timeout/manual episodes, then label success."""
    def __init__(self):
        self.label_step = {}

    def poll(self, s, now):
        if s.mode == 'RECORDING' and len(s.outcomes) % 2 and now-s.started >= max(s.min_seconds, s.seconds*.6):
            return 'e'
        if s.mode == 'TELEOP_RESET':
            return 'h'
        if s.mode == 'SAFE_RESET' and s.safe_since is not None and now-s.safe_since >= .5:
            if s.outcomes:
                ep = len(s.outcomes)-1
                step = self.label_step.get(ep, 0)
                if step < 4:
                    self.label_step[ep] = step+1
                    return 'plsa'[step]
                if len(s.outcomes) >= s.num_episodes:
                    return 'q'
            return 'r'
        return ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=40)
    p.add_argument('--num-episodes', type=int, default=1)
    p.add_argument('--min-seconds', type=float, default=1)
    p.add_argument('--countdown', type=float, default=3)
    p.add_argument('--alignment-timeout', type=float, default=60,
                   help='Startup alignment deadline in seconds (60..120; default 60)')
    p.add_argument('--alignment-tolerance', type=float, default=2.0,
                   help='Startup alignment residual limit (0..2.5, default 2.0; degrees except gripper normalized)')
    p.add_argument('--task', default=TASK)
    p.add_argument('--gripper-display', action=argparse.BooleanOptionalAction, default=False, help='Display-only gripper monitor; no extra serial reads')
    p.add_argument('--gripper-display-hz', type=float, default=4)
    p.add_argument('--j6-telemetry-hz', type=float, default=0,
                   help='J6 Present_Load/Current diagnostic reads per second; 0 disables (default)')
    p.add_argument('--j6-max-closing-lead', type=float, default=None,
                   help='Opt-in J6 position lead cap (normalized units, <8); no validated safe default')
    p.add_argument('--slew', type=float, default=30, help='Maximum goal change: degrees/s; gripper normalized units/s')
    p.add_argument('--rotary-slew', type=float, default=None,
                   help='J1-J5 degrees/s; overrides --slew for rotary joints only')
    p.add_argument('--gripper-slew', type=float, default=None,
                   help='J6 normalized units/s; overrides --slew for gripper only')
    p.add_argument('--camera-config', type=Path, help='Independent OBS input sources; omit for legacy iphone virtual camera')
    p.add_argument('--enable-cameras', help='Comma-separated YAML logical names overriding enabled flags')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--simulate', action='store_true')
    a = p.parse_args()
    if a.enable_cameras and not a.camera_config:
        p.error('--enable-cameras requires --camera-config')
    camera_cfg = None
    image_names = ('iphone',)
    image_shape = (480, 640, 3)
    if a.camera_config:
        from multi_camera import capture_image_shape, load_config
        camera_cfg = load_config(a.camera_config, a.enable_cameras)
        image_names = tuple(camera_cfg['active_cameras'])
        image_shape = capture_image_shape(camera_cfg)
    if (not all(math.isfinite(x) for x in (a.seconds,a.min_seconds,a.countdown,a.slew))
        or not 0 < a.min_seconds <= a.seconds or a.countdown < 0 or a.slew <= 0 or a.num_episodes < 1 or not a.task.strip()):
        p.error('Require 0 < min-seconds <= seconds, countdown >=0, slew >0, episodes >=1 and nonempty task')
    if not math.isfinite(a.gripper_display_hz) or not 2<=a.gripper_display_hz<=5:
        p.error('--gripper-display-hz must be 2..5')
    if not math.isfinite(a.j6_telemetry_hz) or not 0 <= a.j6_telemetry_hz <= RECORD_FPS:
        p.error('--j6-telemetry-hz must be 0..20')
    if any(x is not None and (not math.isfinite(x) or x <= 0)
           for x in (a.rotary_slew, a.gripper_slew)):
        p.error('--rotary-slew and --gripper-slew must be finite and positive')
    if a.j6_max_closing_lead is not None and (not math.isfinite(a.j6_max_closing_lead)
                                              or not 0 < a.j6_max_closing_lead < 8):
        p.error('--j6-max-closing-lead must be >0 and <8 normalized units')
    if not math.isfinite(a.alignment_timeout) or not 60<=a.alignment_timeout<=120:
        p.error('--alignment-timeout must be 60..120 seconds')
    if not math.isfinite(a.alignment_tolerance) or not 0<a.alignment_tolerance<=2.5:
        p.error('--alignment-tolerance must be >0 and <=2.5')
    root = a.root.expanduser().resolve()
    validate_destination(root)
    for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    if not a.execute and not a.simulate:
        hardware_preflight(root)
        print('File/USB-path preflight passed. No camera or serial connection. Use --execute only after approval.')
        return
    if a.execute and not sys.stdin.isatty():
        p.error('--execute requires a real interactive terminal')
    camera = follower = leader = sink = session = display = keyboard = None
    hashes = {}
    configured = False
    error = None
    cleanup_errors = []
    def stop_signal(*_):
        raise KeyboardInterrupt('Termination signal')
    previous_signal = signal.signal(signal.SIGTERM, stop_signal)
    try:
        # Finish heavy imports/allocation before enabling or moving any motors.
        if not a.simulate:
            hashes = hardware_preflight(root)
        dataset = create_dataset(root, a.simulate, image_names, image_shape)
        sink = DatasetSink(dataset, root)
        if a.simulate:
            io = SimulationIO(image_names, image_shape)
            limits = None
        else:
            if camera_cfg:
                from multi_camera import CameraRig
                camera = CameraRig(camera_cfg)
            else:
                from obs_camera import OBSCamera
                camera = OBSCamera()
            camera.connect()  # use the index selected by the OBS adapter
            sample = camera.read_latest()
            if camera_cfg:
                if set(sample) != set(image_names):
                    raise ValueError('Camera rig keys differ from enabled features')
                if any(frame.shape != image_shape for frame in sample.values()):
                    raise ValueError('Camera rig image size differs from Dataset features')
            elif sample.shape != (480,640,3):
                raise ValueError('Camera size mismatch')
            follower, leader = make_hardware(camera)
            limits = calibration_limits(follower)
            original_configure = leader.configure
            def configure_leader():
                nonlocal configured
                configured = True
                original_configure()
            leader.configure = configure_leader
            start_hardware(follower, leader, alignment_timeout=a.alignment_timeout,
                           alignment_tolerance=a.alignment_tolerance,
                           alignment_diagnostics_path=root/'alignment_diagnostics.json')
            io = HardwareIO(follower, leader, camera)
        if a.gripper_display:
            from gripper_display import GripperDisplay
            try:display=GripperDisplay(hz=a.gripper_display_hz)
            except Exception as exc:print(f'Display unavailable: {exc}',flush=True)
        session = Session(io, sink, seconds=a.seconds, num_episodes=a.num_episodes, task=a.task,
                          min_seconds=a.min_seconds, countdown=a.countdown, limits=limits, slew=a.slew,
                          display=display, image_names=image_names, image_shape=image_shape,
                          j6_telemetry_hz=a.j6_telemetry_hz,
                          j6_max_closing_lead=a.j6_max_closing_lead,
                          rotary_slew=a.rotary_slew, gripper_slew=a.gripper_slew)
        atomic_json(root/'capture_config.json', dict(simulated=a.simulate, fps=20, seconds=a.seconds,
            num_episodes=a.num_episodes, task=a.task, countdown=a.countdown, min_seconds=a.min_seconds, use_degrees=True, gripper_unit='normalized_0_100',
            alignment_timeout_s=a.alignment_timeout, alignment_tolerance=a.alignment_tolerance,
            camera_index=None if camera_cfg or camera is None else camera.config.index_or_path,
            cameras=None if camera_cfg is None else
                ({name: cam.identity for name, cam in camera.cameras.items()}
                 if camera is not None else {name: None for name in image_names}),
            image_shape=list(image_shape), image_captured_at_clock='time.perf_counter response time',
            limits=session.limits, slew=a.slew,
            rotary_slew=session.rotary_slew, gripper_slew=session.gripper_slew,
            gripper_display=a.gripper_display,gripper_display_hz=a.gripper_display_hz,
            j6_telemetry_hz=a.j6_telemetry_hz,
            j6_max_closing_lead=a.j6_max_closing_lead,
            action='send_action return: limited goal, not measured position or acknowledgment',
            action_requested='Leader original position', calibration_sha256=hashes))
        print('r=start recording after hold/check/countdown; e=end episode; h=hold; t=teleop; q=quit from SAFE_RESET\n'
              'SAFE_RESET labels: p/P Pick success/failure; l/L Place; s/f/u overall; a adopt; n note\n'
              'Ctrl-C stops follow and saves partial episode. Torque is NOT disabled on exit.\n'
              'After h, observe physical stability before touching objects. Hold cannot be guaranteed on faults.', flush=True)
        scripted = ScriptedKeys()
        keyboard = Keyboard() if a.execute else None
        if keyboard:
            keyboard.__enter__()
        try:
            from lerobot.utils.robot_utils import precise_sleep
            while not session.done:
                tick = time.perf_counter()
                key = keyboard.poll() if keyboard else scripted.poll(session, tick)
                if key and display:
                    display.pause(1.5)
                session.tick(tick, key)
                work = time.perf_counter()-tick
                session.timings.append(dict(time=tick, mode=session.mode, work_s=work))
                precise_sleep(max(0,PERIOD-work))  # no burst catch-up on overruns
        finally:
            if keyboard:
                keyboard.__exit__()
    except BaseException as exc:
        error = exc
        import traceback
        if root.exists():
            (root/'session_error.txt').write_text(traceback.format_exc())
    finally:
        # First stop control and close serial WITHOUT torque-off; only then wait for encoding.
        if session:
            cleanup_errors.extend(session.stop(time.perf_counter()))
        elif configured and follower and follower.bus.is_connected and not isinstance(error, (ConnectionError, OSError)):
            try:
                current = follower.bus.sync_read('Present_Position')
                if all(math.isfinite(v) for v in current.values()):
                    follower.bus.sync_write('Goal_Position', current)
            except Exception as exc:
                cleanup_errors.append(f'Initialisation hold unavailable: {exc}')
        try:
            close_hardware(follower, leader, False)
        except Exception as exc:
            cleanup_errors.append(str(exc))
        try:
            if camera and camera.is_connected:
                camera.disconnect()
        except Exception as exc:
            cleanup_errors.append(str(exc))
        if display:
            try:
                report=display.close()
                report['submit_error']=None if session is None else session.display_error
                atomic_json(root/'gripper_display.json',report)
            except Exception as exc:
                print(f'Display cleanup failed (dataset finalization continues): {exc}',flush=True)
        if session:
            try:
                atomic_json(root/'fault_diagnostics.json', session.diagnostics)
                atomic_json(root/'communication_diagnostics.json', session.io.report())
                if session.j6_telemetry_hz:
                    atomic_json(root/'j6_telemetry.json', dict(
                        frequency_hz=session.j6_telemetry_hz, registers_raw=True,
                        note='Motor telemetry is not fingertip grip force. No automatic contact control.',
                        samples=session.j6_diagnostics))
                if session.j6_max_closing_lead is not None:
                    atomic_json(root/'j6_closing_guard.json', dict(
                        max_lead_normalized=session.j6_max_closing_lead,
                        note='Position lead is not gripping force. A held Goal may still exert force.',
                        cycles=session.j6_guard_events))
            except Exception as exc:
                cleanup_errors.append(f'Diagnostic save: {exc}')
        try:
            if sink:
                sink.close()
        except Exception as exc:
            cleanup_errors.append(str(exc))
        if session:
            atomic_json(root/'episode_outcomes.json', session.outcomes)
            atomic_json(root/'session_events.json', session.events)
            atomic_json(root/'key_events.json', session.key_events)
            if keyboard and hasattr(keyboard,'polls'):
                atomic_json(root/'keyboard_diagnostics.json',dict(
                    reader_thread='control', input_is_tty=True, entered_at=keyboard.entered_at,
                    polls=keyboard.polls, bytes_read=keyboard.bytes_read,
                    decoded_key_events=len(session.key_events),
                    note='Zero key events means no decoded key reached the control loop; terminal delivery is not established.'))
            work = [r['work_s'] for r in session.timings]
            atomic_json(root/'capture_timing.json', dict(cycles=session.timings, frame_times=session.frame_timings, summary=dict(
                cycles=len(work), work_mean_ms=1000*sum(work)/max(1,len(work)),
                work_max_ms=1000*max(work,default=0), overrun_50ms=sum(x>PERIOD for x in work))))
        for path, before in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != before:
                cleanup_errors.append(f'Calibration changed: {path}')
        if root.exists():
            atomic_json(root/'shutdown.json', dict(error=repr(error) if error else None,
                cleanup_errors=cleanup_errors, torque_disabled=False,
                warning='Holding is not guaranteed on communication, torque or power faults. Support arm before power removal.'))
        signal.signal(signal.SIGTERM, previous_signal)
    if error:
        raise error
    if cleanup_errors:
        raise RuntimeError('; '.join(cleanup_errors))
    print(f'Saved {len(session.outcomes)} episodes to {root}. No upload. Support arm before removing power.')


if __name__ == '__main__':
    main()
