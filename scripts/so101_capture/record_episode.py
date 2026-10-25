"""SO-101 shared hardware guards and multi-episode launcher. Hardware requires approval."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback
from contextlib import contextmanager

RECORD_SECONDS = 40
RECORD_FPS = 20

def phase(message):
    print(f"[INIT] {message}", flush=True)

def report_exception(prefix, exc):
    print(f"[ERROR] {prefix}: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exception(type(exc), exc, exc.__traceback__)

NAMES = [n + '.pos' for n in ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')]
FOLLOWER = '/dev/tty.usbmodem5B7B0098201'
LEADER = '/dev/tty.usbmodem5B7B0185961'
STS3215_TICKS_PER_TURN = 4096


def deny_calibration(*args, **kwargs):
    raise RuntimeError('Calibration changes are forbidden. Stop and investigate the mismatch.')


def validate_calibrations(devices):
    # Call only after both buses are open, before either configure() can write.
    for device in devices:
        if not device.is_calibrated:
            raise RuntimeError(f'{device.name}: calibration mismatch; no automatic repair')


def validate_raw_positions_within_calibration(positions, calibration):
    """Reject raw servo positions outside the stored raw calibration limits."""
    out_of_range = []
    for name in sorted(positions):
        position = positions[name]
        range_min = calibration[name].range_min
        range_max = calibration[name].range_max
        if range_min <= position <= range_max:
            continue
        deviation = range_min - position if position < range_min else position - range_max
        direction = "increase" if position < range_min else "decrease"
        deviation_degrees = deviation * 360 / STS3215_TICKS_PER_TURN
        out_of_range.append(
            f"{name}: Present={position}, calibrated_range=[{range_min}, {range_max}], "
            f"deviation={deviation} raw ticks (~{deviation_degrees:.2f} deg). "
            f"With servo power off and the arm supported, reposition this joint so its raw Present "
            f"values {direction} back inside the calibrated range; leave margin from the endpoint. "
            "Do not force the joint or edit calibration to bypass this check."
        )
    if out_of_range:
        raise RuntimeError(
            "Present outside calibrated range before Goal synchronization. "
            "No Goal was written and configure/Alignment did not start:\n"
            + "\n".join(out_of_range)
        )


def validate_destination(root):
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite {root}')
    if not root.parent.is_dir() or not os.access(root.parent, os.W_OK):
        raise ValueError(f'Create a writable parent directory first: {root.parent}')


def calibration_paths():
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
    return [HF_LEROBOT_CALIBRATION/'robots/so_follower/my_follower_arm.json',
            HF_LEROBOT_CALIBRATION/'teleoperators/so_leader/my_leader_arm.json']


def hardware_preflight(root):
    validate_destination(root)
    problems = [f'Missing USB port: {port}' for port in (FOLLOWER, LEADER) if not Path(port).exists()]
    hashes = {}
    for path in calibration_paths():
        if not path.is_file():
            problems.append(f'Missing calibration: {path}')
            continue
        data = json.loads(path.read_text())
        if set(data) != {n.removesuffix('.pos') for n in NAMES}:
            problems.append(f'Unexpected calibration joints: {path}')
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if problems:
        raise RuntimeError('\n'.join(problems))
    return hashes


def make_hardware(camera):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER, id='my_follower_arm', use_degrees=True))
    leader = SO101Leader(SO101LeaderConfig(port=LEADER, id='my_leader_arm', use_degrees=True))
    # The multi-source rig is sampled by capture_session directly. Its OBS
    # inputs are not LeRobot Camera instances.
    follower.cameras = {} if hasattr(camera, 'sample_all') else {'iphone': camera}
    for device in (follower, leader):
        # Instance-local guards; no LeRobot files or global classes are patched.
        device.calibrate = deny_calibration
        device._save_calibration = deny_calibration
        device.bus.write_calibration = deny_calibration
    return follower, leader


def synchronize_positions(follower, leader, target, *, timeout=60, tolerance=2.0, diagnostics_path=None,
                          clock=time.perf_counter, sleep=time.sleep):
    """Ramp goals to a stationary Leader, keeping torque on for the next phase."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Alignment timeout must be positive and finite')
    if not math.isfinite(tolerance) or not 0 < tolerance <= 2.5:
        raise ValueError('Alignment tolerance must be in (0, 2.5]')
    current = follower.bus.sync_read('Present_Position')
    command = current.copy()
    start = clock()
    last_progress = start
    last_error = max(abs(target[k]-current[k]) for k in target)
    settled = 0
    cycle = 0
    cycle_period_s = None
    previous_tick = None
    snapshots = []
    last_snapshot_at = start
    live = target.copy()
    def snapshot(stage, now, reason=None):
        joints = {k: dict(leader_target=target[k], leader_live=live.get(k),
                          follower_measured=current.get(k), sent_goal=command.get(k),
                          target_measured_error=abs(target[k]-current[k]),
                          sent_measured_error=abs(command[k]-current[k]),
                          target_sent_error=abs(target[k]-command[k]),
                          unit='normalized_0_100' if 'gripper' in k else 'degrees')
                  for k in target if k in current and k in command}
        worst = max(joints, key=lambda k:joints[k]['target_measured_error']) if joints else None
        row = dict(stage=stage, elapsed_s=now-start, cycle=cycle,
                   cycle_period_s=cycle_period_s, settled_cycles=settled,
                   worst_joint=worst, worst_error=None if worst is None else joints[worst]['target_measured_error'],
                   reason=reason, joints=joints)
        snapshots.append(row)
        if diagnostics_path is not None:
            path=Path(diagnostics_path)
            tmp=path.with_suffix(path.suffix+'.tmp')
            tmp.write_text(json.dumps(dict(timeout_s=timeout, tolerance=tolerance,
                                            target_slew_per_cycle=.15,
                                            nominal_period_s=.05, snapshots=snapshots),indent=2))
            tmp.replace(path)
        if stage != 'periodic':
            print(f"ALIGNMENT {stage}: elapsed={row['elapsed_s']:.1f}s cycle={cycle} "
                  f"worst={worst} error={row['worst_error']}",flush=True)
    snapshot('start',start)
    print('ALIGNING: keep Leader still. Target slew <=3 deg/s (gripper 3 units/s).', flush=True)
    try:
        while clock()-start < timeout:
            tick = clock()
            cycle += 1
            cycle_period_s = None if previous_tick is None else tick-previous_tick
            previous_tick = tick
            live = leader.bus.sync_read('Present_Position')
            current = follower.bus.sync_read('Present_Position')
            if set(current) != set(target) or set(live) != set(target):
                raise ValueError('Unexpected joint keys during alignment')
            if not all(math.isfinite(v) for v in [*current.values(), *live.values()]):
                raise ValueError('Nonfinite position during alignment')
            if any(abs(live[k]-target[k]) > 3 for k in target):
                raise RuntimeError('Leader moved during alignment; stopping')
            if any(abs(command[k]-current[k]) > 8 for k in target):
                raise RuntimeError('Alignment tracking error >8; stopping')
            error = max(abs(target[k]-current[k]) for k in target)
            if error < last_error-.3:
                last_error, last_progress = error, tick
            if error > 3 and tick-last_progress > 3:
                raise RuntimeError('Alignment stalled for 3 seconds; stopping')
            command = {k: command[k]+max(-.15,min(.15,target[k]-command[k])) for k in target}
            follower.bus.sync_write('Goal_Position', command)
            settled = settled+1 if error < tolerance and max(abs(command[k]-target[k]) for k in target)<.2 else 0
            if tick-last_snapshot_at >= 1:
                snapshot('periodic',tick)
                print(f'ALIGNING: elapsed={tick-start:.1f}s cycle={cycle} '
                      f'worst={snapshots[-1]["worst_joint"]} error={snapshots[-1]["worst_error"]:.3f}',flush=True)
                last_snapshot_at=tick
            if settled >= 10:
                snapshot('complete',clock())
                print('ALIGNMENT COMPLETE. Keeping connection for operation.', flush=True)
                return
            sleep(max(0,.05-(clock()-tick)))
        raise RuntimeError('Alignment timed out; stopping')
    except BaseException as exc:
        snapshot('stopped',clock(),f'{type(exc).__name__}: {exc}')
        raise


def start_hardware(follower, leader, *, alignment_timeout=60, alignment_tolerance=2.0,
                   alignment_diagnostics_path=None, guarded_teleop_start=False,
                   startup_checkpoints=None):
    # Avoid standard connect()'s automatic calibration path. No configure writes
    # occur until BOTH devices pass the existing read-only calibration check.
    phase('Opening Leader serial bus')
    leader.bus.connect()
    phase('Opening Follower serial bus')
    follower.bus.connect()
    phase('Checking existing calibrations (read-only)')
    validate_calibrations((leader, follower))
    current = follower.bus.sync_read('Present_Position')
    target = leader.bus.sync_read('Present_Position')
    for name in current:
        if not math.isfinite(current[name]) or not math.isfinite(target[name]):
            raise ValueError('Nonfinite initial joint position')
        limit = 100 if name == 'gripper' else 120
        difference = abs(current[name] - target[name])
        if difference > limit:
            unit = 'normalized_0_100' if name == 'gripper' else 'degrees'
            raise RuntimeError(
                f'Initial pose mismatch exceeds alignment limit: {name}; '
                f'Leader={target[name]:.3f}, Follower={current[name]:.3f}, '
                f'difference={difference:.3f}, limit={limit} {unit}')
    phase('Configuring Leader')
    leader.configure()
    phase('Preparing Follower current-position hold')
    # Standard follower.configure() may enable torque on exit. Set its current
    # position as goal first, so a stale goal is not used when enabling torque.
    follower.bus.disable_torque()
    current = follower.bus.sync_read('Present_Position')
    if guarded_teleop_start:
        motors = set(follower.bus.motors)
        def raw_snapshot():
            registers = ('Goal_Position', 'Present_Position', 'Torque_Enable', 'Status')
            snapshot = {reg: follower.bus.sync_read(reg, normalize=False, num_retry=0)
                        for reg in registers}
            if any(set(snapshot[reg]) != motors for reg in registers):
                raise RuntimeError('Incomplete six-axis startup readback')
            return snapshot

        def record_snapshot(stage, snapshot):
            if startup_checkpoints is not None:
                startup_checkpoints.append(dict(stage=stage, sampled_at_monotonic=time.perf_counter(),
                                                raw=snapshot))

        before = raw_snapshot()
        record_snapshot('before_goal_write', before)
        if any(before['Torque_Enable'][n] != 0 or before['Status'][n] != 0 for n in motors):
            raise RuntimeError('Torque/Status unexpected before Goal synchronization')
        limits = follower.bus.calibration

        if set(limits) != motors:
            missing = sorted(motors - set(limits))
            extra = sorted(set(limits) - motors)
            raise RuntimeError(
                "Calibration motor mismatch before Goal synchronization: "
                f"missing={missing}, extra={extra}"
            )

        validate_raw_positions_within_calibration(before["Present_Position"], limits)
        target_raw = before['Present_Position'].copy()
        # One broadcast only. A failed transmission has ambiguous delivery: never retry it.
        follower.bus.sync_write('Goal_Position', target_raw, normalize=False, num_retry=0)
        # STS3215 sync_write has no status ACK. Give the bus time to settle
        # before the first six-servo sync_read; this is not a write retry.
        time.sleep(.02)

        def check_readback(stage, *, expected_torque):
            try:
                snap = raw_snapshot()
            except Exception as exc:
                if startup_checkpoints is not None:
                    startup_checkpoints.append(dict(stage=stage, sampled_at_monotonic=time.perf_counter(),
                                                    error=f'{type(exc).__name__}: {exc}'))
                raise  # No configure, Alignment, Goal retry or automatic recovery.
            record_snapshot(stage, snap)
            torques = {snap['Torque_Enable'][n] for n in motors}
            if torques not in ({0}, {1}) or (expected_torque is not None and torques != {expected_torque}):
                raise RuntimeError(f'{stage}: mixed or unexpected Torque_Enable {torques}')
            for n in motors:
                if snap['Status'][n] != 0:
                    raise RuntimeError(f'{stage}: {n} Status={snap["Status"][n]}')
                if abs(snap['Goal_Position'][n] - target_raw[n]) > 5:
                    raise RuntimeError(f'{stage}: {n} Goal partially applied or mismatched')
                present_delta = snap['Present_Position'][n] - target_raw[n]
                if abs(present_delta) > 5:
                    raise RuntimeError(
                        f'{stage}: {n} Present moved unexpectedly: '
                        f'start={target_raw[n]}, now={snap["Present_Position"][n]}, '
                        f'delta={present_delta:+d} raw ticks, allowed=+/-5'
                    )
            if torques == {1} and expected_torque is None:
                phase(f'{stage}: Torque=1 on all axes after Goal write; cause unconfirmed. Arm may move or hold.')
            return snap

        check_readback('20ms after Goal write', expected_torque=None)
        time.sleep(.02)
        check_readback('second Goal readback', expected_torque=None)
    else:
        follower.bus.sync_write('Goal_Position', current)
    phase('Configuring Follower (writes servo registers; torque toggles)')
    try:
        if guarded_teleop_start:
            original_torque_disabled = follower.bus.torque_disabled
            @contextmanager
            def enable_only_after_success(motors=None):
                follower.bus.disable_torque(motors)
                yield
                check_readback('before configure torque enable', expected_torque=0)
                follower.bus.enable_torque(motors)
            follower.bus.torque_disabled = enable_only_after_success
            try:
                follower.configure()
            finally:
                follower.bus.torque_disabled = original_torque_disabled
            check_readback('after configure', expected_torque=1)
        else:
            follower.configure()
    except Exception as exc:
        report_exception('Follower.configure failed; no recording will start', exc)
        # Inspect register values only. Avoid any writes or automatic retries.
        for motor in follower.bus.motors:
            try:
                value = follower.bus.read('Lock', motor)
                phase(f'After failure: {motor} Lock={value}')
            except Exception as read_exc:
                phase(f'After failure: {motor} Lock read failed: {read_exc}')
        raise
    phase('Follower configured; starting controlled alignment (robot may move)')
    synchronize_positions(follower, leader, target, timeout=alignment_timeout,
                          tolerance=alignment_tolerance,
                          diagnostics_path=alignment_diagnostics_path)


def close_hardware(follower, leader, configured):
    errors = []
    for device in (follower, leader):
        if device is None:
            continue
        try:
            if device.bus.is_connected:
                device.bus.disconnect(disable_torque=configured)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            # Close even a partially opened bus or if disabling torque failed.
            # Feetech closePort() dereferences ser; it is None before first open.
            handler = device.bus.port_handler
            if getattr(handler, 'ser', None) is not None:
                try:
                    handler.closePort()
                except Exception as exc:
                    errors.append(str(exc))
    if errors:
        raise RuntimeError('Disconnect failed; check arm power: ' + '; '.join(errors))


def record_loop(dataset, robot, teleop, seconds=RECORD_SECONDS, fps=RECORD_FPS,
                task="Pick up the Muhi bottle and place it into the target container."):
    import numpy as np
    from lerobot.utils.robot_utils import precise_sleep
    start = time.perf_counter()
    timing = []
    while time.perf_counter() - start < seconds:
        tick = time.perf_counter()
        obs = robot.get_observation()
        observed = time.perf_counter()
        requested = teleop.get_action()
        if not all(math.isfinite(requested[n]) and math.isfinite(obs[n]) for n in NAMES):
            raise ValueError('Nonfinite joint position; action not sent')
        read_action = time.perf_counter()
        sent = robot.send_action(requested)
        sent_at = time.perf_counter()
        def vector(values):
            result = np.array([values[n] for n in NAMES], dtype=np.float32)
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite joint data')
            return result
        image = obs['iphone']
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f'Unexpected image contract: {image.shape}, {image.dtype}')
        dataset.add_frame({'observation.state': vector(obs), 'action': vector(sent),
                           'action_requested': vector(requested), 'observation.images.iphone': image,
                           'task': task})
        done = time.perf_counter()
        timing.append(dict(cycle=len(timing), start_s=tick-start, observe_s=observed-tick,
                           leader_s=read_action-observed, send_s=sent_at-read_action,
                           write_s=done-sent_at, work_s=done-tick))
        # Do not catch up by issuing a burst of commands after an overrun.
        precise_sleep(max(0, 1/fps-(time.perf_counter()-tick)))
    return timing, time.perf_counter()-start


class SyntheticCamera:
    def __init__(self):
        self.count = 0

    def read_latest(self):
        import cv2
        import numpy as np
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[:, :320, 2] = 220
        frame[:, 320:, 0] = 220
        x = self.count % 560
        frame[100:180, x:x+80] = [0, 220, 0]
        self.count += 2
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class DummyRobot:
    def __init__(self):
        self.camera = SyntheticCamera()
        self.state = dict.fromkeys(NAMES, 0.)

    def get_observation(self):
        return {**self.state, 'iphone': self.camera.read_latest()}

    def send_action(self, action):
        # Deliberately distinguish request and sent command to test provenance.
        self.state = {k: v * .5 for k, v in action.items()}
        return self.state.copy()


class DummyTeleop:
    def __init__(self):
        self.i = 0

    def get_action(self):
        self.i += 1
        return {n: float((self.i+j) % 40) for j, n in enumerate(NAMES)}


def main():
    # Shared hardware helpers stay here for existing tools/tests.
    from capture_session import main as session_main
    session_main()


if __name__ == '__main__':
    main()
