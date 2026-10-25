"""One offline episode. Default preflight never opens serial ports; --execute requires approval."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

NAMES = [n + '.pos' for n in ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')]
FOLLOWER = '/dev/tty.usbmodem5B7B0098201'
LEADER = '/dev/tty.usbmodem5B7B0185961'


def deny_calibration(*args, **kwargs):
    raise RuntimeError('Calibration changes are forbidden. Stop and investigate the mismatch.')


def validate_calibrations(devices):
    # Call only after both buses are open, before either configure() can write.
    for device in devices:
        if not device.is_calibrated:
            raise RuntimeError(f'{device.name}: calibration mismatch; no automatic repair')


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
    follower.cameras = {'iphone': camera}
    for device in (follower, leader):
        # Instance-local guards; no LeRobot files or global classes are patched.
        device.calibrate = deny_calibration
        device._save_calibration = deny_calibration
        device.bus.write_calibration = deny_calibration
    return follower, leader


def synchronize_positions(follower, leader, target, *, timeout=60, clock=time.perf_counter, sleep=time.sleep):
    """Ramp goals to a stationary Leader, keeping torque on for the next phase."""
    current = follower.bus.sync_read('Present_Position')
    command = current.copy()
    start = clock()
    last_progress = start
    last_error = max(abs(target[k]-current[k]) for k in target)
    settled = 0
    print('ALIGNING: keep Leader still. Target slew <=3 deg/s (gripper 3 units/s).', flush=True)
    while clock()-start < timeout:
        tick = clock()
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
        settled = settled+1 if error < 2 and max(abs(command[k]-target[k]) for k in target)<.2 else 0
        if settled >= 10:
            print('ALIGNMENT COMPLETE. Keeping connection for operation.', flush=True)
            return
        sleep(max(0,.05-(clock()-tick)))
    raise RuntimeError('Alignment timed out; stopping')


def start_hardware(follower, leader):
    # Avoid standard connect()'s automatic calibration path. No configure writes
    # occur until BOTH devices pass the existing read-only calibration check.
    leader.bus.connect()
    follower.bus.connect()
    validate_calibrations((leader, follower))
    current = follower.bus.sync_read('Present_Position')
    target = leader.bus.sync_read('Present_Position')
    for name in current:
        if not math.isfinite(current[name]) or not math.isfinite(target[name]):
            raise ValueError('Nonfinite initial joint position')
        limit = 100 if name == 'gripper' else 120
        if abs(current[name] - target[name]) > limit:
            raise RuntimeError(f'Initial pose mismatch exceeds alignment limit: {name}')
    leader.configure()
    # Standard follower.configure() may enable torque on exit. Set its current
    # position as goal first, so a stale goal is not used when enabling torque.
    follower.bus.disable_torque()
    current = follower.bus.sync_read('Present_Position')
    follower.bus.sync_write('Goal_Position', current)
    follower.configure()
    synchronize_positions(follower, leader, target)


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


def record_loop(dataset, robot, teleop, seconds=40, fps=20):
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
                           'task': 'Move the gripper slowly toward the object and return.'})
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--countdown', type=int, default=10, help='Seconds to wait after hardware setup before recording')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Hardware operation: only after explicit user approval')
    mode.add_argument('--simulate', action='store_true', help='Run the identical recording loop with dummy devices')
    a = p.parse_args()
    if a.countdown < 0:
        p.error('--countdown must be nonnegative')
    root = a.root.expanduser().resolve()
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    import cv2  # Same import order as real recording, before PyAV.
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    validate_destination(root)
    hashes = {} if a.simulate else hardware_preflight(root)
    camera = None
    follower = leader = None
    dataset = None
    configured = False
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    try:
        if not a.simulate:
            from obs_camera import OBSCamera
            camera = OBSCamera()
            camera.connect()
            image = camera.read_latest()
            assert image.shape == (480,640,3)
            if not a.execute:
                print('Preflight OK: USB paths, calibration files, OBS RGB 640x480, new destination. No serial ports opened.')
                return
        features = {k: dict(dtype='float32', shape=(6,), names=NAMES)
                    for k in ('observation.state', 'action', 'action_requested')}
        features['observation.images.iphone'] = dict(dtype='video', shape=(480,640,3), names=['height','width','channels'])
        dataset = LeRobotDataset.create('local/so101_episode', fps=20, root=root,
            robot_type='synthetic_so101' if a.simulate else 'so101_follower', features=features,
            use_videos=True, video_backend='pyav', rgb_encoder=RGBEncoderConfig(vcodec='h264'),
            image_writer_threads=2, image_writer_processes=0)
        if a.simulate:
            robot, teleop = DummyRobot(), DummyTeleop()
        else:
            follower, leader = make_hardware(camera)
            # On failures before any configuration, disconnect without torque writes.
            # Once configure starts, cleanup must attempt to disable torque.
            # Calibration and pose checks occur before this flag changes below.
            original_configure = leader.configure
            def configure_leader():
                nonlocal configured
                configured = True
                original_configure()
            leader.configure = configure_leader
            start_hardware(follower, leader)
            robot, teleop = follower, leader
            for remaining in range(a.countdown, 0, -1):
                print(f'Recording starts in {remaining} seconds. Keep Leader still.', flush=True)
                time.sleep(1)
            print('Recording 10 seconds. Ctrl-C aborts and disconnects; support the arm before power removal.', flush=True)
        timing, elapsed = record_loop(dataset, robot, teleop)
        # Release hardware before potentially slow video encoding.
        if not a.simulate:
            close_hardware(follower, leader, configured)
            follower = leader = None
            camera.disconnect()
            camera = None
        work = np.array([r['work_s'] for r in timing])
        summary = dict(frames=len(timing), elapsed_s=elapsed, effective_fps=len(timing)/elapsed,
                       work_mean_ms=float(work.mean()*1000), work_p95_ms=float(np.percentile(work,95)*1000),
                       work_max_ms=float(work.max()*1000), overrun_50ms=int((work>.05).sum()))
        (root/'capture_timing.json').write_text(json.dumps({'summary':summary,'cycles':timing},indent=2))
        (root/'capture_config.json').write_text(json.dumps(dict(simulated=a.simulate, fps=20, seconds=40,
            use_degrees=True, gripper_unit='normalized_0_100', camera_index=1, image_shape=[480,640,3],
            action='send_action return value', action_requested='leader get_action value',
            calibration_sha256=hashes, opencv=cv2.__version__),indent=2))
        dataset.save_episode()
        dataset.finalize()
        print(json.dumps(summary, indent=2))
    finally:
        try:
            close_hardware(follower, leader, configured)
        finally:
            if camera is not None and camera.is_connected:
                camera.disconnect()
            if dataset is not None and dataset.writer is not None:
                dataset.writer.stop_image_writer()
            for path, before in hashes.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != before:
                    raise RuntimeError(f'Calibration file changed externally: {path}')


if __name__ == '__main__':
    main()
