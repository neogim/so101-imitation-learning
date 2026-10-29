"""Camera-free, recording-free SO-101 teleoperation with existing calibration guards."""
import argparse
import hashlib
import math
from pathlib import Path
import signal
import time

from record_episode import hardware_preflight, make_hardware, start_hardware, close_hardware, NAMES


def teleop_tick(follower, leader, last_sent, *, joint_bounds=None):
    """One monitored control cycle; any failed read prevents the next Goal write."""
    state = {n + '.pos': v for n, v in follower.bus.sync_read('Present_Position', num_retry=0).items()}
    torque = follower.bus.sync_read('Torque_Enable', normalize=False, num_retry=0)
    status = follower.bus.sync_read('Status', normalize=False, num_retry=0)
    motors = {n.removesuffix('.pos') for n in NAMES}
    if (set(state) != set(NAMES) or set(torque) != motors or set(status) != motors
            or any(torque[n] != 1 or status[n] != 0 for n in motors)
            or not all(math.isfinite(v) for v in state.values())):
        raise RuntimeError('Follower state/torque/status invalid; no new Action sent')
    offenders = [n for n in NAMES if abs(state[n]-last_sent[n]) > 8]
    if offenders:
        raise RuntimeError('Tracking error >8; no new Action: ' + ', '.join(
            f'{n}={abs(state[n]-last_sent[n]):.3f} '
            f'{"normalized_0_100" if n == "gripper.pos" else "degrees"}' for n in offenders))
    action = leader.get_action()
    if set(action) != set(NAMES) or not all(math.isfinite(action[n]) for n in NAMES):
        raise ValueError('Nonfinite action; stopping')
    step = 30 * .05
    command = {n: last_sent[n]+max(-step, min(step, action[n]-last_sent[n])) for n in NAMES}
    if joint_bounds is not None:
        for name, (lower, upper) in joint_bounds.items():
            if name not in command or not all(math.isfinite(v) for v in (lower, upper)):
                raise RuntimeError("Invalid joint command bounds")
            if not lower < upper:
                raise RuntimeError("Invalid joint bound order")
            if not lower <= state[name] <= upper:
                raise RuntimeError(f"Measured joint outside allowed interval: {name}")
            # Never issue an Action outside the excursion envelope.
            command[name] = max(lower, min(upper, command[name]))
    sent = follower.send_action(command)
    if set(sent) != set(NAMES) or not all(math.isfinite(sent[n]) for n in NAMES):
        raise RuntimeError('Invalid sent Action; stopping')
    return sent.copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--seconds', type=int, default=60)
    a = p.parse_args()
    if not a.execute or not 1 <= a.seconds <= 300:
        p.error('--execute and seconds in 1..300 required')
    hashes = hardware_preflight(Path('/tmp/so101-teleop-no-output-placeholder'))
    from lerobot.utils.robot_utils import precise_sleep
    follower = leader = None
    configured = False
    def stop(signum, frame):
        raise KeyboardInterrupt('Stop requested')
    signal.signal(signal.SIGTERM, stop)
    try:
        follower, leader = make_hardware(None)
        follower.cameras = {}
        start_hardware(follower, leader, guarded_teleop_start=True)
        configured = True
        print(f'TELEOP ACTIVE for {a.seconds} seconds. No camera, no recording.', flush=True)
        last_sent = {n + '.pos': v for n, v in follower.bus.sync_read('Goal_Position').items()}
        if set(last_sent) != set(NAMES) or not all(math.isfinite(v) for v in last_sent.values()):
            raise RuntimeError('Invalid initial Goal readback; no teleop Action sent')
        start = time.perf_counter()
        while time.perf_counter()-start < a.seconds:
            tick = time.perf_counter()
            last_sent = teleop_tick(follower, leader, last_sent)
            precise_sleep(max(0, .05-(time.perf_counter()-tick)))
    except KeyboardInterrupt:
        print('Stop requested.', flush=True)
    finally:
        try:
            print('CONTROL STOPPED. Support arm before torque-off or power removal; '
                  'servo may still hold its last Goal.', flush=True)
            close_hardware(follower, leader, configured)
        finally:
            for path,digest in hashes.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError(f'Calibration changed: {path}')
    print('TELEOP STOPPED. If initialization completed, torque disable was requested; '
          'physical stopping or holding is not guaranteed.', flush=True)


if __name__ == '__main__':
    main()
