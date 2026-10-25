"""Bounded target-position ramp to a stationary Leader; no calibration writes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

from record_episode import hardware_preflight, make_hardware, validate_calibrations, close_hardware


def ramp(previous, target, dt):
    # Cap even after delayed scheduling: never catch up with a large step.
    step = 3.0 * min(max(dt, 0), .05)
    if set(previous) != set(target) or not all(math.isfinite(x) for x in [*previous.values(), *target.values()]):
        raise ValueError('Invalid joint positions')
    return {k: previous[k] + max(-step, min(step, target[k]-previous[k])) for k in previous}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--log', type=Path, required=True)
    a = p.parse_args()
    if not a.execute:
        p.error('--execute requires explicit hardware authorization')
    hashes = hardware_preflight(a.log)
    follower = leader = None
    configured = False
    events = []
    def stop(signum, frame):
        raise KeyboardInterrupt('Alignment interrupted')
    signal.signal(signal.SIGTERM, stop)
    try:
        follower, leader = make_hardware(None)
        follower.cameras = {}
        leader.bus.connect()
        follower.bus.connect()
        validate_calibrations((leader, follower))
        target = leader.bus.sync_read('Present_Position')
        current = follower.bus.sync_read('Present_Position')
        ramp(current, target, 0)  # finite/key check before ANY configuration
        gap = {k: target[k]-current[k] for k in current}
        print(json.dumps({'initial_difference':gap},indent=2), flush=True)
        if any(abs(v) > (100 if k == 'gripper' else 120) for k,v in gap.items()):
            raise RuntimeError('Initial difference exceeds alignment limit')
        configured = True
        leader.configure()
        follower.bus.disable_torque()
        current = follower.bus.sync_read('Present_Position')
        follower.bus.sync_write('Goal_Position', current)
        follower.configure()
        print('Slow alignment started: target slew <=3 degrees/s, gripper <=3 units/s. Keep Leader still.',flush=True)
        command = current.copy()
        start = time.perf_counter()
        last_progress = start
        last_error = max(abs(target[k]-current[k]) for k in target)
        settled = 0
        while time.perf_counter()-start < 60:
            tick = time.perf_counter()
            live = leader.bus.sync_read('Present_Position')
            current = follower.bus.sync_read('Present_Position')
            ramp(current, live, 0)
            if any(abs(live[k]-target[k]) > 3 for k in target):
                raise RuntimeError('Leader moved; alignment stopped')
            if any(abs(command[k]-current[k]) > 8 for k in command):
                raise RuntimeError('Follower tracking error >8; alignment stopped')
            error = max(abs(target[k]-current[k]) for k in target)
            if error < last_error-.3:
                last_error, last_progress = error, tick
            if error > 3 and tick-last_progress > 3:
                raise RuntimeError('No progress for 3 seconds; alignment stopped')
            command = ramp(command, target, .05)
            follower.bus.sync_write('Goal_Position', command)
            events.append({'t':tick-start,'state':current,'command':command.copy(),'target':target})
            settled = settled+1 if error < 2 and max(abs(command[k]-target[k]) for k in target)<.2 else 0
            if len(events)%40 == 0:
                print(f'Alignment {tick-start:.1f}s: max error {error:.2f}',flush=True)
            if settled >= 10:
                print('Alignment complete. Disabling torque and disconnecting.',flush=True)
                break
            time.sleep(max(0,.05-(time.perf_counter()-tick)))
        else:
            raise RuntimeError('60-second alignment timeout')
    finally:
        try:
            close_hardware(follower, leader, configured)
        finally:
            with a.log.open('x') as f:
                json.dump(events,f,indent=2)
            for path,digest in hashes.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Calibration file changed externally')


if __name__=='__main__':
    main()
