"""Preview independent OBS sources without connecting to a robot."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from PIL import Image

from multi_camera import CameraFault, CameraRig, load_config


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"mean": round(statistics.mean(values), 3),
            "p95": round(ordered[min(len(ordered)-1, int(.95*len(ordered)))], 3),
            "max": round(ordered[-1], 3)}


def request_concurrency(fetches):
    events = []
    for rows in fetches.values():
        for row in rows:
            events.extend(((row['request_started_at'], 1), (row['response_at'], -1)))
    active = peak = 0
    previous = None
    busy = overlap = 0.0
    for stamp, delta in sorted(events):
        if previous is not None:
            duration = stamp - previous
            busy += duration if active else 0.0
            overlap += duration if active >= 2 else 0.0
        active += delta
        peak = max(peak, active)
        previous = stamp
    return {'peak_requests_in_flight': peak,
            'overlap_fraction_of_request_busy_time': round(overlap / busy, 3) if busy else 0.0}


def write_request_timeline(path, fetches, utc_offset_s):
    """Write after the timed loop; no CSV work runs during camera sampling."""
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            'camera', 'request_started_utc', 'response_utc', 'request_ms',
            'image_decode_ms', 'jpeg_bytes'))
        writer.writeheader()
        for name, camera_rows in fetches.items():
            for item in camera_rows:
                utc = lambda stamp: datetime.fromtimestamp(stamp + utc_offset_s,
                                                           timezone.utc).isoformat()
                writer.writerow(dict(camera=name,
                    request_started_utc=utc(item['request_started_at']),
                    response_utc=utc(item['response_at']),
                    request_ms=item['request_ms'],
                    image_decode_ms=item.get('image_decode_ms'),
                    jpeg_bytes=item.get('jpeg_bytes')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-config', type=Path, required=True)
    p.add_argument('--enable-cameras', help='Comma-separated logical names')
    p.add_argument('--seconds', type=float, default=3)
    p.add_argument('--hz', type=float, default=20, help='Fixed-rate preview sampling target')
    p.add_argument('--warmup-seconds', type=float, default=0,
                   help='Allow camera background capture to settle before timed sampling')
    p.add_argument('--diagnose-stale', action='store_true',
                   help='Preview only: count stale/skew samples and continue; never accepts them as frames')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.seconds <= 0 or args.hz <= 0 or args.warmup_seconds < 0 or args.output.exists():
        p.error('Require positive seconds/Hz, nonnegative warmup, and a new output directory')
    cfg = load_config(args.camera_config, args.enable_cameras)
    rig = CameraRig(cfg, collect_timings=True)
    rows = []
    try:
        rig.connect()
        args.output.mkdir(parents=True)
        # Save inspection frames outside the timed loop; PNG encoding is not
        # part of the camera sampling path used by recording or inference.
        preview_frames = rig.read_latest()
        for name, frame in preview_frames.items():
            Image.fromarray(frame).save(args.output / f'{name}.png')
        if args.warmup_seconds:
            time.sleep(args.warmup_seconds)
        started = time.perf_counter()
        utc_offset_s = time.time() - time.perf_counter()
        sample_times = []
        sample_ms = []
        skew_ms = []
        iteration = 0
        fault = None
        failed_samples = 0
        first_failed_sample = None
        while time.perf_counter()-started < args.seconds:
            due = started + iteration / args.hz
            if due > time.perf_counter():
                time.sleep(due-time.perf_counter())
            sampled_at = time.perf_counter()
            sample_times.append(sampled_at)
            try:
                frames, stamps = rig.sample_all()
            except CameraFault as exc:
                reason = str(exc)
                failed_samples += 1
                if first_failed_sample is None:
                    first_failed_sample = reason
                if args.diagnose_stale and ('stale frame' in reason or 'availability pairing' in reason):
                    iteration += 1
                    continue
                fault = reason
                break
            observed_at = time.perf_counter()
            sample_ms.append((observed_at-sampled_at)*1000)
            skew_ms.append((max(stamps.values())-min(stamps.values()))*1000)
            for name in frames:
                rows.append(dict(camera=name, sampled_at_monotonic=stamps[name],
                                 observed_at_monotonic=observed_at))
            iteration += 1
        elapsed = time.perf_counter()-started
        fetches = {name: [row for row in camera.timing_snapshot() if row['response_at'] >= started]
                   for name, camera in rig.cameras.items()}
        camera_state_at_end = rig.timing_brief()
        digests = {name: {entry.get('sampled_at_monotonic', entry['response_at']): entry['content_digest']
                          for entry in camera.timing_snapshot()}
                   for name, camera in rig.cameras.items()}
        for row in rows:
            row['content_digest'] = digests[row['camera']].get(row['sampled_at_monotonic'])
        result = {'identity': {name: camera.identity for name, camera in rig.cameras.items()},
                  'host_utc_offset_s': utc_offset_s,
                  'camera_state_at_end': camera_state_at_end,
                  'image_shape': list(rig.image_shape),
                  'completed': fault is None and failed_samples == 0,
                  'measurement_reached_duration': fault is None,
                  'fault': fault,
                  'failed_samples': failed_samples,
                  'first_failed_sample': first_failed_sample,
                  'target_hz': args.hz, 'warmup_seconds': args.warmup_seconds,
                  'elapsed_s': round(elapsed, 3),
                  'samples': {name: len([r for r in rows if r['camera'] == name])
                              for name in rig.cameras},
                  'unique_source_frames': {name: len({r['sampled_at_monotonic'] for r in rows
                                                     if r['camera'] == name})
                                           for name in rig.cameras},
                  'unique_screenshot_responses': {name: len({r['sampled_at_monotonic'] for r in rows
                                                             if r['camera'] == name})
                                                  for name in rig.cameras},
                  'sampled_fps': {name: round(
                      len({r['sampled_at_monotonic'] for r in rows if r['camera'] == name})
                      / max(.001, elapsed), 2)
                      for name in rig.cameras},
                  'loop_interval_ms': stats([(b-a)*1000 for a,b in zip(sample_times, sample_times[1:])]),
                  'sample_copy_ms': stats(sample_ms),
                  'camera_response_skew_ms': stats(skew_ms),
                  'request_concurrency': request_concurrency(fetches),
                  'camera': {name: {
                      'provider': rig.cameras[name].spec['provider'],
                      'jpeg_quality': rig.cameras[name].spec.get('jpeg_quality', 90),
                      'completed_screenshots': len(fetches[name]),
                      'screenshot_rate_hz': round(len(fetches[name])/max(.001,elapsed), 2),
                      'identical_content_count': sum(not r['content_changed'] for r in fetches[name]),
                      'request_ms': stats([r['request_ms'] for r in fetches[name]]),
                      'request_start_interval_ms': stats([
                          (b['request_started_at']-a['request_started_at'])*1000
                          for a,b in zip(fetches[name],fetches[name][1:])]),
                      'identity_check_ms': stats([
                          r['duration_ms'] for r in rig.cameras[name].identity_timing_snapshot()
                          if r['started_at'] >= started]),
                      'base64_ms': stats([r['base64_ms'] for r in fetches[name]]),
                      'jpeg_bytes': stats([r['jpeg_bytes'] for r in fetches[name]]),
                      'image_decode_ms': stats([r['image_decode_ms'] for r in fetches[name]]),
                      'hash_ms': stats([r['hash_ms'] for r in fetches[name]]),
                      'decode_hash_ms': stats([r['decode_hash_ms'] for r in fetches[name]]),
                      'total_fetch_ms': stats([r['total_ms'] for r in fetches[name]]),
                      'image_age_at_sample_ms': stats([(r['observed_at_monotonic']-r['sampled_at_monotonic'])*1000
                                                       for r in rows if r['camera'] == name]),
                      'reused_screenshot_samples': len([r for r in rows if r['camera']==name])
                          - len({r['sampled_at_monotonic'] for r in rows if r['camera']==name}),
                      'identical_adjacent_samples': sum(
                          a['content_digest'] == b['content_digest']
                          for a,b in zip([r for r in rows if r['camera']==name],
                                         [r for r in rows if r['camera']==name][1:])),
                  } for name in rig.cameras},
                  'sampled_at_clock': 'time.perf_counter after OBS screenshot response',
                  'warning': 'unique_source_frames counts response timestamps, not sensor frames. Identical image content can be a static scene. GetSourceScreenshot request_ms includes OBS rendering/encoding and WebSocket transfer; source exposure timestamps are unavailable.'}
        (args.output / 'preview.json').write_text(json.dumps(result, indent=2) + '\n')
        # Written after measurement; wall-clock timestamps let an operator align
        # a noted iPhone warning with OBS response stalls, without changing 20 Hz.
        write_request_timeline(args.output / 'requests.csv', fetches, utc_offset_s)
        print(json.dumps(result, indent=2))
        if fault or failed_samples:
            raise CameraFault(fault or f'{failed_samples} stale/skew samples rejected during preview')
    finally:
        rig.disconnect()


if __name__ == '__main__':
    main()
