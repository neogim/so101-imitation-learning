"""Lean full-episode v3 verification; decodes video locally without image export."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


NAMES = [f'{name}.pos' for name in (
    'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')]


def verify(root, expected, height, width):
    info = json.loads((root / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0' or info['fps'] != 20 or info['total_episodes'] != expected:
        raise ValueError('Dataset format, FPS or episode count mismatch')
    features = info['features']
    if any(features[k]['names'] != NAMES or features[k]['shape'] != [6]
           for k in ('observation.state', 'action')):
        raise ValueError('Joint feature order/shape mismatch')
    if features['observation.images.iphone']['shape'] != [height, width, 3]:
        raise ValueError('RGB feature shape mismatch')
    if 'action_requested' in features:
        raise ValueError('Leader-only action_requested entered ACT dataset')
    data = sorted((root / 'data').rglob('*.parquet'))
    videos = sorted((root / 'videos/observation.images.iphone').rglob('*.mp4'))
    if len(data) != 1 or not videos:
        raise ValueError('Expected one Parquet and at least one MP4')
    table = pq.read_table(data[0], columns=[
        'index', 'episode_index', 'frame_index', 'timestamp', 'task_index',
        'observation.state', 'action'])
    if len(table) != info['total_frames']:
        raise ValueError('Parquet frame count mismatch')
    counts = Counter()
    for index, row in enumerate(table.to_pylist()):
        ep = int(row['episode_index'])
        frame = counts[ep]
        if (row['index'] != index or row['frame_index'] != frame or
                abs(float(row['timestamp']) - frame / 20) > .002 or row['task_index'] != 0):
            raise ValueError(f'Index/timestamp/task mismatch at global frame {index}')
        for key in ('observation.state', 'action'):
            values = np.asarray(row[key], dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f'Invalid {key} at global frame {index}')
        counts[ep] += 1
    if set(counts) != set(range(expected)):
        raise ValueError('Episode indices not contiguous')
    episode_files = sorted((root / 'meta/episodes').rglob('*.parquet'))
    episodes = [row for path in episode_files for row in pq.read_table(path, columns=[
        'episode_index', 'length', 'videos/observation.images.iphone/from_timestamp',
        'videos/observation.images.iphone/to_timestamp',
        'videos/observation.images.iphone/chunk_index',
        'videos/observation.images.iphone/file_index']).to_pylist()]
    if len(episodes) != expected or any(counts[int(e['episode_index'])] != e['length'] for e in episodes):
        raise ValueError('Episode metadata lengths mismatch')
    video_counts = np.zeros(expected, dtype=int)
    decoded = 0
    for path in videos:
        file_index = int(path.stem.split('-')[-1])
        chunk_index = int(path.parent.name.split('-')[-1])
        local_episodes = [e for e in episodes if
            e['videos/observation.images.iphone/file_index'] == file_index and
            e['videos/observation.images.iphone/chunk_index'] == chunk_index]
        if not local_episodes:
            raise ValueError(f'Video has no episode metadata: {path}')
        starts = np.asarray([e['videos/observation.images.iphone/from_timestamp'] for e in local_episodes])
        stops = np.asarray([e['videos/observation.images.iphone/to_timestamp'] for e in local_episodes])
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if (stream.height, stream.width) != (height, width):
                raise ValueError('MP4 dimensions mismatch')
            for frame in container.decode(video=0):
                timestamp = float(frame.pts * frame.time_base)
                local = int(np.searchsorted(starts, timestamp + 1e-5, side='right') - 1)
                if local < 0 or local >= len(local_episodes) or not starts[local] - 1e-5 <= timestamp < stops[local] - 1e-5:
                    raise ValueError(f'MP4 frame outside episode interval: {path} frame={decoded}')
                video_counts[int(local_episodes[local]['episode_index'])] += 1
                decoded += 1
    if decoded != len(table) or any(video_counts[ep] != counts[ep] for ep in counts):
        raise ValueError('MP4/Parquet per-episode frame mismatch')
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('local/so101_verify', root=root, video_backend='pyav')
    if len(ds) != len(table):
        raise ValueError('LeRobotDataset API length mismatch')
    for index in (0, len(ds) // 2, len(ds) - 1):
        item = ds[index]
        if tuple(item['observation.images.iphone'].shape) != (3, height, width):
            raise ValueError('LeRobotDataset API RGB readback mismatch')
    return dict(episodes=expected, frames=len(table), fps=20, image_shape=[height, width, 3],
                per_episode_frames=[counts[i] for i in range(expected)],
                video_frames=decoded, state_shape=[6], action_shape=[6],
                gripper_unit='normalized_0_100', rotary_unit='degrees',
                action_requested_present=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--expected', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args.root.resolve(), args.expected, args.height, args.width)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'per_episode_frames'}))


if __name__ == '__main__':
    main()
