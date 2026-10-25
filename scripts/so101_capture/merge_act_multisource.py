"""Build one local v3 dataset from an audited, explicitly adopted manifest."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


KEEP = ('observation.state', 'action', 'observation.images.iphone')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    for name in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[name] = '1'
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    selected = json.loads(args.selected.read_text())
    if len(selected) != args.expected or len({(r['source'], r['episode_index']) for r in selected}) != len(selected):
        raise ValueError('Selected episode count or uniqueness mismatch')
    by_source = defaultdict(dict)
    for item in selected:
        by_source[Path(item['source'])][int(item['episode_index'])] = int(item['frames'])
    reference_features = None
    for source in by_source:
        info = json.loads((source / 'meta/info.json').read_text())
        if info['codebase_version'] != 'v3.0' or info['fps'] != 20:
            raise ValueError(f'Incompatible source: {source}')
        features = {key: info['features'][key] for key in KEEP}
        if reference_features is None:
            reference_features = features
        elif reference_features != features:
            raise ValueError(f'Feature mismatch: {source}')
    dataset = LeRobotDataset.create(f'local/muhi_act_success_{len(selected):03d}_20261022',
        fps=20, root=args.output, robot_type='so101_follower', features=reference_features,
        use_videos=True, video_backend='pyav', rgb_encoder=RGBEncoderConfig(vcodec='h264'),
        image_writer_threads=2)
    lineage = []
    task = 'Pick up the Muhi bottle and place it into the target container.'
    try:
        for source, desired in by_source.items():
            data_files = sorted((source / 'data').rglob('*.parquet'))
            video_files = sorted((source / 'videos/observation.images.iphone').rglob('*.mp4'))
            if len(data_files) != 1 or len(video_files) != 1:
                raise ValueError(f'Expected one Parquet and MP4: {source}')
            info = json.loads((source / 'meta/info.json').read_text())
            rows = pq.read_table(data_files[0], columns=[
                'episode_index', 'frame_index', 'timestamp', 'observation.state', 'action']).to_pylist()
            counts = {ep: 0 for ep in desired}
            with av.open(str(video_files[0])) as video:
                stream = video.streams.video[0]
                if (stream.width, stream.height) != (640, 480):
                    raise ValueError(f'Video size mismatch: {source}')
                decoded = 0
                for row, video_frame in zip(rows, video.decode(video=0), strict=True):
                    decoded += 1
                    ep = int(row['episode_index'])
                    if ep not in desired:
                        continue
                    index = counts[ep]
                    if int(row['frame_index']) != index or abs(float(row['timestamp']) - index / 20) > .002:
                        raise ValueError(f'Frame/timestamp mismatch: {source} ep={ep} frame={index}')
                    state = np.asarray(row['observation.state'], dtype=np.float32)
                    action = np.asarray(row['action'], dtype=np.float32)
                    if state.shape != (6,) or action.shape != (6,) or not (
                        np.isfinite(state).all() and np.isfinite(action).all()
                    ):
                        raise ValueError(f'Invalid joint data: {source} ep={ep} frame={index}')
                    dataset.add_frame({
                        'observation.state': state, 'action': action,
                        'observation.images.iphone': video_frame.to_ndarray(format='rgb24'),
                        'task': task})
                    counts[ep] += 1
                    if counts[ep] == desired[ep]:
                        dataset.save_episode(parallel_encoding=False)
                        lineage.append(dict(episode_index=len(lineage), source=str(source),
                            source_episode_index=ep, frames=counts[ep]))
                        print(f'Saved {len(lineage)}/{args.expected}: {source.name} ep={ep} frames={counts[ep]}', flush=True)
                if decoded != len(rows) or decoded != info['total_frames']:
                    raise ValueError(f'Video/Parquet count mismatch: {source}')
            if any(counts[ep] != desired[ep] for ep in desired):
                raise ValueError(f'Incomplete selected episode: {source}')
    finally:
        dataset.finalize()
    if len(lineage) != args.expected or sum(x['frames'] for x in lineage) != sum(x['frames'] for x in selected):
        raise ValueError('Merged totals differ from selection')
    (args.output / 'source_lineage.json').write_text(json.dumps(lineage, indent=2) + '\n')
    print(f'Merged {len(lineage)} episodes, {sum(x["frames"] for x in lineage)} frames')


if __name__ == '__main__':
    main()
