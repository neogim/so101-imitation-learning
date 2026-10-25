"""Create a NEW success-only ACT dataset; never edits raw source or uploads."""
import argparse
import json
import os
from pathlib import Path


def export(source, destination, labels_path=None):
    from record_episode import validate_destination
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Derived output must be outside the raw source directory')
    validate_destination(destination)
    import cv2
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if labels_path is not None:
        from offline_labels import validate_labels, digest
        labels_path = Path(labels_path).expanduser().resolve()
        labels = validate_labels(json.loads(labels_path.read_text()), source)
        selected = [r['episode_index'] for r in labels if r['adopt']]
        label_sha256 = digest(labels_path)
    else:
        outcomes = json.loads((source/'episode_outcomes.json').read_text())
        selected = [r['episode_index'] for r in outcomes if r['adopt'] is True
                    and all(r[k]=='success' for k in ('pick','place','overall'))
                    and r['end_reason'] in ('manual','time_limit')]
        label_sha256 = None
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('No eligible successful episodes, or duplicate outcome IDs')
    info = json.loads((source/'meta/info.json').read_text())
    if any(ep not in range(info['total_episodes']) for ep in selected):
        raise ValueError('Outcome episode IDs outside source dataset')
    keep = ('observation.state','action','observation.images.iphone')
    features = {k:info['features'][k] for k in keep}
    reader = LeRobotDataset('local/source', root=source, video_backend='pyav')
    output = LeRobotDataset.create('local/so101_act', fps=info['fps'], root=destination,
        robot_type=info['robot_type'], features=features, use_videos=True, video_backend='pyav',
        rgb_encoder=RGBEncoderConfig(vcodec='h264'), image_writer_threads=2)
    lineage = []
    try:
        # Reading only selected episodes also prevents failed/reset frames entering stats.
        for ep in selected:
            count = 0
            for i in range(len(reader)):
                # Numeric index lookup avoids decoding unselected episodes.
                if int(reader.hf_dataset[i]['episode_index']) != ep:
                    continue
                row = reader[i]
                frame = {k: np.asarray(row[k]).copy() for k in keep[:2]}
                frame[keep[2]] = (np.asarray(row[keep[2]]).transpose(1,2,0)*255).round().astype(np.uint8)
                frame['task'] = row['task']
                output.add_frame(frame)
                count += 1
            if not count:
                raise ValueError(f'Empty selected episode {ep}')
            output.save_episode(parallel_encoding=False)
            lineage.append(dict(episode_index=len(lineage), source_episode_index=ep, frames=count))
    finally:
        output.finalize()
    (destination/'source_lineage.json').write_text(json.dumps(dict(source=str(source), episodes=lineage,
        labels_file=None if labels_path is None else str(labels_path),labels_sha256=label_sha256,
        features=list(keep), success_filter=True, statistics='recomputed on selected episodes'),indent=2))
    return lineage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',type=Path,help='Separate offline labels file; raw outcomes remain unchanged')
    a=p.parse_args()
    for k in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
        os.environ[k]='1'
    print(json.dumps(export(a.source.expanduser().resolve(), a.output.expanduser().resolve(),a.labels),indent=2))


if __name__=='__main__':
    main()
