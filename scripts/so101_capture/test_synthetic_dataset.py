"""Offline 200-frame dataset round-trip. Never imports robot/camera drivers."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'
    import av
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / 'dataset'
    names = [x+'.pos' for x in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']]
    features = {k: dict(dtype='float32', shape=(6,), names=names) for k in ('observation.state', 'action')}
    features['observation.images.iphone'] = dict(dtype='video', shape=(480, 640, 3), names=['height', 'width', 'channels'])
    ds = LeRobotDataset.create('local/synthetic_so101', fps=20, root=root,
                              robot_type='synthetic_so101', features=features,
                              use_videos=True, video_backend='pyav',
                              rgb_encoder=RGBEncoderConfig(vcodec='h264'),
                              image_writer_threads=2, image_writer_processes=0)
    def values(i):
        state = np.array([i/20, -i/40, i/80, 0, 10, i%101], dtype=np.float32)
        action = state + np.array([1, 2, 3, 4, 5, 0], dtype=np.float32)
        return state, action
    for i in range(200):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :320, 0] = 220
        rgb[:, 320:, 2] = 220
        rgb[100:180, (i*2)%560:(i*2)%560+80] = [0, 220, 0]
        state, action = values(i)
        ds.add_frame({'observation.state': state, 'action': action,
                      'observation.images.iphone': rgb, 'task': 'Synthetic validation only; no hardware.'})
    ds.save_episode()
    ds.finalize()
    reader = LeRobotDataset('local/synthetic_so101', root=root, video_backend='pyav')
    assert len(reader) == 200
    for i in range(200):
        row = reader[i]
        state, action = values(i)
        np.testing.assert_allclose(row['observation.state'], state)
        np.testing.assert_allclose(row['action'], action)
        np.testing.assert_allclose(float(row['timestamp']), i/20, atol=1e-5)
        assert int(row['frame_index']) == i and int(row['episode_index']) == 0
        rgb = np.asarray(row['observation.images.iphone'])
        assert rgb.shape == (3, 480, 640)
        assert rgb[0, 300, 100] > .7 and rgb[2, 300, 100] < .1
        assert rgb[2, 300, 500] > .7 and rgb[0, 300, 500] < .1
    parquets = list((root/'data').rglob('*.parquet'))
    assert parquets and sum(pq.read_metadata(f).num_rows for f in parquets) == 200
    videos = list((root/'videos').rglob('*.mp4'))
    assert len(videos) == 1
    with av.open(str(videos[0])) as container:
        stream = container.streams.video[0]
        assert stream.width == 640 and stream.height == 480
        assert stream.codec_context.name == 'h264'
        frames = list(container.decode(video=0))
        assert len(frames) == 200
        pts = [float(f.pts*f.time_base) for f in frames]
        np.testing.assert_allclose(pts, np.arange(200)/20, atol=1e-5)
    verification = args.output/'verification'
    subprocess.run([sys.executable, str(Path(__file__).with_name('verify_dataset.py')), str(root),
                    '--output', str(verification), '--body-unit', 'degrees'], check=True)
    report = json.loads((verification/'report.json').read_text())
    assert report['errors'] == [] and report['frames'] == 200
    assert (verification/'viewer.html').is_file()
    result = dict(frames=200, episodes=1, fps=20, shape=[480,640,3], codec='h264',
                  parquet_rows=200, video_frames=200, verified=True)
    (args.output/'roundtrip.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
