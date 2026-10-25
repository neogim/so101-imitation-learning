"""Validate the simulated launcher's command provenance, Parquet and video count."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    import av
    import numpy as np
    import pyarrow.parquet as pq
    tables = [pq.read_table(f) for f in sorted((args.root/'data').rglob('*.parquet'))]
    rows = [row for table in tables for row in table.to_pylist()]
    assert rows
    for i, row in enumerate(rows):
        requested = np.array(row['action_requested'])
        sent = np.array(row['action'])
        np.testing.assert_allclose(sent, requested*.5)
        expected = np.zeros(6) if i == 0 else np.array(rows[i-1]['action'])
        np.testing.assert_allclose(row['observation.state'], expected)
        np.testing.assert_allclose(row['timestamp'], i/20, atol=1e-5)
        assert row['frame_index'] == i and row['episode_index'] == 0
    files = list((args.root/'videos').rglob('*.mp4'))
    assert len(files) == 1
    with av.open(str(files[0])) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (640,480)
        assert stream.codec_context.name == 'h264'
        frames = list(container.decode(video=0))
        assert len(frames) == len(rows)
        np.testing.assert_allclose([float(f.pts*f.time_base) for f in frames], np.arange(len(rows))/20, atol=1e-5)
    print(json.dumps({'frames':len(rows), 'sent_vs_requested_verified':True, 'observation_before_action_verified':True,
                      'video_pts_verified':True},indent=2))


if __name__ == '__main__':
    main()
