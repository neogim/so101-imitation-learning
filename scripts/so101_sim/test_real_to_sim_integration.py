"""Optional MuJoCo + pyarrow offline smoke test with generated synthetic data."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("real_to_sim", HERE / "real_to_sim.py")
r2s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2s)


class RealToSimIntegration(unittest.TestCase):
    def test_synthetic_single_episode_mujoco(self):
        import mujoco  # installed in isolated test environment
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "raw"
            (root / "meta").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            info = {"features": {
                "observation.state": {"names": r2s.STATE_NAMES},
                "action": {"names": r2s.STATE_NAMES}
            }}
            (root / "meta/info.json").write_text(json.dumps(info))
            values = [[0., 0., 0., 0., 0., 50.]] * 3
            rows = pa.table({
                "observation.state": pa.array(values),
                "action": pa.array(values),
                "timestamp": pa.array([0.0, 0.05, 0.10])
            })
            pq.write_table(rows, root / "data/chunk-000/episode_000000.parquet")
            out = Path(d) / "new-output"
            args = argparse.Namespace(
                dataset_root=root, episode=0, stride=1, max_frames=3,
                urdf=r2s.DEFAULT_URDF,
                config=HERE / "workcell.example.json",
                output=out, render=False, real_images=False,
                render_every=1, width=320, height=240
            )
            result = r2s.run(args)
            self.assertEqual(len(result["frames"]), 3)
            self.assertEqual(result["mode"], "OFFLINE_KINEMATIC_REPLAY_ONLY_NOT_PHYSICS_VALIDATION")
            self.assertTrue((out / "report.json").is_file())
            self.assertFalse(result["calibrated"])
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                r2s.run(args)


    def test_v3_shared_parquet_episode_boundaries(self):
        """Two episodes share file-000; selecting episode 1 must never read ep 0."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "v3"
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            info = {
                "codebase_version": "v3.0",
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "fps": 20,
                "features": {"observation.state": {"names": r2s.STATE_NAMES},
                             "action": {"names": r2s.STATE_NAMES}},
            }
            (root / "meta/info.json").write_text(json.dumps(info))
            meta = pa.table({
                "episode_index": [0, 1],
                "length": [2, 3],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
                "dataset_from_index": [0, 2],
                "dataset_to_index": [2, 5],
                "videos/observation.images.side/chunk_index": [0, 0],
                "videos/observation.images.side/file_index": [0, 0],
                "videos/observation.images.side/from_timestamp": [0.0, .1],
                "videos/observation.images.side/to_timestamp": [.1, .25],
            })
            pq.write_table(meta, root / "meta/episodes/chunk-000/file-000.parquet")
            def state(v):
                return [float(v), 0., 0., 0., 0., 50.]
            data = pa.table({
                "episode_index": [0, 0, 1, 1, 1],
                "frame_index": [0, 1, 0, 1, 2],
                "observation.state": [state(v) for v in (90, 90, 1, 2, 3)],
                "action": [state(v) for v in (90, 90, 1, 2, 3)],
                "timestamp": [0.0, 0.05, 0.0, 0.05, 0.1],
            })
            pq.write_table(data, root / "data/chunk-000/file-000.parquet")
            selected = r2s.read_episode(root, 1, stride=1, max_frames=3)
            self.assertEqual([x["state"][0] for x in selected["rows"]], [1, 2, 3])
            self.assertEqual([x["frame_index"] for x in selected["rows"]], [0, 1, 2])
            self.assertEqual(selected["metadata"]["length"], 3)
            args = argparse.Namespace(
                dataset_root=root, episode=1, stride=1, max_frames=3,
                urdf=r2s.DEFAULT_URDF, config=HERE / "workcell.example.json",
                output=Path(d) / "v3-output", render=False, real_images=False,
                render_every=1, width=320, height=240,
            )
            result = r2s.run(args)
            self.assertEqual(len(result["frames"]), 3)
            self.assertEqual(result["frames"][0]["frame_index"], 0)
            with self.assertRaisesRegex(ValueError, "Expected one metadata row"):
                r2s.read_episode(root, 2, stride=1, max_frames=3)

    def test_v3_metadata_missing_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(json.dumps({
                "codebase_version": "v3.0",
                "features": {"observation.state": {"names": r2s.STATE_NAMES},
                             "action": {"names": r2s.STATE_NAMES}},
            }))
            with self.assertRaisesRegex(ValueError, "Missing v3 episode metadata"):
                r2s.read_episode(root, 0, stride=1, max_frames=2)


if __name__ == "__main__":
    unittest.main()
