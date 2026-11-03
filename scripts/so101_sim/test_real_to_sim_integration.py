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


if __name__ == "__main__":
    unittest.main()
