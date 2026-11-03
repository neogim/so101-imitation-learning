"""Offline validation tests; no LeRobot or hardware import is needed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class TestVlaLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).with_name("train_smolvla.py")
        spec = importlib.util.spec_from_file_location("test_vla_launcher_target", script)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_preflight_requires_info(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "meta/info.json"):
                self.mod.dataset_preflight(Path(d))

    def test_preflight_reports_cameras(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "meta"
            folder.mkdir()
            (folder / "info.json").write_text(json.dumps({
                "total_episodes": 2, "total_frames": 100,
                "features": {"action": {}, "observation.state": {},
                             "observation.images.side": {}, "observation.images.wrist": {}}
            }))
            result = self.mod.dataset_preflight(Path(d))
            self.assertEqual(result["episodes"], 2)
            self.assertEqual(result["cameras"], ["observation.images.side", "observation.images.wrist"])

    def test_preflight_rejects_missing_actions(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "meta"
            folder.mkdir()
            (folder / "info.json").write_text(json.dumps({
                "features": {"observation.state": {}, "observation.images.side": {}}
            }))
            with self.assertRaisesRegex(ValueError, "action"):
                self.mod.dataset_preflight(Path(d))


if __name__ == "__main__":
    unittest.main()
