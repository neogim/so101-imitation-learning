"""Unit tests requiring only the Python standard library; never access hardware."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("real_to_sim.py")
spec = importlib.util.spec_from_file_location("so101_real_to_sim", SCRIPT)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


class RealToSimOfflineTests(unittest.TestCase):
    def setUp(self):
        self.config = sim.load_config(Path(__file__).with_name("workcell.example.json"))

    def test_workcell_uncalibrated(self):
        self.assertFalse(self.config["calibrated"])
        self.assertEqual(set(self.config["cameras"]), {"side", "wrist"})
        self.assertEqual(len(sim.STATE_NAMES), 6)

    def test_joint_degrees_to_radians_identity(self):
        mapped = sim.joint_radians([0, 90, -45, 20, 180, 50], self.config)
        self.assertAlmostEqual(mapped["shoulder_lift"], math.pi / 2)
        self.assertAlmostEqual(mapped["elbow_flex"], -math.pi / 4)
        self.assertNotIn("gripper", mapped)

    def test_joint_sign_offset(self):
        self.config["joints"]["sign"]["shoulder_pan"] = -1
        self.config["joints"]["offset_deg"]["shoulder_pan"] = 10
        self.assertAlmostEqual(sim.joint_radians([20, 0, 0, 0, 0, 0], self.config)["shoulder_pan"],
                               math.radians(-10))

    def test_invalid_config_rejected(self):
        self.config["cameras"]["wrist"]["pos_m"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "invalid camera pose"):
            sim.validate_config(self.config)

    def test_invalid_dataset_metadata_rejected_without_mujoco(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(json.dumps(
                {"features": {"observation.state": {"names": ["wrong"]}, "action": {"names": sim.STATE_NAMES}}}
            ))
            with self.assertRaisesRegex(ValueError, "Unexpected joint names"):
                sim.read_episode(root, 0, 20, 30)

    def test_empty_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(json.dumps(
                {"features": {"observation.state": {"names": sim.STATE_NAMES},
                              "action": {"names": sim.STATE_NAMES}}}
            ))
            with self.assertRaisesRegex(ValueError, "Expected exactly one Parquet"):
                sim.read_episode(root, 0, 20, 30)


if __name__ == "__main__":
    unittest.main()
