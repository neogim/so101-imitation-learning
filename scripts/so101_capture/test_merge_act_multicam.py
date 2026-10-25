import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "so101-act-merge-test-hf"))

import merge_act_multicam as merger
from verify_act_multicam import verify


class MergeActMulticamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / "sources"
        cls.root.mkdir()
        cls.source = cls.root / "muhi-side-wrist-production-30-synthetic"
        cls._create_source(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def _create_source(root):
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        joints = merger.JOINTS
        features = {
            "observation.state": {"dtype": "float32", "shape": (6,), "names": joints},
            "action": {"dtype": "float32", "shape": (6,), "names": joints},
            "action_requested": {"dtype": "float32", "shape": (6,), "names": joints},
            "observation.images.side": {"dtype": "video", "shape": (240, 320, 3),
                                                "names": ["height", "width", "channels"]},
            "observation.images.wrist": {"dtype": "video", "shape": (240, 320, 3),
                                                 "names": ["height", "width", "channels"]},
        }
        dataset = LeRobotDataset.create(
            "local/synthetic_multicam_source", fps=20, root=root,
            robot_type="synthetic_so101", features=features, use_videos=True,
            video_backend="pyav", rgb_encoder=RGBEncoderConfig(vcodec="h264"),
            image_writer_threads=1,
        )
        side = np.zeros((240, 320, 3), dtype=np.uint8)
        side[..., 0] = 240
        wrist = np.zeros((240, 320, 3), dtype=np.uint8)
        wrist[..., 2] = 240
        labels = []
        outcomes = [
            ("manual", "success", "success", "success", True),
            ("interrupted", "unknown", "unknown", "unknown", False),
            ("manual", "success", "success", "success", False),
        ]
        for episode, outcome in enumerate(outcomes):
            for frame in range(2):
                values = np.full(6, episode + frame / 10, dtype=np.float32)
                dataset.add_frame({
                    "observation.state": values,
                    "action": values + 1,
                    "action_requested": values + 2,
                    "observation.images.side": side,
                    "observation.images.wrist": wrist,
                    "task": merger.TASK,
                })
            dataset.save_episode(parallel_encoding=False)
            labels.append({
                "episode_index": episode, "frames": 2, "end_reason": outcome[0],
                "pick": outcome[1], "place": outcome[2], "overall": outcome[3],
                "adopt": outcome[4], "notes": "",
            })
        dataset.finalize()
        (root / "episode_outcomes.json").write_text(json.dumps(labels))
        (root / "capture_config.json").write_text(json.dumps({
            "fps": 20, "rotary_slew": 40, "gripper_slew": 30,
            "j6_max_closing_lead": 3, "image_shape": [240, 320, 3],
        }))

    def test_selection_dry_run_and_output(self):
        plan = merger.make_plan(self.root, None, expected_episodes=1, expected_frames=2)
        self.assertEqual(len(plan["selected"]), 1)
        self.assertEqual(len(plan["excluded"]), 2)
        self.assertEqual(plan["selected"][0]["source_episode_index"], 0)
        output = Path(self.temp.name) / "merged"
        self.assertFalse(output.exists())  # Planning/dry-run is read-only.
        before = merger.fingerprints(plan["sessions"])
        merger.merge(plan, output, "local/synthetic_merged", 7, 0.5, before)
        result = verify(output, 1, 2)
        self.assertTrue(result["lerobot_dataset_readback"])
        info = json.loads((output / "meta/info.json").read_text())
        self.assertNotIn("action_requested", info["features"])
        self.assertEqual(set(result["image_keys"]), set(merger.IMAGE_KEYS))

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset("local/synthetic_merged", root=output, video_backend="pyav")
        item = dataset[0]
        side = item["observation.images.side"].numpy()
        wrist = item["observation.images.wrist"].numpy()
        self.assertGreater(float(side[0].mean()), float(side[2].mean()))
        self.assertGreater(float(wrist[2].mean()), float(wrist[0].mean()))

    def test_duplicate_session_and_overwrite_rejected(self):
        with self.assertRaises(merger.MergeError):
            merger.determine_sessions(self.root, [self.source, self.source])
        existing = Path(self.temp.name) / "already_exists"
        existing.mkdir()
        plan = merger.make_plan(self.root, None, 1, 2)
        with self.assertRaises(FileExistsError):
            merger.merge(plan, existing, "local/no", 7, 0.5, {})

    def test_video_frame_mismatch_detected(self):
        original = merger.video_frames

        def shortened(paths, shape):
            frames = list(original(paths, shape))
            if "observation.images.wrist" in str(paths[0]):
                frames = frames[:-1]
            yield from frames

        with mock.patch.object(merger, "video_frames", shortened):
            with self.assertRaises(merger.MergeError):
                merger.validate_session(self.source)


if __name__ == "__main__":
    unittest.main()
