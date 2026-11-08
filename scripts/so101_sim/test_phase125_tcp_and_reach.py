"""Synthetic offline tests for episode TCP extraction and MuJoCo Reach."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_episode_tcp_trajectory as extract
import real_to_sim as r2s


class TCPExtractorTests(unittest.TestCase):
    def test_synthetic_recording_is_timestamped_and_preserves_actions(self):
        import mujoco
        spec = mujoco.MjSpec.from_file(str(r2s.DEFAULT_URDF))
        model = spec.compile()
        data = mujoco.MjData(model)
        rows = [
            {"frame_index": i, "timestamp": i / 20.,
             "state": [0., 10. + i, -20., 15., 0., 50.],
             "action": [0., 10. + i + 1, -20., 15., 0., 60.]}
            for i in range(3)]
        result = extract.extract_tcp_rows(model, data, rows)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["tcp_proxy_body"], "gripper_link")
        self.assertEqual(result[2]["frame_index"], 2)
        self.assertEqual(result[1]["timestamp_s"], .05)
        self.assertEqual(result[0]["logged_action_joint6_raw"][-1], 60.)
        self.assertEqual(len(result[0]["tcp_proxy_quat_wxyz"]), 4)
        self.assertAlmostEqual(np.linalg.norm(result[0]["tcp_proxy_quat_wxyz"]), 1., places=5)
        self.assertFalse(np.allclose(result[0]["tcp_proxy_xyz_base_m"],
                                     result[2]["tcp_proxy_xyz_base_m"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = extract.export_rows(
                result, Path(tmp) / "export", episode=4, dataset=Path("fake"),
                urdf=r2s.DEFAULT_URDF, config=extract.default_joint_config(),
                body_name="gripper_link")
            self.assertTrue(path.is_file())
            self.assertTrue((path.parent / "tcp_trajectory.csv").is_file())
            with self.assertRaises(ValueError):
                extract.export_rows(result, path.parent, episode=4, dataset=Path("fake"),
                                    urdf=r2s.DEFAULT_URDF,
                                    config=extract.default_joint_config(), body_name="gripper_link")
        with self.assertRaises(ValueError):
            extract.extract_tcp_rows(model, data, rows[::-1])
        with self.assertRaisesRegex(ValueError, "Missing"):
            extract.extract_tcp_rows(model, data, rows, body_name="not_a_tcp")


class ReachEnvTests(unittest.TestCase):
    def test_reward_and_reproducible_goal(self):
        from so101_reach_env import SO101ReachEnv, reach_reward
        self.assertEqual(reach_reward(0.01), (-0.01, True))
        self.assertEqual(reach_reward(0.1), (-0.1, False))
        with self.assertRaises(ValueError):
            reach_reward(float("nan"))
        env = SO101ReachEnv(max_steps=3, frame_skip=2)
        try:
            obs, info = env.reset(seed=23)
            goal1 = info["goal_base_m"]
            self.assertEqual(obs.shape, (14,))
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(env.action_space.contains(np.zeros(5, dtype=np.float32)))
            obs2, info2 = env.reset(seed=23)
            np.testing.assert_allclose(info2["goal_base_m"], goal1)
            np.testing.assert_allclose(obs2, obs)
            for i in range(3):
                obs, reward, done, truncated, info = env.step(np.zeros(5))
                self.assertEqual(obs.shape, (14,))
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward))
                self.assertEqual(info["status"], "REACH_ONLY_NOT_PICK")
                if done or truncated:
                    break
            self.assertTrue(done or truncated)
            with self.assertRaises(ValueError):
                env.step([np.nan] * 5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
