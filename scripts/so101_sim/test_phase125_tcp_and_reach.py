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


    def test_near_goal_sampling_and_pd_oracle(self):
        from so101_reach_env import SO101ReachEnv
        from diagnose_reach_pd import run_diagnostic
        env = SO101ReachEnv(goal_joint_delta_rad=.35, frame_skip=2)
        try:
            _, info = env.reset(seed=73)
            self.assertEqual(info["goal_mode"], "near")
            self.assertTrue(np.all(np.abs(env.goal_joint_angles - env.targets) <= .35 + 1e-9))
            _, again = env.reset(seed=73)
            np.testing.assert_allclose(again["goal_base_m"], info["goal_base_m"])
        finally:
            env.close()
        result = run_diagnostic(seed=73, episodes=2,
                                near_goal_joint_rad=.35, max_steps=20,
                                frame_skip=2)
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all(np.isfinite(r["final_tcp_distance_m"])
                            for r in result["results"]))
        self.assertTrue(all(np.isfinite(r["final_joint_error_rad_l2"])
                            for r in result["results"]))
        # Success is deliberately not asserted: this test diagnoses PD control.

    def test_pd_trace_csv_and_graphs(self):
        import csv
        import json
        from diagnose_reach_pd_trace import run_trace
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pd_trace"
            record = run_trace(seed=31, episode=0, out=out,
                               near_goal_joint_rad=.35,
                               max_steps=3, frame_skip=2)
            self.assertEqual(record["physics_substeps"], 6)
            self.assertEqual(record["joint_names"], list(r2s.JOINTS))
            self.assertEqual(len(record["saturated_substep_fraction_per_joint"]), 5)
            with (out / "pd_trace.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), record["physics_substeps"])
            self.assertIn("shoulder_pan_torque_applied_nm", rows[0])
            self.assertIn("shoulder_pan_saturated", rows[0])
            self.assertIn("contact_pairs", rows[0])
            self.assertEqual(json.loads((out / "summary.json").read_text())["seed"], 31)
            for name in ("joint_angles.png", "joint_torques.png",
                         "joint_velocities.png", "tcp_distance_contacts.png"):
                self.assertGreater((out / name).stat().st_size, 1000)
            with self.assertRaises(ValueError):
                run_trace(seed=31, episode=0, out=out, max_steps=1, frame_skip=1)


    def test_compiled_collision_inventory_and_contact_ablation(self):
        from inspect_reach_collisions import run_audit
        report = run_audit(seed=101, episode=0, max_steps=3, frame_skip=2)
        print("MUJOCO_COLLISION_GEOMS:", [(g["geom_id"], g["body_name"], g["mesh_name"]) for g in report["geoms"]])
        print("MUJOCO_INITIAL_CONTACTS:", report["baseline_contact_enabled"]["initial_contact_geom_pairs"])
        self.assertGreater(report["model"]["ngeom"], 0)
        self.assertEqual(len(report["geoms"]), report["model"]["ngeom"])
        self.assertTrue(all(g["body_name"] for g in report["geoms"]))
        full = report["baseline_contact_enabled"]
        no_contact = report["contact_disabled_diagnostic_only"]
        np.testing.assert_allclose(full["goal_joint_rad"], no_contact["goal_joint_rad"])
        np.testing.assert_allclose(full["goal_tcp_base_m"], no_contact["goal_tcp_base_m"])
        self.assertEqual(no_contact["contact_substep_fraction"], 0.)
        self.assertTrue(np.isfinite(full["final_tcp_distance_m"]))
        self.assertTrue(np.isfinite(no_contact["final_tcp_distance_m"]))
        long_report = run_audit(seed=101, episode=0, max_steps=150, frame_skip=10)
        for mode in ("baseline_contact_enabled", "contact_disabled_diagnostic_only"):
            item = long_report[mode]
            print("MUJOCO_CONTACT_ABLATION:", mode, "reach=", item["success"],
                  "final_dist=", round(item["final_tcp_distance_m"], 5),
                  "joint_err=", round(item["final_joint_error_l2_rad"], 5),
                  "saturation=", [round(v, 3) for v in item["torque_saturation_fraction"]],
                  "contact_frac=", round(item["contact_substep_fraction"], 3))

    def test_selective_base_shoulder_contact_exclusions(self):
        import csv
        from compare_base_shoulder_contacts import compare_pairs, save_report, CASES
        report = compare_pairs(seed=101, episode=0, max_steps=3, frame_skip=2)
        ref = report["results"]["baseline"]
        remasked = report["results"]["reencoded_masks_control"]
        self.assertEqual(ref["initial_contact_geom_pairs"],
                         remasked["initial_contact_geom_pairs"])
        self.assertEqual(set(CASES), set(report["results"]) - {"all_contacts_disabled_reference"})
        for name, expected in (
            ("exclude_geom0_6", (0, 6)),
            ("exclude_geom1_6", (1, 6)),
            ("exclude_geom2_6", (2, 6)),
        ):
            contacts = {tuple(x["geom_id_pair"])
                        for x in report["results"][name]["initial_contact_geom_pairs"]}
            self.assertNotIn(expected, contacts)
            self.assertGreater(report["results"][name]["initial_contact_points"], 0)
        final_contacts = report["results"]["exclude_all_three"]["initial_contact_geom_pairs"]
        self.assertEqual(final_contacts, [])
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "comparison"
            path = save_report(report, target)
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 7)
            self.assertTrue((target / "pair_ablation.json").exists())
        full_run = compare_pairs(seed=101, episode=0, max_steps=150, frame_skip=10)
        for name, entry in full_run["results"].items():
            print("MUJOCO_PAIR_ABLATION:", name,
                  "success=", entry["success"],
                  "final_tcp_m=", round(entry["final_tcp_distance_m"], 5),
                  "joint_err_l2=", round(entry["final_joint_error_l2_rad"], 5),
                  "saturation=", [round(v, 3) for v in entry["torque_saturation_fraction"]],
                  "contact_frac=", round(entry["contact_substep_fraction"], 3))



    def test_compiled_collision_snapshot_and_primitive_proxy(self):
        from visualize_base_shoulder_contacts import collision_snapshot, save_snapshot
        from compare_collision_proxies import compare_proxies, export
        geom_map, vertices, contacts = collision_snapshot()
        self.assertEqual(len(vertices), 4)
        self.assertEqual(set(vertices), {0, 1, 2, 6})
        self.assertTrue(all(v.shape[1] == 3 for v in vertices.values()))
        self.assertTrue(all(np.isfinite(v).all() for v in vertices.values()))
        self.assertTrue(any(x["pair"] == [0, 6] for x in contacts))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "collision_plot"
            snap = save_snapshot(path)
            self.assertGreater((path / "base_shoulder_contacts_3d.png").stat().st_size, 1000)
            self.assertTrue((path / "geometry_snapshot.json").exists())
            self.assertGreater(len(snap["contact_points"]), 0)
            report = compare_proxies(seed=101, episode=0, max_steps=3,
                                     frame_skip=2, scale=.65)
            self.assertEqual(len(report["results"]), 5)
            original = report["results"]["original_meshes"]
            self.assertFalse(original["primitive_proxies"])
            for key, entry in report["results"].items():
                self.assertTrue(np.isfinite(entry["final_tcp_distance_m"]))
                np.testing.assert_allclose(entry["goal_joint_rad"], original["goal_joint_rad"])
                if key != "original_meshes":
                    self.assertTrue(entry["primitive_proxies"])
            dest = export(report, Path(tmp) / "proxies")
            self.assertTrue(dest.is_file())
            self.assertTrue((dest.parent / "proxy_comparison.json").is_file())
        full = compare_proxies(seed=101, episode=0, max_steps=150, frame_skip=10, scale=.65)
        for key, entry in full["results"].items():
            print("MUJOCO_PRIMITIVE_PROXY:", key,
                  "final_tcp_m=", round(entry["final_tcp_distance_m"], 5),
                  "joint_err_l2=", round(entry["final_joint_error_l2_rad"], 5),
                  "contact_frac=", round(entry["contact_substep_fraction"], 3),
                  "mean_sat=", round(float(np.mean(entry["torque_saturation_fraction"])), 3))



if __name__ == "__main__":
    unittest.main()
