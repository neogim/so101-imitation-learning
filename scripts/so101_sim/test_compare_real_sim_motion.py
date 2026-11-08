"""Synthetic tests for side-by-side replay helpers (no robot or video required)."""
import sys
import unittest
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_real_sim_motion import (sample_evenly, comparison_panel,
                                     pixel_to_base_plane, video_object_layout, MuhiGripperAttachment, read_object_targets, set_visual_gripper)


class MotionComparisonTests(unittest.TestCase):
    def test_sample_five_spread_across_twenty(self):
        rows = [{"frame_index": i} for i in range(20)]
        chosen = sample_evenly(rows, 5)
        self.assertEqual(len(chosen), 5)
        self.assertEqual(chosen[0]["frame_index"], 0)
        self.assertEqual(chosen[-1]["frame_index"], 19)
        self.assertEqual(len(set(item["frame_index"] for item in chosen)), 5)

    def test_sample_limits_and_invalid(self):
        self.assertEqual(len(sample_evenly([1, 2], 5)), 2)
        with self.assertRaises(ValueError):
            sample_evenly([], 5)
        with self.assertRaises(ValueError):
            sample_evenly([1], 0)

    def test_object_pixel_target_matches_projected_camera_ray(self):
        T = np.array([[-.61650, .43981, -.65307, .49516],
                      [.75783, .10642, -.64372, .38170],
                      [-.21361, -.89176, -.39891, .28123],
                      [0, 0, 0, 1]])
        K = {"fx": 543.173, "fy": 542.957, "cx": 316.395, "cy": 229.548}
        p = pixel_to_base_plane(T, K, 255., 360., .019)
        xyz = T[:3, :3].T @ (p - T[:3, 3])
        self.assertAlmostEqual(K["fx"]*xyz[0]/xyz[2]+K["cx"], 255., delta=0.02)
        self.assertAlmostEqual(K["fy"]*xyz[1]/xyz[2]+K["cy"], 360., delta=0.02)
        self.assertAlmostEqual(p[2], .019, places=5)
        layout = video_object_layout(T, K)
        self.assertEqual(layout["muhi_yaw_deg"], 45.)
        self.assertEqual(len(layout["cup_xy"]), 2)
        with self.assertRaises(ValueError):
            pixel_to_base_plane(T, K, float("nan"), 310., .055)

    @unittest.skipUnless(importlib.util.find_spec("mujoco") is not None,
                         "MuJoCo unavailable")
    def test_attach_follow_and_release_keeps_world_pose(self):
        import mujoco
        xml = """<mujoco><worldbody>
          <body name="gripper_link" pos="0 0 0.15">
            <joint name="slide" type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".01" mass=".01"/>
          </body>
          <geom name="r2s_muhi_body" type="sphere" pos=".02 0 .15" size=".012" contype="0" conaffinity="0"/>
          <geom name="r2s_muhi_cap" type="sphere" pos=".04 0 .15" size=".008" contype="0" conaffinity="0"/>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        obj = MuhiGripperAttachment(model, data)
        original = data.geom_xpos[obj.ids[0]].copy()
        obj.attach()
        self.assertAlmostEqual(obj.initial_distance_m, .02, places=5)
        data.qpos[0] = .10
        mujoco.mj_forward(model, data)
        obj.follow()
        np.testing.assert_allclose(data.geom_xpos[obj.ids[0]],
                                   original + [.10, 0., 0.], atol=1e-7)
        obj.detach()
        data.qpos[0] = .20
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.geom_xpos[obj.ids[0]],
                                   original + [.10, 0., 0.], atol=1e-7)

    def test_curved_tabletop_front_profile_matches_three_measured_depths(self):
        import compare_id1_eye_to_hand as scene
        self.assertAlmostEqual(scene.tabletop_front_y_m(-.4), .2175, places=6)
        self.assertAlmostEqual(scene.tabletop_front_y_m(0.), .2525, places=6)
        self.assertAlmostEqual(scene.tabletop_front_y_m(.4), .2175, places=6)
        strips = scene.tabletop_front_strip_boxes()
        self.assertGreater(len(strips), 20)
        self.assertTrue(all(name.startswith("r2s_table_front_") for name, _, _ in strips))
        self.assertLessEqual(max(pos[1] + size[1] for _, pos, size in strips), .2525 + 1e-9)
        self.assertAlmostEqual(max(pos[1] + size[1] for _, pos, size in strips),
                               .2525, delta=.001)
        with self.assertRaises(ValueError):
            scene.tabletop_front_y_m(.41)

    def test_measured_table_footprint_and_unaffected_props(self):
        import mujoco
        import compare_id1_eye_to_hand as scene
        spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
        spec.compiler.discardvisual = False
        scene.add_approximate_workcell(spec)
        model = spec.compile()
        table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                     "r2s_preview_table")
        self.assertEqual(model.geom_type[table_id], int(mujoco.mjtGeom.mjGEOM_BOX))
        np.testing.assert_allclose(model.geom_size[table_id], [.4, .235, .02], atol=1e-8)
        self.assertAlmostEqual(model.geom_pos[table_id, 1], -.0175, places=6)
        from compare_real_sim_motion import table_geom_ids
        self.assertGreater(len(table_geom_ids(model)), 20)
        self.assertAlmostEqual(scene.tabletop_front_y_m(-.4), .2175, places=6)
        self.assertAlmostEqual(scene.tabletop_front_y_m(0.), .2525, places=6)
        self.assertAlmostEqual(scene.tabletop_front_y_m(.4), .2175, places=6)
        self.assertAlmostEqual(-.2525, -.505 / 2, places=6)
        muhi_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    "r2s_muhi_body")
        self.assertGreaterEqual(muhi_id, 0)

    def test_mount_diagnostic_is_not_a_pose_correction(self):
        import mujoco
        import inspect_table_robot_mount as mount
        xml = """<mujoco><worldbody>
          <geom name="r2s_preview_table" type="box" size=".4 .255 .02" pos=".4 0 -.02"/>
          <geom name="r2s_table_leg_mm" type="box" size=".02 .02 .1"/>
          <geom name="r2s_table_leg_mp" type="box" size=".02 .02 .1"/>
          <geom name="r2s_table_leg_pm" type="box" size=".02 .02 .1"/>
          <geom name="r2s_table_leg_pp" type="box" size=".02 .02 .1"/>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        initial_positions = model.geom_pos.copy()
        report = mount.analyze(model)
        self.assertEqual(report["status"], "DIAGNOSTIC_ONLY_NOT_REAL_CLAMP_CALIBRATION")
        self.assertAlmostEqual(report["edge_signed_inward_distance_m"]["x-"], 0., places=7)
        self.assertFalse(report["robot_anchor_inside_table_footprint"] is False)
        np.testing.assert_allclose(model.geom_pos, initial_positions)

    def test_cup_photo_estimated_height_is_shorter_than_muhi_length(self):
        import mujoco
        import compare_id1_eye_to_hand as scene
        spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
        scene.add_approximate_workcell(spec)
        model = spec.compile()
        cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_cup_wall_00")
        self.assertGreaterEqual(cup, 0)
        # Full uncut wall height including the base plate is approximately 92mm.
        top_m = model.geom_pos[cup, 2] + model.geom_size[cup, 2]
        np.testing.assert_allclose(top_m, .092, atol=1e-6)
        self.assertLess(top_m, .10)

    def test_three_click_axis_projects_same_body_and_cap_ends(self):
        import mujoco
        from compare_real_sim_motion import muhi_axis_reprojection
        xml = """<mujoco><worldbody>
        <geom name="r2s_muhi_body" type="cylinder"
          pos="-.0175 0 1" size=".019 .0325"/>
        <geom name="r2s_muhi_cap" type="cylinder"
          pos=".0325 0 1" size=".019 .0175"/>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        T = np.eye(4)
        K = {"fx": 500., "fy": 500., "cx": 320., "cy": 240.}
        annotations = {
            "muhi_body_end_px": [295., 240.],
            "muhi_cap_end_px": [345., 240.],
        }
        diag = muhi_axis_reprojection(model, T, K, annotations)
        np.testing.assert_allclose(diag["reprojection_error_px"], [0., 0.], atol=1e-5)

    def test_read_per_video_object_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "objects.json"
            path.write_text(json.dumps({
                "image_size": [640, 480],
                "muhi_center_px": [250, 355],
                "cup_center_px": [375, 300],
                "muhi_yaw_deg": -20,
            }), encoding="utf-8")
            muhi, cup, yaw = read_object_targets(path)
            self.assertEqual(muhi, (250., 355.))
            self.assertEqual(cup, (375., 300.))
            self.assertEqual(yaw, -20.)
            path.write_text(json.dumps({
                "image_size": [640, 480],
                "muhi_center_px": [700, 355],
                "cup_center_px": [375, 300],
                "muhi_yaw_deg": -20,
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_object_targets(path)

    @unittest.skipUnless(importlib.util.find_spec("mujoco") is not None,
                         "MuJoCo unavailable")
    def test_j6_normalized_gripper_replay(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string("""<mujoco>
          <worldbody><body><joint name="gripper" type="hinge"
          range="-0.174533 1.74533" limited="true"/>
          <geom type="box" size=".01 .01 .01" mass=".01"/></body></worldbody>
          </mujoco>""")
        data = mujoco.MjData(model)
        at_closed = set_visual_gripper(model, data, 0.)
        at_open = set_visual_gripper(model, data, 100.)
        self.assertGreater(at_open, at_closed)
        midpoint = set_visual_gripper(model, data, 50.)
        self.assertAlmostEqual(midpoint, (at_closed + at_open)/2, places=6)
        inverted = set_visual_gripper(model, data, 0., invert=True)
        self.assertAlmostEqual(inverted, at_open, places=6)
        with self.assertRaises(ValueError):
            set_visual_gripper(model, data, 150.)

    def test_legacy_three_click_json_suffix_is_accepted(self):
        payload = {
            "image_size": [640, 480],
            "muhi_center_px": [250, 355],
            "cup_center_px": [375, 300],
            "muhi_yaw_deg": -20,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.json"
            path.write_text(json.dumps(payload) + "\\n", encoding="utf-8")
            self.assertEqual(read_object_targets(path),
                             ((250., 355.), (375., 300.), -20.))
            path.write_text(json.dumps(payload) + "BAD", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                read_object_targets(path)

    def test_real_sim_panel(self):
        real = Image.new("RGB", (224, 168), "red")
        sim = Image.new("RGB", (640, 480), "blue")
        panel = comparison_panel(real, sim, "test")
        self.assertEqual(panel.size, (1280, 508))
        self.assertEqual(tuple(np.asarray(panel)[28, 0]), (255, 0, 0))
        self.assertEqual(tuple(np.asarray(panel)[28, 640]), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
