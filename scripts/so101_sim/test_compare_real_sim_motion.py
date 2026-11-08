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


class RobotOverlayGuiTests(unittest.TestCase):
    def test_joint_offset_degrees_are_mapped_to_radians(self):
        import real_to_sim as r2s
        from compare_real_sim_robot_overlay_gui import joint_config
        conf = joint_config([0, 2, -3, 4, -5])
        angles = r2s.joint_radians([0, 0, 0, 0, 0, 50], conf)
        self.assertAlmostEqual(angles["shoulder_lift"], np.deg2rad(2))
        self.assertAlmostEqual(angles["elbow_flex"], np.deg2rad(-3))
        self.assertTrue(all(conf["joints"]["sign"][key] == 1
                            for key in r2s.JOINTS))
        with self.assertRaises(ValueError):
            joint_config([0, 0, 0, 0, 100])
        with self.assertRaises(ValueError):
            joint_config([0, 0, float("nan"), 0, 0])

    def test_overlay_selects_original_episode_frame_numbers(self):
        from compare_real_sim_robot_overlay_gui import selected_frame_rows
        ep = {"rows": [{"frame_index": idx, "state": [idx] * 6}
                       for idx in range(300)]}
        output = selected_frame_rows(ep, (0, 24, 40, 52), comparison_stride=5)
        self.assertEqual([row["state"][0] for row in output.values()],
                         [0, 120, 200, 260])
        with self.assertRaisesRegex(ValueError, "Missing"):
            selected_frame_rows(ep, [61], comparison_stride=5)
        with self.assertRaises(ValueError):
            selected_frame_rows(ep, [24, 24], comparison_stride=5)

    def test_saved_robot_candidate_scene_references(self):
        from types import SimpleNamespace
        from compare_real_sim_robot_overlay_gui import (
            robot_pose_payload, read_robot_pose, matches_scene)
        old = Path("outputs/real2sim/episode4-objects-20261108-004859.json")
        cup = Path("outputs/real2sim/episode4-cup-gui-20261108-131429.json")
        saved = robot_pose_payload(
            [0, 1, -1, 2, 0], 4, old, str(cup),
            (4, 5, 12, 7, 0, -9.375), frame_index=40, overlay_alpha=.45)
        args = SimpleNamespace(
            episode=4, object_layout_json=old, cup_pose_json=cup,
            table_extra_left_cm=4, table_extra_front_cm=5,
            table_base_x_cm=12, table_base_y_cm=7, table_base_z_cm=0,
            table_yaw_delta_deg=-9.375)
        matches_scene(saved, args)
        self.assertEqual(saved["source_frame_index"], 40)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            self.assertEqual(read_robot_pose(path)["joint_offset_deg"]["elbow_flex"], -1)
        args.table_base_x_cm = 14
        with self.assertRaisesRegex(ValueError, "table_base_x_cm"):
            matches_scene(saved, args)


class MotionComparisonTests(unittest.TestCase):
    def test_cup_diameter_and_height_controls_preserve_other_objects(self):
        import mujoco
        from align_cup_gui import cup_geom_ids, set_cup_dimensions
        parts = ["<geom name='r2s_cup_base' type='cylinder' size='.05 .001' pos='.2 .1 .001'/>"]
        import math
        for i in range(16):
            a = 2 * math.pi * i / 16
            x, y = .2 + .049 * math.cos(a), .1 + .049 * math.sin(a)
            parts.append(f"<geom name='r2s_cup_wall_{i:02d}' type='box' "
                         f"size='.010 .0015 .045' pos='{x} {y} .047'/>")
        parts.append("<geom name='r2s_muhi_body' type='sphere' size='.01' pos='0 0 .04'/>")
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>" + "".join(parts) + "</worldbody></mujoco>")
        ids = cup_geom_ids(model)
        before_positions, before_sizes = model.geom_pos.copy(), model.geom_size.copy()
        set_cup_dimensions(model, ids, before_positions[ids], before_sizes[ids],
                           diameter_m=.08, height_m=.105, xyz_m=[-.032, -.063, -.001])
        self.assertAlmostEqual(model.geom_size[ids[0], 0], .04, places=6)
        self.assertAlmostEqual(model.geom_size[ids[1], 2], .0525, places=6)
        self.assertAlmostEqual(model.geom_pos[ids[1], 2], .002 + .105 / 2 - .001, places=6)
        self.assertAlmostEqual(
            np.linalg.norm(model.geom_pos[ids[1], :2] -
                           (before_positions[ids[0], :2] + [-.032, -.063])),
            .049 * .8, places=6)
        self.assertTrue(np.allclose(model.geom_pos[-1], before_positions[-1]))
        self.assertTrue(np.allclose(model.geom_size[-1], before_sizes[-1]))
        set_cup_dimensions(model, ids, before_positions[ids], before_sizes[ids],
                           diameter_m=.100, height_m=.090)
        np.testing.assert_allclose(model.geom_pos[ids], before_positions[ids])
        np.testing.assert_allclose(model.geom_size[ids], before_sizes[ids])

    def test_cup_dimensions_saved_and_old_json_default(self):
        from align_cup_gui import alignment_payload, read_cup_payload
        info = alignment_payload([-.032, -.063, -.001], Path("objects.json"), 4,
                                 (12, 7, -9.375, 0), diameter_m=.078, height_m=.102)
        self.assertAlmostEqual(info["cup_diameter_m"], .078)
        self.assertAlmostEqual(info["cup_height_m"], .102)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cup.json"
            path.write_text(json.dumps(info), encoding="utf-8")
            self.assertAlmostEqual(read_cup_payload(path)["cup_height_m"], .102)
            info.pop("cup_diameter_m")
            info.pop("cup_height_m")
            path.write_text(json.dumps(info), encoding="utf-8")
            self.assertNotIn("cup_height_m", read_cup_payload(path))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            alignment_payload([0, 0, 0], Path("objects.json"), 4,
                              (12, 7, 0, 0), diameter_m=-1)

    def test_gui_cup_delta_preserves_other_geometries(self):
        import mujoco
        from align_cup_gui import cup_geom_ids, set_cup_delta
        geoms = ["<geom name='r2s_cup_base' type='sphere' size='.01' pos='.2 -.1 .001'/>"]
        geoms += [f"<geom name='r2s_cup_wall_{i:02d}' type='sphere' size='.01' pos='.2 -.1 .04'/>"
                  for i in range(16)]
        geoms += ["<geom name='r2s_muhi_body' type='sphere' size='.01'/>",
                  "<geom name='r2s_preview_table' type='box' size='.4 .25 .02'/>"]
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody>" +
                                               "".join(geoms) + "</worldbody></mujoco>")
        ids = cup_geom_ids(model)
        original = model.geom_pos[ids].copy()
        before = model.geom_pos.copy()
        set_cup_delta(model, ids, original, [.10, -.07, .02])
        np.testing.assert_allclose(model.geom_pos[ids], original + [.10, -.07, .02])
        other = [i for i in range(model.ngeom) if i not in ids]
        np.testing.assert_allclose(model.geom_pos[other], before[other])
        set_cup_delta(model, ids, original, [0., 0., 0.])
        np.testing.assert_allclose(model.geom_pos[ids], original)

    def test_fixed_measured_size_overrides_old_gui_size(self):
        from compare_id1_eye_to_hand import CUP_DIAMETER_M, CUP_HEIGHT_M
        from align_cup_gui import alignment_payload
        payload = alignment_payload([-.032, -.063, -.001], Path("old.json"),
                                    4, (12, 7, -9.375, 0))
        self.assertEqual(payload["cup_diameter_m"], CUP_DIAMETER_M)
        self.assertEqual(payload["cup_height_m"], CUP_HEIGHT_M)

    def test_gui_saved_cup_pose_schema_and_validation(self):
        from align_cup_gui import alignment_payload, read_cup_payload
        payload = alignment_payload([.12, -.03, .015], Path("outputs/real2sim/objects.json"),
                                    4, (12., 7., -9.375, 0.))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cup_pose.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = read_cup_payload(path)
            self.assertEqual(loaded["cup_delta_base_m"], [.12, -.03, .015])
            self.assertEqual(loaded["coordinate_frame"], "so101_base")
            self.assertEqual(loaded["table_yaw_delta_deg"], -9.375)
            path.write_text('{"schema": "other", "cup_delta_base_m":[0,0,0]}')
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                read_cup_payload(path)
        with self.assertRaises(ValueError):
            alignment_payload([float("nan"), 0, 0], Path("a"), 4, (12, 7, 0, 0))

    def test_perspective_projection_uses_depth_not_image_scale(self):
        from compare_real_sim_motion import project_base_xyz_to_side
        T = np.eye(4)
        intr = {"fx":500., "fy":500., "cx":320., "cy":240.}
        near = project_base_xyz_to_side(T, intr, [0.05, 0., 0.5])
        far = project_base_xyz_to_side(T, intr, [0.05, 0., 1.0])
        self.assertAlmostEqual(near[0] - 320., 2 * (far[0] - 320.))

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

    def test_user_measured_cup_and_muhi_dimensions(self):
        import mujoco
        import compare_id1_eye_to_hand as scene
        spec = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
        scene.add_approximate_workcell(spec)
        model = spec.compile()
        cup = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_cup_wall_00")
        self.assertGreaterEqual(cup, 0)
        # Cup wall extends 83mm above its 2mm baseline offset.
        top_m = model.geom_pos[cup, 2] + model.geom_size[cup, 2]
        np.testing.assert_allclose(top_m, .085, atol=1e-6)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_cup_base")
        np.testing.assert_allclose(model.geom_size[base_id, 0], .039, atol=1e-8)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_muhi_body")
        cap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_muhi_cap")
        np.testing.assert_allclose(model.geom_size[body_id, 0], .0175, atol=1e-8)
        np.testing.assert_allclose(model.geom_size[cap_id, 0], .0175, atol=1e-8)
        self.assertAlmostEqual(2*(model.geom_size[body_id, 1]+model.geom_size[cap_id, 1]),
                               .097, places=8)

    def test_refined_cup_contact_keeps_old_muhi_and_uses_new_pixel(self):
        import tempfile
        import json
        import refine_cup_contact as refine
        from compare_real_sim_motion import read_object_targets
        payload = {
            "image_size": [640, 480], "muhi_center_px": [401., 399.],
            "muhi_body_end_px": [375., 455.], "muhi_cap_end_px": [430., 345.],
            "muhi_yaw_deg": -80.691, "cup_center_px": [159., 325.],
        }
        result = refine.add_cup_contact(payload, [140., 345.])
        self.assertEqual(result["cup_center_px"], [159., 325.])
        self.assertEqual(result["cup_contact_px"], [140., 345.])
        self.assertEqual(result["muhi_yaw_deg"], payload["muhi_yaw_deg"])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "objects.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            muhi, cup, yaw = read_object_targets(path)
            self.assertEqual(muhi, (401., 399.))
            self.assertEqual(cup, (140., 345.))
            self.assertEqual(yaw, -80.691)

    def test_cup_click_height_changes_backprojected_position(self):
        from compare_real_sim_motion import video_object_layout
        cam = np.eye(4)
        cam[:3, :3] = [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]
        cam[:3, 3] = [0., 0., .5]
        K = {"fx":500., "fy":500., "cx":320., "cy":240.}
        a = video_object_layout(cam, K, (400., 400.), (180., 300.), 0., .055)
        b = video_object_layout(cam, K, (400., 400.), (180., 300.), 0., .001)
        self.assertGreater(np.linalg.norm(np.asarray(a["cup_xy"]) - b["cup_xy"]), .001)

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
