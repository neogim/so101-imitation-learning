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
