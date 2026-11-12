"""Regression tests for the optional SO-101 Wrist camera rig.

Run in the existing macOS conda environment. No physical robot.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_camera_sim as wrist


class WristCameraRigTests(unittest.TestCase):
    def test_validate_mount_rejects_units_and_nonfinite(self):
        self.assertEqual(wrist.validate_mount((0, 0, -.048), (0, 0, 0))[0],
                         (0., 0., -.048))
        for pos, rpy in [
            ((0, 0, 600), (0, 0, 0)),
            ((math.nan, 0, 0), (0, 0, 0)),
            ((0, 0, 0), (0, math.inf, 0)),
            ((0, 0), (0, 0, 0)),
        ]:
            with self.subTest(pos=pos, rpy=rpy), self.assertRaises(ValueError):
                wrist.validate_mount(pos, rpy)

    def test_photographed_j5_j6_bracket_is_contiguous(self):
        # Bracket foot and stem must overlap; otherwise the rig floats.
        self.assertEqual(wrist.DEFAULT_MOUNT_POS, (0.0, 0.0, -0.048))
        self.assertLessEqual(wrist.BRACKET_STEM_CENTER_Z - 0.029,
                             wrist.BRACKET_BASE_Z + 0.006)
        self.assertGreaterEqual(wrist.BRACKET_STEM_CENTER_Z + 0.029,
                                wrist.BRACKET_TOP_Z - 0.004)
        self.assertLess(wrist.CAMERA_Z, wrist.LENS_Z)
        self.assertLess(wrist.LENS_Z, wrist.PCB_Z)
        # Board and L-bracket must physically meet; 0.004 is top half-thickness
        # and 0.003 is PCB half-thickness (metres).
        self.assertLessEqual(wrist.PCB_Z - 0.003,
                             wrist.BRACKET_TOP_Z + 0.004)
        self.assertLessEqual(wrist.PCB_Z - 0.003,
                             wrist.LENS_Z + 0.012)

    def test_mujoco_compile_camera_follows_joint_and_colliders_exist(self):
        import mujoco

        spec = mujoco.MjSpec()
        gripper = spec.worldbody.add_body(name="gripper_link", pos=[0, 0, .2])
        gripper.add_joint(name="wrist_roll", type=mujoco.mjtJoint.mjJNT_HINGE,
                          axis=[0, 0, 1])
        gripper.add_geom(name="robot_mount", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                         size=[.012], mass=.1, contype=0, conaffinity=0)
        wrist.install(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        qid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_roll")
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, wrist.CAMERA)
        self.assertGreaterEqual(cid, 0)
        for name in wrist.COLLISION_NAMES:
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertGreaterEqual(geom, 0)
            self.assertEqual(model.geom_contype[geom], 1)
        mujoco.mj_forward(model, data)
        before_pos = data.cam_xpos[cid].copy()
        before_rot = data.cam_xmat[cid].copy()
        data.qpos[model.jnt_qposadr[qid]] = math.pi / 2
        mujoco.mj_forward(model, data)
        after_pos = data.cam_xpos[cid].copy()
        after_rot = data.cam_xmat[cid].copy()
        # A centered camera can sit on the J5 rotation axis: position need
        # not change even though its optical orientation DOES follow J5.
        self.assertGreater(np.linalg.norm(before_rot - after_rot), 1e-3)
        self.assertAlmostEqual(float(np.linalg.norm(after_pos - [.0, .0, .2])),
                               float(np.linalg.norm(before_pos - [.0, .0, .2])),
                               delta=1e-4)

    def test_duplicate_rig_rejected(self):
        import mujoco
        spec = mujoco.MjSpec()
        gripper = spec.worldbody.add_body(name="gripper_link")
        gripper.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.01])
        wrist.install(spec)
        with self.assertRaises(ValueError):
            wrist.install(spec)


if __name__ == "__main__":
    unittest.main()
