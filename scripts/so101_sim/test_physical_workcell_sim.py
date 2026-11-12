#!/usr/bin/env python3
"""Synthetic physics checks; no LeRobot dataset or real hardware required."""
from __future__ import annotations

import unittest

import mujoco
import numpy as np

import physical_workcell_sim as physical


class PhysicalWorkcellTest(unittest.TestCase):
    def make_model(self):
        spec = mujoco.MjSpec()
        spec.compiler.discardvisual = False
        spec.worldbody.add_geom(
            name="r2s_preview_table", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, 0, -.015], size=[.3, .3, .01],
            contype=0, conaffinity=0)
        spec.worldbody.add_geom(
            name="r2s_muhi_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[.05, 0, .0175], quat=[.70710678, .70710678, 0, 0],
            size=[.0175, .031525, 0], contype=0, conaffinity=0)
        spec.worldbody.add_geom(
            name="r2s_muhi_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[.05, .0485, .0175], quat=[.70710678, .70710678, 0, 0],
            size=[.0175, .016975, 0], contype=0, conaffinity=0)
        physical.install_dynamic_muhi(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        return model, data

    def test_dynamic_bottle_has_free_joint_and_collision(self):
        model, data = self.make_model()
        adr = physical.configure_dynamic_muhi(model, data)
        self.assertEqual(model.jnt_type[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              physical.DYNAMIC_JOINT)], mujoco.mjtJoint.mjJNT_FREE)
        np.testing.assert_allclose(data.qpos[adr:adr+3], [.05, .016975, .0175],
                                   atol=1e-6)
        self.assertEqual(model.geom_contype[
            physical.geom_id(model, "r2s_muhi_body")], 0)
        self.assertEqual(model.geom_rgba[
            physical.geom_id(model, "r2s_muhi_body"), 3], 0)
        self.assertEqual(model.geom_contype[
            physical.geom_id(model, physical.DYNAMIC_GEOMS[0])], 1)

    def test_muhi_falls_and_touches_table(self):
        model, data = self.make_model()
        adr = physical.configure_dynamic_muhi(model, data)
        physical.prepare_environment(model)
        data.qpos[adr+2] = .10
        mujoco.mj_forward(model, data)
        for _ in range(500):
            mujoco.mj_step(model, data)
        self.assertLess(data.qpos[adr+2], .10)
        self.assertGreater(data.qpos[adr+2], -.03)
        self.assertTrue(np.isfinite(data.qpos).all())
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                physical.DYNAMIC_BODY)
        kinds, _, _ = physical.summarize_contacts(model, data, body)
        self.assertGreater(kinds["muhi_table"], 0)


if __name__ == "__main__":
    unittest.main()
