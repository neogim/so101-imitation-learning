"""Collision-safe table guard tests, synthetic MuJoCo only (no robot hardware)."""
import unittest

import mujoco
import numpy as np

from table_collision_guard import TableCollisionGuard, interpolate_joint_segment


class TableCollisionGuardTests(unittest.TestCase):
    def setup_scene(self):
        spec=mujoco.MjSpec()
        spec.compiler.discardvisual=False
        spec.worldbody.add_geom(
            name="r2s_preview_table",type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0,0,-.03],size=[.3,.3,.02],
            contype=0,conaffinity=0)
        robot=spec.worldbody.add_body(name="upper_arm_link",pos=[0,0,.10])
        robot.add_joint(name="approach",type=mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=[0,0,1],range=[-.3,.3],limited=True)
        robot.add_geom(name="guarded_tip",type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[.02,0,0],mass=.1,contype=1,conaffinity=1)
        model=spec.compile()
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        return model,data

    def test_swept_segment(self):
        segment=interpolate_joint_segment([0],[.05],max_delta_rad=.01)
        self.assertEqual(len(segment),5)
        self.assertAlmostEqual(segment[-1][0],.05)

    def test_rejects_unsafe_start(self):
        model,data=self.setup_scene()
        guard=TableCollisionGuard(model,data,[0],clearance_mm=3)
        result=guard.inspect([-.15])
        self.assertFalse(result["clearance_satisfied"])
        target,info=guard.filter_step([-.15],[0])
        self.assertEqual(info["status"],"unsafe_start")
        self.assertAlmostEqual(target[0],-.15)

    def test_truncates_collision_crossing(self):
        model,data=self.setup_scene()
        guard=TableCollisionGuard(model,data,[0],clearance_mm=3)
        target,info=guard.filter_step([0],[-.15])
        self.assertEqual(info["status"],"shortened")
        self.assertGreater(target[0],-.15)
        self.assertTrue(guard.inspect(target)["clearance_satisfied"])
        self.assertAlmostEqual(data.qpos[0],0)

    def test_original_display_only_geoms_still_checked(self):
        model,data=self.setup_scene()
        table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"r2s_preview_table")
        self.assertEqual(model.geom_contype[table],0)
        guard=TableCollisionGuard(model,data,[0])
        self.assertLess(guard.inspect([-.09])["minimum_distance_mm"],0)


if __name__=="__main__":
    unittest.main()
