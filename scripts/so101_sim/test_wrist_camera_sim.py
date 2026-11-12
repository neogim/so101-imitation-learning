"""Focused tests for the SO-101 J5/J6 Wrist camera L-bracket.

Never accesses real hardware. Uses a tiny synthetic MuJoCo scene.
"""
from __future__ import annotations

import ast
import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrist_camera_sim as wrist


class WristCameraTests(unittest.TestCase):
    def test_mount_validation_and_defaults(self):
        self.assertEqual(wrist.validate_mount(wrist.DEFAULT_MOUNT_POS,(0,0,0))[0],
                         (0.0,0.024,-0.045))
        for xyz,rpy in [((0,0,100),(0,0,0)),
                        ((math.nan,0,0),(0,0,0)),
                        ((0,0,0),(0,math.inf,0)),
                        ((0,0),(0,0,0))]:
            with self.subTest(xyz=xyz,rpy=rpy),self.assertRaises(ValueError):
                wrist.validate_mount(xyz,rpy)

    def test_visual_L_is_contiguous(self):
        self.assertLessEqual(abs(wrist.RISER_POS[2]-wrist.FOOT_POS[2]),
                             0.020+0.004)
        self.assertLessEqual(abs(wrist.RISER_POS[2]-wrist.TOP_POS[2]),
                             0.020+0.004)
        self.assertLessEqual(abs(wrist.TOP_POS[1]-wrist.RISER_POS[1]),
                             0.023+0.007)
        self.assertLessEqual(abs(wrist.BOARD_POS[1]-wrist.TOP_POS[1]),
                             0.023+0.016)

    def test_optical_camera_looks_at_target(self):
        origin=wrist.BOARD_POS
        target=wrist.DEFAULT_LOOK_AT
        x,y,z=wrist.camera_axes(origin,target)
        v=wrist.normalize(tuple(target[i]-origin[i] for i in range(3)))
        self.assertAlmostEqual(sum(-z[i]*v[i] for i in range(3)),1.0,places=6)
        self.assertAlmostEqual(sum(x[i]*y[i] for i in range(3)),0.0,places=6)
        quat=wrist.quat_from_basis(x,y,z)
        self.assertAlmostEqual(sum(a*a for a in quat),1.0,places=6)

    def test_unique_collision_ids_static(self):
        tree=ast.parse(inspect.getsource(wrist.install))
        ids=[]
        for call in (n for n in ast.walk(tree) if isinstance(n,ast.Call)):
            if not isinstance(call.func,ast.Attribute) or call.func.attr!="add_geom":
                continue
            named=next((kw.value for kw in call.keywords if kw.arg=="name"),None)
            if isinstance(named,ast.Subscript):
                self.assertEqual(named.value.id,"COLLISION_NAMES")
                ids.append(named.slice.value)
        self.assertCountEqual(ids,range(len(wrist.COLLISION_NAMES)))

    def test_mujoco_build_and_j5_follow(self):
        import mujoco
        spec=mujoco.MjSpec()
        gripper=spec.worldbody.add_body(name="gripper_link",pos=[0,0,.2])
        gripper.add_joint(name="wrist_roll",type=mujoco.mjtJoint.mjJNT_HINGE,
                          axis=[0,0,1])
        gripper.add_geom(name="robot_mount",type=mujoco.mjtGeom.mjGEOM_SPHERE,
                         size=[.012],mass=.1,contype=0,conaffinity=0)
        wrist.install(spec)
        model=spec.compile()
        data=mujoco.MjData(model)
        cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,wrist.CAMERA)
        self.assertGreaterEqual(cid,0)
        self.assertGreaterEqual(mujoco.mj_name2id(
            model,mujoco.mjtObj.mjOBJ_CAMERA,wrist.MOUNT_INSPECTION_CAMERA),0)
        for name in wrist.COLLISION_NAMES:
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
            self.assertGreaterEqual(gid,0)
            self.assertEqual(model.geom_contype[gid],1)
            self.assertEqual(model.geom_group[gid],3)
        for name in ("foot","vertical","top","pcb","lens"):
            gid=mujoco.mj_name2id(
                model,mujoco.mjtObj.mjOBJ_GEOM,"r2s_wrist_visual_"+name)
            self.assertGreaterEqual(gid,0)
            self.assertEqual(model.geom_contype[gid],0)
            self.assertEqual(model.geom_group[gid],0)
        mujoco.mj_forward(model,data)
        before=data.cam_xmat[cid].copy()
        qid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"wrist_roll")
        data.qpos[model.jnt_qposadr[qid]]=math.pi/2
        mujoco.mj_forward(model,data)
        self.assertGreater(np.linalg.norm(data.cam_xmat[cid]-before),1e-3)

    def test_install_twice_fails(self):
        import mujoco
        spec=mujoco.MjSpec()
        g=spec.worldbody.add_body(name="gripper_link")
        g.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.01])
        wrist.install(spec)
        with self.assertRaises(ValueError):
            wrist.install(spec)


if __name__=="__main__":
    unittest.main()
