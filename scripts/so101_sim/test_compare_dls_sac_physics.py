#!/usr/bin/env python3
"""Synthetic, hardware-free checks for DLS/SAC physical comparison."""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

import compare_dls_sac_physics as comparison


class PhysicalComparisonTests(unittest.TestCase):
    def test_load_two_policy_recorded_joint_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"comparison_rollout.csv"
            with path.open("w",newline="",encoding="utf-8") as fp:
                writer=csv.DictWriter(fp,fieldnames=["policy","stage",*comparison.JOINT_COLUMNS])
                writer.writeheader()
                for policy in comparison.POLICIES:
                    for stage in ("muhi","cup","home"):
                        row={"policy":policy,"stage":stage}
                        row.update({name:0.1 for name in comparison.JOINT_COLUMNS})
                        writer.writerow(row)
            output=comparison.load_comparison(path)
            self.assertEqual(tuple(output),comparison.POLICIES)
            self.assertEqual(len(output["sac"][0]),3)
            row,target,index=comparison.target_sequence(output["dls"],10,
                                              np.array([-.2]*5),np.array([.2]*5))
            self.assertEqual(row["stage"],"home")
            np.testing.assert_allclose(target,np.array([.1]*5))
            self.assertEqual(index,2)

    def test_missing_policy_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"rollout.csv"
            with path.open("w",newline="",encoding="utf-8") as fp:
                writer=csv.DictWriter(fp,fieldnames=["policy","stage",*comparison.JOINT_COLUMNS])
                writer.writeheader()
                writer.writerow({"policy":"dls","stage":"muhi",
                                 **{x:0. for x in comparison.JOINT_COLUMNS}})
            with self.assertRaises(ValueError):
                comparison.load_comparison(path)

    def test_pd_changes_joint_through_step_not_teleport(self):
        spec=mujoco.MjSpec()
        body=spec.worldbody.add_body(name="arm",pos=[0,0,.1])
        body.add_joint(name="j1",type=mujoco.mjtJoint.mjJNT_HINGE,
                       axis=[0,0,1],range=[-1,1],limited=True)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=[.03,.01,.01],
                      pos=[.09,0,0],mass=.1)
        model=spec.compile()
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        before=float(data.qpos[0])
        comparison.pd_step(model,data,np.array([0]),np.array([0]),np.array([.4]),
                           kp=12,kd=1.1,max_torque=3,
                           gripper_qid=None,gripper_did=None,gripper_goal=None)
        self.assertGreater(float(data.qpos[0]),before)
        self.assertLess(float(data.qpos[0]),.4)


if __name__=="__main__":
    unittest.main()
