"""The same local safety shield must work for any controller proposal."""
import unittest

import numpy as np

from coordinated_joint_guard import joint_candidates, coordinate_step, find_static_safe_seed


class HalfspaceSweptGuard:
    """Toy wall: do not allow J2 + J3 to be negative."""
    def inspect(self,q):
        gap=float((q[1]+q[2])*1000.)
        return {"minimum_distance_mm":gap,
                "nearest_pair":("gripper","table"),
                "clearance_satisfied":gap >= -1e-8}

    def filter_step(self,current,target):
        q=np.asarray(current,float)
        target=np.asarray(target,float)
        # This toy obstacle is convex, so straight-line interpolation needs
        # only start/end; real guard separately samples swept geometry.
        if not self.inspect(q)["clearance_satisfied"]:
            return q.copy(),{"status":"unsafe_start",**self.inspect(q)}
        if not self.inspect(target)["clearance_satisfied"]:
            return q.copy(),{"status":"blocked",**self.inspect(target)}
        return target.copy(),{"status":"accepted",**self.inspect(target)}


class CoupledShieldTests(unittest.TestCase):
    def setUp(self):
        self.guard=HalfspaceSweptGuard()
        self.q=np.zeros(5)
        self.low=np.full(5,-.2)
        self.high=np.full(5,.2)

    def test_generates_elbow_and_wrist_alternatives(self):
        proposals=joint_candidates(self.q,self.q+[0,-.04,0,0,0],
                                    self.low,self.high)
        self.assertTrue(any(p[2]>.01 for p in proposals))
        self.assertTrue(any(p[3]>.01 for p in proposals))
        self.assertTrue(any(p[1]<0 and p[2]>0 for p in proposals))

    def test_recovers_from_collision_proposal_using_elbow(self):
        proposed=np.array([0,-.04,0,0,0],float)
        next_q,info=coordinate_step(
            self.guard,self.q,proposed,self.low,self.high,
            lambda x: float(np.sum((x-np.array([0,-.04,.04,0,0]))**2)),
            action_weight=.01)
        self.assertEqual(info["status"],"coordinated")
        self.assertGreater(next_q[2],0)
        self.assertTrue(self.guard.inspect(next_q)["clearance_satisfied"])

    def test_static_safe_seed_is_separate_reinitialized_baseline(self):
        initial=np.array([0,-.06,0,0,0],float)
        seed,report=find_static_safe_seed(
            self.guard,initial,self.low,self.high,max_delta_rad=.12)
        self.assertIsNotNone(seed)
        self.assertEqual(report["status"],"FOUND_STATIC_SAFE_SIM_INITIALIZATION")
        self.assertTrue(self.guard.inspect(seed)["clearance_satisfied"])
        self.assertFalse(np.allclose(seed,initial))
        self.assertIn("NOT planned",report["warning"])

    def test_static_seed_returns_none_if_no_safe_poses(self):
        class AlwaysBlocked:
            def inspect(self,q):
                return {"clearance_satisfied":False,"minimum_distance_mm":-9}
        seed,report=find_static_safe_seed(
            AlwaysBlocked(),self.q,self.low,self.high,max_delta_rad=.12)
        self.assertIsNone(seed)
        self.assertEqual(report["safe_configurations"],0)

    def test_rejects_unsafe_start_without_teleport(self):
        initial=np.array([0,-.10,0,0,0],float)
        q,info=coordinate_step(
            self.guard,initial,np.zeros(5),
            self.low,self.high,lambda x:0.)
        self.assertEqual(info["status"],"unsafe_start")
        np.testing.assert_array_equal(q,initial)


if __name__=="__main__":
    unittest.main()
