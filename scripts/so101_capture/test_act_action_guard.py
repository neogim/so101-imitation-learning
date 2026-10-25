"""No-hardware tests for proposed ACT Action filtering."""

import unittest

from act_action_guard import ActionGuard, GuardConfig


class ActionGuardTest(unittest.TestCase):
    def make(self, lead=None):
        config = GuardConfig(limits=((-180, 180),) * 5 + ((0, 100),),
                             speed_per_s=(30,) * 6, gripper_closing_lead=lead)
        guard = ActionGuard(config)
        guard.reset_for_new_episode([0, 0, 0, 0, 0, 50])
        return guard

    def test_range_and_speed_are_separate(self):
        guard = self.make()
        goal, reasons = guard.propose([200, 0, 0, 0, 0, -2], [0, 0, 0, 0, 0, 50])
        self.assertEqual(goal.tolist(), [1.5, 0, 0, 0, 0, 48.5])
        self.assertIn('range:gripper.pos', reasons)
        self.assertIn('speed:gripper.pos', reasons)

    def test_optional_closing_lead(self):
        guard = self.make(lead=2)
        goal, reasons = guard.propose([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 50])
        self.assertEqual(goal[5], 48.5)
        self.assertIn('gripper_closing_lead', reasons)

    def test_tracking_latches_before_new_goal(self):
        guard = self.make()
        goal, reasons = guard.propose([0, 0, 0, 0, 0, 50], [9, 0, 0, 0, 0, 50])
        self.assertIsNone(goal)
        self.assertIn('tracking_error:shoulder_pan.pos', reasons[0])
        self.assertIsNone(guard.propose([0, 0, 0, 0, 0, 50], [0, 0, 0, 0, 0, 50])[0])

    def test_communication_and_stale_latch(self):
        for kwargs, expected in [({'communication_ok': False}, 'communication_fault'),
                                 ({'sample_age_s': .11}, 'stale_measurement')]:
            with self.subTest(kwargs=kwargs):
                guard = self.make()
                goal, reasons = guard.propose([0, 0, 0, 0, 0, 50], [0, 0, 0, 0, 0, 50], **kwargs)
                self.assertIsNone(goal)
                self.assertEqual(reasons, [expected])

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            GuardConfig(limits=((-180, 180),) * 5 + ((-10, 100),), speed_per_s=(30,) * 6)
        with self.assertRaises(ValueError):
            GuardConfig(limits=((-180, 180),) * 5 + ((0, 100),), speed_per_s=(30,) * 6,
                        tracking_error_limit=9)

    def test_initial_position_must_be_within_limits(self):
        guard = self.make()
        with self.assertRaises(ValueError):
            guard.reset_for_new_episode([181, 0, 0, 0, 0, 50])

    def test_chunk_boundary_jump_is_rate_limited_with_following_state(self):
        guard = self.make()
        measured = [0, 0, 0, 0, 0, 50]
        for requested in ([0, -45, 0, 0, 0, 50], [0, 45, 0, 0, 0, 50]):
            goal, _ = guard.propose(requested, measured)
            self.assertIsNotNone(goal)
            self.assertLessEqual(abs(goal[1] - measured[1]), 1.5)
            measured = goal.tolist()  # ideal synthetic follower; no motor command

    def test_every_frame_replanning_never_bypasses_limits(self):
        guard = self.make()
        measured = [0, 0, 0, 0, 0, 50]
        for frame in range(100):
            extreme = 300 if frame % 2 else -300
            raw = [extreme] * 5 + [200 if frame % 2 else -100]
            goal, reasons = guard.propose(raw, measured)
            self.assertIsNotNone(goal)
            self.assertTrue(reasons)
            self.assertTrue(all(-180 <= goal[i] <= 180 for i in range(5)))
            self.assertTrue(0 <= goal[5] <= 100)
            self.assertTrue(all(abs(goal[i]-measured[i]) <= 1.5 for i in range(6)))
            measured = goal.tolist()


if __name__ == '__main__':
    unittest.main()
