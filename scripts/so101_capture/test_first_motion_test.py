"""All first-motion tests use only a synthetic bus and fake time."""

import unittest

from first_motion_test import SyntheticIO, run_micro


class FakeClock:
    def __init__(self):
        self.now = 0.

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0., seconds)


class MicroMotionTests(unittest.TestCase):
    def run_case(self, io, *, keypoll=lambda: ''):
        time = FakeClock()
        return run_micro(io, joint='wrist_flex.pos', delta_deg=1, cycles=20,
                         clock=time.clock, sleep=time.sleep, keypoll=keypoll)

    def test_synthetic_success_is_small_and_returns(self):
        io = SyntheticIO()
        result = self.run_case(io)
        self.assertTrue(result['completed'], result['fault'])
        self.assertEqual(io.writes, 20)
        self.assertEqual(io.position['wrist_flex.pos'], 45.)
        self.assertTrue(all(abs(r['limited'][3] - 45.) <= 1. for r in result['cycles']))
        self.assertTrue(all(r['full_cycle_ms'] <= 50 for r in result['cycles']))

    def test_torque_off_prevents_every_write(self):
        io = SyntheticIO(torque=False)
        result = self.run_case(io)
        self.assertFalse(result['completed'])
        self.assertEqual(io.writes, 0)

    def test_position_read_timeout_prevents_every_write(self):
        io = SyntheticIO(fail_read_at=5)
        result = self.run_case(io)
        self.assertFalse(result['completed'])
        self.assertEqual(io.writes, 0)
        self.assertTrue(result['communication']['latched'])

    def test_write_timeout_latches_without_retry_or_return(self):
        io = SyntheticIO(fail_write_at=3)
        result = self.run_case(io)
        self.assertFalse(result['completed'])
        self.assertEqual(io.writes, 3)
        self.assertTrue(result['communication']['latched'])
        self.assertEqual(len(result['cycles']), 2)

    def test_operator_stop_sends_nothing(self):
        io = SyntheticIO()
        result = self.run_case(io, keypoll=lambda: 'q')
        self.assertFalse(result['completed'])
        self.assertEqual(io.writes, 0)

    def test_existing_goal_mismatch_sends_nothing(self):
        io = SyntheticIO()
        io.last_goal['wrist_flex.pos'] += 2
        result = self.run_case(io)
        self.assertFalse(result['completed'])
        self.assertEqual(io.writes, 0)

    def test_cycle_overrun_stops_after_first_write(self):
        time = FakeClock()

        class SlowIO(SyntheticIO):
            def send(self, values):
                result = super().send(values)
                time.sleep(.06)
                return result

        io = SlowIO()
        result = run_micro(io, joint='wrist_flex.pos', delta_deg=1, cycles=20,
                           clock=time.clock, sleep=time.sleep)
        self.assertFalse(result['completed'])
        self.assertEqual(io.writes, 1)


if __name__ == '__main__':
    unittest.main()
