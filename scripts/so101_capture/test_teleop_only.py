"""Synthetic startup and control-loop checks; never opens serial ports."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from record_episode import NAMES, start_hardware, validate_raw_positions_within_calibration
from teleop_only import teleop_tick


MOTORS = [n.removesuffix('.pos') for n in NAMES]


class FakeBus:
    def __init__(self, *, auto_torque=False, partial=False, read_fault=False,
                 moved=False, bad_status=False, mixed_torque=False):
        self.motors = {n: object() for n in MOTORS}
        self.calibration = {n: SimpleNamespace(range_min=100, range_max=3000) for n in MOTORS}
        self.present = dict.fromkeys(MOTORS, 1000)
        self.goal = dict.fromkeys(MOTORS, 0)
        self.torque = dict.fromkeys(MOTORS, 0)
        self.status = dict.fromkeys(MOTORS, 0)
        self.auto_torque, self.partial, self.read_fault = auto_torque, partial, read_fault
        self.moved, self.bad_status, self.mixed_torque = moved, bad_status, mixed_torque
        self.writes = 0
        self.enabled = 0

    def connect(self):
        pass

    def disable_torque(self, motors=None):
        self.torque = dict.fromkeys(MOTORS, 0)

    def enable_torque(self, motors=None):
        self.enabled += 1
        self.torque = dict.fromkeys(MOTORS, 1)

    def sync_read(self, reg, **kwargs):
        if self.read_fault and self.writes and reg == 'Goal_Position':
            raise ConnectionError('synthetic read failure')
        return {'Present_Position': self.present, 'Goal_Position': self.goal,
                'Torque_Enable': self.torque, 'Status': self.status}[reg].copy()

    def sync_write(self, reg, values, **kwargs):
        self.writes += 1
        self.goal = values.copy()
        if self.partial:
            self.goal[MOTORS[0]] = 0
        if self.auto_torque:
            self.torque = dict.fromkeys(MOTORS, 1)
        if self.moved:
            self.present[MOTORS[0]] += 6
        if self.bad_status:
            self.status[MOTORS[0]] = 1
        if self.mixed_torque:
            self.torque[MOTORS[0]] = 0 if self.auto_torque else 1


class TeleopSafetyTests(unittest.TestCase):
    def test_raw_calibration_range_boundaries(self):
        calibration = {'elbow_flex': SimpleNamespace(range_min=907, range_max=3122)}
        for position in (907, 2000, 3122):
            with self.subTest(position=position):
                validate_raw_positions_within_calibration({'elbow_flex': position}, calibration)
        for position, deviation, direction in ((3123, 1, 'decrease'), (906, 1, 'increase')):
            with self.subTest(position=position):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf'Present={position}.*deviation={deviation} raw ticks.*{direction}',
                ):
                    validate_raw_positions_within_calibration({'elbow_flex': position}, calibration)

    def test_raw_calibration_range_reports_observed_upper_overrun(self):
        calibration = {'elbow_flex': SimpleNamespace(range_min=907, range_max=3122)}
        with self.assertRaisesRegex(RuntimeError, r'Present=3132.*deviation=10 raw ticks \(~0\.88 deg\)'):
            validate_raw_positions_within_calibration({'elbow_flex': 3132}, calibration)

    def devices(self, **kwargs):
        bus = FakeBus(**kwargs)
        follower = SimpleNamespace(bus=bus, is_calibrated=True, name='follower')
        leader = SimpleNamespace(bus=FakeBus(), is_calibrated=True, name='leader', configure=Mock())
        def configure():
            with bus.torque_disabled():
                pass
        from contextlib import contextmanager
        @contextmanager
        def standard_context(motors=None):
            bus.disable_torque(motors)
            try:
                yield
            finally:
                bus.enable_torque(motors)
        bus.torque_disabled = standard_context
        follower.configure = Mock(side_effect=configure)
        return follower, leader

    def start(self, follower, leader):
        with patch('record_episode.synchronize_positions') as alignment, patch('record_episode.time.sleep'):
            start_hardware(follower, leader, guarded_teleop_start=True)
        return alignment

    def test_goal_success_and_auto_torque_one(self):
        for auto in (False, True):
            with self.subTest(auto_torque=auto):
                follower, leader = self.devices(auto_torque=auto)
                alignment = self.start(follower, leader)
                self.assertEqual(follower.bus.writes, 1)
                self.assertEqual(follower.bus.enabled, 1)
                alignment.assert_called_once()

    def test_startup_checkpoints_record_six_axes_and_torque_one(self):
        follower, leader = self.devices(auto_torque=True)
        checkpoints = []
        with patch('record_episode.synchronize_positions'), patch('record_episode.time.sleep'):
            start_hardware(follower, leader, guarded_teleop_start=True,
                           startup_checkpoints=checkpoints)
        self.assertEqual(checkpoints[0]['stage'], 'before_goal_write')
        self.assertEqual(set(checkpoints[1]['raw']['Goal_Position']), set(MOTORS))
        self.assertEqual(set(checkpoints[1]['raw']['Torque_Enable'].values()), {1})

    def test_first_readback_waits_and_failed_read_never_configures(self):
        follower, leader = self.devices(read_fault=True)
        checkpoints = []
        waits = []
        first_goal_read = follower.bus.sync_read
        def read(reg, **kwargs):
            if follower.bus.writes and reg == 'Goal_Position':
                self.assertEqual(waits, [.02])
            return first_goal_read(reg, **kwargs)
        follower.bus.sync_read = read
        with patch('record_episode.time.sleep', side_effect=waits.append), \
             patch('record_episode.synchronize_positions') as alignment:
            with self.assertRaises(ConnectionError):
                start_hardware(follower, leader, guarded_teleop_start=True,
                               startup_checkpoints=checkpoints)
        self.assertEqual(follower.bus.writes, 1)
        self.assertIn('synthetic read failure', checkpoints[-1]['error'])
        follower.configure.assert_not_called()
        alignment.assert_not_called()

    def test_partial_goal_and_failed_read_stop_before_configure(self):
        for fault in ('partial', 'read_fault', 'moved', 'bad_status', 'mixed_torque'):
            follower, leader = self.devices(**{fault: True}, auto_torque=True)
            with self.assertRaises((RuntimeError, ConnectionError)):
                self.start(follower, leader)
            follower.configure.assert_not_called()
            self.assertEqual(follower.bus.enabled, 0)

    def test_motion_fault_reports_raw_delta_and_threshold(self):
        follower, leader = self.devices(moved=True, auto_torque=True)
        with self.assertRaisesRegex(RuntimeError, r'delta=\+6 raw ticks, allowed=\+/-5'):
            self.start(follower, leader)
        follower.configure.assert_not_called()

    def test_configure_exception_does_not_reenable_torque_or_align(self):
        follower, leader = self.devices(auto_torque=True)
        def fail():
            with follower.bus.torque_disabled():
                raise RuntimeError('synthetic configure failure')
        follower.configure.side_effect = fail
        with patch('record_episode.synchronize_positions') as alignment, patch('record_episode.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'synthetic configure failure'):
                start_hardware(follower, leader, guarded_teleop_start=True)
        self.assertEqual(follower.bus.enabled, 0)
        alignment.assert_not_called()

    def test_ambiguous_goal_write_failure_is_not_retried(self):
        follower, leader = self.devices()
        original = follower.bus.sync_write
        def fail_after_delivery(*args, **kwargs):
            original(*args, **kwargs)
            raise ConnectionError('synthetic ambiguous delivery')
        follower.bus.sync_write = fail_after_delivery
        with patch('record_episode.synchronize_positions') as alignment:
            with self.assertRaises(ConnectionError):
                start_hardware(follower, leader, guarded_teleop_start=True)
        self.assertEqual(follower.bus.writes, 1)
        follower.configure.assert_not_called()
        alignment.assert_not_called()

    def test_teleop_slew_and_tracking_stop(self):
        follower, leader = self.devices()
        follower.bus.present = dict.fromkeys(MOTORS, 0)
        follower.bus.torque = dict.fromkeys(MOTORS, 1)
        follower.send_action = Mock(side_effect=lambda x: x)
        leader.get_action = Mock(return_value=dict.fromkeys(NAMES, 40.))
        sent = teleop_tick(follower, leader, dict.fromkeys(NAMES, 0.))
        self.assertEqual(set(sent.values()), {1.5})
        follower.bus.present[MOTORS[0]] = 10
        with self.assertRaisesRegex(RuntimeError, 'Tracking error >8'):
            teleop_tick(follower, leader, sent)
        follower.send_action.assert_called_once()

    def test_teleop_communication_failure_blocks_action(self):
        follower, leader = self.devices()
        follower.bus.torque = dict.fromkeys(MOTORS, 1)
        follower.bus.present = dict.fromkeys(MOTORS, 0)
        follower.bus.read_fault = True
        follower.bus.writes = 1
        follower.bus.sync_read = Mock(side_effect=ConnectionError('synthetic read failure'))
        follower.send_action = Mock()
        with self.assertRaises(ConnectionError):
            teleop_tick(follower, leader, dict.fromkeys(NAMES, 0.))
        follower.send_action.assert_not_called()

    def test_teleop_torque_anomaly_blocks_action(self):
        follower, leader = self.devices()
        follower.bus.present = dict.fromkeys(MOTORS, 0)
        follower.bus.torque = dict.fromkeys(MOTORS, 1)
        follower.bus.torque[MOTORS[0]] = 0
        follower.send_action = Mock()
        with self.assertRaisesRegex(RuntimeError, 'state/torque/status invalid'):
            teleop_tick(follower, leader, dict.fromkeys(NAMES, 0.))
        follower.send_action.assert_not_called()


if __name__ == '__main__':
    unittest.main()
