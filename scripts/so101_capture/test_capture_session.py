"""Deterministic state/safety tests: no devices or dataset downloads."""
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock
from capture_session import Session, SimulationIO, DatasetSink, HardwareIO
from record_episode import NAMES
from j6_closing_guard import limit_j6_closing
from j6_contact_estimate import ContactCalibration, estimate_contact


class Sink:
    def __init__(self):
        self.items = []
        self.saved = 0
    def check(self):
        pass
    def submit(self, kind, value=None):
        self.items.append((kind, value))
        if kind == 'save':
            self.saved += 1


class SessionTests(unittest.TestCase):
    def make(self, **kwargs):
        io, sink = SimulationIO(), Sink()
        s = Session(io, sink, seconds=.5, min_seconds=.1, countdown=0, num_episodes=3, **kwargs)
        return s, io, sink

    def ready(self, s, t=0):
        for i in range(12):
            s.tick(t+i*.05)
        s.tick(t+.6, 'r')

    def test_three_episodes_early_timeout_and_indices(self):
        s, io, sink = self.make()
        t = 0
        for ep in range(3):
            self.ready(s, t)
            for i in range(1,12):
                now = t+.6+i*.05
                s.tick(now, 'e' if ep == 1 and i == 4 else '')
                if s.mode == 'TELEOP_RESET':
                    break
            self.assertEqual(s.mode, 'TELEOP_RESET')
            s.tick(now+.05, 'h')
            t = now+.1
        self.assertEqual([r['episode_index'] for r in s.outcomes], [0,1,2])
        self.assertEqual([r['end_reason'] for r in s.outcomes], ['time_limit','manual','time_limit'])
        self.assertTrue(all(r['overall']=='unknown' and not r['adopt'] for r in s.outcomes))
        self.assertEqual(sum(k=='frame' for k,_ in sink.items), sum(r['frames'] for r in s.outcomes))

    def test_safe_hold_never_follows_or_writes(self):
        s, io, sink = self.make()
        s.tick(0)
        io.request = Mock(side_effect=AssertionError('Leader read in SAFE_RESET'))
        before = len(io.sent)
        for i in range(1,10):
            s.tick(i*.05)
        self.assertEqual(len(io.sent), before)
        io.request.assert_not_called()
        self.assertFalse(sink.items)

    def test_multicamera_not_sampled_in_safe_reset_or_after_episode_finish(self):
        s, io, sink = self.make(image_names=('side', 'wrist'), image_shape=(480, 640, 3))
        io.image_names = ('side', 'wrist')
        io.image_shape = (480, 640, 3)
        original_images = io.images
        io.images = Mock(side_effect=AssertionError('camera sampled outside RECORDING'))
        for i in range(12):
            s.tick(i*.05)
        io.images.assert_not_called()

        io.images = Mock(side_effect=original_images)
        s.tick(.6, 'r')
        s.tick(.65)
        self.assertEqual(s.mode, 'RECORDING')
        recorded_calls = io.images.call_count
        s.tick(.71, 'e')
        self.assertEqual(s.mode, 'TELEOP_RESET')
        self.assertEqual(io.images.call_count, recorded_calls)
        self.assertEqual(sink.saved, 1)

    def test_resume_large_difference_refused(self):
        s, io, sink = self.make()
        for i in range(12):
            s.tick(i*.05)
        io.request = lambda: dict.fromkeys(NAMES, 10.)
        count = len(io.sent)
        s.tick(.6, 'r')
        self.assertEqual(s.mode,'SAFE_RESET')
        self.assertEqual(len(io.sent), count)

    def test_slew_limit_and_requested_distinct(self):
        s, io, sink = self.make()
        self.ready(s)
        io.request = lambda: dict.fromkeys(NAMES, 20.)
        s.tick(.65)
        row = [v for k,v in sink.items if k=='frame'][-1]
        self.assertAlmostEqual(float(row['action'][0]), 1.5)
        self.assertEqual(float(row['action_requested'][0]), 20.)

    def test_rotary_and_gripper_slew_override_separately(self):
        s, io, sink = self.make(rotary_slew=40, gripper_slew=30)
        io.positions['gripper.pos'] = 50
        self.ready(s)
        io.request = lambda: dict.fromkeys(NAMES, 20.)
        s.tick(.65)
        row = [v for kind, v in sink.items if kind == 'frame'][-1]
        self.assertAlmostEqual(float(row['action'][0]), 2.0)
        self.assertAlmostEqual(float(row['action'][-1]), 48.5)
        self.assertEqual((s.rotary_slew, s.gripper_slew), (40, 30))
        legacy, _, _ = self.make(slew=30)
        self.assertEqual((legacy.rotary_slew, legacy.gripper_slew), (30, 30))

    def test_separate_slew_keeps_j6_guard_and_tracking_stop(self):
        s, io, _ = self.make(rotary_slew=40, gripper_slew=30, j6_max_closing_lead=3)
        io.positions['gripper.pos'] = 50
        self.ready(s)
        io.request = lambda: {**dict.fromkeys(NAMES, 0.), 'gripper.pos': 40.}
        for now in (.65, .70, .75):
            io.positions['gripper.pos'] = 50
            s.tick(now)
        self.assertEqual(s.j6_guard_events[-1]['reason'], 'closing_lead_limited')
        before = len(io.sent)
        io.positions['gripper.pos'] = 56
        with self.assertRaisesRegex(RuntimeError, 'Tracking error >8'):
            s.tick(.80)
        self.assertEqual(len(io.sent), before)

    def test_j6_telemetry_disabled_by_default_and_sampled_at_configured_rate(self):
        s, io, _ = self.make()
        io.j6_register = Mock(side_effect=AssertionError('diagnostic read when disabled'))
        self.ready(s)
        s.tick(.65)
        self.assertEqual(s.j6_diagnostics, [])
        s2, io2, _ = self.make(j6_telemetry_hz=5)
        io2.j6_register = Mock(side_effect=lambda register: {'Present_Load': 12,
                                                               'Present_Current': 34}[register])
        self.ready(s2)
        for tick in (.65, .70, .75, .80, .85, .90):
            s2.tick(tick)
        self.assertEqual(len(s2.j6_diagnostics), 2)
        self.assertEqual(io2.j6_register.call_count, 4)
        row = s2.j6_diagnostics[-1]
        self.assertEqual((row['Present_Load'], row['Present_Current']), (12, 34))
        self.assertEqual(row['unit'], 'normalized_0_100')
        self.assertEqual(set(row['reads']), {'Present_Load', 'Present_Current'})

    def test_j6_telemetry_failure_latches_and_blocks_goal(self):
        s, io, _ = self.make(j6_telemetry_hz=5)
        self.ready(s)
        before = len(io.sent)
        io.j6_register = Mock(side_effect=ConnectionError('J6 read timeout'))
        with self.assertRaisesRegex(ConnectionError, 'Present_Load'):
            s.tick(.85)
        self.assertEqual(len(io.sent), before)
        self.assertTrue(s.io.latched)
        self.assertIn('error', s.j6_diagnostics[-1])
        io.j6_register = Mock(return_value=0)
        with self.assertRaises(ConnectionError):
            s.tick(.90)
        self.assertEqual(len(io.sent), before)
        io.j6_register.assert_not_called()

    def test_j6_hardware_read_targets_only_gripper_without_write(self):
        bus = SimpleNamespace(read=Mock(return_value=17), sync_write=Mock())
        hardware = HardwareIO(SimpleNamespace(bus=bus), None, None)
        self.assertEqual(hardware.j6_register('Present_Load'), 17)
        self.assertEqual(hardware.j6_register('Present_Current'), 17)
        self.assertEqual(bus.read.call_count, 2)
        bus.read.assert_any_call('Present_Load', 'gripper', normalize=False, num_retry=0)
        bus.read.assert_any_call('Present_Current', 'gripper', normalize=False, num_retry=0)
        bus.sync_write.assert_not_called()

    def test_j6_closing_guard_limits_only_additional_close(self):
        self.assertEqual(limit_j6_closing(measured=22.653, previous_goal=14.072,
            requested=8.534, proposed_goal=12.572, max_lead=3)[0], 14.072)
        self.assertEqual(limit_j6_closing(measured=30, previous_goal=29,
            requested=20, proposed_goal=27.5, max_lead=3)[0], 27.5)
        self.assertEqual(limit_j6_closing(measured=22, previous_goal=20,
            requested=25, proposed_goal=21.5, max_lead=3)[0], 21.5)

    def test_j6_guard_records_actual_action_and_requested_action(self):
        s, io, sink = self.make(j6_max_closing_lead=3)
        io.positions['gripper.pos'] = 50
        self.ready(s)
        io.request = lambda: {**dict.fromkeys(NAMES, 0.), 'gripper.pos': 40.}
        s.tick(.65)
        frame = [v for kind, v in sink.items if kind == 'frame'][-1]
        self.assertEqual(float(frame['action'][-1]), 48.5)
        self.assertEqual(float(frame['action_requested'][-1]), 40.)
        # Lagging measurement freezes only further closing on the third cycle.
        io.positions['gripper.pos'] = 50
        s.tick(.70)
        io.positions['gripper.pos'] = 50
        s.tick(.75)
        self.assertEqual(s.j6_guard_events[-1]['reason'], 'closing_lead_limited')
        self.assertEqual(s.j6_guard_events[-1]['guarded_goal'], io.sent[-1]['gripper.pos'])

    def test_contact_estimate_requires_calibration_fresh_complete_data(self):
        row = dict(completed_monotonic=1.0, Present_Current=30, Present_Load=-20,
            measured_position=25, previous_sent_goal=20, leader_requested_this_cycle=15)
        self.assertFalse(estimate_contact(row, now=1.05)['valid'])
        cal = ContactCalibration(10, 0, 10, 10, 3, .1)
        self.assertTrue(estimate_contact(row, now=1.05, calibration=cal)['contact_suspected'])
        self.assertEqual(estimate_contact(row, now=1.3, calibration=cal)['reason'], 'stale')
        self.assertFalse(estimate_contact({}, now=1.05, calibration=cal)['valid'])

    def test_multicamera_dropout_blocks_current_goal(self):
        s, io, _ = self.make(image_names=('overhead', 'side'))
        import numpy as np
        io.images = Mock(return_value=(
            {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in s.image_names},
            {name: 0.0 for name in s.image_names}))
        self.ready(s)
        before = len(io.sent)
        io.camera = SimpleNamespace(timing_brief=lambda: {
            'side': {'latest_image_timestamp': .5, 'image_age_ms': 109.,
                     'last_fetch': {'request_ms': 12., 'image_decode_ms': 1.}}})
        io.images = Mock(side_effect=ConnectionError('side camera unavailable'))
        with self.assertRaisesRegex(ConnectionError, 'Camera failure'):
            s.tick(.65)
        self.assertEqual(len(io.sent), before)
        cycle = s.io.report()['cycle_history'][-1]
        self.assertFalse(cycle['ok'])
        self.assertIn('image_check_started_monotonic', cycle)
        self.assertIn('image_check_finished_monotonic', cycle)
        self.assertEqual(cycle['cameras']['side']['image_age_ms'], 109.)
        reads = [r for r in s.io.report()['read_history'] if r['cycle'] == cycle['cycle']]
        self.assertEqual({(r['role'], r['register']) for r in reads},
                         {('follower', 'Present_Position'), ('follower', 'Torque_Enable'),
                          ('leader', 'Present_Position')})
        self.assertTrue(all(r['callback_started_monotonic'] <= r['callback_finished_monotonic']
                            for r in reads))

    def test_camera_diagnostic_failure_preserves_original_fault(self):
        s, io, _ = self.make(image_names=('side', 'wrist'))
        import numpy as np
        io.images = lambda: ({name: np.zeros((480, 640, 3), dtype=np.uint8)
                              for name in s.image_names}, {name: 0.0 for name in s.image_names})
        self.ready(s)
        before = len(io.sent)
        io.images = Mock(side_effect=ConnectionError('side stale frame'))
        io.camera = SimpleNamespace(timing_brief=Mock(side_effect=RuntimeError('diagnostic failed')))
        with self.assertRaisesRegex(ConnectionError, 'Camera failure'):
            s.tick(.65)
        self.assertEqual(len(io.sent), before)
        self.assertIn('diagnostic failed', s.io.report()['cycle_history'][-1]['camera_timing_error'])

    def test_drift_torque_nonfinite_and_range_faults(self):
        for fault in ('drift','torque','nan','range'):
            s, io, sink = self.make()
            s.tick(0)
            count = len(io.sent)
            if fault == 'torque':
                io.torque = False
            else:
                io.positions[NAMES[0]] = {'drift':3., 'nan':float('nan'),'range':200.}[fault]
            with self.assertRaises((ValueError,RuntimeError)):
                s.tick(.05)
            self.assertEqual(len(io.sent), count)

    def test_interrupted_episode_preserved_and_not_adopted(self):
        s, io, sink = self.make()
        self.ready(s)
        self.assertEqual(s.stop(.65), [])
        self.assertEqual(s.outcomes[0]['end_reason'],'interrupted')
        self.assertFalse(s.outcomes[0]['adopt'])
        self.assertEqual(s.mode,'STOPPED')
        self.assertEqual(sink.saved,1)

    def test_minimum_duration_and_premature_quit(self):
        s, io, sink = self.make()
        self.ready(s)
        s.tick(.65,'e')
        self.assertEqual(s.mode,'RECORDING')
        s.tick(.7,'q')
        self.assertEqual(s.mode,'RECORDING')
        self.assertFalse(s.done)

    def test_gap_no_burst(self):
        s, io, sink = self.make()
        self.ready(s)
        count = len(io.sent)
        with self.assertRaisesRegex(RuntimeError,'Control gap'):
            s.tick(1.)
        self.assertEqual(len(io.sent),count)

    def test_label_requires_explicit_success(self):
        s,io,sink=self.make()
        self.ready(s)
        s.tick(.75,'e'); s.tick(.8,'h')
        s.tick(.85,'a')
        self.assertFalse(s.outcomes[0]['adopt'])
        for i,key in enumerate('plsa'):
            s.tick(.9+i*.05,key)
        self.assertTrue(s.outcomes[0]['adopt'])
        s.tick(1.1,'f')
        self.assertFalse(s.outcomes[0]['adopt'])

    def test_countdown_movement_cancels_without_send(self):
        s,io,sink=self.make()
        s.countdown=3
        self.ready(s)
        count=len(io.sent)
        io.request=lambda:dict.fromkeys(NAMES, 10.)
        s.tick(.65)
        self.assertEqual(s.mode,'SAFE_RESET')
        self.assertEqual(len(io.sent),count)

    def test_unavailable_hold_preserves_partial_without_claiming_success(self):
        s, io, sink = self.make()
        self.ready(s)
        io.state = Mock(side_effect=OSError('no status packet'))
        count = len(io.sent)
        errors = s.stop(.65)
        self.assertTrue(any('Hold unavailable' in e for e in errors))
        self.assertEqual(len(io.sent), count)
        self.assertEqual(sink.saved, 1)
        self.assertEqual(s.outcomes[0]['end_reason'], 'interrupted')

    def test_bounded_queue_refuses_silent_drop(self):
        import queue
        sink = DatasetSink.__new__(DatasetSink)
        sink.error = None
        sink.queue = queue.Queue(1)
        sink.submit('frame', {'one': 1})
        with self.assertRaisesRegex(RuntimeError, 'queue full'):
            sink.submit('frame', {'two': 2})
        self.assertEqual(sink.queue.get(), ('frame', {'one': 1}))

    def test_legacy_record_loop_action_provenance(self):
        import numpy as np
        from record_episode import record_loop, DummyRobot, DummyTeleop
        dataset = Mock()
        record_loop(dataset, DummyRobot(), DummyTeleop(), seconds=.1)
        self.assertGreaterEqual(dataset.add_frame.call_count, 1)
        for call in dataset.add_frame.call_args_list:
            frame = call.args[0]
            np.testing.assert_allclose(frame['action'], frame['action_requested']*.5)

    def test_writer_error_finalizes_and_surfaces(self):
        with tempfile.TemporaryDirectory() as d:
            ds=Mock()
            ds.add_frame.side_effect=ValueError('injected write fault')
            sink=DatasetSink(ds,Path(d))
            sink.submit('frame',{})
            sink.thread.join(2)
            with self.assertRaisesRegex(RuntimeError,'writer failed'):
                sink.close()
            ds.finalize.assert_called_once()
            self.assertTrue((Path(d)/'writer_error.txt').exists())

    def test_slow_save_does_not_block_control(self):
        with tempfile.TemporaryDirectory() as d:
            import threading
            gate=threading.Event()
            ds=Mock()
            ds.has_pending_frames.return_value=False
            ds.save_episode.side_effect=lambda **_:gate.wait(2)
            sink=DatasetSink(ds,Path(d))
            try:
                sink.submit('save')
                s,io,_=self.make()
                s.sink=sink
                s.tick(0)
                start=time.perf_counter()
                for i in range(1,10):
                    s.tick(i*.05)
                self.assertLess(time.perf_counter()-start,.1)
                self.assertEqual(sink.saved,0)
            finally:
                gate.set()
                sink.close()


if __name__=='__main__':
    unittest.main()
