"""No LeRobot hardware object is constructed by these tests."""

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from act_action_guard import NAMES
from act_handoff_launcher import (HandoffController, HardwareAdapter,
                                  finalize_handoff_run, read_only_startup_preflight,
                                  raise_run_fault, save_prediction_input_images, JOINTS)
from teleop_act_handoff import HandoffFault


class FakeIO:
    def __init__(self):
        self.position = {n: 20.0 for n in NAMES}
        self.position[NAMES[-1]] = 50.0
        self.goal = self.position.copy()
        self.torque = dict.fromkeys(NAMES, 1)
        self.status = dict.fromkeys(NAMES, 0)
        self.writes = []
        self.requests = 0
        self.fail_read = False
        self.configure_calls = self.torque_calls = 0

    def limits(self):
        return {n: (0, 100) if n == NAMES[-1] else (-90, 90) for n in NAMES}

    def snapshot(self):
        import time
        if self.fail_read:
            raise ConnectionError('injected Present_Position timeout')
        return dict(present=self.position.copy(), goal=self.goal.copy(),
                    torque=self.torque.copy(), status=self.status.copy(),
                    sampled_at=time.perf_counter(), sample_age_s=0.001)

    def request(self):
        self.requests += 1
        return self.position.copy()

    def image(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def image_sample(self):
        import time
        return self.image(), time.perf_counter(), dict(index=0, backend=1200, device_name=None)

    def calibration_ok(self):
        return True

    def write(self, goal):
        self.writes.append(goal.copy())
        self.goal = goal.copy()
        self.position = goal.copy()  # ideal synthetic follower
        return goal.copy()


class FakePredictor:
    def __init__(self):
        self.calls = 0
        self.offset = 0

    def predict(self, state, image):
        self.calls += 1
        refreshed = self.offset == 0
        self.offset = (self.offset+1)%5
        return np.full(6, 80.0), refreshed

    def reset(self):
        self.offset = 0


class FakeReadOnlyPreflight:
    def __init__(self):
        self.raw = {register: dict.fromkeys(JOINTS, value) for register,value in
                    (('Goal_Position',100),('Present_Position',100),
                     ('Torque_Enable',0),('Status',0))}
        self.bounds = {name:(50,200) for name in JOINTS}
        self.calibrated = {register:dict.fromkeys(NAMES,20.0)
                           for register in ('Goal_Position','Present_Position')}
        self.matches = True
        self.closed = False
        self.writes = 0

    def open(self):return {'read_only':True}
    def calibration_matches(self):return self.matches
    def limits(self):return self.bounds
    def read_raw(self):return {k:v.copy() for k,v in self.raw.items()}
    def read_calibrated_positions(self):return self.calibrated
    def close_without_writes(self):self.closed = True


class ControllerTests(unittest.TestCase):
    def test_continuous_act_moves_six_axes_and_operator_stop_closes_writer(self):
        io, policy = FakeIO(), FakePredictor()
        policy.predict = lambda state, image: (np.asarray([25.0] * 5 + [45.0]), True)
        ctl = HandoffController(io, policy, observe_only=False, continuous_act=True,
                                act_seconds=10, act_speed_deg_s=10,
                                act_gripper_speed_s=20)
        ctl.tick(); ctl.tick('x'); ctl.tick()
        self.assertEqual(ctl.phase, 'ACT')
        writes = len(io.writes)
        with self.assertRaises(HandoffFault):
            ctl.gate.send_teleop(io.goal)
        ctl.tick()
        self.assertEqual(len(io.writes), writes + 1)
        self.assertAlmostEqual(io.goal[NAMES[0]], 20.5)
        self.assertAlmostEqual(io.goal[NAMES[5]], 49.0)
        self.assertEqual(ctl.events[-1]['pre_send_present'][NAMES[0]], 20.0)
        ctl.tick('q')
        self.assertEqual(ctl.phase, 'STOPPED')
        self.assertEqual(ctl.gate.owner, 'STOPPED')
        self.assertEqual(ctl.continuous_completion['reason'], 'operator_stop')
        self.assertEqual(ctl.continuous_completion['goal_write_attempts_since_handoff'], 1)
        with self.assertRaises(HandoffFault):
            ctl.gate.send_act([25.0] * 6, io.position, sample_age_s=0)
        self.assertEqual(len(io.writes), writes + 1)

    def test_continuous_act_time_limit_and_faults(self):
        for cause in ('time', 'status', 'communication', 'tracking'):
            with self.subTest(cause=cause):
                io, policy = FakeIO(), FakePredictor()
                policy.predict = lambda state, image: (np.asarray([25.0] * 5 + [45.0]), True)
                ctl = HandoffController(io, policy, observe_only=False, continuous_act=True)
                ctl.tick(); ctl.tick('x'); ctl.tick()
                writes = len(io.writes)
                if cause == 'time':
                    ctl.act_started_at -= 10.01
                    ctl.tick()
                    self.assertEqual(ctl.phase, 'STOPPED')
                    self.assertEqual(ctl.continuous_completion['reason'], 'time_limit')
                else:
                    if cause == 'status': io.status[NAMES[0]] = 1
                    if cause == 'communication': io.fail_read = True
                    if cause == 'tracking': io.position[NAMES[0]] += 9
                    with self.assertRaises((HandoffFault, ConnectionError)):
                        ctl.tick()
                    self.assertEqual(ctl.phase, 'FAULT')
                self.assertEqual(len(io.writes), writes)

    def test_continuous_act_requires_bounded_speed_and_duration(self):
        io, policy = FakeIO(), FakePredictor()
        for seconds in (90, 120):
            ctl = HandoffController(io, policy, observe_only=False,
                                    continuous_act=True, act_seconds=seconds)
            self.assertEqual(ctl.act_seconds, seconds)
        for options in (dict(act_seconds=0), dict(act_seconds=120.01),
                        dict(act_speed_deg_s=31), dict(act_gripper_speed_s=0)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                HandoffController(io, policy, observe_only=False,
                                  continuous_act=True, **options)

    def test_continuous_act_operator_can_stop_before_handoff(self):
        io, policy = FakeIO(), FakePredictor()
        ctl = HandoffController(io, policy, observe_only=False, continuous_act=True)
        ctl.tick()
        writes = len(io.writes)
        ctl.tick('q')
        self.assertEqual(ctl.phase, 'STOPPED')
        self.assertEqual(ctl.gate.owner, 'STOPPED')
        self.assertEqual(len(io.writes), writes)

    def test_hardware_writer_refuses_relative_target_clipping(self):
        adapter = HardwareAdapter()
        adapter.follower = Mock()
        adapter.follower.config.max_relative_target = 1.0
        with self.assertRaisesRegex(RuntimeError, 'may clip Goal'):
            adapter.write(dict.fromkeys(NAMES, 20.0))
        self.assertEqual(adapter.goal_writes, 0)
        adapter.follower.send_action.assert_not_called()

    def test_hardware_adapter_reuses_successful_teleop_startup(self):
        adapter = HardwareAdapter()
        follower, leader = Mock(), Mock()
        camera = Mock()
        with patch('obs_camera.OBSCamera', return_value=camera), \
             patch('record_episode.make_hardware', return_value=(follower, leader)) as make, \
             patch('record_episode.start_hardware') as start:
            adapter.open()
        camera.connect.assert_called_once()
        make.assert_called_once_with(camera)
        start.assert_called_once_with(follower, leader, guarded_teleop_start=True,
                                      startup_checkpoints=adapter.startup_checkpoints)

    def test_camera_sample_pairs_frame_and_timestamp_without_hardware(self):
        import threading
        import time
        adapter = HardwareAdapter()
        camera = Mock()
        camera.is_connected = True
        camera.thread.is_alive.return_value = True
        camera.frame_lock = threading.Lock()
        camera.latest_frame = np.full((480, 640, 3), 23, dtype=np.uint8)
        camera.latest_timestamp = time.perf_counter()
        camera.config.index_or_path = 0
        camera.config.backend = 1200
        adapter.camera = camera
        rgb, captured_at, info = adapter.image_sample()
        self.assertEqual(rgb[0, 0, 0], 23)
        self.assertEqual(captured_at, camera.latest_timestamp)
        self.assertEqual(info, dict(index=0, backend=1200, device_name=None))
        camera.latest_frame[:] = 99
        self.assertEqual(rgb[0, 0, 0], 23)
        camera.latest_timestamp -= .11
        with self.assertRaises(TimeoutError):
            adapter.image_sample()

    def test_prestartup_raw_register_gate_refuses_anomalies_without_writes(self):
        cases = ('normal','calibration','torque','status','present_range',
                 'stored_goal_range','goal_present_gap')
        for case in cases:
            with self.subTest(case=case):
                reader=FakeReadOnlyPreflight()
                if case=='calibration':reader.matches=False
                if case=='torque':reader.raw['Torque_Enable'][JOINTS[0]]=1
                if case=='status':reader.raw['Status'][JOINTS[1]]=1
                if case=='present_range':reader.raw['Present_Position'][JOINTS[2]]=250
                if case=='stored_goal_range':reader.raw['Goal_Position'][JOINTS[3]]=0
                if case=='goal_present_gap':reader.calibrated['Goal_Position'][NAMES[4]]=35.0
                if case in ('normal','goal_present_gap'):
                    report=read_only_startup_preflight(reader)
                    self.assertEqual(report['result'],'ok')
                    if case=='goal_present_gap':
                        self.assertEqual(report['goal_present_gap'][NAMES[4]],15.0)
                else:
                    with self.assertRaises(Exception) as caught:read_only_startup_preflight(reader)
                    self.assertEqual(caught.exception.diagnostics['result'],'refused')
                    if case!='calibration':
                        self.assertIn('first_raw',caught.exception.diagnostics)
                self.assertTrue(reader.closed)
                self.assertEqual(reader.writes,0)

    def test_sync_only_preflight_accepts_stale_goal_but_not_torque_or_status(self):
        reader=FakeReadOnlyPreflight()
        reader.raw['Goal_Position']=dict.fromkeys(JOINTS,0)
        report=read_only_startup_preflight(reader,goal_sync_only=True)
        self.assertEqual(report['result'],'ok')
        self.assertEqual(report['raw_goal_present_gap'][JOINTS[0]],-100)
        self.assertEqual(reader.writes,0)
        for register in ('Torque_Enable','Status'):
            reader=FakeReadOnlyPreflight()
            reader.raw['Goal_Position']=dict.fromkeys(JOINTS,0)
            reader.raw[register][JOINTS[0]]=1
            with self.assertRaises(HandoffFault):
                read_only_startup_preflight(reader,goal_sync_only=True)
            self.assertEqual(reader.writes,0)

    def setup_controller(self, observe_only=False):
        io, policy = FakeIO(), FakePredictor()
        controller = HandoffController(io, policy, observe_only=observe_only, observe_frames=1)
        controller.tick()
        controller.tick('x')
        self.assertEqual(controller.phase, 'CHECK')
        return io, policy, controller

    def test_observation_only_blocks_both_writers(self):
        io, policy, ctl = self.setup_controller(observe_only=True)
        writes = len(io.writes)
        requests = io.requests
        ctl.tick()  # CHECK and warm-up; no Goal
        self.assertEqual(ctl.phase, 'OBSERVE_ACT')
        ctl.tick()  # one real observation-only ACT cycle
        self.assertEqual(ctl.phase, 'OBSERVED')
        self.assertEqual(ctl.gate.owner, 'CHECK')
        self.assertEqual(len(io.writes), writes)
        self.assertEqual(io.requests, requests)
        with self.assertRaises(HandoffFault):
            ctl.gate.send_teleop(io.goal)
        with self.assertRaises(HandoffFault):
            ctl.gate.send_act([80] * 6, io.position, sample_age_s=0)
        self.assertEqual(policy.calls, 2)

    def test_handoff_calibration_read_fault_latches_and_preserves_exception(self):
        io, policy = FakeIO(), FakePredictor()
        ctl = HandoffController(io, policy, observe_only=True, observe_frames=3)
        ctl.tick(); ctl.tick('x')
        writes = len(io.writes)
        def failed_calibration():
            raise RuntimeError("Failed to read 'Homing_Offset' on id_=6 after 1 tries. "
                               "[RxPacketError] Input voltage error!")
        io.calibration_ok = failed_calibration
        with self.assertRaisesRegex(RuntimeError, 'Input voltage error'):
            ctl.tick()  # CHECK must still perform full calibration verification
        self.assertEqual(ctl.phase, 'FAULT')
        self.assertEqual(ctl.gate.owner, 'FAULT')
        self.assertEqual(len(io.writes), writes)
        self.assertEqual(ctl.events[-1]['event'], 'fault')
        self.assertIn('Homing_Offset', ctl.events[-1]['reason'])
        with self.assertRaises(HandoffFault):
            ctl.tick()
        self.assertEqual(len(io.writes), writes)

    def test_observation_uses_verified_handoff_without_calibration_reads(self):
        io, policy = FakeIO(), FakePredictor()
        checks = []
        def calibrated():
            checks.append(True)
            return True
        io.calibration_ok = calibrated
        ctl = HandoffController(io, policy, observe_only=True, observe_frames=3)
        ctl.tick(); ctl.tick('x'); ctl.tick()
        self.assertEqual(len(checks), 2)  # before and after warm-up in CHECK
        for cycle in range(3):
            ctl.tick()
            self.assertEqual(len(checks), 2)
        self.assertEqual(ctl.phase, 'OBSERVED')

    def test_observation_status_and_communication_fault_stop_without_writes(self):
        for fault in ('status', 'communication'):
            with self.subTest(fault=fault):
                io, policy = FakeIO(), FakePredictor()
                ctl = HandoffController(io, policy, observe_only=True, observe_frames=3)
                ctl.tick(); ctl.tick('x'); ctl.tick()
                writes = len(io.writes)
                if fault == 'status':
                    io.status[NAMES[-1]] = 1
                else:
                    io.fail_read = True
                with self.assertRaises((HandoffFault, ConnectionError)):
                    ctl.tick()
                self.assertEqual(ctl.phase, 'FAULT')
                self.assertEqual(ctl.gate.owner, 'FAULT')
                self.assertEqual(len(io.writes), writes)
                with self.assertRaises(HandoffFault):
                    ctl.tick()
                self.assertEqual(len(io.writes), writes)

    def test_cleanup_records_separate_fault_without_losing_primary_error(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            args = Namespace(output=output, observe_only=True, checkpoint=Path('checkpoint'),
                             first_motion_trial=False)
            io = Mock(goal_writes=0, startup_checkpoints=[])
            io.close.side_effect = RuntimeError('serial close failed')
            session = Mock(check_image=None, events=[{'event':'fault','reason':'voltage'}],
                           phase='FAULT', handoff_write_baseline=0)
            session.gate.owner = 'FAULT'
            original = RuntimeError("Failed to read 'Homing_Offset' on id_=6: Input voltage error")
            errors = finalize_handoff_run(args, io, session,
                                          [{'mode_before':'OBSERVE_ACT','work_ms':1,
                                            'interval_ms':50}], {}, original)
            report = __import__('json').loads((output/'handoff_report.json').read_text())
            self.assertIn('serial close failed', errors[0])
            self.assertEqual(report['original_exception']['message'], str(original))
            self.assertEqual(report['original_exception']['mode_before'], 'OBSERVE_ACT')
            self.assertEqual(report['fault_state'], 'FAULT')
            self.assertEqual(report['post_handoff_goal_write_attempts'], 0)
            self.assertEqual(report['cleanup_errors'], errors)
            io.write.assert_not_called()
            with self.assertRaises(RuntimeError) as caught:
                raise_run_fault(original, errors)
            self.assertIs(caught.exception, original)

    def test_report_write_failure_is_explicit(self):
        import io as textio
        import json
        with TemporaryDirectory() as directory:
            output = Path(directory)
            args = Namespace(output=output, observe_only=True, checkpoint=Path('checkpoint'),
                             first_motion_trial=False)
            device = Mock(goal_writes=0, startup_checkpoints=[])
            original_write = Path.write_text
            def failing_write(path, data, *a, **kw):
                if path.name == 'handoff_report.json':
                    raise OSError('disk full')
                return original_write(path, data, *a, **kw)
            stderr = textio.StringIO()
            original = RuntimeError('primary voltage fault')
            with patch.object(Path, 'write_text', failing_write), patch('sys.stderr', stderr):
                errors = finalize_handoff_run(args, device, None, [], {}, original)
            self.assertIn('REPORT SAVE FAILED', errors[-1])
            self.assertIn('disk full', stderr.getvalue())
            self.assertIn('primary voltage fault', stderr.getvalue())

    def test_observation_only_five_frame_refresh_never_sends(self):
        io, policy = FakeIO(), FakePredictor()
        ctl = HandoffController(io, policy, observe_only=True, observe_frames=11)
        ctl.tick();ctl.tick('x')
        writes = len(io.writes)
        ctl.tick()
        for _ in range(11):ctl.tick()
        self.assertEqual(ctl.phase,'OBSERVED')
        self.assertEqual(len(io.writes),writes)
        rows=[r for r in ctl.events if r['event']=='observe_act']
        self.assertEqual([r['refreshed'] for r in rows],
                         [True,False,False,False,False,True,False,False,False,False,True])
        self.assertEqual(ctl.events[-1]['post_handoff_goal_write_attempts'],0)
        self.assertEqual([frame for frame, _ in ctl.prediction_input_images], [0, 5, 10])
        self.assertEqual(ctl.prediction_input_images[0][1].shape, (480, 640, 3))
        self.assertEqual(rows[0]['camera']['index'], 0)
        self.assertEqual(set(rows[0]['prediction_goal_abs_error']), set(NAMES))
        self.assertEqual(set(rows[0]['present']), set(NAMES))
        self.assertEqual(set(rows[0]['goal']), set(NAMES))
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from PIL import Image
        with TemporaryDirectory() as directory:
            save_prediction_input_images(ctl, Path(directory))
            self.assertEqual(len(list(Path(directory).glob('act_input_*.png'))), 6)
            self.assertEqual(Image.open(Path(directory)/'act_input_000_224x168.png').size,
                             (224, 168))

    def test_observation_only_40_frames_no_goal_writes(self):
        io, policy = FakeIO(), FakePredictor()
        ctl = HandoffController(io, policy, observe_only=True, observe_frames=40)
        ctl.tick(); ctl.tick('x')
        writes = len(io.writes)
        ctl.tick()  # warm-up, both writers blocked
        for _ in range(40):
            ctl.tick()
        self.assertEqual(ctl.phase, 'OBSERVED')
        self.assertEqual(len(io.writes), writes)
        self.assertEqual(ctl.observed_frames, 40)
        self.assertEqual(sum(r['refreshed'] for r in ctl.events if r['event'] == 'observe_act'), 8)

    def test_slow_observation_inference_rechecks_fresh_state_without_writes(self):
        import time
        io, policy, ctl = self.setup_controller(observe_only=True)
        ctl.tick()  # warm-up CHECK
        writes = len(io.writes)
        original = policy.predict
        def delayed(*args):
            time.sleep(.11)
            return original(*args)
        policy.predict = delayed
        ctl.tick()
        self.assertEqual(ctl.phase, 'OBSERVED')
        self.assertEqual(len(io.writes), writes)
        row = next(r for r in ctl.events if r['event'] == 'observe_act')
        self.assertGreater(row['input_age_after_inference_ms'], 100)
        self.assertLess(row['post_inference_sample_age_ms'], 100)

    def test_post_inference_communication_failure_stops_without_writes(self):
        io, policy, ctl = self.setup_controller(observe_only=True)
        ctl.tick()  # warm-up CHECK
        writes = len(io.writes)
        def fail_after_prediction(*args):
            io.fail_read = True
            return np.full(6, 80.0), True
        policy.predict = fail_after_prediction
        with self.assertRaises(ConnectionError):
            ctl.tick()
        self.assertEqual(ctl.phase, 'FAULT')
        self.assertEqual(len(io.writes), writes)

    def test_five_action_chunks_and_first_goal_slew(self):
        io, policy, ctl = self.setup_controller()
        writes = len(io.writes)
        ctl.tick()  # readbacks and live inference; still no ACT write
        self.assertEqual(ctl.phase, 'ACT')
        self.assertEqual(len(io.writes), writes)
        self.assertEqual(io.requests, 1)  # Leader request stopped
        for index in range(11):
            previous = io.goal.copy()
            ctl.tick()
            self.assertTrue(all(abs(io.goal[n]-previous[n]) <= 1.5 for n in NAMES))
        rows = [row for row in ctl.events if row['event'] == 'act_sent']
        self.assertEqual([row['chunk_offset'] for row in rows], [0,1,2,3,4,0,1,2,3,4,0])
        self.assertEqual([row['refreshed'] for row in rows],
                         [True,False,False,False,False,True,False,False,False,False,True])
        self.assertEqual(len(io.writes), writes+11)
        self.assertEqual(io.configure_calls, 0)
        self.assertEqual(io.torque_calls, 0)

    def first_motion_controller(self):
        io, policy = FakeIO(), FakePredictor()
        policy.predict = lambda state, image: (np.asarray([22.0] * 5 + [50.0]), True)
        ctl = HandoffController(io, policy, observe_only=False, first_motion_trial=True)
        ctl.tick(); ctl.tick('x'); ctl.tick()  # CHECK warm-up; no ACT Goal
        self.assertEqual(ctl.phase, 'ACT')
        return io, policy, ctl

    def test_first_motion_rejects_large_prediction_before_write(self):
        io, policy = FakeIO(), FakePredictor()  # 80 deg is too far
        ctl = HandoffController(io, policy, observe_only=False, first_motion_trial=True)
        ctl.tick(); ctl.tick('x')
        writes = len(io.writes)
        with self.assertRaisesRegex(HandoffFault, 'prediction too far'):
            ctl.tick()
        self.assertEqual(len(io.writes), writes)

    def test_first_motion_speed_displacement_and_gripper_hold(self):
        io, policy, ctl = self.first_motion_controller()
        baseline = io.goal.copy()
        policy.predict = lambda state, image: (np.asarray([22.0] * 5 + [0.0]), True)
        writes = len(io.writes)
        for _ in range(6):
            ctl.tick()
        for goal in io.writes[writes:]:
            self.assertTrue(all(abs(goal[n]-baseline[n]) <= 1.0 for n in NAMES[:5]))
            self.assertEqual(goal[NAMES[5]], baseline[NAMES[5]])
        self.assertAlmostEqual(io.goal[NAMES[0]]-baseline[NAMES[0]], 0.9)
        self.assertEqual(len(io.writes), writes+6)
        ctl.tick()
        self.assertEqual(ctl.phase, 'STOPPED')
        self.assertEqual(ctl.gate.owner, 'STOPPED')
        self.assertEqual(ctl.trial_completion['reason'], 'cumulative_displacement_limit')
        self.assertEqual(ctl.trial_completion['goal_write_attempts_since_handoff'], 6)
        self.assertEqual(ctl.trial_completion['final_sent_goal'], io.goal)
        self.assertEqual(ctl.trial_completion['last_recorded_present'], io.position)
        self.assertEqual(len(io.writes), writes+6)
        with self.assertRaises(HandoffFault):
            ctl.gate.send_act([22.0] * 5 + [50.0], io.position, sample_age_s=0)
        self.assertEqual(len(io.writes), writes+6)

    def test_first_motion_prepared_chunk_consumes_actions_zero_through_four(self):
        class SmallQueue(FakePredictor):
            def predict(self, state, image):
                self.calls += 1
                refreshed = self.offset == 0
                self.offset = (self.offset+1) % 5
                return np.asarray([22.0] * 5 + [50.0]), refreshed

        io, policy = FakeIO(), SmallQueue()
        ctl = HandoffController(io, policy, observe_only=False, first_motion_trial=True)
        ctl.tick(); ctl.tick('x'); ctl.tick()  # warm-up and prepare index 0
        self.assertEqual(policy.calls, 2)
        for _ in range(6):
            ctl.tick()
        rows = [row for row in ctl.events if row['event'] == 'act_sent']
        self.assertEqual([row['chunk_offset'] for row in rows], [0, 1, 2, 3, 4, 0])
        self.assertEqual([row['refreshed'] for row in rows], [True, False, False, False, False, True])
        self.assertEqual(policy.calls, 7)  # first Action was prepared in CHECK

    def test_first_motion_one_second_stops_without_new_goal(self):
        io, policy, ctl = self.first_motion_controller()
        writes = len(io.writes)
        ctl.act_started_at -= 1.01
        ctl.tick()
        self.assertEqual(ctl.phase, 'STOPPED')
        self.assertEqual(ctl.gate.owner, 'STOPPED')
        self.assertEqual(ctl.trial_completion['reason'], 'time_limit')
        self.assertEqual(len(io.writes), writes)

    def test_first_motion_total_travel_blocks_oscillation(self):
        io, policy, ctl = self.first_motion_controller()
        writes = len(io.writes)
        policy.predict = lambda state, image: (
            np.asarray(([18.0] if (len(io.writes)-writes) % 2 else [22.0]) * 5 + [50.0]), True)
        for _ in range(6):
            ctl.tick()
        self.assertLess(abs(io.goal[NAMES[0]]-20.0), 1.0)
        self.assertAlmostEqual(ctl.gate.total_displacement[0], .9)
        ctl.tick()
        self.assertEqual(ctl.phase, 'STOPPED')
        self.assertEqual(ctl.trial_completion['reason'], 'cumulative_displacement_limit')
        self.assertEqual(len(io.writes), writes+6)

    def test_first_motion_slow_inference_and_communication_stop(self):
        import time
        for fault in ('slow', 'communication'):
            with self.subTest(fault=fault):
                io, policy, ctl = self.first_motion_controller()
                writes = len(io.writes)
                if fault == 'slow':
                    ctl.tick()  # Consume the precomputed first Action safely.
                    writes = len(io.writes)
                    def slow(state, image):
                        time.sleep(.11)
                        return np.asarray([22.0] * 5 + [50.0]), True
                    policy.predict = slow
                else:
                    io.fail_read = True
                with self.assertRaises((HandoffFault, ConnectionError)):
                    ctl.tick()
                self.assertEqual(len(io.writes), writes)
                self.assertEqual(ctl.phase, 'FAULT')

    def test_first_motion_initial_gap_and_sensor_faults_block_goals(self):
        for fault in ('initial_gap', 'torque', 'camera'):
            with self.subTest(fault=fault):
                io, policy = FakeIO(), FakePredictor()
                policy.predict = lambda state, image: (np.asarray([22.0] * 5 + [50.0]), True)
                ctl = HandoffController(io, policy, observe_only=False, first_motion_trial=True)
                ctl.tick(); ctl.tick('x')
                writes = len(io.writes)
                if fault == 'initial_gap':
                    io.position[NAMES[0]] += 1.6
                elif fault == 'torque':
                    io.torque[NAMES[0]] = 0
                else:
                    io.image = Mock(side_effect=RuntimeError('synthetic stale camera'))
                with self.assertRaises((HandoffFault, RuntimeError)):
                    ctl.tick()
                self.assertEqual(ctl.phase, 'FAULT')
                self.assertEqual(len(io.writes), writes)

    def test_first_motion_rotary_initial_gap_only_changes_start_check(self):
        for gap, allowed in ((1.143, True), (1.5, True), (1.501, False)):
            with self.subTest(gap=gap):
                io, policy = FakeIO(), FakePredictor()
                policy.predict = lambda state, image: (np.asarray([22.0] * 5 + [50.0]), True)
                ctl = HandoffController(io, policy, observe_only=False, first_motion_trial=True)
                ctl.tick(); ctl.tick('x')
                baseline_writes = len(io.writes)
                io.position[NAMES[3]] += gap
                if allowed:
                    ctl.tick()
                    self.assertEqual(ctl.phase, 'ACT')
                    self.assertEqual(len(io.writes), baseline_writes)
                else:
                    with self.assertRaisesRegex(HandoffFault, 'rotary 1.5'):
                        ctl.tick()
                    self.assertEqual(ctl.phase, 'FAULT')
                    self.assertEqual(len(io.writes), baseline_writes)

    def test_first_motion_gripper_initial_gap_is_separate_from_rotary(self):
        for gap, allowed in ((1.0204081632653, True), (1.5, True), (1.6, False)):
            with self.subTest(gap=gap):
                io, policy = FakeIO(), FakePredictor()
                policy.predict = lambda state, image: (np.asarray([22.0] * 5 + [50.0]), True)
                ctl = HandoffController(io, policy, observe_only=False, first_motion_trial=True)
                ctl.tick(); ctl.tick('x')
                baseline_writes = len(io.writes)
                io.position[NAMES[5]] += gap
                if allowed:
                    ctl.tick()
                    self.assertEqual(ctl.phase, 'ACT')
                    self.assertEqual(len(io.writes), baseline_writes)
                    ctl.tick()
                    self.assertEqual(io.writes[-1][NAMES[5]], 50.0)
                else:
                    with self.assertRaisesRegex(HandoffFault, 'gripper 1.5'):
                        ctl.tick()
                    self.assertEqual(ctl.phase, 'FAULT')
                    self.assertEqual(len(io.writes), baseline_writes)

    def test_first_motion_gripper_movement_limit_remains_one_normalized_unit(self):
        io, policy, ctl = self.first_motion_controller()
        writes = len(io.writes)
        io.position[NAMES[5]] += 1.1
        with self.assertRaisesRegex(HandoffFault, 'gripper moved while Goal was held'):
            ctl.tick()
        self.assertEqual(ctl.phase, 'FAULT')
        self.assertEqual(len(io.writes), writes)

    def test_first_motion_rechecks_gap_after_prepared_inference(self):
        io, policy = FakeIO(), FakePredictor()
        def predict(state, image):
            if ctl.predictor.calls == 1:
                io.position[NAMES[5]] += 1.6
            ctl.predictor.calls += 1
            return np.asarray([22.0] * 5 + [50.0]), True
        policy.predict = predict
        ctl = HandoffController(io, policy, observe_only=False, first_motion_trial=True)
        ctl.tick(); ctl.tick('x')
        writes = len(io.writes)
        with self.assertRaisesRegex(HandoffFault, 'gripper 1.5'):
            ctl.tick()
        self.assertEqual(ctl.phase, 'FAULT')
        self.assertEqual(len(io.writes), writes)

    def test_bad_torque_and_status_before_handoff_no_write(self):
        for field, joint, value in (('torque', NAMES[0], 0), ('status', NAMES[2], 1)):
            with self.subTest(field=field):
                io, _, ctl = self.setup_controller()
                getattr(io, field)[joint] = value
                writes = len(io.writes)
                with self.assertRaises(HandoffFault):ctl.tick()
                self.assertEqual(ctl.phase, 'FAULT')
                self.assertEqual(len(io.writes), writes)

    def test_bad_position_and_goal_before_handoff_no_write(self):
        for field, value in (('position', 29.0), ('goal', 22.0)):
            with self.subTest(field=field):
                io, _, ctl = self.setup_controller()
                getattr(io, field)[NAMES[0]] = value
                writes = len(io.writes)
                with self.assertRaises(HandoffFault):ctl.tick()
                self.assertEqual(len(io.writes), writes)

    def test_communication_failure_after_switch_latches(self):
        io, _, ctl = self.setup_controller()
        ctl.tick()
        io.fail_read = True
        writes = len(io.writes)
        with self.assertRaises(ConnectionError):ctl.tick()
        self.assertEqual(ctl.phase, 'FAULT')
        self.assertEqual(len(io.writes), writes)
        io.fail_read = False
        with self.assertRaises(HandoffFault):ctl.tick()
        self.assertEqual(len(io.writes), writes)

    def test_inference_exception_and_cycle_gap_do_not_reinitialize(self):
        for failure in ('predict', 'gap'):
            with self.subTest(failure=failure):
                io, policy, ctl = self.setup_controller()
                writes = len(io.writes)
                if failure == 'predict':
                    def broken(*args):raise RuntimeError('injected model failure')
                    policy.predict = broken
                else:
                    ctl.last_tick -= 0.3
                with self.assertRaises((RuntimeError, HandoffFault)):ctl.tick()
                self.assertEqual(len(io.writes), writes)
                self.assertEqual(io.configure_calls, 0)
                self.assertEqual(io.torque_calls, 0)
                self.assertEqual(ctl.gate.owner, 'FAULT')


if __name__ == '__main__':unittest.main()
