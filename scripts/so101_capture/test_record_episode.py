"""Safety checks without opening camera/serial devices."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from record_episode import deny_calibration, start_hardware, synchronize_positions, validate_destination, close_hardware, NAMES


class SafetyTests(unittest.TestCase):
    def test_cleanup_unopened_follower_still_closes_leader(self):
        follower, leader = Mock(), Mock()
        follower.bus.is_connected = False
        follower.bus.port_handler.ser = None
        leader.bus.is_connected = True
        close_hardware(follower, leader, False)
        follower.bus.port_handler.closePort.assert_not_called()
        leader.bus.disconnect.assert_called_once_with(disable_torque=False)
        leader.bus.port_handler.closePort.assert_called_once()

    def devices(self):
        follower, leader = Mock(), Mock()
        for d in (follower, leader):
            d.is_calibrated = True
            d.bus.sync_read.return_value = {n.removesuffix('.pos'): 0 for n in NAMES}
        return follower, leader

    def test_mismatch_never_configures_or_writes(self):
        for which in (0, 1):
            devices = self.devices()
            devices[which].is_calibrated = False
            with self.assertRaisesRegex(RuntimeError, 'calibration mismatch'):
                start_hardware(*devices)
            for device in devices:
                device.configure.assert_not_called()
                device.bus.sync_write.assert_not_called()
                device.bus.write_calibration.assert_not_called()
                device.calibrate.assert_not_called()

    def test_pose_mismatch_never_configures(self):
        follower, leader = self.devices()
        leader.bus.sync_read.return_value['shoulder_pan'] = 130
        with self.assertRaisesRegex(RuntimeError, 'pose mismatch'):
            start_hardware(follower, leader)
        follower.configure.assert_not_called()
        leader.configure.assert_not_called()
        follower.bus.sync_write.assert_not_called()

    def test_explicit_calibration_guard(self):
        with self.assertRaises(RuntimeError):
            deny_calibration()

    def test_existing_destination_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError):
                validate_destination(Path(root))

    def test_alignment_ramps_and_keeps_connected(self):
        follower, leader = self.devices()
        state = {'shoulder_pan': 0.}
        target = {'shoulder_pan': 20.}
        now = [0.]
        commands = []
        follower.bus.sync_read.side_effect = lambda _: state.copy()
        leader.bus.sync_read.return_value = target
        def write(_, command):
            commands.append(command['shoulder_pan'])
            state.update(command)
        follower.bus.sync_write.side_effect = write
        synchronize_positions(follower, leader, target, clock=lambda:now[0], sleep=lambda dt:now.__setitem__(0,now[0]+dt))
        self.assertAlmostEqual(state['shoulder_pan'],20)
        self.assertTrue(all(abs(b-a)<=.1500001 for a,b in zip([0.]+commands, commands)))
        follower.bus.disconnect.assert_not_called()

    def test_alignment_stops_if_leader_moves(self):
        follower, leader = self.devices()
        follower.bus.sync_read.return_value = {'x':0.}
        leader.bus.sync_read.return_value = {'x':10.}
        with self.assertRaisesRegex(RuntimeError,'Leader moved'):
            synchronize_positions(follower,leader,{'x':0.})
        follower.bus.sync_write.assert_not_called()

    def test_alignment_stall_stops(self):
        follower, leader = self.devices()
        follower.bus.sync_read.return_value = {'x':0.}
        leader.bus.sync_read.return_value = {'x':6.}
        now=[0.]
        with self.assertRaisesRegex(RuntimeError,'stalled'):
            synchronize_positions(follower,leader,{'x':6.},clock=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt))

    def test_alignment_60_seconds_insufficient_at_slow_cycle_120_succeeds(self):
        for timeout, expected in ((60,'timeout'),(120,'success')):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory:
                follower, leader = self.devices()
                state={'shoulder_pan.pos':0.,'gripper.pos':47.}
                target={'shoulder_pan.pos':100.,'gripper.pos':47.}
                now=[0.];commands=[]
                follower.bus.sync_read.side_effect=lambda _:state.copy()
                leader.bus.sync_read.return_value=target.copy()
                def write(_,goal):
                    commands.append(goal.copy());state.update(goal)
                follower.bus.sync_write.side_effect=write
                path=Path(directory)/'alignment_diagnostics.json'
                kw=dict(timeout=timeout,diagnostics_path=path,clock=lambda:now[0],
                        sleep=lambda dt:now.__setitem__(0,now[0]+max(.1,dt)))
                if expected=='timeout':
                    with self.assertRaisesRegex(RuntimeError,'timed out'):
                        synchronize_positions(follower,leader,target,**kw)
                else:synchronize_positions(follower,leader,target,**kw)
                data=json.loads(path.read_text())
                stages=[r['stage'] for r in data['snapshots']]
                self.assertEqual(stages[0],'start')
                self.assertEqual(stages[-1],'stopped' if expected=='timeout' else 'complete')
                self.assertIn('periodic',stages)
                first=data['snapshots'][0]['joints']['shoulder_pan.pos']
                final=data['snapshots'][-1]['joints']['shoulder_pan.pos']
                self.assertEqual(first['target_measured_error'],100.)
                self.assertLess(final['target_measured_error'],first['target_measured_error'])
                self.assertTrue(all(abs(b['shoulder_pan.pos']-a['shoulder_pan.pos'])<=.1500001
                                    for a,b in zip([state | {'shoulder_pan.pos':0.}]+commands,commands)))
                self.assertEqual(data['snapshots'][-1]['worst_joint'],'shoulder_pan.pos')

    def test_logged_wrist_residual_is_configurable_only_for_alignment(self):
        target={'wrist_flex':78.1978021978022}
        measured={'wrist_flex':76.17582417582418}
        for tolerance, expected in ((2.0,'timeout'),(2.5,'complete')):
            with self.subTest(tolerance=tolerance), tempfile.TemporaryDirectory() as directory:
                follower,leader=self.devices()
                follower.bus.sync_read.side_effect=lambda _:measured.copy()
                leader.bus.sync_read.return_value=target.copy()
                now=[0.]
                path=Path(directory)/'alignment.json'
                kwargs=dict(timeout=3,tolerance=tolerance,diagnostics_path=path,
                            clock=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+max(dt,.05)))
                if expected=='timeout':
                    with self.assertRaisesRegex(RuntimeError,'timed out'):
                        synchronize_positions(follower,leader,target,**kwargs)
                else:synchronize_positions(follower,leader,target,**kwargs)
                report=json.loads(path.read_text())
                self.assertEqual(report['tolerance'],tolerance)
                self.assertEqual(report['snapshots'][-1]['stage'],
                                 'stopped' if expected=='timeout' else 'complete')
                self.assertAlmostEqual(report['snapshots'][-1]['joints']['wrist_flex']['target_measured_error'],
                                       2.021978021978029)
                self.assertTrue(all(abs(call.args[1]['wrist_flex']-measured['wrist_flex'])<8
                                    for call in follower.bus.sync_write.call_args_list))

    def test_residual_below_three_allows_countdown_with_existing_slew(self):
        from capture_session import Session,SimulationIO
        from test_capture_session import Sink
        io=SimulationIO();s=Session(io,Sink(),countdown=0,seconds=1)
        io.request=lambda:{n:(2.021978021978029 if n=='wrist_flex.pos' else 0.) for n in NAMES}
        for i in range(12):s.tick(i*.05)
        s.tick(.6,'r')
        self.assertEqual(s.mode,'RECORDING')
        self.assertEqual(s.key_events[-1]['decision'],'ACCEPTED: COUNTDOWN')
        self.assertAlmostEqual(s.last_sent['wrist_flex.pos'],1.5)
        s.tick(.65)
        self.assertAlmostEqual(s.last_sent['wrist_flex.pos'],2.021978021978029)
        s.tick(.7)  # Existing previous-goal tracking error remains below 8.
        self.assertEqual(s.mode,'RECORDING')


if __name__ == '__main__':
    unittest.main()
