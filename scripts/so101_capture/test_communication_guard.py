"""No hardware: bus-shaped mocks and complete Session fault injection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from communication_guard import GuardedIO, CommunicationFault
from capture_session import Session, SimulationIO, HardwareIO, atomic_json
from test_capture_session import Sink


class CommunicationTests(unittest.TestCase):
    def run_failure(self, register, recovered, recording=False):
        io=SimulationIO();s=Session(io,Sink(),countdown=0)
        for i in range(12):s.tick(i*.05)
        if recording:s.tick(.6,'r')
        fn='state' if register=='Present_Position' else 'torque_ok'
        value=io.state() if fn=='state' else True
        failing=Mock(side_effect=[ConnectionError('no status packet'),value if recovered else ConnectionError('still no packet')])
        setattr(io,fn,failing)
        sends=len(io.sent)
        with self.assertRaises(CommunicationFault):s.tick(.65)
        self.assertTrue(s.io.latched)
        self.assertEqual(failing.call_count,2)
        self.assertEqual(len(io.sent),sends)
        # Even explicitly requested keys cannot clear the latch.
        for key in ('r','t'):
            with self.assertRaises(ConnectionError):s.tick(.7,key)
        with self.assertRaises(CommunicationFault):s.io.send(value)
        errors=s.stop(.75)
        self.assertTrue(errors)
        self.assertEqual(len(io.sent),sends)
        self.assertEqual(failing.call_count,2)
        self.assertEqual(s.mode,'STOPPED')
        d=s.io.report();f=d['failures'][0]
        self.assertEqual(f['failure']['register'],register)
        self.assertEqual(f['failure']['servo_ids'],[1,2,3,4,5,6])
        self.assertEqual(f['failure']['mode'],'RECORDING' if recording else 'SAFE_RESET')
        self.assertIsNotNone(f['failure']['last_success_monotonic'])
        self.assertEqual(f['diagnostic_retry']['ok'],recovered)
        self.assertTrue(any(not x['ok'] for x in d['cycle_history']))
        self.assertGreaterEqual(f['failure']['duration_s'],0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'communication_diagnostics.json';atomic_json(p,d)
            self.assertTrue(json.loads(p.read_text())['latched'])
        if recording:self.assertEqual(s.outcomes[-1]['end_reason'],'interrupted')
        return s

    def test_position_transient(self):self.run_failure('Present_Position',True)
    def test_position_persistent(self):self.run_failure('Present_Position',False)
    def test_torque_transient(self):self.run_failure('Torque_Enable',True)
    def test_torque_persistent(self):self.run_failure('Torque_Enable',False)
    def test_recording_preservation(self):self.run_failure('Present_Position',True,True)

    def test_teleop_reset_leader_transient_gets_one_read_only_retry(self):
        io=SimulationIO();g=GuardedIO(io)
        g.context={'mode':'TELEOP_RESET','cycle':7}
        expected=io.request()
        io.request=Mock(side_effect=[ConnectionError('temporary timeout'),expected])
        self.assertEqual(g.request(),expected)
        self.assertFalse(g.latched)
        self.assertEqual(io.request.call_count,2)
        self.assertEqual(io.sent,[])
        report=g.report()
        self.assertEqual(len(report['transient_recoveries']),1)
        self.assertFalse(report['transient_recoveries'][0]['first_failure']['ok'])
        self.assertTrue(report['transient_recoveries'][0]['recovery']['ok'])

    def test_teleop_reset_leader_persistent_fault_latches_before_diagnostic(self):
        io=SimulationIO();g=GuardedIO(io)
        g.context={'mode':'TELEOP_RESET','cycle':8}
        io.request=Mock(side_effect=ConnectionError('no status packet'))
        with self.assertRaises(CommunicationFault):g.request()
        self.assertTrue(g.latched)
        # Initial read, bounded control retry, then discarded diagnostic read.
        self.assertEqual(io.request.call_count,3)
        self.assertEqual(io.sent,[])

    def test_recording_leader_transient_still_latches_on_first_failure(self):
        io=SimulationIO();g=GuardedIO(io)
        g.context={'mode':'RECORDING','cycle':9}
        expected=io.request()
        io.request=Mock(side_effect=[ConnectionError('temporary timeout'),expected])
        with self.assertRaises(CommunicationFault):g.request()
        self.assertTrue(g.latched)
        # Second call is diagnostic only; its recovered value is never used.
        self.assertEqual(io.request.call_count,2)
        self.assertEqual(io.sent,[])

    def test_latched_before_diagnostic_attempt(self):
        io=SimulationIO();g=GuardedIO(io);calls=[]
        def fail_then_check():
            if not calls:
                calls.append(1);raise ConnectionError('first failure')
            self.assertTrue(g.latched)
            with self.assertRaises(CommunicationFault):g.send(io.positions)
            return io.positions.copy()
        io.state=fail_then_check
        with self.assertRaises(CommunicationFault):g.state()
        self.assertEqual(io.sent,[])

    def test_bounded_history_and_success_timings(self):
        g=GuardedIO(SimulationIO())
        for _ in range(300):g.state();g.torque_ok()
        self.assertEqual(len(g.history),256)
        self.assertFalse(g.latched)
        for name in ('Present_Position','Torque_Enable'):
            self.assertEqual(g.summary['follower:'+name]['attempts'],300)
            self.assertEqual(g.summary['follower:'+name]['failures'],0)

    def test_actual_adapter_retry_and_timeout_settings_without_ports(self):
        robot,leader,camera=Mock(),Mock(),Mock()
        robot.bus.port_handler.getBaudRate.return_value=1_000_000
        robot.bus.port_handler.ser.timeout=0
        robot.bus.port_handler.packet_timeout=50.51
        robot.config.num_read_retries=2
        robot.bus.sync_read.side_effect=[{'a':1},{'a':1}]
        io=HardwareIO(robot,leader,camera)
        io.state();io.torque_ok()
        self.assertEqual(robot.bus.sync_read.call_args_list[0].kwargs,{'num_retry':0})
        self.assertEqual(robot.bus.sync_read.call_args_list[1].kwargs,{'normalize':False,'num_retry':0})
        m=io.communication_settings('Present_Position','follower')
        self.assertAlmostEqual(m['calculated_packet_timeout_ms'],50.51)
        self.assertEqual(m['effective_num_read_retries'],0)

    def test_read_during_cleanup_failure_never_sends(self):
        io=SimulationIO();s=Session(io,Sink());s.tick(0);count=len(io.sent)
        io.state=Mock(side_effect=ConnectionError('cleanup read failed'))
        self.assertTrue(s.stop(.1))
        self.assertTrue(s.io.latched)
        self.assertEqual(len(io.sent),count)

    def test_missing_timeout_metadata_does_not_hide_read_fault(self):
        io=SimulationIO()
        io.communication_settings=Mock(side_effect=AttributeError('port unavailable'))
        io.state=Mock(side_effect=ConnectionError('no packet'))
        g=GuardedIO(io)
        with self.assertRaises(CommunicationFault):g.state()
        self.assertTrue(g.latched)
        self.assertEqual(io.state.call_count,2)
        self.assertIn('unavailable',g.failures[0]['failure']['settings'])

    def test_send_failure_latches_without_retry(self):
        io=SimulationIO();g=GuardedIO(io)
        io.send=Mock(side_effect=ConnectionError('write failed'))
        with self.assertRaises(CommunicationFault):g.send(io.positions)
        with self.assertRaises(CommunicationFault):g.send(io.positions)
        io.send.assert_called_once()


if __name__=='__main__':unittest.main()
