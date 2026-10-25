import threading
import time
import unittest
from unittest.mock import Mock
from gripper_display import GripperDisplay,render
from capture_session import Session,SimulationIO
from test_capture_session import Sink
from record_episode import NAMES


class DisplayTests(unittest.TestCase):
    def sample(self):
        return dict(mode='RECORDING',cycle=1,measured=24.2,previous_sent=16.7,error=7.5,
                    requested=9.,sent=15.2,measured_at=10.,measured_utc=0.,event='normal')

    def test_fields_units_and_stale(self):
        line=render(self.sample(),10.1)
        for value in ('FRESH','normalized_0_100','measured=24.200','prev_goal=16.700',
                      'error_prev=7.500','leader_now=9.000','sent_now=15.200','age_ms=100','NOT a grip-force'):
            self.assertIn(value,line)
        self.assertIn('STALE',render(self.sample(),11.))
        s=self.sample();s['communication_fault']=True
        self.assertIn('STALE',render(s,10.01))

    def test_slow_writer_full_queue_does_not_block_producer(self):
        entered,release=threading.Event(),threading.Event()
        def slow(_):entered.set();release.wait(2)
        d=GripperDisplay(writer=slow,hz=5)
        try:
            d.submit(self.sample());self.assertTrue(entered.wait(1))
            start=time.perf_counter()
            for i in range(1000):s=self.sample();s['cycle']=i;d.submit(s)
            self.assertLess(time.perf_counter()-start,.1)
            self.assertGreater(d.dropped,0)
            self.assertEqual(d.queue.qsize(),1)
            self.assertEqual(d.last_sample['cycle'],999)
        finally:release.set();d.close()

    def test_close_is_bounded_when_output_stalls(self):
        entered,release=threading.Event(),threading.Event()
        def slow(_):entered.set();release.wait(2)
        d=GripperDisplay(writer=slow)
        d.submit(self.sample());self.assertTrue(entered.wait(1))
        try:
            start=time.perf_counter();r=d.close()
            self.assertLess(time.perf_counter()-start,.5)
            self.assertTrue(r['worker_still_running'])
        finally:release.set();d.thread.join(1)

    def test_final_fault_sample_drained(self):
        lines=[];d=GripperDisplay(writer=lines.append)
        s=self.sample();s['event']='Tracking error';d.submit(s);d.close()
        self.assertTrue(any('Tracking error' in x for x in lines))
        self.assertTrue(any('STREAM_ENDED' in x and 'STALE' in x for x in lines))

    def test_writer_error_is_isolated(self):
        def broken(_):raise OSError('output unavailable')
        d=GripperDisplay(writer=broken);d.submit(self.sample());d.thread.join(1)
        self.assertIn('output unavailable',d.error)
        d.submit(self.sample());d.close()

    def test_retained_sample_becomes_stale_without_new_data(self):
        now=[10.];lines=[];done=threading.Event()
        def writer(line):
            lines.append(line)
            now[0]=11.
            if 'STALE' in line:done.set()
        d=GripperDisplay(writer=writer,clock=lambda:now[0],hz=5)
        try:
            d.submit(self.sample())
            self.assertTrue(done.wait(1))
            self.assertIn('FRESH',lines[0])
            self.assertTrue(any('STALE' in line for line in lines))
        finally:d.close()

    def test_display_rate_configuration(self):
        for hz in (1,6,float('nan')):
            with self.assertRaises(ValueError):GripperDisplay(hz=hz,writer=lambda _:None)

    def make(self, display=None):
        io=SimulationIO()
        for name in ('state','torque_ok','request','send'):
            setattr(io,name,Mock(wraps=getattr(io,name)))
        s=Session(io,Sink(),countdown=0,seconds=1,min_seconds=.1,display=display)
        for i in range(12):s.tick(i*.05)
        s.tick(.6,'r')
        return s,io

    def test_no_extra_reads_or_action_changes(self):
        d=Mock();a,ia=self.make();b,ib=self.make(d)
        for name in ('state','torque_ok','request','send'):
            self.assertEqual(getattr(ia,name).call_args_list,getattr(ib,name).call_args_list)
        sample=d.submit.call_args.args[0]
        self.assertEqual(sample['sent'],0.)
        self.assertEqual(sample['requested'],0.)
        self.assertEqual(sample['error'],0.)
        self.assertEqual(a.frames,b.frames)

    def test_tracking_failure_snapshot_before_cleanup(self):
        d=Mock();s,io=self.make(d)
        io.positions[NAMES[-1]]=8.5;count=io.send.call_count
        with self.assertRaisesRegex(RuntimeError,'gripper.pos'):s.tick(.65)
        x=d.submit.call_args.args[0]
        self.assertEqual(x['error'],8.5)
        self.assertEqual(x['previous_sent'],0.)
        self.assertIsNone(x['sent']);self.assertIsNone(x['requested'])
        self.assertIn('Tracking error',x['event'])
        self.assertEqual(io.send.call_count,count)
        s.stop(.7)
        self.assertEqual(s.outcomes[-1]['end_reason'],'interrupted')

    def test_communication_failure_keeps_old_timestamp_and_no_send(self):
        d=Mock();s,io=self.make(d)
        previous=d.submit.call_args.args[0]['measured_at'];count=io.send.call_count
        io.state.side_effect=ConnectionError('no packet')
        with self.assertRaises(ConnectionError):s.tick(.65)
        x=d.submit.call_args.args[0]
        self.assertEqual(x['measured_at'],previous)
        self.assertTrue(x['communication_fault'])
        self.assertIsNone(x['error'])
        self.assertIn('STALE',render(x,time.perf_counter()))
        s.stop(.7);self.assertEqual(io.send.call_count,count)

    def test_submit_failure_does_not_interrupt_recording(self):
        d=Mock();d.submit.side_effect=OSError('display unavailable')
        s,io=self.make(d)
        self.assertEqual(s.mode,'RECORDING')
        self.assertEqual(s.frames,1)
        self.assertIsNone(s.display)
        self.assertIn('display unavailable',s.display_error)


if __name__=='__main__':unittest.main()
