import contextlib
import io as textio
import json
from pathlib import Path
import tempfile
import unittest
from capture_session import Session, SimulationIO, atomic_json
from test_capture_session import Sink
from record_episode import NAMES
from gripper_contact_model import GripperContactModel, ContactStop


class DiagnosticTests(unittest.TestCase):
    def session(self):
        io=SimulationIO()
        return Session(io,Sink(),countdown=10),io

    def test_ten_seconds_display_and_no_follow(self):
        s,io=self.session()
        out=textio.StringIO()
        with contextlib.redirect_stdout(out):
            for i in range(12):s.tick(i*.05)
            s.tick(.6,'r')
            sent=len(io.sent)
            for i in range(1,200):s.tick(.6+i*.05)
            self.assertEqual(len(io.sent),sent)
            self.assertEqual(s.mode,'COUNTDOWN')
            s.tick(10.6)
        for n in range(11):self.assertIn(f'COUNTDOWN: {n} seconds',out.getvalue())
        self.assertIn('RECORDING STARTED NOW',out.getvalue())
        self.assertEqual(s.mode,'RECORDING')

    def test_reference_movement_even_when_follower_difference_small(self):
        s,io=self.session()
        io.request=lambda:dict.fromkeys(NAMES,2.)
        for i in range(12):s.tick(i*.05)
        s.tick(.6,'r')
        # -2 remains within 3 of Follower but moved 4 from countdown origin.
        io.request=lambda:{n:(-2. if n!=NAMES[-1] else 2.) for n in NAMES}
        count=len(io.sent)
        s.tick(.65)
        self.assertEqual(s.mode,'SAFE_RESET')
        self.assertEqual(len(io.sent),count)

    def test_actual_tail_and_synthetic_unsaved_cycle(self):
        f=json.loads(Path(__file__).with_name('fixtures').joinpath('muhi_pilot_tail.json').read_text())
        s,io=self.session()
        s.mode='RECORDING';s.started=0;s.hold=io.state()
        for prev,row in zip(f['rows'],f['rows'][1:]):
            s.last_sent=dict(zip(NAMES,prev['action']))
            io.positions=dict(zip(NAMES,row['observation.state']))
            io.request=lambda row=row:dict(zip(NAMES,row['action_requested']))
            s.tick(row['timestamp'])
        # Conservative synthetic extrapolation: same measured position next tick.
        # This is NOT the missing real failing measurement.
        s.last_sent=dict(zip(NAMES,f['rows'][-1]['action']))
        io.positions=dict(zip(NAMES,f['rows'][-1]['observation.state']))
        count=len(io.sent)
        with self.assertRaisesRegex(RuntimeError,'gripper.pos'):
            s.tick(27.05)
        self.assertEqual(len(io.sent),count)
        d=s.diagnostics[-1]
        self.assertEqual(d['offending_joints'],['gripper.pos'])
        self.assertAlmostEqual(d['joints']['gripper.pos']['abs_error'],9.09764576,places=5)
        self.assertIsNone(d['current_leader_requested'])
        self.assertEqual(d['joints'][NAMES[0]]['unit'],'degrees')
        before=len(s.events)
        oldgrip=s.last_sent['gripper.pos']
        s.stop(27.06)
        self.assertEqual([x['mode'] for x in s.events[before:]],['STOPPING','STOPPED'])
        self.assertEqual(io.sent[-1]['gripper.pos'],oldgrip)
        self.assertEqual(s.outcomes[-1]['end_reason'],'interrupted')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fault_diagnostics.json';atomic_json(p,s.diagnostics)
            self.assertEqual(json.loads(p.read_text())[0]['mode'],'RECORDING')

    def test_body_threshold_unchanged(self):
        s,io=self.session();s.mode='TELEOP_RESET';s.hold=io.state();s.last_sent=io.state()
        io.positions[NAMES[0]]=8.01
        with self.assertRaisesRegex(RuntimeError,'shoulder_pan.pos'):s.tick(0)
        self.assertEqual(io.sent,[])


class ContactModelTests(unittest.TestCase):
    def test_saved_trace_contact_latches_before_hard_error(self):
        f=json.loads(Path(__file__).with_name('fixtures').joinpath('muhi_pilot_tail.json').read_text())
        m=GripperContactModel();goal=f['rows'][0]['action'][-1];latched=None
        for i,row in enumerate(f['rows'][1:]):
            try:
                goal=m.step(now=i*.05,measured=row['observation.state'][-1],previous_goal=goal,requested=row['action_requested'][-1])
            except ContactStop as exc:
                self.assertTrue('timeout' in str(exc) or 'hard error' in str(exc));break
            if m.latched is not None:
                if latched is None:latched=goal
                self.assertEqual(goal,latched)
        self.assertIsNotNone(latched)

    def test_contact_plant_freezes_then_stops_without_opening(self):
        m=GripperContactModel();measured=35.;goal=35.;latched=None
        for i in range(30):
            # Ideal response with a bottle preventing closure below 25.
            measured=max(25.,goal)
            try:
                goal=m.step(now=i*.05,measured=measured,previous_goal=goal,requested=10.)
            except ContactStop as exc:
                self.assertIn('timeout',str(exc))
                self.assertIsNotNone(latched)
                return
            if m.latched is not None:
                latched=m.latched
                self.assertEqual(goal,latched)
        self.fail('Contact did not time out')

    def test_open_requires_two_samples_and_never_recloses(self):
        m=GripperContactModel()
        self.assertEqual(m.step(now=0,measured=25,previous_goal=22,requested=10),22)
        self.assertEqual(m.step(now=.05,measured=25,previous_goal=22,requested=30),22)
        self.assertEqual(m.step(now=.1,measured=25,previous_goal=22,requested=30),23.5)
        self.assertEqual(m.step(now=.15,measured=25,previous_goal=23.5,requested=10),23.5)

    def test_timeout_latency_and_bad_data_stop(self):
        m=GripperContactModel();m.step(now=0,measured=25,previous_goal=22,requested=10)
        for i in range(1,10):m.step(now=i*.05,measured=25,previous_goal=22,requested=10)
        with self.assertRaisesRegex(ContactStop,'timeout'):m.step(now=.5,measured=25,previous_goal=22,requested=10)
        for kw in ({'communication_ok':False},{'measured':float('nan')},{'measured':40}):
            args=dict(now=0,measured=25,previous_goal=22,requested=10);args.update(kw)
            with self.assertRaises(ContactStop):GripperContactModel().step(**args)
        m=GripperContactModel();m.step(now=0,measured=25,previous_goal=22,requested=10)
        with self.assertRaisesRegex(ContactStop,'Stale'):m.step(now=.1,measured=25,previous_goal=22,requested=10)


if __name__=='__main__':unittest.main()
