"""Toy plants compare hypotheses, not real force/grasp safety."""
import json
from pathlib import Path
import unittest
from gripper_contact_model import GripperContactModel,ContactStop
from gripper_lag_model import LagAwareContactModel


def scenario(cls, kind):
    m=cls();position=50.;goal=50.;rows=[];first=None
    for i in range(80):
        delta=.3*(goal-position)
        if kind=='slow_free':delta=max(-.01,min(.01,delta))
        position+=delta
        if kind=='bottle':position=max(25.,position)
        if kind=='jam':position=max(40.,position)
        observed=round(position,1) if kind=='slow_free' else position
        try:new=m.step(now=i*.05,measured=observed,previous_goal=goal,requested=10.)
        except ContactStop as exc:
            return dict(kind=kind,contact=first,stop=str(exc),stop_s=i*.05,rows=rows)
        rows.append(dict(t=i*.05,measured=position,goal=new))
        goal=new
        if m.latched is not None and first is None:first=i*.05
    return dict(kind=kind,contact=first,stop=None,rows=rows)


class OptionTests(unittest.TestCase):
    def test_normal_lag_false_positive_improves(self):
        self.assertIsNotNone(scenario(GripperContactModel,'normal')['contact'])
        self.assertIsNone(scenario(LagAwareContactModel,'normal')['contact'])
        self.assertIsNone(scenario(LagAwareContactModel,'normal')['stop'])

    def test_bottle_and_jam_both_treated_as_unverified(self):
        for kind in ('bottle','jam'):
            r=scenario(LagAwareContactModel,kind)
            self.assertIsNotNone(r['contact'])
            self.assertIn('timeout',r['stop'])

    def test_slow_free_false_positive_is_not_hidden(self):
        self.assertIsNotNone(scenario(LagAwareContactModel,'slow_free')['contact'])

    def test_faults_latched_even_after_recovery(self):
        for kw in ({'communication_ok':False},{'measured':40.},{'now':.2}):
            m=LagAwareContactModel();m.step(now=0,measured=25,previous_goal=25,requested=25)
            args=dict(now=.05,measured=25,previous_goal=25,requested=25);args.update(kw)
            with self.assertRaises(ContactStop):m.step(**args)
            with self.assertRaises(ContactStop):m.step(now=.25,measured=25,previous_goal=25,requested=25)

    def test_explicit_open_never_recloses(self):
        m=LagAwareContactModel()
        m.step(now=0,measured=25,previous_goal=22,requested=10)
        m.step(now=.05,measured=25,previous_goal=22,requested=10)
        self.assertEqual(m.step(now=.1,measured=25,previous_goal=22,requested=10),22)
        self.assertEqual(m.step(now=.15,measured=25,previous_goal=22,requested=30),22)
        self.assertAlmostEqual(m.step(now=.2,measured=25,previous_goal=22,requested=30),23.5)
        self.assertAlmostEqual(m.step(now=.25,measured=25,previous_goal=23.5,requested=10),23.5)

    def test_no_automatic_open_by_position_limit(self):
        m=LagAwareContactModel()
        self.assertEqual(m.step(now=0,measured=27,previous_goal=22,requested=10),22)

    def test_closing_lead_limited_without_auto_open(self):
        m=LagAwareContactModel()
        goal=m.step(now=0,measured=50,previous_goal=50,requested=10)
        self.assertGreaterEqual(goal,47.)  # no more than 3 ahead of measurement
        m=LagAwareContactModel()
        self.assertEqual(m.step(now=0,measured=50,previous_goal=45,requested=10),45)

    def test_hard_error_still_stops_and_latches(self):
        m=LagAwareContactModel()
        with self.assertRaisesRegex(ContactStop,'Hard error limit 8'):
            m.step(now=0,measured=50,previous_goal=41,requested=10)
        with self.assertRaisesRegex(ContactStop,'Fault latched'):
            m.step(now=.05,measured=41,previous_goal=41,requested=41)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);a=p.parse_args()
    if a.report:
        if a.report.exists():raise FileExistsError(a.report)
        a.report.write_text(json.dumps({cls.__name__:[scenario(cls,k) for k in ('normal','bottle','jam','slow_free')]
                                       for cls in (GripperContactModel,LagAwareContactModel)},indent=2))
    else:unittest.main(argv=['test_gripper_options'])
