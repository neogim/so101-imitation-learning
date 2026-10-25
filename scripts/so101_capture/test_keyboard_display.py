"""Terminal input and display integration without robot hardware."""
import os
import pty
import sys
import time
import unittest
from unittest.mock import patch

from capture_session import Keyboard, Session, SimulationIO
from gripper_display import GripperDisplay
from test_capture_session import Sink
from record_episode import NAMES


class KeyboardDisplayTests(unittest.TestCase):
    def test_pty_r_survives_safe_transition(self):
        for enabled in (False, True):
            with self.subTest(display=enabled):
                master, slave = pty.openpty()
                stream = os.fdopen(os.dup(slave), 'r', buffering=1)
                display=GripperDisplay(writer=lambda _:None) if enabled else None
                try:
                    with patch.object(sys, 'stdin', stream):
                        with Keyboard() as keyboard:
                            session = Session(SimulationIO(), Sink(), countdown=1,display=display)
                            session.tick(0)  # Initial hold enters SAFE_RESET.
                            os.write(master, b'r')
                            key = keyboard.poll()
                            self.assertEqual(key, 'r')
                            session.tick(.05, key)
                            self.assertEqual(session.key_events[-1]['decision'],
                                             'REFUSED: hold has not been monitored for 0.5 seconds')
                            os.write(master, b'r')
                            for i in range(2,11):session.tick(i*.05)
                            session.tick(.55, keyboard.poll())
                            self.assertEqual(session.mode, 'COUNTDOWN')
                            self.assertEqual(session.key_events[-1]['decision'], 'ACCEPTED: COUNTDOWN')
                finally:
                    if display:display.close()
                    stream.close(); os.close(slave); os.close(master)

    def test_keys_in_all_modes_with_display_on_off(self):
        for enabled in (False, True):
            with self.subTest(display=enabled):
                display = GripperDisplay(writer=lambda _: None) if enabled else None
                try:
                    s=Session(SimulationIO(),Sink(),countdown=.1,seconds=.2,
                              min_seconds=.05,display=display)
                    for i in range(11):s.tick(i*.05)
                    s.tick(.55,'r')
                    self.assertEqual(s.mode,'COUNTDOWN')
                    s.tick(.6)
                    s.tick(.65)
                    self.assertEqual(s.mode,'RECORDING')
                    s.tick(.75,'e')
                    self.assertEqual(s.mode,'TELEOP_RESET')
                    s.tick(.8,'h')
                    self.assertEqual(s.mode,'SAFE_RESET')
                    for i in range(17,27):s.tick(i*.05)
                    s.tick(1.35,'t')
                    self.assertEqual(s.mode,'TELEOP_RESET')
                    self.assertEqual([e['key'] for e in s.key_events],
                                     [repr(k) for k in 'reht'])
                    self.assertTrue(all(e['decision'].startswith('ACCEPTED') for e in s.key_events))
                finally:
                    if display: display.close()

    def test_pose_and_save_refusals_are_logged(self):
        io=SimulationIO();s=Session(io,Sink(),countdown=1)
        for i in range(11):s.tick(i*.05)
        io.request=lambda:dict.fromkeys(NAMES,10.)
        s.tick(.55,'r')
        self.assertIn('pose difference',s.key_events[-1]['decision'])
        self.assertEqual(s.mode,'SAFE_RESET')
        io.request=lambda:dict.fromkeys(NAMES,0.)
        s.outcomes.append(dict(episode_index=0))
        s.tick(.6,'r')
        self.assertIn('still saving',s.key_events[-1]['decision'])
        self.assertEqual(s.mode,'SAFE_RESET')

    def test_terminal_same_line_and_pause(self):
        lines=[]
        d=GripperDisplay(writer=lines.append,hz=5)
        d.terminal=True
        sample=dict(mode='SAFE_RESET',cycle=1,measured=47.041,
                    previous_sent=47.041,error=0.,requested=None,sent=None,
                    measured_at=time.perf_counter(),measured_utc=time.time(),
                    measurement_this_cycle=True)
        try:
            d.pause(.4)
            d.submit(sample)
            time.sleep(.22)
            self.assertFalse(lines)
            time.sleep(.25)
            self.assertTrue(lines)
            self.assertTrue(lines[0].startswith('\r\x1b[2KGRIP FRESH'))
            self.assertNotIn('\n',lines[0])
        finally:d.close()


if __name__ == '__main__':unittest.main()
