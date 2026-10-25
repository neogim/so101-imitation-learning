"""Display-only bounded mailbox. No device access, no print/stdout locks."""
import os
import queue
import threading
import time
from datetime import datetime, timezone


def render(sample, now, stale_after=.5):
    acquired=sample.get('measured_at')
    age=None if acquired is None else max(0.,now-acquired)
    stale=sample.get('stream_ended',False) or sample.get('communication_fault',False) or not sample.get('measurement_this_cycle',True) or age is None or age>stale_after
    def number(key):
        value=sample.get(key)
        return '--' if value is None else f'{value:.3f}'
    stamp=sample.get('measured_utc')
    utc='--' if stamp is None else datetime.fromtimestamp(stamp,timezone.utc).isoformat(timespec='milliseconds')
    return (f"[GRIP {'STALE' if stale else 'FRESH'}] mode={sample['mode']} cycle={sample['cycle']} "
            f"unit=normalized_0_100 measured={number('measured')} prev_goal={number('previous_sent')} "
            f"error_prev={number('error')} error_limit=8 leader_now={number('requested')} sent_now={number('sent')} "
            f"measured_utc={utc} t_mono={number('measured_at')} "
            f"age_ms={'--' if age is None else f'{age*1000:.0f}'} "
            f"event={sample.get('event','normal')} | NOT a grip-force safety indication\n")


def render_status(sample, now):
    """Compact terminal status; render() remains the full diagnostic format."""
    acquired=sample.get('measured_at')
    age=None if acquired is None else max(0.,now-acquired)
    stale=(sample.get('stream_ended',False) or sample.get('communication_fault',False)
           or not sample.get('measurement_this_cycle',True) or age is None or age>.5)
    def n(key):
        value=sample.get(key)
        return '--' if value is None else f'{value:.2f}'
    stamp=sample.get('measured_utc')
    utc='--' if stamp is None else datetime.fromtimestamp(stamp,timezone.utc).strftime('%H:%M:%S.%f')[:-3]
    return (f"GRIP {'STALE' if stale else 'FRESH'} {sample['mode']} "
            f"M={n('measured')} P={n('previous_sent')} E={n('error')} "
            f"L={n('requested')} S={n('sent')} "
            f"t={utc} age={'--' if age is None else f'{age*1000:.0f}'}ms unit=normalized_0_100 force=unknown")


class GripperDisplay:
    def __init__(self, hz=4, writer=None, clock=time.perf_counter):
        if not 2<=hz<=5:raise ValueError('Display Hz must be in [2,5]')
        self.period=1/hz;self.clock=clock
        self.queue=queue.Queue(maxsize=1)
        self.stop_event=threading.Event()
        self.pause_until=0.
        self.dropped=0;self.rendered=0;self.error=None;self.last_sample=None
        # Use raw writes to a separate stderr descriptor, never Python's stdout
        # lock used by mode/stop messages. Blocking output stalls only this daemon.
        self.fd=os.dup(2) if writer is None else None
        self.terminal=self.fd is not None and os.isatty(self.fd)
        self.writer=writer or (lambda line: os.write(self.fd,line.encode('utf-8')))
        self.thread=threading.Thread(target=self._run,name='gripper-display',daemon=True)
        self.thread.start()

    def pause(self, seconds=1.5):
        # The control thread never waits for terminal output; key messages stay visible.
        self.pause_until=max(self.pause_until,self.clock()+seconds)

    def submit(self, sample):
        self.last_sample=dict(sample)
        # At most two non-waiting attempts. The display may lose samples; control never waits.
        for _ in range(2):
            try:self.queue.put_nowait(dict(sample));return
            except queue.Full:
                try:self.queue.get_nowait();self.dropped+=1
                except queue.Empty:pass
        self.dropped+=1

    def _run(self):
        latest=None
        try:
            while True:
                try:latest=self.queue.get_nowait()
                except queue.Empty:pass
                if latest is not None:
                    now=self.clock()
                    if now >= self.pause_until or self.stop_event.is_set():
                        if self.terminal and not latest.get('stream_ended'):
                            line=render_status(latest,now)
                            # ANSI clear-to-end keeps successive updates on one row.
                            self.writer('\r\x1b[2K'+line)
                        else:
                            self.writer(render(latest,now))
                        self.rendered+=1
                if self.stop_event.is_set():break
                # Re-render retained sample so aging becomes visible when producer stops.
                self.stop_event.wait(self.period)
        except Exception as exc:self.error=repr(exc)
        finally:
            if self.fd is not None:os.close(self.fd)

    def close(self):
        if self.last_sample is not None:
            final=dict(self.last_sample)
            final['stream_ended']=True
            final['event']='STREAM_ENDED (historical sample; monitoring ended); '+final.get('event','normal')
            self.submit(final)
        self.stop_event.set()
        self.thread.join(timeout=.3)
        return dict(rendered=self.rendered,dropped=self.dropped,error=self.error,
                    worker_still_running=self.thread.is_alive(),last_sample=self.last_sample)
