"""One-thread IO guard: first read failure permanently inhibits new commands.
Diagnostic retries never return data to control. No reset/rearm operation exists.
"""
from collections import deque
from datetime import datetime, timezone
import time


class CommunicationFault(ConnectionError):
    pass


class GuardedIO:
    def __init__(self, io, clock=time.perf_counter):
        self.io, self.clock = io, clock
        self.latched = False
        self.context = {}
        self.history = deque(maxlen=256)
        self.cycles = deque(maxlen=128)
        self.failures = []
        self.transient_recoveries = []
        self.last_success = {}
        self.summary = {}
        self.start = clock()

    def metadata(self, register, role):
        result = dict(register=register, role=role,
                      servo_ids=[6] if register in ('Present_Load', 'Present_Current') else [1,2,3,4,5,6],
                      extra_read_retries=0, diagnostic_retry_limit=1)
        provider = getattr(self.io, 'communication_settings', None)
        try:
            result['settings'] = provider(register, role) if provider else {'device':'synthetic', 'packet_timeout_ms':None}
        except Exception as exc:
            result['settings'] = {'unavailable':repr(exc)}
        return result

    def _attempt(self, callback, register, role, diagnostic):
        start = self.clock()
        metadata = self.metadata(register, role)
        record = dict(self.context, **metadata,
                      started_monotonic=start, session_elapsed_s=start-self.start,
                      diagnostic=diagnostic, attempt=2 if diagnostic else 1,
                      last_success_monotonic=self.last_success.get(role+':'+register),
                      last_any_read_success_monotonic=max(self.last_success.values(),default=None))
        callback_started = self.clock()
        record['callback_started_monotonic'] = callback_started
        try:
            value = callback()
        except Exception as exc:
            record.update(ok=False, exception_type=type(exc).__name__, exception=str(exc))
            raise
        else:
            record['ok'] = True
            self.last_success[role+':'+register] = self.clock()
            return value
        finally:
            record['callback_finished_monotonic'] = self.clock()
            record['callback_duration_s'] = record['callback_finished_monotonic'] - callback_started
            record['finished_monotonic'] = self.clock()
            record['duration_s'] = record['finished_monotonic']-start
            self.history.append(record)
            stats = self.summary.setdefault(role+':'+register,dict(attempts=0, failures=0, duration_sum_s=0., duration_max_s=0.))
            stats['attempts'] += 1
            stats['failures'] += int(not record.get('ok',False))
            stats['duration_sum_s'] += record['duration_s']
            stats['duration_max_s'] = max(stats['duration_max_s'],record['duration_s'])

    def _read(self, callback, register, role='follower', transient_retries=0):
        if self.latched:
            raise CommunicationFault('Communication fault latched; no further control reads or sends')
        try:
            return self._attempt(callback,register,role,False)
        except Exception as exc:
            # Only callers in a non-recording mode may opt into this bounded
            # read-only retry. No Goal can be sent until the retry succeeds.
            if transient_retries:
                first_failure = dict(self.history[-1])
                try:
                    value = self._attempt(callback,register,role,False)
                except Exception:
                    pass
                else:
                    self.transient_recoveries.append(dict(
                        first_failure=first_failure, recovery=dict(self.history[-1])))
                    return value
            # Latch BEFORE any optional diagnostic access. Never clear on recovery.
            self.latched = True
            failure = dict(failure=dict(self.history[-1]), utc=datetime.now(timezone.utc).isoformat(),
                           reads_before_failure=list(self.history), cycles_before_failure=list(self.cycles))
            self.failures.append(failure)
            try:
                self._attempt(callback,register,role,True)
            except Exception:
                pass
            failure['diagnostic_retry'] = dict(self.history[-1])
            raise CommunicationFault(f'{role} {register}: communication fault latched; no new Action: {exc}') from exc

    def state(self):
        return self._read(self.io.state,'Present_Position')

    def torque_ok(self):
        return self._read(self.io.torque_ok,'Torque_Enable')

    def request(self):
        # Episode video encoding runs concurrently in TELEOP_RESET and can
        # briefly starve the USB read. RECORDING and SAFE_RESET retain their
        # first-failure latch behavior.
        retries = 1 if self.context.get('mode') == 'TELEOP_RESET' else 0
        return self._read(self.io.request,'Present_Position','leader',retries)

    def j6_telemetry(self):
        # Same control thread and bus as state/torque/action; a failed read latches
        # the existing communication fault before another Goal can be sent.
        return {register: self._read(lambda r=register: self.io.j6_register(r), register)
                for register in ('Present_Load', 'Present_Current')}

    def send(self, value):
        if self.latched:
            raise CommunicationFault('New Action forbidden after communication failure')
        try:
            return self.io.send(value)
        except Exception as exc:
            self.latched = True
            self.failures.append(dict(kind='send_failure',context=dict(self.context),exception=repr(exc),
                                      time_monotonic=self.clock(),retry=False))
            raise CommunicationFault('Action send failed; further sends forbidden') from exc

    def image(self):
        return self.io.image()

    def images(self):
        if self.latched:
            raise CommunicationFault('IO fault latched; no further images or Action')
        try:
            return self.io.images()
        except Exception as exc:
            self.latched = True
            self.failures.append(dict(kind='camera_failure',context=dict(self.context),
                                      exception=repr(exc),time_monotonic=self.clock(),retry=False))
            raise CommunicationFault('Camera failure; further Actions forbidden') from exc

    def camera_timing(self):
        camera = getattr(self.io, 'camera', None)
        snapshot = getattr(camera, 'timing_brief', None)
        return snapshot() if snapshot else None

    def report(self):
        return dict(latched=self.latched, automatic_resume=False, diagnostic_retry_limit=1,
                    teleop_reset_leader_read_retries=1,
                    read_history=list(self.history), cycle_history=list(self.cycles),
                    summary=self.summary, failures=self.failures,
                    transient_recoveries=self.transient_recoveries,
                    warning='No new goals after communication fault. Servo may continue toward/hold its last goal. Physical stopping and holding are NOT guaranteed.')
