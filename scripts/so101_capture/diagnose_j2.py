"""Read-only Follower J2 diagnostics; never configure, calibrate or change torque."""
import json
import time

from record_episode import make_hardware


def main():
    follower, _ = make_hardware(None)
    bus = follower.bus
    samples = []
    try:
        # No all-motor handshake required for an isolated read-only diagnosis.
        bus.connect(handshake=False)
        for i in range(3):
            sample = {'sample': i}
            for register in ('Present_Voltage', 'Present_Temperature', 'Present_Position', 'Present_Load', 'Present_Current', 'Status', 'Torque_Enable'):
                try:
                    sample[register] = bus.read(register, 'shoulder_lift', normalize=False)
                except Exception as exc:
                    sample[register] = {'error': str(exc)}
            samples.append(sample)
            time.sleep(.2)
        print(json.dumps(samples, indent=2), flush=True)
    finally:
        # Direct close: deliberately no torque-disable writes, even on failures.
        if getattr(bus.port_handler, 'ser', None) is not None:
            bus.port_handler.closePort()


if __name__ == '__main__':
    main()
