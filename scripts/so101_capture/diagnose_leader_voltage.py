"""Read-only STS3215 voltage diagnosis for the SO-101 Leader.

No Teleoperator.connect(), calibration, configure, Goal, or torque operation.
Feetech voltage register values are in 0.1 V units. A failed read is unknown,
never a healthy measurement. Run on hardware only after separate authorization.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


DEFAULT_PORT = "/dev/tty.usbmodem5B7B0185961"
REGISTERS = ("Present_Voltage", "Status", "Min_Voltage_Limit", "Max_Voltage_Limit")
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def diagnose(bus) -> dict:
    """Read each register once, without retry or any write API."""
    rows = []
    for expected_id, motor in enumerate(MOTORS, start=1):
        if motor not in bus.motors or bus.motors[motor].id != expected_id:
            raise ValueError(f"Unexpected Leader motor mapping: id={expected_id} {motor}")
        values, errors = {}, {}
        for register in REGISTERS:
            try:
                values[register] = int(bus.read(register, motor, normalize=False, num_retry=0))
            except Exception as exc:
                values[register] = None
                errors[register] = f"{type(exc).__name__}: {exc}"
        voltage = values["Present_Voltage"]
        low = values["Min_Voltage_Limit"]
        high = values["Max_Voltage_Limit"]
        status = values["Status"]
        reasons = []
        if errors:
            reasons.append("read_failure")
        if status is not None and status != 0:
            reasons.append("status_nonzero")
        if None not in (low, high) and low >= high:
            reasons.append("invalid_voltage_limits")
        if None not in (voltage, low, high) and not low <= voltage <= high:
            reasons.append("voltage_outside_configured_limits")
        rows.append({
            "id": expected_id, "motor": motor,
            "present_voltage_raw": voltage,
            "present_voltage_v": None if voltage is None else voltage / 10,
            "status_raw": status,
            "min_voltage_limit_raw": low,
            "min_voltage_limit_v": None if low is None else low / 10,
            "max_voltage_limit_raw": high,
            "max_voltage_limit_v": None if high is None else high / 10,
            "status_voltage_bit": None if status is None else bool(status & 1),
            "errors": errors,
            "result": "abnormal" if reasons else "within_configured_limits",
            "reasons": reasons,
        })
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "port": bus.port,
        "voltage_scale_v_per_raw": 0.1,
        "status_voltage_error_bit": 1,
        "overall": "abnormal" if any(row["reasons"] for row in rows) else "within_configured_limits",
        "note": "A normal snapshot does not rule out transient voltage faults. No motor write was requested.",
        "motors": rows,
    }


def save(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "leader_voltage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (output / "leader_voltage.csv").open("w", newline="") as file:
        fields = ("id", "motor", "present_voltage_raw", "present_voltage_v", "status_raw",
                  "status_voltage_bit", "min_voltage_limit_raw", "min_voltage_limit_v",
                  "max_voltage_limit_raw", "max_voltage_limit_v", "result", "reasons", "errors")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in report["motors"]:
            writer.writerow({**row, "reasons": ";".join(row["reasons"]),
                             "errors": json.dumps(row["errors"], ensure_ascii=False)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Separately approved hardware read")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Hardware access requires --execute and separate operator approval")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not Path(args.port).exists():
        parser.error(f"Leader serial port not found: {args.port}")
    typed = input("READ-ONLY Leader voltage diagnosis; no servo write. Type READ ONLY: ")
    if typed.strip() != "READ ONLY":
        parser.error("Operator did not authorize this read-only run")

    # SOLeader constructor loads configuration only. Never call leader.connect():
    # it can calibrate and configure. bus.connect(handshake=False) opens the port
    # and sets host timeout; it skips ping/firmware checks and all servo writes.
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader

    leader = SOLeader(SOLeaderTeleopConfig(port=args.port, id="my_leader_arm", use_degrees=True))
    bus = leader.bus
    try:
        bus.connect(handshake=False)
        report = diagnose(bus)
    finally:
        # bus.disconnect() defaults to disable_torque=True, which WRITES.
        # Direct port close cannot send Goal, torque, or configuration writes.
        if getattr(bus.port_handler, "ser", None) is not None:
            bus.port_handler.closePort()
    save(report, args.output)
    print("ID  motor             voltage V  min V  max V  status  result")
    for row in report["motors"]:
        fmt = lambda value: "?" if value is None else str(value)
        print(f"{row['id']:>2}  {row['motor']:<16}  {fmt(row['present_voltage_v']):>9}  "
              f"{fmt(row['min_voltage_limit_v']):>5}  {fmt(row['max_voltage_limit_v']):>5}  "
              f"{fmt(row['status_raw']):>6}  {row['result']}")
        for register, error in row["errors"].items():
            print(f"    {register}: {error}")
    print(f"Overall: {report['overall']}; saved to {args.output}")
    if report["overall"] != "within_configured_limits":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
