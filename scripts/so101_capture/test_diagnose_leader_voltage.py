"""Hardware-free tests for the read-only Leader voltage probe."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import sys

from diagnose_leader_voltage import MOTORS, REGISTERS, diagnose, main, save


class FakeBus:
    def __init__(self, *, values=None, faults=None):
        self.port = "/dev/fake-leader"
        self.motors = {name: type("Motor", (), {"id": i})()
                       for i, name in enumerate(MOTORS, 1)}
        self.values = values or {}
        self.faults = faults or {}
        self.write = MagicMock(side_effect=AssertionError("write forbidden"))
        self.sync_write = MagicMock(side_effect=AssertionError("sync_write forbidden"))
        self.calls = []

    def read(self, register, motor, *, normalize, num_retry):
        assert register in REGISTERS and normalize is False and num_retry == 0
        self.calls.append((motor, register))
        if (motor, register) in self.faults:
            raise self.faults[motor, register]
        defaults = {"Present_Voltage": 72, "Status": 0,
                    "Min_Voltage_Limit": 50, "Max_Voltage_Limit": 84}
        return self.values.get((motor, register), defaults[register])


class LeaderVoltageTests(unittest.TestCase):
    def test_normal_six_ids_raw_scaling_no_writes(self):
        bus = FakeBus()
        report = diagnose(bus)
        self.assertEqual(report["overall"], "within_configured_limits")
        self.assertEqual(len(bus.calls), 24)
        self.assertEqual([x["id"] for x in report["motors"]], list(range(1, 7)))
        self.assertEqual(report["motors"][5]["present_voltage_v"], 7.2)
        bus.write.assert_not_called()
        bus.sync_write.assert_not_called()

    def test_one_axis_voltage_and_status_abnormal(self):
        bus = FakeBus(values={("gripper", "Present_Voltage"): 47,
                              ("gripper", "Status"): 1})
        report = diagnose(bus)
        self.assertEqual(report["overall"], "abnormal")
        self.assertEqual(sum(x["result"] == "abnormal" for x in report["motors"]), 1)
        self.assertEqual(report["motors"][5]["status_voltage_bit"], True)
        self.assertIn("voltage_outside_configured_limits", report["motors"][5]["reasons"])

    def test_single_read_failure_is_unknown_and_recorded(self):
        bus = FakeBus(faults={("gripper", "Present_Voltage"): RuntimeError("Input voltage error")})
        report = diagnose(bus)
        row = report["motors"][5]
        self.assertIsNone(row["present_voltage_v"])
        self.assertEqual(row["result"], "abnormal")
        self.assertIn("Present_Voltage", row["errors"])
        self.assertEqual(len(bus.calls), 24)

    def test_communication_fault_and_json_csv(self):
        bus = FakeBus(faults={("shoulder_pan", "Status"): ConnectionError("no status packet")})
        report = diagnose(bus)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result"
            save(report, output)
            loaded = json.loads((output / "leader_voltage.json").read_text())
            with (output / "leader_voltage.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(loaded["overall"], "abnormal")
            self.assertEqual(len(rows), 6)
            self.assertIn("no status packet", rows[0]["errors"])
            self.assertEqual(rows[0]["result"], "abnormal")
            with self.assertRaises(FileExistsError):
                save(report, output)
        bus.write.assert_not_called()
        bus.sync_write.assert_not_called()

    def test_main_connects_without_handshake_and_closes_without_writes(self):
        from lerobot.teleoperators.so_leader import so_leader, config_so_leader

        bus = FakeBus()
        bus.connect = MagicMock()
        bus.port_handler = MagicMock()
        bus.port_handler.ser = object()
        fake_leader = type("Leader", (), {"bus": bus})()
        with tempfile.TemporaryDirectory() as temp:
            port = Path(temp) / "fake-port"
            port.touch()
            output = Path(temp) / "result"
            args = ["diagnose_leader_voltage.py", "--execute", "--port", str(port),
                    "--output", str(output)]
            with (patch.object(sys, "argv", args), patch("builtins.input", return_value="READ ONLY"),
                  patch.object(so_leader, "SOLeader", return_value=fake_leader),
                  patch.object(config_so_leader, "SOLeaderTeleopConfig", return_value=object())):
                main()
            bus.connect.assert_called_once_with(handshake=False)
            bus.port_handler.closePort.assert_called_once_with()
            bus.write.assert_not_called()
            bus.sync_write.assert_not_called()
            self.assertTrue((output / "leader_voltage.json").is_file())


if __name__ == "__main__":
    unittest.main()
