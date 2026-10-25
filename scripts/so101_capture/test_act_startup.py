"""Pure synthetic startup tests. No robot, camera, or serial port is opened."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from act_startup import ACTStartup, LeRobotStartupDevice, MOTORS, StartupFault, make_act_guard


class FakeDevice:
    def __init__(self, fault=None):
        self.fault = fault
        self.events = []
        self.present = dict(zip(MOTORS, (1481, 1633, 2894, 2087, 833, 2161)))
        self.goal = dict.fromkeys(MOTORS, 0)
        self.torque = dict.fromkeys(MOTORS, 0)
        self.status = dict.fromkeys(MOTORS, 0)
        self.bounds = {name: (0, 4095) for name in MOTORS}
        self.healthy = True
        self.reads = 0

    def read_raw(self):
        self.reads += 1
        self.events.append("read")
        if self.fault == "read_before_goal" and self.reads == 1:
            self.healthy = False
            raise ConnectionError("synthetic read timeout")
        if self.fault == "read_after_goal" and self.reads == 3:
            self.healthy = False
            raise ConnectionError("synthetic read timeout")
        if self.fault == "late_torque" and self.reads == 4:
            self.torque[MOTORS[0]] = 1
        if self.fault == "post_config_torque" and self.reads == 5:
            self.torque[MOTORS[0]] = 1
        return {"Goal_Position": self.goal.copy(), "Present_Position": self.present.copy(),
                "Torque_Enable": self.torque.copy(), "Status": self.status.copy()}

    def calibration_bounds_raw(self):
        return self.bounds.copy()

    def calibration_matches(self):
        return self.fault != "calibration"

    def check_initial_alignment(self):
        self.events.append("initial_alignment_check")

    def configure_leader(self):
        self.events.append("configure_leader")

    def disable_follower_torque(self):
        self.events.append("disable_torque")
        self.torque = dict.fromkeys(MOTORS, 0)

    def write_present_as_goals(self, target):
        self.events.append("write_goal")
        if self.fault == "write_timeout":
            self.healthy = False
            raise ConnectionError("synthetic ambiguous write")
        self.goal = target.copy()
        if self.fault == "torque_after_goal":
            self.torque[MOTORS[0]] = 1
        if self.fault == "move_after_goal":
            self.present[MOTORS[0]] += 20
        if self.fault == "post_sync_gap":
            self.present[MOTORS[0]] += 4
            self.goal[MOTORS[0]] -= 4

    def configure_follower_without_enable(self):
        self.events.append("configure_follower")
        if self.fault == "configure_error":
            raise RuntimeError("synthetic config failure")

    def enable_follower_torque(self):
        self.events.append("enable_torque")
        if self.fault == "partial_enable":
            self.torque[MOTORS[0]] = 1
            raise ConnectionError("synthetic partial enable")
        self.torque = dict.fromkeys(MOTORS, 1)

    def align_with_recording_routine(self):
        self.events.append("align")
        if self.fault == "alignment_error":
            raise RuntimeError("synthetic alignment stall")

    def read_calibrated_state(self):
        self.events.append("read_state")
        if self.fault == "state_timeout":
            self.healthy = False
            raise ConnectionError("synthetic state timeout")
        return {name: float(self.present[name]) for name in MOTORS}

    def read_calibrated_goal(self):
        self.events.append("read_goal")
        if self.fault == "tracking":
            self.goal[MOTORS[0]] += 9
        return {name: float(self.goal[name]) for name in MOTORS}

    def communication_healthy(self):
        return self.healthy


class ACTStartupTests(unittest.TestCase):
    @staticmethod
    def run_case(fault=None):
        fake = FakeDevice(fault)
        startup = ACTStartup(fake)
        return fake, startup

    def test_requires_explicit_authorization_before_any_write(self):
        fake, startup = self.run_case()
        with self.assertRaises(StartupFault):
            startup.check_before_action()
        self.assertEqual(fake.events, [])
        fake, startup = self.run_case()
        with self.assertRaises(StartupFault):
            startup.run()
        self.assertNotIn("write_goal", fake.events)
        self.assertNotIn("enable_torque", fake.events)

    def test_hardware_adapter_cannot_start_full_sequence(self):
        adapter = LeRobotStartupDevice(object(), object())
        with self.assertRaisesRegex(StartupFault, "Hardware startup is disabled"):
            ACTStartup(adapter).run(operator_approved=True, arm_supported=True, workspace_clear=True)

    def test_in_range_stale_goal_is_allowed_only_before_sync(self):
        fake, startup = self.run_case()
        fake.bounds = {name: (100, 4000) for name in MOTORS}
        fake.goal = dict.fromkeys(MOTORS, 120)
        result = startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
        self.assertTrue(result.ready)
        self.assertEqual(result.checkpoints[0]['raw']['Goal_Position'][MOTORS[0]],120)
        self.assertEqual(result.checkpoints[2]['raw']['Goal_Position'][MOTORS[0]],fake.present[MOTORS[0]])

    def test_out_of_range_stale_goal_refuses_before_any_write(self):
        fake, startup = self.run_case()
        fake.bounds = {name: (100, 4000) for name in MOTORS}
        with self.assertRaisesRegex(StartupFault,'stored Goal outside calibrated raw range'):
            startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
        self.assertNotIn('disable_torque',fake.events)
        self.assertNotIn('write_goal',fake.events)
        self.assertNotIn('configure_leader',fake.events)

    def test_sync_only_old_goal_zero_writes_once_and_never_arms(self):
        fake, startup = self.run_case()
        fake.bounds = {name: (100, 4000) for name in MOTORS}
        startup.sleep=lambda _:None
        result=startup.run_hardware_goal_sync_only(operator_approved=True,
            arm_supported=True,workspace_clear=True,power_stop_ready=True)
        self.assertFalse(result.ready)
        self.assertEqual(fake.events.count('write_goal'),1)
        self.assertEqual(fake.goal,fake.present)
        for forbidden in ('disable_torque','configure_leader','configure_follower','enable_torque','align'):
            self.assertNotIn(forbidden,fake.events)
        self.assertEqual([x['phase'] for x in result.checkpoints][-1], 'after_goal_500ms')

    def test_sync_only_unexpected_torque_stops_without_retry(self):
        for fault in ('torque_after_goal','late_torque','read_after_goal',
                      'write_timeout','post_sync_gap'):
            with self.subTest(fault=fault):
                fake, startup = self.run_case(fault)
                fake.bounds = {name: (100, 4000) for name in MOTORS}
                startup.sleep=lambda _:None
                with self.assertRaises((StartupFault,ConnectionError)):
                    startup.run_hardware_goal_sync_only(operator_approved=True,
                        arm_supported=True,workspace_clear=True,power_stop_ready=True)
                self.assertEqual(fake.events.count('write_goal'),1)
                self.assertNotIn('enable_torque',fake.events)
                self.assertNotIn('align',fake.events)

    def test_normal_order_and_all_checkpoints(self):
        fake, startup = self.run_case()
        result = startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
        self.assertTrue(result.ready)
        self.assertEqual([x["phase"] for x in result.checkpoints], [
            "before_initialization", "before_goal_write", "after_goal_write", "before_configure",
            "after_configure_before_enable", "after_configure", "before_alignment",
            "before_first_act_action"])
        self.assertLess(fake.events.index("write_goal"), fake.events.index("configure_follower"))
        self.assertLess(fake.events.index("configure_follower"), fake.events.index("enable_torque"))
        self.assertLess(fake.events.index("enable_torque"), fake.events.index("align"))

    def test_goal_faults_never_reach_configure_or_enable(self):
        for fault in ("calibration", "read_before_goal", "write_timeout", "read_after_goal",
                      "torque_after_goal", "late_torque", "move_after_goal", "post_sync_gap"):
            with self.subTest(fault=fault):
                fake, startup = self.run_case(fault)
                with self.assertRaises((StartupFault, ConnectionError)):
                    startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
                self.assertFalse(startup.result.ready)
                self.assertNotIn("configure_follower", fake.events)
                self.assertNotIn("enable_torque", fake.events)
                self.assertNotIn("align", fake.events)

    def test_configure_error_does_not_enable_torque_in_finally(self):
        fake, startup = self.run_case("configure_error")
        with self.assertRaises(RuntimeError):
            startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
        self.assertNotIn("enable_torque", fake.events)
        self.assertNotIn("align", fake.events)
        self.assertEqual(fake.torque, dict.fromkeys(MOTORS, 0))

    def test_post_config_torque_change_blocks_explicit_enable(self):
        fake, startup = self.run_case("post_config_torque")
        with self.assertRaises(StartupFault):
            startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
        self.assertNotIn("enable_torque", fake.events)

    def test_partial_enable_or_alignment_fault_never_enters_action(self):
        for fault in ("partial_enable", "alignment_error"):
            with self.subTest(fault=fault):
                fake, startup = self.run_case(fault)
                with self.assertRaises((RuntimeError, ConnectionError)):
                    startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
                self.assertFalse(startup.result.ready)
                self.assertNotIn("read_state", fake.events)

    def test_pre_action_tracking_or_communication_fault_latches(self):
        for fault in ("tracking", "state_timeout"):
            with self.subTest(fault=fault):
                fake, startup = self.run_case(fault)
                with self.assertRaises((StartupFault, ConnectionError)):
                    startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
                self.assertFalse(startup.result.ready)
                with self.assertRaises(StartupFault):
                    startup.check_before_action()

    def test_existing_guard_keeps_speed_and_tracking_limits(self):
        limits = {name + ".pos": (-180, 180) for name in MOTORS}
        limits["gripper.pos"] = (0, 100)
        state = {name: 10.0 for name in MOTORS}
        guard = make_act_guard(limits, state, state)
        self.assertEqual(guard.config.speed_per_s, (30.0,) * 6)
        self.assertEqual(guard.config.tracking_error_limit, 8.0)
        action, _ = guard.propose([100.0] * 6, [10.0] * 6)
        self.assertAlmostEqual(action[0], 11.5)
        stopped, _ = guard.propose([100.0] * 6, [20.0] * 6)
        self.assertIsNone(stopped)

    def test_lerobot_adapter_configuration_matches_register_order_without_enable(self):
        class Bus:
            def __init__(self):
                self.motors = dict.fromkeys(MOTORS)
                self.writes = []
                self.torque = dict.fromkeys(MOTORS, 0)

            def disable_torque(self, **_):
                self.writes.append("disable")

            def configure_motors(self):
                self.write("Return_Delay_Time", MOTORS[0], 0)

            def write(self, register, motor, value, **_):
                self.writes.append((register, motor, value))

            def sync_read(self, register, **_):
                return self.torque.copy() if register == "Torque_Enable" else dict.fromkeys(MOTORS, 0)

        class Follower:
            def __init__(self, bus):
                self.bus = bus
                self.config = type("Config", (), dict(position_p_coefficient=1,
                    position_i_coefficient=2, position_d_coefficient=3))()

        bus = Bus()
        adapter = LeRobotStartupDevice(Follower(bus), object())
        adapter.configure_follower_without_enable()
        names = [row[0] for row in bus.writes if isinstance(row, tuple)]
        self.assertEqual(names[:5], ["Return_Delay_Time", "Operating_Mode",
                                    "P_Coefficient", "I_Coefficient", "D_Coefficient"])
        self.assertEqual(names[-3:], ["Max_Torque_Limit", "Protection_Current", "Overload_Torque"])
        self.assertNotIn("Torque_Enable", names)
        self.assertEqual(bus.writes[0], "disable")

        class FailedBus(Bus):
            def write(self, register, motor, value, **kwargs):
                if register == "I_Coefficient":
                    raise ConnectionError("synthetic configuration timeout")
                super().write(register, motor, value, **kwargs)

        failed = FailedBus()
        with self.assertRaises(ConnectionError):
            LeRobotStartupDevice(Follower(failed), object()).configure_follower_without_enable()
        self.assertEqual(failed.torque, dict.fromkeys(MOTORS, 0))
        self.assertNotIn("Torque_Enable", [row[0] for row in failed.writes if isinstance(row, tuple)])

    def test_json_has_six_axis_raw_checkpoints_and_refuses_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "startup.json"
            fake = FakeDevice()
            startup = ACTStartup(fake, diagnostics_path=path, sleep=lambda _: None)
            startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
            data = json.loads(path.read_text())
            self.assertTrue(data["ready"])
            self.assertEqual(len(data["checkpoints"]), 8)
            for checkpoint in data["checkpoints"]:
                self.assertEqual(checkpoint["validation"], "ok")
                for register in ("Goal_Position", "Present_Position", "Torque_Enable", "Status"):
                    self.assertEqual(set(checkpoint["raw"][register]), set(MOTORS))
            with self.assertRaises(FileExistsError):
                ACTStartup(FakeDevice(), diagnostics_path=path).run(
                    operator_approved=True, arm_supported=True, workspace_clear=True)

    def test_goal_torque_on_json_stops_before_configure(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "startup.json"
            fake = FakeDevice("torque_after_goal")
            startup = ACTStartup(fake, diagnostics_path=path, sleep=lambda _: None)
            with self.assertRaises(StartupFault):
                startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
            data = json.loads(path.read_text())
            self.assertFalse(data["ready"])
            self.assertIn("after_goal_write", data["latched_fault"])
            self.assertEqual(data["checkpoints"][-1]["validation"], "failed")
            self.assertEqual(data["checkpoints"][-1]["raw"]["Torque_Enable"][MOTORS[0]], 1)
            self.assertEqual(data["fault_readback"]["raw"]["Torque_Enable"][MOTORS[0]], 1)
            self.assertNotIn("configure_follower", fake.events)
            self.assertNotIn("align", fake.events)

    def test_partial_torque_enable_records_state_and_no_alignment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "startup.json"
            fake = FakeDevice("partial_enable")
            startup = ACTStartup(fake, diagnostics_path=path, sleep=lambda _: None)
            with self.assertRaises(ConnectionError):
                startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
            data = json.loads(path.read_text())
            self.assertIn("explicit_torque_enable", data["stage"])
            self.assertEqual(data["fault_readback"]["raw"]["Torque_Enable"][MOTORS[0]], 1)
            self.assertEqual(data["fault_readback"]["raw"]["Torque_Enable"][MOTORS[1]], 0)
            self.assertEqual(data["last_confirmed_raw"]["Torque_Enable"][MOTORS[0]], 0)
            self.assertNotIn("align", fake.events)
            self.assertFalse(data["ready"])

    def test_fault_readback_failure_marks_current_state_unknown(self):
        class FailedReadback(FakeDevice):
            def read_raw(self):
                if "write_goal" in self.events:
                    raise ConnectionError("no status packet")
                return super().read_raw()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "startup.json"
            fake = FailedReadback()
            startup = ACTStartup(fake, diagnostics_path=path, sleep=lambda _: None)
            with self.assertRaises(ConnectionError):
                startup.run(operator_approved=True, arm_supported=True, workspace_clear=True)
            data = json.loads(path.read_text())
            self.assertEqual(data["fault_readback"]["current_state"], "unknown")
            self.assertEqual(data["last_confirmed_raw"]["Torque_Enable"][MOTORS[0]], 0)
            self.assertNotIn("configure_follower", fake.events)


if __name__ == "__main__":
    unittest.main()
