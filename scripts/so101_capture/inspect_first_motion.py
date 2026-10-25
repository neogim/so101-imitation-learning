"""Read-only SO-101 Follower pose, stored goals and torque registers.

Uses bus.connect(), never Robot.connect()/configure(), sync_write() or torque
changes. Disconnect preserves torque state. No camera or Leader connection.
"""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from act_action_guard import NAMES, UNITS
from first_motion_test import HardwareIO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from record_episode import calibration_paths

    io = HardwareIO("wrist_flex.pos")  # Joint name is metadata only; no write call.
    opened = False
    try:
        connection = io.open()
        metadata = {"port": connection["port"], "id": connection["id"],
                    "use_degrees": connection["use_degrees"], "read_only": True}
        opened = True
        torque = io.robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=0)
        operating_mode = io.robot.bus.sync_read("Operating_Mode", normalize=False, num_retry=0)
        angular_resolution = io.robot.bus.sync_read("Angular_Resolution", normalize=False, num_retry=0)
        status = io.robot.bus.sync_read("Status", normalize=False, num_retry=0)
        goal_raw = io.robot.bus.sync_read("Goal_Position", normalize=False, num_retry=0)
        present_raw = io.robot.bus.sync_read("Present_Position", normalize=False, num_retry=0)
        goal = io.goal()
        present = io.state()
        limits = io.limits()
        if set(goal) != set(NAMES) or set(present) != set(NAMES) or len(torque) != 6:
            raise RuntimeError("Incomplete Follower register read")
        rows = []
        for index, name in enumerate(NAMES):
            key = name.removesuffix(".pos")
            calibration = io.robot.bus.calibration[key]
            rows.append({"joint": name, "unit": UNITS[index],
                         "torque_enabled": int(torque[key]),
                         "operating_mode": int(operating_mode[key]),
                         "angular_resolution": int(angular_resolution[key]),
                         "status": int(status[key]),
                         "goal_register_decoded": int(goal_raw[key]),
                         "present_register_decoded": int(present_raw[key]),
                         "goal": goal[name], "present": present[name],
                         "goal_minus_present": goal[name] - present[name],
                         "calibrated_limits": list(limits[index]),
                         "calibration": {"range_min": calibration.range_min,
                                         "range_max": calibration.range_max,
                                         "homing_offset": calibration.homing_offset,
                                         "drive_mode": calibration.drive_mode}})
        result = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "metadata": metadata,
                  "calibration_sha256": hashlib.sha256(calibration_paths()[0].read_bytes()).hexdigest(),
                  "communication": io.communication_settings("Present_Position", "follower"),
                  "joint_rows": rows,
                  "warning": "Read only. Torque may still be enabled and the servo may hold or move toward its existing goal. No position holding or physical stop is guaranteed."}
    finally:
        if opened or io.robot is not None:
            io.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
