#!/usr/bin/env python3
"""READ-ONLY OBS side ArUco + SO-101 Follower USB joint telemetry.

No Leader connection, no torque enable/disable, no Goal_Position writes.
Never run alongside a process owning the Follower serial port.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-config", type=Path, default=Path("/tmp/so101-aruco-640.yaml"))
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--hz", type=float, default=5)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--follower-port", default=None)
    p.add_argument("--execute-readonly", action="store_true", help="Explicit permission to open Follower USB; never writes servo registers")
    a = p.parse_args()
    if not a.execute_readonly or not (1 <= a.seconds <= 60) or not (0 < a.hz <= 10) or a.out_dir.exists():
        p.error("Require --execute-readonly, --seconds 1..60, --hz (0,10], and a NEW --out-dir")

    import cv2
    import numpy as np
    from multi_camera import CameraRig, load_config
    from record_episode import FOLLOWER, NAMES
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    port = a.follower_port or FOLLOWER
    if not Path(port).exists():
        p.error(f"Follower USB does not exist: {port}")
    if not a.camera_config.is_file():
        p.error(f"Camera configuration missing: {a.camera_config}")

    cfg = load_config(a.camera_config, enabled="side")
    rig = CameraRig(cfg)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    follower = SO101Follower(SO101FollowerConfig(port=port, id="my_follower_arm", use_degrees=True))
    # No follower.connect()/configure()/calibrate(): open serial bus directly.
    # No hardware write calls in this script, including on disconnect.
    a.out_dir.mkdir(parents=True)
    fields = ["sample_at_perf_s", "follower_read_start_perf_s", "follower_read_end_perf_s",
              "side_host_timestamp_perf_s", "side_image_age_ms", "ids",
              "j2_follower_deg", "j3_follower_deg", "j2_image_relative_deg",
              "j3_image_relative_deg", "j0_angle_image_deg", "j1_angle_image_deg",
              "j2_angle_image_deg", "all_joint_positions_json"]
    rows = []
    connected_camera = False
    connected_bus = False
    failure = None
    try:
        rig.connect()
        connected_camera = True
        follower.bus.connect()
        connected_bus = True
        # check calibrated reading mode; never attempt calibration.
        if not follower.is_calibrated:
            raise RuntimeError("Stored calibration not loaded or invalid. No repair attempted.")
        print("READ-ONLY: Follower USB open; no movement commands will be sent.", flush=True)
        start = time.perf_counter()
        period = 1 / a.hz
        k = 0
        while time.perf_counter() - start < a.seconds:
            due = start + k * period
            if due > time.perf_counter():
                time.sleep(due-time.perf_counter())
            k += 1
            img, stamp = rig.cameras["side"].sample(max_age_s=.1, transient_wait_s=.1, freeze_timeout_s=30)
            detect = time.perf_counter()
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            angles = {}
            if ids is not None:
                for xy, id_row in zip(corners, ids):
                    ident = int(id_row[0])
                    if ident in (0,1,2):
                        pts = xy.reshape(4,2)
                        v = pts[1] - pts[0]
                        angles[ident] = math.degrees(math.atan2(float(v[1]),float(v[0])))
            read_start = time.perf_counter()
            raw = follower.bus.sync_read("Present_Position", num_retry=0)
            read_end = time.perf_counter()
            values = {name+".pos": value for name,value in raw.items()}
            if set(values) != set(NAMES) or not all(math.isfinite(value) for value in values.values()):
                raise RuntimeError("Incomplete/nonfinite follower readings")
            def relative(a,b):
                return round((angles[b]-angles[a]+180)%360-180,3) if a in angles and b in angles else ""
            rows.append({
                "sample_at_perf_s": round(detect,6),
                "follower_read_start_perf_s": round(read_start,6),
                "follower_read_end_perf_s": round(read_end,6),
                "side_host_timestamp_perf_s": round(stamp,6),
                "side_image_age_ms": round((detect-stamp)*1000,2),
                "ids": ",".join(map(str,sorted(angles))),
                "j2_follower_deg": values["shoulder_lift.pos"],
                "j3_follower_deg": values["elbow_flex.pos"],
                "j2_image_relative_deg": relative(0,1),
                "j3_image_relative_deg": relative(1,2),
                "j0_angle_image_deg": angles.get(0,""),
                "j1_angle_image_deg": angles.get(1,""),
                "j2_angle_image_deg": angles.get(2,""),
                "all_joint_positions_json": json.dumps(values)
            })
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Passing disable_torque=False prevents motor state modifications at shutdown.
        if connected_bus:
            follower.bus.disconnect(disable_torque=False)
        if connected_camera:
            rig.disconnect()
        if rows:
            with (a.out_dir/"samples.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        summary={"mode":"read-only","camera":"side","samples":len(rows),
                 "all_three_samples":sum(r["ids"]=="0,1,2" for r in rows),
                 "failure":failure,
                 "caution":"Camera host screenshot response is not an exposure timestamp. 2D angles are not true J2/J3 angles."}
        (a.out_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
