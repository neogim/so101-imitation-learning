#!/usr/bin/env python3
"""Guarded SO-101 teleop plus timestamped follower and OBS ArUco samples.

Hardware MOVES ONLY with --execute. Uses existing teleop_only safety checks.
A single thread owns both serial buses; the camera is read in a worker thread.
Camera failure stops teleoperation. Outputs are diagnostic, not 3D calibration.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import signal
import threading
import time

from record_episode import hardware_preflight, make_hardware, start_hardware, close_hardware, NAMES
from teleop_only import teleop_tick


class CameraReader:
    def __init__(self, index, ids):
        import cv2
        self.cv2 = cv2
        self.ids = ids
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}; check OBS Virtual Camera")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="aruco-camera")

    def start(self):
        self.thread.start()

    def _loop(self):
        import numpy as np
        try:
            while not self.stopping.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("Camera frame read failed")
                stamp = time.perf_counter()
                corners, ids, _ = self.detector.detectMarkers(frame)
                observed = {}
                if ids is not None:
                    for corners_one, id_one in zip(corners, ids):
                        ident = int(id_one[0])
                        if ident not in self.ids:
                            continue
                        xy = corners_one.reshape(4, 2)
                        direction = xy[1] - xy[0]
                        deg = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
                        lengths = [float(np.linalg.norm(xy[(i+1)%4]-xy[i])) for i in range(4)]
                        observed[ident] = dict(angle_image_deg=deg, min_side_px=min(lengths),
                                               corners_px=xy.round(2).tolist())
                with self.lock:
                    self.latest = (stamp, observed)
        except BaseException as exc:
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"

    def snapshot(self):
        with self.lock:
            if self.error:
                raise RuntimeError(f"Camera failed: {self.error}")
            return self.latest

    def close(self):
        self.stopping.set()
        self.cap.release()
        if self.thread.is_alive():
            self.thread.join(timeout=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-cameras", action="store_true", help="Enumerate camera indices without opening robot")
    ap.add_argument("--execute", action="store_true", help="Explicit approval for active teleoperation")
    ap.add_argument("--camera-index", type=int, default=None)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/so101-aruco-live"))
    ap.add_argument("--max-frame-age-ms", type=float, default=500)
    a = ap.parse_args()
    if a.list_cameras:
        import cv2
        for index in range(12):
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                ok, frame = cap.read()
                print(f"index={index}, readable={ok}, shape={None if not ok else frame.shape}", flush=True)
            cap.release()
        return
    if not a.execute or a.camera_index is None or not 1 <= a.seconds <= 300 or a.max_frame_age_ms <= 0:
        ap.error("Use --execute --camera-index N --seconds 1..300 (positive max age)")
    if a.out_dir.exists():
        ap.error(f"Output exists, refusing overwrite: {a.out_dir}")
    hashes = hardware_preflight(a.out_dir)
    a.out_dir.mkdir(parents=True)
    (a.out_dir / "metadata.json").write_text(json.dumps({
        "camera_index": a.camera_index, "seconds": a.seconds,
        "units": "degrees for five joints, gripper normalized_0_100",
        "warning": "Image angles are only 2D. No calibrated 3D marker pose or joint-zero proof.",
    }, indent=2))
    camera = CameraReader(a.camera_index, (0,1,2))
    follower = leader = None
    configured = False
    stop_requested = False
    def stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    from lerobot.utils.robot_utils import precise_sleep
    csv_path = a.out_dir / "samples.csv"
    fields = ["sample_perf_counter_s", "camera_perf_counter_s", "camera_age_ms",
              "j2_follower_deg", "j3_follower_deg", "j2_goal_deg", "j3_goal_deg"]
    for marker in (0,1,2):
        fields += [f"id{marker}_angle_image_deg",f"id{marker}_min_side_px",f"id{marker}_corners_json"]
    fields += ["j2_image_relative_deg","j3_image_relative_deg"]
    try:
        camera.start()
        deadline = time.perf_counter() + 5
        while camera.snapshot() is None:
            if time.perf_counter() > deadline:
                raise RuntimeError("No OBS camera frame after 5 seconds; no motor startup")
            time.sleep(.03)
        follower, leader = make_hardware(None)
        follower.cameras = {}
        start_hardware(follower, leader, guarded_teleop_start=True)
        configured = True
        last_sent = {name + ".pos": v for name,v in follower.bus.sync_read("Goal_Position").items()}
        if set(last_sent)!=set(NAMES) or not all(math.isfinite(v) for v in last_sent.values()):
            raise RuntimeError("Invalid initial Goal; no action sent")
        print("TELEOP ACTIVE. Support arm, move ONLY small J2/J3 increments, stop with Ctrl-C.", flush=True)
        with csv_path.open("w",newline="",encoding="utf-8") as output:
            writer = csv.DictWriter(output,fieldnames=fields)
            writer.writeheader()
            start = time.perf_counter()
            while not stop_requested and time.perf_counter()-start < a.seconds:
                tick = time.perf_counter()
                image_sample = camera.snapshot()
                if image_sample is None:
                    raise RuntimeError("No camera sample")
                stamp, markers = image_sample
                if (tick-stamp)*1000 > a.max_frame_age_ms:
                    raise RuntimeError("Stale camera frame: stopping before next Goal command")
                # This is the only call allowed to access robot control.
                last_sent = teleop_tick(follower,leader,last_sent)
                now = time.perf_counter()
                follower_state = {name+".pos":v for name,v in follower.bus.sync_read("Present_Position",num_retry=0).items()}
                if set(follower_state)!=set(NAMES) or not all(math.isfinite(v) for v in follower_state.values()):
                    raise RuntimeError("Invalid follower telemetry, stopping")
                row = {"sample_perf_counter_s":round(now,6),
                       "camera_perf_counter_s":round(stamp,6),
                       "camera_age_ms":round((now-stamp)*1000,2),
                       "j2_follower_deg":follower_state["shoulder_lift.pos"],
                       "j3_follower_deg":follower_state["elbow_flex.pos"],
                       "j2_goal_deg":last_sent["shoulder_lift.pos"],
                       "j3_goal_deg":last_sent["elbow_flex.pos"]}
                for i in (0,1,2):
                    data = markers.get(i)
                    row[f"id{i}_angle_image_deg"]=data["angle_image_deg"] if data else ""
                    row[f"id{i}_min_side_px"]=round(data["min_side_px"],2) if data else ""
                    row[f"id{i}_corners_json"]=json.dumps(data["corners_px"]) if data else ""
                def relative(first,second):
                    if first not in markers or second not in markers:
                        return ""
                    return round((markers[second]["angle_image_deg"]-markers[first]["angle_image_deg"]+180)%360-180,3)
                row["j2_image_relative_deg"]=relative(0,1)
                row["j3_image_relative_deg"]=relative(1,2)
                writer.writerow(row)
                output.flush()
                precise_sleep(max(0,0.05-(time.perf_counter()-tick)))
    except KeyboardInterrupt:
        print("Interrupted",flush=True)
    finally:
        print("CONTROL STOPPED. Support arm before torque-off. No physical holding guarantee.",flush=True)
        try:
            close_hardware(follower, leader, configured)
        finally:
            camera.close()
            for path,digest in hashes.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError(f"Calibration changed: {path}")
    print(f"Saved {csv_path}",flush=True)


if __name__=="__main__":
    main()
