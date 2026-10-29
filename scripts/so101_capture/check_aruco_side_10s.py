#!/usr/bin/env python3
"""10-second ArUco check on the configured OBS side source. No servo access."""
import argparse
import json
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera-config", type=Path, default=Path("scripts/so101_capture/cameras.local.yaml"))
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.seconds <= 60 or args.out_dir.exists():
        ap.error("Require --seconds 1..60 and a NEW --out-dir")
    import cv2
    import numpy as np
    from PIL import Image
    from multi_camera import CameraRig, load_config
    if not hasattr(cv2, "aruco"):
        ap.error("OpenCV must support cv2.aruco")
    cfg = load_config(args.camera_config, enabled="side")
    rig = CameraRig(cfg)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    counts = {0: 0, 1: 0, 2: 0}
    frames = 0
    simultaneous = 0
    best = (-1, None, None)
    last_timestamp = None
    fault = None
    try:
        rig.connect()
        args.out_dir.mkdir(parents=True)
        end = time.perf_counter() + args.seconds
        while time.perf_counter() < end:
            try:
                frame, stamp = rig.cameras["side"].sample(
                    max_age_s=.1, transient_wait_s=.1, freeze_timeout_s=30)
            except Exception as exc:
                fault = f"{type(exc).__name__}: {exc}"
                break
            if stamp == last_timestamp:
                time.sleep(.02)
                continue
            last_timestamp = stamp
            frames += 1
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            corners, ids, _ = detector.detectMarkers(bgr)
            visible = set() if ids is None else {int(v) for v in ids.flatten()} & set(counts)
            for v in visible:
                counts[v] += 1
            simultaneous += (len(visible) == 3)
            if len(visible) > best[0]:
                best = (len(visible), frame.copy(), (corners, ids))
            time.sleep(.01)
    finally:
        rig.disconnect()
    if best[1] is not None:
        annotated = cv2.cvtColor(best[1], cv2.COLOR_RGB2BGR)
        corners, ids = best[2]
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        cv2.imwrite(str(args.out_dir / "best_detected.png"), annotated)
    result = {
        "camera": "side", "source": str(args.camera_config),
        "seconds_requested": args.seconds, "unique_frames": frames,
        "id_counts": {str(k): v for k, v in counts.items()},
        "id_rates": {str(k): round(v/frames, 3) if frames else None for k,v in counts.items()},
        "all_three_count": simultaneous,
        "all_three_rate": round(simultaneous/frames, 3) if frames else None,
        "best_frame_marker_count": best[0],
        "camera_fault": fault,
        "note": "No robot or servo access. OBS screenshot response time is not exposure time."
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("Output:", args.out_dir / "best_detected.png")
    return 0 if not fault else 2

if __name__ == "__main__":
    raise SystemExit(main())
