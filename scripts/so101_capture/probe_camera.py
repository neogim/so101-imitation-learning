"""Camera-only probe; no robot imports or servo access. Run with Conda lerobot Python."""
import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("index", type=int)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--rotation", type=int, choices=[0, 90, 180, -90], default=0)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.seconds <= 0:
        p.error("--seconds must be positive")
    import cv2
    import numpy as np
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    a.output.mkdir(parents=True, exist_ok=False)
    camera = OpenCVCamera(OpenCVCameraConfig(
        index_or_path=a.index, width=a.width, height=a.height, fps=a.camera_fps,
        backend=1200, color_mode="rgb", rotation=a.rotation,
    ))
    times, repeats, previous = [], 0, None
    try:
        camera.connect()
        start = time.perf_counter()
        while time.perf_counter() - start < a.seconds:
            frame = camera.async_read(timeout_ms=2000)
            times.append(time.perf_counter())
            repeats += int(previous is not None and np.array_equal(previous, frame))
            previous = frame.copy()
            if len(times) == 1:
                cv2.imwrite(str(a.output / "first.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(a.output / "last.png"), cv2.cvtColor(previous, cv2.COLOR_RGB2BGR))
        cap = camera.videocapture
        result = {
            "index": a.index, "backend": cap.getBackendName(),
            "reported_fps": cap.get(cv2.CAP_PROP_FPS), "rgb_shape": list(previous.shape),
            "received_frames": len(times), "identical_adjacent_frames": repeats,
            "measured_delivery_fps": (len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None,
            "delivery_intervals_s": np.diff(times).tolist(),
            "note": "Host delivery cadence, not iPhone exposure time. Inspect first/last.png with a readable text and left/right marker.",
        }
        (a.output / "probe.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        if camera.is_connected:
            camera.disconnect()


if __name__ == "__main__":
    main()
