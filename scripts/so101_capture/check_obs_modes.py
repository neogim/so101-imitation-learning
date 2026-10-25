"""Camera-only comparison; opens index 1 exclusively. No robot imports."""
import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--with-pyav", action="store_true", help="Check camera alongside the recording video library")
    a = p.parse_args()
    import cv2
    if a.with_pyav:
        import av
        print(f"PyAV {av.__version__} loaded alongside OpenCV {cv2.__version__}")
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from obs_camera import OBSCamera

    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    for fps in (None, 30, 20):
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        try:
            if not cap.isOpened():
                raise RuntimeError("OBS index 1 cannot open")
            setters = {}
            if fps is not None:
                setters = dict(width=cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
                               height=cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480),
                               fps=cap.set(cv2.CAP_PROP_FPS, fps))
            shapes, times = [], []
            for _ in range(30):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("OBS read failed")
                times.append(time.perf_counter())
                shapes.append(tuple(frame.shape))
            results.append(dict(mode=f"raw_set_{fps}", setters=setters,
                                reported=[cap.get(k) for k in (3, 4, 5)],
                                actual_shapes=sorted(set(shapes)), delivery_fps=29/(times[-1]-times[0])))
        finally:
            cap.release()
    for mode in ("standard", "adapter"):
        cam = OBSCamera() if mode == "adapter" else OpenCVCamera(OpenCVCameraConfig(
            index_or_path=1, backend=1200, width=640, height=480, fps=30))
        try:
            cam.connect()
            times = []
            for _ in range(90):
                frame = cam.async_read(timeout_ms=2000)
                assert frame.shape == (480, 640, 3), frame.shape
                times.append(time.perf_counter())
            cv2.imwrite(str(a.output / f"{mode}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            results.append(dict(mode=mode, ok=True, shape=list(frame.shape),
                                frames=len(times), delivery_fps=89/(times[-1]-times[0])))
        except Exception as e:
            results.append(dict(mode=mode, ok=False, error=str(e)))
        finally:
            if cam.is_connected:
                cam.disconnect()
    (a.output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if not results[-1].get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
