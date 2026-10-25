"""Probe only the specified, already identified OBS index; never enumerate cameras."""
import argparse
import json
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--index", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
import cv2

a.output.mkdir(parents=True, exist_ok=False)
cap = cv2.VideoCapture(a.index, cv2.CAP_AVFOUNDATION)
times, shapes = [], []
try:
    if not cap.isOpened():
        raise RuntimeError("OBS device could not open")
    properties = dict(width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                      height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT), fps=cap.get(cv2.CAP_PROP_FPS))
    start = time.perf_counter()
    while time.perf_counter()-start < 5:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("OBS frame read failed")
        times.append(time.perf_counter())
        shapes.append(list(frame.shape))
        if len(times) == 1:
            cv2.imwrite(str(a.output / "first.png"), frame)
    cv2.imwrite(str(a.output / "last.png"), frame)
    result = dict(index=a.index, reported=properties, actual_shapes=sorted({tuple(s) for s in shapes}),
                  frames=len(times), delivery_fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None)
    (a.output / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
finally:
    cap.release()
