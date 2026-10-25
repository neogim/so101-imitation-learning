"""Local OBS adapter: fixed index 1, RGB 640x480; no global patches."""
import logging

import cv2
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


class OBSCamera(OpenCVCamera):
    def __init__(self):
        super().__init__(OpenCVCameraConfig(
            index_or_path=1, backend=1200, width=640, height=480,
            fps=30, color_mode="rgb", rotation=0, warmup_s=1,
        ))

    def _configure_capture_settings(self):
        # OBS controls output dimensions/FPS. AVFoundation's CAP_PROP values may
        # describe its advertised format instead of the actual sample buffers.
        cap = self.videocapture
        logging.info("OBS advertised width=%s height=%s fps=%s (not used for shape)",
                     cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                     cap.get(cv2.CAP_PROP_FPS))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("OBS index 1 did not deliver a frame")
        # Validate actual pixels on connect AND every background frame using the
        # parent's strict shape/channel checks. No resize, crop or implicit flip.
        self._postprocess_image(frame)
