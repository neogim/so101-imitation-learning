"""Pure unit tests for the robot-free two-camera display."""
import inspect
import threading
import time
import unittest
import numpy as np
from preview_side_wrist_live import render
from multi_camera import CameraRig

class FakeCamera:
    def __init__(self, age=0.01, error=None):
        self.lock=threading.Lock()
        self.frame=np.full((12,16,3),80,dtype=np.uint8)
        self.timestamp=time.perf_counter()-age
        self.last_sampled_timestamp=123.0
        self.error=error

class PreviewTests(unittest.TestCase):
    def test_render_does_not_consume_capture_frame(self):
        camera=FakeCamera()
        original=camera.last_sampled_timestamp
        image=render(camera,"side",(160,120),0.1)
        self.assertEqual(image.shape,(160,160,3))
        self.assertEqual(camera.last_sampled_timestamp,original)
    def test_stale_and_failed_camera_still_render(self):
        for camera in (FakeCamera(age=0.2),FakeCamera(error="offline")):
            self.assertEqual(render(camera,"wrist",(160,120),0.1).shape,(160,160,3))
    def test_pair_retry_is_bounded(self):
        self.assertEqual(inspect.signature(CameraRig.sample_all).parameters["pairing_wait_s"].default,0.12)

if __name__ == "__main__":
    unittest.main()
