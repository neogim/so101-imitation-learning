"""No OBS server, camera, robot, or motor connection."""
import base64
import csv
from io import BytesIO
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from PIL import Image
import numpy as np
import yaml

from multi_camera import (AVFoundationCamera, CameraFault, CameraRig, OBSSourceCamera,
                          capture_image_shape, image_keys,
                          load_config, obs_websocket_password, verify_source)
from compare_camera_previews import flatten
from preview_cameras import write_request_timeline


def spec(name, number):
    return {'enabled': True, 'provider': 'obs_source', 'source_name': name,
            'source_uuid': f'00000000-0000-0000-0000-{number:012d}',
            'expected_settings': {'device': f'device-{number}'}}


class FakeOBS:
    sources = {}
    screenshot_requests = []
    def __init__(self, **kwargs):
        self.sequence = 0
        self.closed = False

    def request(self, kind, data=None):
        if kind == 'GetInputList':
            return {'inputs': [{'inputName': n, 'inputUuid': s['source_uuid'],
                                'inputKind': 'video_capture_device'} for n, s in self.sources.items()]}
        if kind == 'GetInputSettings':
            return {'inputSettings': self.sources[data['inputName']]['expected_settings']}
        if kind == 'GetSourceScreenshot':
            self.screenshot_requests.append(dict(data))
            self.sequence += 1
            image = Image.new('RGB', (data['imageWidth'], data['imageHeight']),
                              (self.sequence % 255, 20, 30))
            data = BytesIO()
            image.save(data, format='JPEG')
            return {'imageData': 'data:image/jpeg;base64,' +
                    base64.b64encode(data.getvalue()).decode()}
        raise AssertionError(kind)

    def close(self):
        self.closed = True


class MultiCameraTests(unittest.TestCase):
    class PairCamera:
        def __init__(self, timestamps):
            self.timestamps = list(timestamps)
            self.calls = 0

        def sample(self, transient_wait_s=.05, newer_than=None, **_):
            self.calls += 1
            for stamp in self.timestamps:
                if newer_than is None or stamp > newer_than:
                    self.timestamps.remove(stamp)
                    return np.zeros((1, 1, 3), dtype=np.uint8), stamp
            raise CameraFault('no newer frame within bounded wait')

    def make_pair_rig(self, side, wrist):
        rig = CameraRig.__new__(CameraRig)
        rig.cameras = {'side': self.PairCamera(side), 'wrist': self.PairCamera(wrist)}
        return rig

    def test_pairing_accepts_initial_pair_within_75ms(self):
        now = time.perf_counter()
        rig = self.make_pair_rig([now-.030], [now-.080])
        _, stamps = rig.sample_all()
        self.assertLessEqual(max(stamps.values())-min(stamps.values()), .075)
        self.assertEqual(rig.cameras['side'].calls, 1)
        self.assertEqual(rig.cameras['wrist'].calls, 1)

    def test_pairing_refreshes_only_older_camera_and_recovers(self):
        now = time.perf_counter()
        rig = self.make_pair_rig([now-.005], [now-.090, now-.020])
        _, stamps = rig.sample_all()
        self.assertLessEqual(max(stamps.values())-min(stamps.values()), .075)
        self.assertEqual(rig.cameras['side'].calls, 1)
        self.assertEqual(rig.cameras['wrist'].calls, 2)

    def test_pairing_fails_when_older_camera_never_updates(self):
        now = time.perf_counter()
        rig = self.make_pair_rig([now-.005], [now-.090])
        with self.assertRaisesRegex(CameraFault, 'no newer frame|availability pairing'):
            rig.sample_all(pairing_wait_s=.005)

    def test_request_timeline_written_after_preview_with_utc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'requests.csv'
            write_request_timeline(path, {'side': [{
                'request_started_at': 10., 'response_at': 10.02,
                'request_ms': 20., 'image_decode_ms': .5, 'jpeg_bytes': 1234}]},
                utc_offset_s=1_700_000_000.)
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['camera'], 'side')
            self.assertEqual(rows[0]['request_ms'], '20.0')
            self.assertTrue(rows[0]['response_utc'].endswith('+00:00'))

    def test_obs_wrist_comparison_config_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = directory/'compare.yaml'
            config.write_text(yaml.safe_dump({'capture': {'width': 320, 'height': 240},
                'cameras': {'side': spec('side', 1), 'wrist': {
                    **spec('InnoMaker_Wrist', 2),
                    'expected_settings': {'device': '0x11240000c456366'}}}}))
            cfg = load_config(config, 'side,wrist')
            self.assertEqual(capture_image_shape(cfg), (240, 320, 3))
            self.assertEqual(cfg['active_cameras']['wrist']['provider'], 'obs_source')
            (directory/'preview.json').write_text(__import__('json').dumps({
                'completed': False, 'fault': 'side: stale frame 0.148s',
                'failed_samples': 1, 'camera': {'side': {
                    'provider': 'obs_source', 'jpeg_quality': 70,
                    'request_ms': {'mean': 6, 'p95': 8, 'max': 111},
                    'image_age_at_sample_ms': {'p95': 40, 'max': 99}}},
                'camera_response_skew_ms': {'p95': 20, 'max': 70},
                'loop_interval_ms': {'p95': 50, 'max': 52}}))
            row = flatten('B', 1, directory)
            self.assertFalse(row['completed'])
            self.assertTrue(row['abnormal_stop'])
            self.assertEqual(row['side_request_ms_mean'], 6)
            self.assertEqual(row['side_jpeg_quality'], 70)
            self.assertEqual(row['side_request_ms_max'], 111)
            self.assertEqual(row['camera_response_skew_ms_max'], 70)

    def test_obs_password_env_file_and_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / '.env'
            env_file.write_text('OTHER=ignored\nOBS_WEBSOCKET_PASSWORD="file-secret"\n')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(obs_websocket_password(env_file), 'file-secret')
                with patch.dict(os.environ, {'OBS_WEBSOCKET_PASSWORD': 'process-secret'}):
                    self.assertEqual(obs_websocket_password(env_file), 'process-secret')
                env_file.unlink()
                with self.assertRaisesRegex(CameraFault, 'environment or project-root .env'):
                    obs_websocket_password(env_file)
                with patch.dict(os.environ, {'OBS_WEBSOCKET_PASSWORD': 'process-secret'}):
                    self.assertEqual(obs_websocket_password(env_file), 'process-secret')
                env_file.write_text('OBS_WEBSOCKET_PASSWORD=\n')
                with self.assertRaisesRegex(CameraFault, 'OBS_WEBSOCKET_PASSWORD is not set'):
                    obs_websocket_password(env_file)

    def make_config(self, count=3):
        entries = {name: spec(name, i) for i, name in enumerate(
            ('overhead', 'side', 'wrist')[:count], 1)}
        FakeOBS.sources = entries
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cameras.yaml'
            path.write_text(yaml.safe_dump({'obs': {}, 'cameras': entries}))
            return load_config(path)

    def test_one_two_three_camera_selection(self):
        cfg = self.make_config()
        self.assertEqual(image_keys(cfg), ('observation.images.overhead',
                                           'observation.images.side',
                                           'observation.images.wrist'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cameras.yaml'
            path.write_text(yaml.safe_dump({'cameras': cfg['cameras']}))
            self.assertEqual(image_keys(load_config(path, 'side')), ('observation.images.side',))
            self.assertEqual(len(image_keys(load_config(path, 'side,overhead'))), 2)

    def test_mixed_obs_and_avfoundation_selection(self):
        entries = {name: spec(name, i) for i, name in enumerate(('overhead', 'side'), 1)}
        entries['wrist'] = {'enabled': False, 'provider': 'avfoundation_unique_id',
                            'unique_id': '0x11240000c456366'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cameras.yaml'
            path.write_text(yaml.safe_dump({'capture': {'width': 320, 'height': 240},
                                            'cameras': entries}))
            two = load_config(path)
            three = load_config(path, 'overhead,side,wrist')
        self.assertEqual(image_keys(two), ('observation.images.overhead',
                                           'observation.images.side'))
        self.assertEqual(image_keys(three)[-1], 'observation.images.wrist')
        self.assertEqual(capture_image_shape(three), (240, 320, 3))
        self.assertEqual(type(CameraRig(three).cameras['wrist']).__name__, 'AVFoundationCamera')

    def test_identity_snapshot_preserves_native_device_evidence(self):
        rig = CameraRig.__new__(CameraRig)
        camera = SimpleNamespace(lock=threading.Lock(),
            identity={'logical_name': 'wrist', 'unique_id': 'usb-id'},
            native_diagnostics=['SO101_AVFOUNDATION_DIAGNOSTIC {"event":"selected"}'])
        rig.cameras = {'wrist': camera}
        result = rig.identity_snapshot()
        self.assertEqual(result['wrist']['identity']['unique_id'], 'usb-id')
        self.assertIn('selected', result['wrist']['native_diagnostics'][0])

    def test_avfoundation_transient_stale_waits_for_new_fresh_frame(self):
        camera = AVFoundationCamera('wrist', {'unique_id': 'synthetic'}, {},
                                    image_shape=(2, 2, 3))
        old = np.zeros((2, 2, 3), dtype=np.uint8)
        fresh = np.ones((2, 2, 3), dtype=np.uint8)
        with camera.lock:
            camera.frame = old
            camera.timestamp = time.perf_counter() - .110
            camera.last_changed = camera.timestamp

        def publish_fresh():
            time.sleep(.01)
            with camera.lock:
                camera.frame = fresh
                camera.timestamp = time.perf_counter()
                camera.last_changed = camera.timestamp

        publisher = threading.Thread(target=publish_fresh)
        publisher.start()
        frame, stamp = camera.sample(transient_wait_s=.05)
        publisher.join()
        np.testing.assert_array_equal(frame, fresh)
        self.assertLess(time.perf_counter()-stamp, .1)

    def test_avfoundation_persistent_stale_never_returns_old_frame(self):
        camera = AVFoundationCamera('wrist', {'unique_id': 'synthetic'}, {},
                                    image_shape=(2, 2, 3))
        with camera.lock:
            camera.frame = np.zeros((2, 2, 3), dtype=np.uint8)
            camera.timestamp = time.perf_counter() - .110
            camera.last_changed = camera.timestamp
        with self.assertRaisesRegex(CameraFault, 'stale frame.*after waiting'):
            camera.sample(transient_wait_s=.005)

    def test_obs_transient_stale_waits_for_changed_fresh_frame(self):
        camera = OBSSourceCamera('side', spec('side', 1), {}, collect_timings=True,
                                 image_shape=(2, 2, 3))
        old = np.zeros((2, 2, 3), dtype=np.uint8)
        fresh = np.ones((2, 2, 3), dtype=np.uint8)
        with camera.lock:
            camera.frame = old
            camera.timestamp = time.perf_counter() - .141
            camera.last_changed = camera.timestamp

        def publish_fresh():
            time.sleep(.01)
            with camera.lock:
                camera.frame = fresh
                camera.timestamp = time.perf_counter()
                camera.last_changed = camera.timestamp

        publisher = threading.Thread(target=publish_fresh)
        publisher.start()
        started = time.perf_counter()
        frame, stamp = camera.sample(transient_wait_s=.05)
        elapsed = time.perf_counter() - started
        publisher.join()
        np.testing.assert_array_equal(frame, fresh)
        self.assertLess(elapsed, .05)
        self.assertLess(time.perf_counter()-stamp, .1)

    def test_obs_identical_screenshot_does_not_advance_freshness(self):
        camera = OBSSourceCamera('side', spec('side', 1), {}, collect_timings=True,
                                 image_shape=(2, 2, 3))
        image = Image.new('RGB', (2, 2), (1, 2, 3))
        encoded = BytesIO()
        image.save(encoded, format='JPEG')
        payload = 'data:image/jpeg;base64,' + base64.b64encode(encoded.getvalue()).decode()
        camera.client = Mock()
        camera.client.request.return_value = {'imageData': payload}
        camera._fetch()
        first_timestamp = camera.timestamp
        time.sleep(.002)
        camera._fetch()
        self.assertEqual(camera.timestamp, first_timestamp)
        self.assertFalse(camera.fetch_timings[-1]['content_changed'])

    def test_obs_persistent_stale_never_returns_old_frame(self):
        camera = OBSSourceCamera('side', spec('side', 1), {}, image_shape=(2, 2, 3))
        with camera.lock:
            camera.frame = np.zeros((2, 2, 3), dtype=np.uint8)
            camera.timestamp = time.perf_counter() - .141
            camera.last_changed = camera.timestamp
        with self.assertRaisesRegex(CameraFault, 'stale or reused frame.*changed fresh frame'):
            camera.sample(transient_wait_s=.005)

    def test_obs_same_published_frame_is_not_sampled_twice(self):
        camera = OBSSourceCamera('side', spec('side', 1), {}, image_shape=(2, 2, 3))
        with camera.lock:
            camera.frame = np.zeros((2, 2, 3), dtype=np.uint8)
            camera.timestamp = time.perf_counter()
            camera.last_changed = camera.timestamp
        camera.sample(transient_wait_s=0)
        with self.assertRaisesRegex(CameraFault, 'stale or reused frame'):
            camera.sample(transient_wait_s=.005)

    def test_independent_samples_and_identity(self):
        cfg = self.make_config(2)
        with patch('multi_camera.OBSClient', FakeOBS):
            rig = CameraRig(cfg)
            try:
                rig.connect()
                images, stamps = rig.sample_all()
                self.assertEqual(set(images), {'overhead', 'side'})
                self.assertEqual(set(stamps), set(images))
                self.assertEqual(images['side'].shape, (480, 640, 3))
                self.assertIsNot(images['side'], images['overhead'])
                brief = rig.timing_brief()
                self.assertGreaterEqual(brief['side']['image_age_ms'], 0)
                self.assertIn('request_started_at', brief['side']['last_fetch'])
                self.assertIn('response_at', brief['side']['last_fetch'])
                self.assertIn('image_decode_ms', brief['side']['last_fetch'])
                rig.cameras['side'].stop_event.set()
                rig.cameras['side'].thread.join(timeout=1)
                with rig.cameras['side'].lock:
                    rig.cameras['side'].active_request_started_at = time.perf_counter() - .05
                self.assertGreaterEqual(rig.timing_brief()['side']['active_request_age_ms'], 50)
            finally:
                rig.disconnect()

    def test_underlying_device_swap_rejected(self):
        cfg = self.make_config(1)
        fake = FakeOBS()
        FakeOBS.sources['overhead'] = {**FakeOBS.sources['overhead'],
                                       'expected_settings': {'device': 'swapped'}}
        with self.assertRaises(CameraFault):
            verify_source(fake, 'overhead', cfg['active_cameras']['overhead'])

    def test_screenshot_size_quality_and_stale_guard_unchanged(self):
        cfg = self.make_config(1)
        FakeOBS.screenshot_requests = []
        cfg['active_cameras']['overhead']['jpeg_quality'] = 70
        cfg['capture'] = {'width': 448, 'height': 336}
        with patch('multi_camera.OBSClient', FakeOBS):
            rig = CameraRig(cfg, collect_timings=True)
            try:
                rig.connect()
                frame, _ = rig.cameras['overhead'].sample()
                self.assertEqual(frame.shape, (336, 448, 3))
                self.assertEqual(FakeOBS.screenshot_requests[0]['imageCompressionQuality'], 70)
                self.assertEqual(FakeOBS.screenshot_requests[0]['imageWidth'], 448)
                self.assertEqual(FakeOBS.screenshot_requests[0]['imageHeight'], 336)
                with rig.cameras['overhead'].lock:
                    rig.cameras['overhead'].timestamp = time.perf_counter() - .101
                with self.assertRaisesRegex(CameraFault, 'stale or reused frame'):
                    rig.cameras['overhead'].sample(transient_wait_s=0)
            finally:
                rig.disconnect()

        cfg['cameras']['overhead']['jpeg_quality'] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cameras.yaml'
            path.write_text(yaml.safe_dump({'cameras': cfg['cameras']}))
            with self.assertRaisesRegex(ValueError, 'jpeg_quality'):
                load_config(path)

    def test_capture_shape_defaults_and_rejects_mismatch(self):
        cfg = self.make_config(2)
        self.assertEqual(capture_image_shape(cfg), (480, 640, 3))
        cfg['capture'] = {'width': 320, 'height': 240}
        self.assertEqual(capture_image_shape(cfg), (240, 320, 3))
        cfg['capture'] = {'width': 448, 'height': 480}
        with self.assertRaisesRegex(ValueError, 'capture width/height'):
            capture_image_shape(cfg)

    def test_recreated_source_uuid_and_wifi_ip_rejected(self):
        cfg = self.make_config(1)
        fake = FakeOBS()
        original = FakeOBS.sources['overhead']
        cfg['active_cameras']['overhead']['expected_settings'] = {'cur_dev_ip': '192.0.2.10'}
        try:
            FakeOBS.sources['overhead'] = {**original,
                'source_uuid': '00000000-0000-0000-0000-000000000099'}
            with self.assertRaisesRegex(CameraFault, 'UUID changed or missing'):
                verify_source(fake, 'overhead', cfg['active_cameras']['overhead'])
            FakeOBS.sources['overhead'] = {**original,
                'expected_settings': {'cur_dev_ip': '192.0.2.11'}}
            with self.assertRaisesRegex(CameraFault, 'device setting cur_dev_ip changed'):
                verify_source(fake, 'overhead', cfg['active_cameras']['overhead'])
        finally:
            FakeOBS.sources['overhead'] = original

    def test_stale_image_rejected(self):
        cfg = self.make_config(1)
        with patch('multi_camera.OBSClient', FakeOBS):
            rig = CameraRig(cfg)
            try:
                rig.connect()
                cam = rig.cameras['overhead']
                with cam.lock:
                    cam.timestamp = time.perf_counter() - 1
                with self.assertRaisesRegex(CameraFault, 'stale'):
                    cam.sample(transient_wait_s=0)
            finally:
                rig.disconnect()

    def test_dataset_features_follow_enabled_cameras(self):
        from capture_session import create_dataset, NAMES
        with tempfile.TemporaryDirectory() as tmp:
            for names, shape in ((('overhead', 'side'), (336, 448, 3)),
                                 (('overhead', 'side', 'wrist'), (480, 640, 3))):
                root = Path(tmp) / f'dataset-{len(names)}'
                dataset = create_dataset(root, True, names, shape)
                frame = np.zeros(shape, dtype=np.uint8)
                values = np.zeros(len(NAMES), dtype=np.float32)
                row = {'observation.state': values, 'action': values,
                       'action_requested': values, 'task': 'synthetic'}
                row.update({f'observation.images.{name}': frame.copy() for name in names})
                dataset.add_frame(row)
                dataset.save_episode(parallel_encoding=False)
                dataset.finalize()
                import json
                info = json.loads((root/'meta/info.json').read_text())
                self.assertEqual(info['total_episodes'], 1)
                self.assertEqual(info['total_frames'], 1)
                self.assertEqual({k for k in info['features'] if k.startswith('observation.images.')},
                                 {f'observation.images.{name}' for name in names})
                self.assertTrue(all(info['features'][f'observation.images.{name}']['shape'] == list(shape)
                                    for name in names))

    def test_act_predictor_uses_exact_logical_image_keys(self):
        import torch
        from act_action_guard import NAMES
        from act_handoff_launcher import ACTPredictor
        class Policy:
            config = SimpleNamespace(chunk_size=50, temporal_ensemble_coeff=None,
                                     n_action_steps=50)
            _action_queue = ()
            seen = None
            def reset(self): pass
            def select_action(self, observation):
                self.seen = set(observation)
                return torch.zeros((1, 6), dtype=torch.float32)
        for names, shape in ((('overhead', 'side'), (240, 320, 3)),
                             (('overhead', 'side'), (336, 448, 3)),
                             (('overhead', 'side', 'wrist'), (480, 640, 3))):
            image = np.zeros(shape, dtype=np.uint8)
            policy = Policy()
            keys = tuple(f'observation.images.{name}' for name in names)
            with patch('observe_act_read_only.load_policy', return_value=(policy, lambda x:x, lambda x:x)):
                predictor = ACTPredictor(device='cpu', checkpoint=Path('/unused'),
                                         image_keys=keys, image_shape=shape)
            action, _ = predictor.predict(dict.fromkeys(NAMES, 0.),
                                          {name: image for name in names})
            self.assertEqual(action.shape, (6,))
            self.assertTrue({'observation.state', *keys} <= policy.seen)
            with self.assertRaisesRegex(ValueError, 'camera keys'):
                predictor.predict(dict.fromkeys(NAMES, 0.), {'iphone': image})


if __name__ == '__main__':
    unittest.main()
