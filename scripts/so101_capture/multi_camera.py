"""Independent OBS input capture selected by stable source UUID, not camera index.

OBS WebSocket v5 must be enabled locally. Credentials come from the process
environment or the project-root .env; never put them in camera YAML.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
from PIL import Image
import yaml


class CameraFault(RuntimeError):
    pass


def obs_websocket_password(env_file: Path | None = None) -> str:
    """Use an explicit process value first, then the project-root .env."""
    password = os.environ.get("OBS_WEBSOCKET_PASSWORD")
    if password:
        return password
    path = env_file or Path(__file__).resolve().parents[2] / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.strip().partition("=")
            if not separator or key.strip() != "OBS_WEBSOCKET_PASSWORD":
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                return value
    raise CameraFault("OBS_WEBSOCKET_PASSWORD is not set; set it in the environment or project-root .env")


def load_config(path: str | Path, enabled: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or not isinstance(cfg.get("cameras"), dict):
        raise ValueError("Expected a cameras mapping in YAML")
    capture_image_shape(cfg)
    selected = None if enabled is None else {x.strip() for x in enabled.split(",") if x.strip()}
    if selected is not None and (not selected or selected - cfg["cameras"].keys()):
        raise ValueError("Unknown or empty enabled camera list")
    cameras = {}
    for name, spec in cfg["cameras"].items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(spec, dict):
            raise ValueError("Camera names must be identifiers with mapping settings")
        if not isinstance(spec.get("enabled"), bool):
            raise ValueError(f"{name}: enabled must be boolean")
        if selected is not None and name not in selected:
            continue
        if selected is None and not spec["enabled"]:
            continue
        provider = spec.get("provider")
        if provider == "obs_source":
            if not spec.get("source_name") or not spec.get("source_uuid"):
                raise ValueError(f"{name}: source_name and source_uuid are required")
            uuid.UUID(spec["source_uuid"])
            expected = spec.get("expected_settings")
            if not isinstance(expected, dict) or not expected:
                raise ValueError(f"{name}: expected_settings must identify the actual device")
            quality = spec.get("jpeg_quality", 90)
            if type(quality) is not int or not 1 <= quality <= 100:
                raise ValueError(f"{name}: jpeg_quality must be an integer from 1 to 100")
        elif provider == "avfoundation_unique_id":
            if name != "wrist" or not spec.get("unique_id"):
                raise ValueError("Direct AVFoundation capture requires wrist unique_id")
        else:
            raise ValueError(f"{name}: unsupported camera provider {provider}")
        cameras[name] = spec
    ids = [s.get("source_uuid", s.get("unique_id")) for s in cameras.values()]
    if not cameras or len(set(ids)) != len(ids):
        raise ValueError("No cameras or duplicated OBS source UUID")
    cfg["active_cameras"] = cameras
    return cfg


def image_keys(cfg: dict) -> tuple[str, ...]:
    return tuple(f"observation.images.{name}" for name in cfg["active_cameras"])


def capture_image_shape(cfg: dict) -> tuple[int, int, int]:
    """The shared RGB shape for all enabled OBS sources; legacy default is 480x640."""
    capture = cfg.get("capture", {})
    if not isinstance(capture, dict):
        raise ValueError("capture must be a mapping")
    width, height = capture.get("width", 640), capture.get("height", 480)
    if (type(width) is not int or type(height) is not int or
            (width, height) not in ((640, 480), (448, 336), (320, 240))):
        raise ValueError("capture width/height must be 640x480, 448x336, or 320x240")
    return height, width, 3


class OBSClient:
    def __init__(self, host="127.0.0.1", port=4455, timeout_s=2.0):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("OBS WebSocket must be local")
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError("Install websocket-client into isolated .local/camera-deps") from exc
        self.timeout_s = timeout_s
        self.websocket = websocket
        self.ws = websocket.create_connection(
            f"ws://{host}:{port}", timeout=timeout_s, http_proxy_host=None)
        try:
            hello = json.loads(self.ws.recv())
            if hello.get("op") != 0:
                raise CameraFault("Unexpected OBS WebSocket hello")
            identify = {"rpcVersion": 1}
            auth = hello.get("d", {}).get("authentication")
            if auth:
                password = obs_websocket_password()
                secret = base64.b64encode(hashlib.sha256(
                    (password + auth["salt"]).encode()).digest()).decode()
                identify["authentication"] = base64.b64encode(hashlib.sha256(
                    (secret + auth["challenge"]).encode()).digest()).decode()
            self.ws.send(json.dumps({"op": 1, "d": identify}))
            if json.loads(self.ws.recv()).get("op") != 2:
                raise CameraFault("OBS WebSocket authentication failed")
        except BaseException:
            self.ws.close()
            raise

    def request(self, kind: str, data: dict | None = None) -> dict:
        request_id = uuid.uuid4().hex
        self.ws.send(json.dumps({"op": 6, "d": {"requestType": kind,
            "requestId": request_id, "requestData": data or {}}}))
        # OBS can interleave asynchronous events with request responses.
        # Ignore only event messages; never accept a response for another request.
        deadline = time.monotonic() + self.timeout_s
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraFault(f"OBS {kind}: timed out waiting for response")
                self.ws.settimeout(remaining)
                try:
                    response = json.loads(self.ws.recv())
                except self.websocket.WebSocketTimeoutException as exc:
                    raise CameraFault(f"OBS {kind}: timed out waiting for response") from exc
                if response.get("op") == 5:
                    continue  # Event; not the response to this request.
                result = response.get("d", {})
                if response.get("op") != 7 or result.get("requestId") != request_id:
                    raise CameraFault(
                        f"Unexpected OBS response for {kind}: "
                        f"op={response.get('op')}, requestId={result.get('requestId')}")
                if not result.get("requestStatus", {}).get("result"):
                    raise CameraFault(f"OBS {kind}: {result.get('requestStatus')}")
                return result.get("responseData", {})
        finally:
            self.ws.settimeout(self.timeout_s)

    def close(self):
        self.ws.close()


def verify_source(client: OBSClient, name: str, spec: dict) -> dict:
    inputs = client.request("GetInputList").get("inputs", [])
    matches = [x for x in inputs if x.get("inputName") == spec["source_name"]
               and x.get("inputUuid") == spec["source_uuid"]]
    if len(matches) != 1:
        raise CameraFault(f"{name}: OBS source name/UUID changed or missing")
    settings = client.request("GetInputSettings", {"inputName": spec["source_name"]})
    for key, value in spec["expected_settings"].items():
        if settings.get("inputSettings", {}).get(key) != value:
            raise CameraFault(f"{name}: OBS underlying device setting {key} changed")
    return {"logical_name": name, "source_name": spec["source_name"],
            "source_uuid": spec["source_uuid"], "input_kind": matches[0].get("inputKind")}


class OBSSourceCamera:
    def __init__(self, name: str, spec: dict, connection: dict, collect_timings=False,
                 image_shape=(480, 640, 3)):
        self.name, self.spec, self.connection = name, spec, connection
        self.image_shape = image_shape
        self.client = None
        self.thread = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.timestamp = None
        self.last_sampled_timestamp = None
        self.last_changed = None
        self.last_digest = None
        self.error = None
        self.identity = None
        self.collect_timings = collect_timings
        self.fetch_timings = []
        self.identity_timings = []
        self.last_fetch_timing = None
        self.active_request_started_at = None
        self.content_digest = None

    def _fetch(self):
        started = time.perf_counter()
        with self.lock:
            self.active_request_started_at = started
        data = self.client.request("GetSourceScreenshot", {
            "sourceName": self.spec["source_name"], "imageFormat": "jpg",
            "imageWidth": self.image_shape[1], "imageHeight": self.image_shape[0],
            "imageCompressionQuality": self.spec.get("jpeg_quality", 90)})
        response_at = time.perf_counter()
        payload = data["imageData"].split(",", 1)[1]
        raw = base64.b64decode(payload, validate=True)
        decoded_at = time.perf_counter()
        frame = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).copy()
        if frame.shape != self.image_shape:
            raise CameraFault(f"{self.name}: wrong frame shape {frame.shape}")
        image_decoded_at = time.perf_counter()
        digest = hashlib.sha256(frame.tobytes()).digest()
        done_at = time.perf_counter()
        with self.lock:
            changed = digest != self.content_digest
            if changed:
                self.last_changed = response_at
                # OBS provides only the screenshot response time, not a device
                # capture timestamp. Publish it only when pixels changed so a
                # repeated OBS/DroidCam image cannot masquerade as a new frame.
                self.frame, self.timestamp = frame, response_at
            self.content_digest = digest
            self.last_digest = digest
            self.active_request_started_at = None
            self.last_fetch_timing = {
                "request_started_at": started, "response_at": response_at,
                "decode_finished_at": image_decoded_at,
                "request_ms": (response_at-started)*1000,
                "base64_ms": (decoded_at-response_at)*1000,
                "image_decode_ms": (image_decoded_at-decoded_at)*1000,
            }
            if self.collect_timings:
                self.fetch_timings.append({
                    "request_started_at": started,
                    "response_at": response_at,
                    "request_ms": (response_at-started)*1000,
                    "base64_ms": (decoded_at-response_at)*1000,
                    "image_decode_ms": (image_decoded_at-decoded_at)*1000,
                    "hash_ms": (done_at-image_decoded_at)*1000,
                    "decode_hash_ms": (done_at-decoded_at)*1000,
                    "total_ms": (done_at-started)*1000,
                    "jpeg_bytes": len(raw),
                    "content_digest": digest.hex(),
                    "content_changed": changed,
                })

    def connect(self):
        self.client = OBSClient(**self.connection)
        try:
            self.identity = verify_source(self.client, self.name, self.spec)
            self._fetch()
            self.thread = threading.Thread(target=self._run, daemon=True,
                                           name=f"obs-{self.name}")
            self.thread.start()
        except BaseException:
            self.client.close()
            self.client = None
            raise

    def _run(self):
        last_identity_check = time.perf_counter()
        while not self.stop_event.is_set():
            started = time.perf_counter()
            try:
                if started-last_identity_check >= 2.0:
                    verify_source(self.client, self.name, self.spec)
                    if self.collect_timings:
                        self.identity_timings.append({"started_at": started,
                            "duration_ms": (time.perf_counter()-started)*1000})
                    last_identity_check = started
                self._fetch()
            except BaseException as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                return
            self.stop_event.wait(max(0, .04 - (time.perf_counter() - started)))

    def sample(self, max_age_s=.10, freeze_timeout_s=10.0, transient_wait_s=.05,
               newer_than=None):
        """Wait briefly for a changed, fresh OBS frame; never reuse the stale one."""
        deadline = time.perf_counter() + transient_wait_s
        while True:
            with self.lock:
                error = self.error
                frame = self.frame
                timestamp = self.timestamp
                last_changed = self.last_changed
            if error or frame is None or timestamp is None:
                raise CameraFault(f"{self.name}: {error or 'no frame'}")
            now = time.perf_counter()
            age = now - timestamp
            unused = (timestamp != self.last_sampled_timestamp
                      and (newer_than is None or timestamp > newer_than))
            if age <= max_age_s and unused:
                if last_changed is not None and now-last_changed > freeze_timeout_s:
                    raise CameraFault(
                        f"{self.name}: repeated identical screenshot for {freeze_timeout_s}s"
                    )
                with self.lock:
                    self.last_sampled_timestamp = timestamp
                return frame.copy(), timestamp
            if now >= deadline:
                raise CameraFault(
                    f"{self.name}: stale or reused frame {age:.3f}s after waiting "
                    f"{transient_wait_s:.3f}s for a changed fresh frame"
                )
            time.sleep(min(.002, deadline-now))

    def disconnect(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.client:
            self.client.close()

    def timing_snapshot(self):
        with self.lock:
            return list(self.fetch_timings)

    def identity_timing_snapshot(self):
        return list(self.identity_timings)


class AVFoundationCamera(OBSSourceCamera):
    """Direct USB capture; the native helper opens the exact AVFoundation uniqueID."""

    def __init__(self, name, spec, connection, collect_timings=False,
                 image_shape=(480, 640, 3)):
        super().__init__(name, spec, connection, collect_timings, image_shape)
        self.process = None
        self.native_diagnostics = []
        self.stderr_thread = None

    def _collect_stderr(self):
        for raw in iter(self.process.stderr.readline, b""):
            text = raw.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            with self.lock:
                self.native_diagnostics.append(text)
            print(f"CAMERA {self.name} NATIVE: {text}", file=sys.stderr, flush=True)

    def connect(self):
        source = Path(__file__).with_name("avfoundation_wrist.swift")
        binary = Path("/tmp/so101-avfoundation-wrist")
        if not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime:
            env = os.environ.copy()
            env["CLANG_MODULE_CACHE_PATH"] = "/tmp/so101-clang-cache"
            env["SWIFT_MODULE_CACHE_PATH"] = "/tmp/so101-swift-cache"
            subprocess.run(["swiftc", str(source), "-o", str(binary)],
                           env=env, check=True, capture_output=True)
        self.process = subprocess.Popen([str(binary), self.spec["unique_id"]],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0)
        self.stderr_thread = threading.Thread(target=self._collect_stderr, daemon=True,
                                              name=f"avfoundation-{self.name}-stderr")
        self.stderr_thread.start()
        try:
            deadline = time.perf_counter() + 5.0
            while not self._fetch(initial=True):
                if time.perf_counter() > deadline:
                    raise CameraFault(f"{self.name}: direct capture startup frames remained stale")
            self.identity = {"logical_name": self.name, "input_kind": "avfoundation_direct",
                             "unique_id": self.spec["unique_id"],
                             "native_diagnostics": self.native_diagnostics}
            self.thread = threading.Thread(target=self._run, daemon=True,
                                           name=f"avfoundation-{self.name}")
            self.thread.start()
            deadline = time.perf_counter() + 3.0
            while time.perf_counter() < deadline:
                with self.lock:
                    age = (time.perf_counter() - self.timestamp
                           if self.timestamp is not None else float("inf"))
                    error = self.error
                if error:
                    raise CameraFault(f"{self.name}: {error}")
                if age <= .075:
                    break
                time.sleep(.01)
            else:
                raise CameraFault(f"{self.name}: direct capture did not produce a fresh frame")
        except BaseException:
            self.disconnect()
            raise

    def _read_exact(self, length, timeout_s=2.0):
        data = bytearray()
        deadline = time.perf_counter() + timeout_s
        while len(data) < length:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise CameraFault(f"{self.name}: AVFoundation frame timeout")
            part = os.read(self.process.stdout.fileno(), length - len(data))
            if not part:
                raise CameraFault(f"{self.name}: AVFoundation capture exited")
            data.extend(part)
        return data

    def _fetch(self, initial=False):
        started = time.perf_counter()
        with self.lock:
            self.active_request_started_at = started
        stamp_ns, width, height, length = struct.unpack(
            "<QIII", self._read_exact(20, timeout_s=8.0 if initial else 2.0))
        if not (0 < width <= 1920 and 0 < height <= 1080 and length == width * height * 4):
            raise CameraFault(f"{self.name}: invalid AVFoundation frame header")
        raw = self._read_exact(length)
        received_at = time.perf_counter()
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
        rgb = bgra[:, :, [2, 1, 0]]
        if (height, width) != self.image_shape[:2]:
            frame = np.asarray(Image.fromarray(rgb).resize(
                (self.image_shape[1], self.image_shape[0]), Image.Resampling.BILINEAR)).copy()
        else:
            frame = rgb.copy()
        decoded_at = time.perf_counter()
        # macOS Python perf_counter and Darwin CLOCK_MONOTONIC have distinct
        # epochs. Convert the helper's acquisition clock to the rig's clock.
        clock_offset = time.perf_counter() - time.clock_gettime(time.CLOCK_MONOTONIC)
        capture_at = stamp_ns / 1e9 + clock_offset
        if initial and received_at - capture_at > .1:
            with self.lock:
                self.active_request_started_at = None
            return False  # Drain startup frames; none is exposed to sample().
        # A single delayed capture is not necessarily camera failure.
        # Discard it without updating the published frame or its timestamp.
        # CameraRig.sample_all() still enforces fresh frames (100 ms) and
        # cross-camera skew (75 ms); persistent delays therefore remain faults.
        if capture_at > received_at:
            raise CameraFault(f"{self.name}: invalid AVFoundation timestamp "
                              f"(age={received_at-capture_at:.3f}s)")
        if received_at - capture_at > .1:
            with self.lock:
                self.active_request_started_at = None
            return False
        # Like OBS, use the host receipt time for cross-camera sample/skew checks.
        # Retain the earlier AVFoundation callback time in timing diagnostics.
        timestamp = received_at
        digest = hashlib.sha256(frame.tobytes()).digest()
        with self.lock:
            changed = digest != self.content_digest
            if changed:
                self.last_changed = timestamp
            self.content_digest = self.last_digest = digest
            self.frame, self.timestamp = frame, timestamp
            self.active_request_started_at = None
            self.last_fetch_timing = {
                "request_started_at": started, "response_at": received_at,
                "decode_finished_at": decoded_at,
                "request_ms": (received_at-started)*1000,
                "image_decode_ms": (decoded_at-received_at)*1000,
                "capture_to_host_ms": (received_at-capture_at)*1000,
            }
            if self.collect_timings:
                self.fetch_timings.append({
                    "request_started_at": started, "response_at": received_at,
                    "sampled_at_monotonic": timestamp,
                    "capture_to_host_ms": (received_at - capture_at) * 1000,
                    "request_ms": (received_at - started) * 1000,
                    "base64_ms": 0.0,
                    "image_decode_ms": (decoded_at - received_at) * 1000,
                    "hash_ms": 0.0,
                    "decode_hash_ms": (decoded_at - received_at) * 1000,
                    "total_ms": (decoded_at - started) * 1000,
                    "jpeg_bytes": 0, "content_digest": digest.hex(),
                    "content_changed": changed,
                })
        return True

    def sample(self, max_age_s=.10, freeze_timeout_s=10.0, transient_wait_s=.05,
               newer_than=None):
        """Wait briefly for a fresh direct-capture frame; never return the stale one."""
        deadline = time.perf_counter() + transient_wait_s
        while True:
            with self.lock:
                error = self.error
                frame = self.frame
                timestamp = self.timestamp
                last_changed = self.last_changed
            if error or frame is None or timestamp is None:
                raise CameraFault(f"{self.name}: {error or 'no frame'}")
            now = time.perf_counter()
            age = now - timestamp
            if age <= max_age_s and (newer_than is None or timestamp > newer_than):
                if last_changed is not None and now-last_changed > freeze_timeout_s:
                    raise CameraFault(
                        f"{self.name}: repeated identical screenshot for {freeze_timeout_s}s"
                    )
                return frame.copy(), timestamp
            if now >= deadline:
                raise CameraFault(
                    f"{self.name}: stale frame {age:.3f}s after waiting "
                    f"{transient_wait_s:.3f}s for a fresh frame"
                )
            # The capture thread publishes only frames whose capture-to-host age
            # is <=100 ms. A delayed frame is discarded in _fetch(), so this
            # loop cannot relabel or return it as fresh.
            time.sleep(min(.002, deadline-now))

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self._fetch()
            except BaseException as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                return

    def disconnect(self):
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        if self.thread:
            self.thread.join(timeout=3)
        if self.process:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        if self.stderr_thread:
            self.stderr_thread.join(timeout=1)


class CameraRig:
    def __init__(self, cfg: dict, collect_timings=False):
        self.cfg = cfg
        self.is_connected = False
        connection = cfg.get("obs", {})
        self.image_shape = capture_image_shape(cfg)
        self.cameras = {name: (AVFoundationCamera if spec["provider"] == "avfoundation_unique_id"
                               else OBSSourceCamera)(name, spec, connection, collect_timings,
                                                      self.image_shape)
                        for name, spec in cfg["active_cameras"].items()}

    def connect(self):
        connected = []
        try:
            for camera in self.cameras.values():
                camera.connect()
                connected.append(camera)
                print(f"CAMERA {camera.name}: {camera.identity}", flush=True)
            self.is_connected = True
        except BaseException:
            for camera in reversed(connected):
                camera.disconnect()
            raise

    def sample_all(self, pairing_wait_s=.12):
        """Build one bounded pair using comparable host-availability times.\n\n        The extra wait absorbs transient scheduling jitter; it does not relax\n        the 100ms freshness bound, 75ms skew, or reused-frame rejection.

        OBS exposes no device capture timestamp, so pairing it with AVFoundation's
        device callback time would mix clock semantics. Both camera adapters publish
        the host receipt/response time here. This bounds availability skew, not true
        optical capture skew.
        """
        deadline = time.perf_counter() + pairing_wait_s
        samples = {}
        for name, camera in self.cameras.items():
            samples[name] = camera.sample(
                transient_wait_s=max(0., deadline-time.perf_counter()))
        while True:
            times = {name: stamp for name, (_, stamp) in samples.items()}
            now = time.perf_counter()
            max_age = max(now-stamp for stamp in times.values())
            skew = max(times.values())-min(times.values())
            if max_age <= .100 and skew <= .075:
                return {name: frame for name, (frame, _) in samples.items()}, times
            older = min(times, key=times.get)
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                raise CameraFault(
                    f"Camera host-availability pairing failed within {pairing_wait_s:.3f}s; "
                    f"max_age={max_age:.3f}s, skew={skew:.3f}s, timestamps={times}")
            samples[older] = self.cameras[older].sample(
                transient_wait_s=remaining, newer_than=times[older])

    def timing_brief(self):
        """Small, image-free snapshot for control-cycle fault diagnostics."""
        now = time.perf_counter()
        result = {}
        for name, camera in self.cameras.items():
            with camera.lock:
                stamp = camera.timestamp
                result[name] = {
                    "latest_image_timestamp": stamp,
                    "image_age_ms": None if stamp is None else (now-stamp)*1000,
                    "active_request_started_at": camera.active_request_started_at,
                    "active_request_age_ms": None if camera.active_request_started_at is None
                                             else (now-camera.active_request_started_at)*1000,
                    "last_fetch": None if camera.last_fetch_timing is None
                                  else dict(camera.last_fetch_timing),
                    "error": camera.error,
                }
        return result

    def identity_snapshot(self):
        result = {}
        for name, camera in self.cameras.items():
            with camera.lock:
                result[name] = {
                    "identity": None if camera.identity is None else dict(camera.identity),
                    "native_diagnostics": list(getattr(camera, "native_diagnostics", ())),
                }
        return result

    def read_latest(self):
        return self.sample_all()[0]

    def disconnect(self):
        for camera in self.cameras.values():
            camera.disconnect()
        self.is_connected = False
