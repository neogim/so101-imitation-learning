"""SO-101 ACT observation-only probe. Never configures motors or sends Actions.

Hardware mode opens only the configured Follower serial port and OBS index 1.
It does not call follower.connect(), configure(), send_action(), sync_write(),
enable_torque(), disable_torque(), or a torque-changing disconnect().
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from act_action_guard import ActionGuard, GuardConfig, NAMES


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/so101/muhi-act-success-055-168x224"
CHECKPOINT = ROOT / "outputs/so101-act-muhi-055-step20000-resume14000/checkpoints/018000/pretrained_model"
FOLLOWER_PORT = "/dev/tty.usbmodem5B7B0098201"
PERIOD = .05


class SyntheticReader:
    def __init__(self, state):
        from record_episode import SyntheticCamera
        self.state = np.asarray(state, dtype=np.float32)
        self.camera = SyntheticCamera()
        self.last_torque = {n.removesuffix(".pos"): 1 for n in NAMES}

    def open(self):
        return {}

    def read(self):
        t0 = time.perf_counter()
        state = self.state.copy()
        t1 = time.perf_counter()
        torque = True
        t2 = time.perf_counter()
        image = self.camera.read_latest()
        t3 = time.perf_counter()
        return state, image, torque, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, 0.0

    def limits(self, offline_limits):
        return tuple(tuple(offline_limits[n]) for n in NAMES)

    def close(self):
        pass


class ReadOnlyHardwareReader:
    def __init__(self):
        self.robot = None
        self.camera = None
        self.calibration_hash = None
        self.last_torque = None

    def open(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from obs_camera import OBSCamera
        from record_episode import calibration_paths, deny_calibration
        if not Path(FOLLOWER_PORT).exists():
            raise FileNotFoundError(f"Follower port absent: {FOLLOWER_PORT}")
        calibration = calibration_paths()[0]
        if not calibration.is_file():
            raise FileNotFoundError(f"Follower calibration absent: {calibration}")
        self.calibration_hash = hashlib.sha256(calibration.read_bytes()).hexdigest()
        self.robot = SO101Follower(SO101FollowerConfig(
            port=FOLLOWER_PORT, id="my_follower_arm", use_degrees=True,
            disable_torque_on_disconnect=False,
        ))
        self.robot.calibrate = deny_calibration
        self.robot._save_calibration = deny_calibration
        self.robot.bus.write_calibration = deny_calibration
        # Bus.connect() opens the serial port and handshakes via reads.
        # Robot.connect() is deliberately forbidden: it calls configure().
        self.robot.bus.connect()
        if not self.robot.is_calibrated:
            raise RuntimeError("Follower calibration mismatch; read-only probe stopped")
        self.camera = OBSCamera()  # hard-coded, previously confirmed index 1
        self.camera.connect()
        return {"follower_port": FOLLOWER_PORT, "calibration_sha256": self.calibration_hash,
                "camera_index": 1, "camera_rotation_degrees": 0, "use_degrees": True}

    def limits(self, _):
        # Existing calibrated motor endpoints, converted with installed LeRobot code.
        from capture_session import calibration_limits
        values = calibration_limits(self.robot)
        values["gripper.pos"] = (max(0, values["gripper.pos"][0]),
                                 min(100, values["gripper.pos"][1]))
        return tuple(tuple(values[n]) for n in NAMES)

    def read(self):
        if self.robot is None or self.camera is None:
            raise RuntimeError("Reader not open")
        t0 = time.perf_counter()
        values = self.robot.bus.sync_read("Present_Position", num_retry=0)
        t1 = time.perf_counter()
        torque = self.robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=0)
        self.last_torque = torque.copy()
        t2 = time.perf_counter()
        if set(values) != {n.removesuffix(".pos") for n in NAMES}:
            raise RuntimeError("Unexpected Follower joint order/keys")
        state = np.asarray([values[n.removesuffix(".pos")] for n in NAMES], dtype=np.float32)
        torque_ok = len(torque) == 6 and all(value == 1 for value in torque.values())
        image = self.camera.read_latest(max_age_ms=100)
        t3 = time.perf_counter()
        with self.camera.frame_lock:
            image_timestamp = self.camera.latest_timestamp
        camera_age_ms = (t3 - image_timestamp) * 1000 if image_timestamp is not None else math.inf
        return state, image, torque_ok, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, camera_age_ms

    def close(self):
        errors = []
        if self.camera is not None and self.camera.is_connected:
            try:
                self.camera.disconnect()
            except Exception as exc:
                errors.append(f"camera disconnect: {exc}")
        if self.robot is not None and self.robot.bus.is_connected:
            try:
                self.robot.bus.disconnect(disable_torque=False)
            except Exception as exc:
                errors.append(f"serial close: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))


def load_policy(device, checkpoint=None, image_keys=('observation.images.iphone',)):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    selected_checkpoint = Path(checkpoint).resolve() if checkpoint is not None else CHECKPOINT
    if not selected_checkpoint.is_dir() or (checkpoint is None and not DATASET.is_dir()):
        raise FileNotFoundError("Selected ACT checkpoint or pinned 55-demo dataset missing")
    config = PreTrainedConfig.from_pretrained(selected_checkpoint, local_files_only=True)
    config.device = device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if (config.chunk_size, config.n_action_steps, config.temporal_ensemble_coeff) != (50, 50, None):
        raise ValueError("Unexpected ACT chunk configuration")
    if (set(config.input_features) != {"observation.state", *image_keys}
            or set(config.output_features) != {"action"}
            or tuple(config.input_features["observation.state"].shape) != (6,)
            or any(tuple(config.input_features[key].shape) != (3, 168, 224)
                   for key in image_keys)):
        raise ValueError("ACT feature mismatch")
    policy = ACTPolicy.from_pretrained(selected_checkpoint, config=config, local_files_only=True, strict=True)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(selected_checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def read_only_cycle(reader, policy, pre, post, guard, device, frame_index):
    from lerobot.policies.utils import prepare_observation_for_inference
    t0 = time.perf_counter()
    state, image, torque_ok, state_ms, torque_ms, camera_ms, age_ms = reader.read()
    t1 = time.perf_counter()
    if state.shape != (6,) or not np.isfinite(state).all():
        raise RuntimeError("Invalid six-axis Follower state")
    if image.shape != (480, 640, 3) or image.dtype != np.uint8 or age_ms > 100:
        raise RuntimeError(f"Unexpected/stale OBS RGB image: {image.shape}, {image.dtype}, {age_ms:.1f}ms")
    # Identical to resize_act_dataset.py: PIL BILINEAR, no crop/flip/rotation.
    small = np.array(Image.fromarray(image).resize((224, 168), Image.Resampling.BILINEAR), copy=True)
    t2 = time.perf_counter()
    obs = prepare_observation_for_inference({"observation.state": state,
                                             "observation.images.iphone": small}, torch.device(device))
    t3 = time.perf_counter()
    refreshed = len(policy._action_queue) == 0
    normalized = policy.select_action(pre(obs))
    t4 = time.perf_counter()
    predicted = post(normalized).detach().cpu().numpy().reshape(-1)
    t5 = time.perf_counter()
    if predicted.shape != (6,) or not np.isfinite(predicted).all():
        raise RuntimeError("Nonfinite or malformed ACT Action")
    if guard.last_goal is None:
        guard.reset_for_new_episode(state)
    # Hypothetical only: live arm does not follow these candidate goals.
    limited, reasons = guard.propose(predicted, state, sample_age_s=age_ms / 1000)
    t6 = time.perf_counter()
    timings = dict(state_read_ms=state_ms, torque_read_ms=torque_ms, camera_read_ms=camera_ms,
                   camera_age_ms=age_ms, resize_ms=(t2-t1)*1000, prepare_ms=(t3-t2)*1000,
                   model_ms=(t4-t3)*1000, post_ms=(t5-t4)*1000, guard_ms=(t6-t5)*1000,
                   total_work_ms=(t6-t0)*1000)
    return state, image, small, torque_ok, predicted, limited, reasons, refreshed, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "hardware"), default="synthetic")
    parser.add_argument("--hardware-read-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--n-action-steps", type=int, choices=(1, 5, 10, 50), default=50,
                        help="Observation-only replanning interval; checkpoint weights are unchanged")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source == "hardware" and not args.hardware_read_only:
        raise ValueError("Hardware observation requires --hardware-read-only")
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        raise ValueError("Observation duration must be in (0, 60] seconds")
    if args.output.exists():
        raise FileExistsError(args.output)
    policy, pre, post = load_policy(args.device)
    policy.config.n_action_steps = args.n_action_steps
    if args.source == "hardware":
        reader = ReadOnlyHardwareReader()
        offline_limits = None
    else:
        limits_doc = json.loads((ROOT / "outputs/so101-act-muhi-055-step20000-resume14000/dry_run_limits.json").read_text())
        # Synthetic initial state within the validated training envelope.
        reader = SyntheticReader([-60, -80, 80, 45, -90, 20])
        offline_limits = limits_doc["joint_limits"]
    metadata, rows, fault, limits, start = {}, [], None, None, None
    args.output.mkdir(parents=True)
    try:
        metadata = reader.open()
        limits = reader.limits(offline_limits)
        guard = ActionGuard(GuardConfig(limits=limits, speed_per_s=(30,) * 6,
                                          gripper_closing_lead=None, period_s=PERIOD))
        policy.reset()
        start = time.perf_counter()
        previous_tick = None
        count = math.ceil(args.seconds / PERIOD)
        with torch.inference_mode():
            for index in range(count):
                target_time = start + index * PERIOD
                time.sleep(max(0, target_time-time.perf_counter()))
                tick = time.perf_counter()
                state, image, small, torque_ok, predicted, limited, reasons, refreshed, timings = read_only_cycle(
                    reader, policy, pre, post, guard, args.device, index)
                if index == 0:
                    Image.fromarray(image).save(args.output / "observed_rgb_640x480.png")
                    Image.fromarray(small).save(args.output / "model_rgb_224x168.png")
                row = dict(frame_index=index, elapsed_s=tick-start,
                           chunk_start_frame=(index//args.n_action_steps)*args.n_action_steps,
                           chunk_offset=index%args.n_action_steps, chunk_refreshed=int(refreshed),
                           torque_all_enabled=int(torque_ok),
                           hypothetical_guard_latched=int(limited is None), reasons=";".join(reasons),
                           loop_interval_ms=None if previous_tick is None else (tick-previous_tick)*1000,
                           **timings)
                previous_tick = tick
                for label, values in (("state", state), ("predicted", predicted)):
                    row.update({f"{label}.{name}": float(values[i]) for i, name in enumerate(NAMES)})
                if limited is not None:
                    row.update({f"hypothetical_limited.{name}": float(limited[i]) for i, name in enumerate(NAMES)})
                row["cycle_elapsed_ms"] = (time.perf_counter()-tick)*1000
                rows.append(row)
    except BaseException as exc:
        fault = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            reader.close()  # disable_torque=False; do not change motor state
        except BaseException as exc:
            fault = fault or f"Cleanup {type(exc).__name__}: {exc}"
        fields = ["frame_index", "elapsed_s", "chunk_start_frame", "chunk_offset", "chunk_refreshed",
                  "torque_all_enabled",
                  "hypothetical_guard_latched", "reasons", "state_read_ms", "torque_read_ms",
                  "camera_read_ms", "camera_age_ms", "resize_ms", "prepare_ms", "model_ms",
                  "post_ms", "guard_ms", "total_work_ms", "cycle_elapsed_ms", "loop_interval_ms"]
        for label in ("state", "predicted", "hypothetical_limited"):
            fields.extend(f"{label}.{name}" for name in NAMES)
        with (args.output / "observations.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary = {"source": args.source, "device": args.device,
                   "dataset": str(DATASET), "checkpoint": str(CHECKPOINT),
                   "n_action_steps": args.n_action_steps,
                   "frames": len(rows), "target_fps": 20, "fault": fault,
                   "metadata": metadata, "limits": dict(zip(NAMES, limits)) if limits is not None else None,
                   "last_torque_registers": reader.last_torque,
                   "torque_disabled_frames": sum(not r["torque_all_enabled"] for r in rows),
                   "recorded_metrics": [key for key in fields if key.endswith("_ms")],
                   "note": "Observation only. No Goal_Position or Action write. Hypothetical tracking differences cannot be interpreted as live tracking errors."}
        if rows:
            summary["timing_max_ms"] = {k: max(r[k] for r in rows if r[k] is not None)
                                        for k in fields if k.endswith("_ms")}
            summary["over_50ms_cycles"] = sum(r["cycle_elapsed_ms"] > 50 for r in rows)
            summary["chunk_refresh_count"] = sum(r["chunk_refreshed"] for r in rows)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if fault:
        raise RuntimeError(fault)


if __name__ == "__main__":
    main()
