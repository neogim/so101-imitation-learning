"""Offline ACT replay from a local v3 dataset. No robot/camera connection code."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import av
import numpy as np
import pyarrow.parquet as pq
import torch

from act_action_guard import ActionGuard, GuardConfig, NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limits-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--n-action-steps", type=int, choices=(1, 5, 10, 50), default=50)
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_frames_per_episode < 0:
        raise ValueError("max-frames-per-episode must be nonnegative")
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.utils import prepare_observation_for_inference

    dataset, checkpoint = args.dataset.resolve(), args.checkpoint.resolve()
    if "muhi-act-success-055-168x224" not in dataset.name or "018000" not in checkpoint.parts:
        raise ValueError("Expected 55-episode dataset and 018000 checkpoint; refusing model mix-up")
    info = json.loads((dataset / "meta/info.json").read_text())
    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.device = args.device
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if (cfg.chunk_size, cfg.n_action_steps, cfg.temporal_ensemble_coeff) != (50, 50, None):
        raise ValueError("ACT chunk selection differs from the trained checkpoint")
    cfg.n_action_steps = args.n_action_steps  # inference only; saved checkpoint is untouched
    if set(cfg.input_features) != {"observation.state", "observation.images.iphone"} or set(cfg.output_features) != {"action"}:
        raise ValueError("Unexpected ACT feature contract")
    policy = ACTPolicy.from_pretrained(checkpoint, config=cfg, local_files_only=True, strict=True)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    limits_doc = json.loads(args.limits_json.read_text())
    if limits_doc.get("hardware_authorized") is not False:
        raise ValueError("This dry-run requires hardware_authorized=false")
    if limits_doc.get("units") != ["degrees"] * 5 + ["normalized_0_100"]:
        raise ValueError("Joint unit mismatch")
    guard = ActionGuard(GuardConfig(
        limits=tuple(tuple(limits_doc["joint_limits"][n]) for n in NAMES),
        speed_per_s=tuple(limits_doc["speed_per_s"][n] for n in NAMES),
        gripper_closing_lead=limits_doc["gripper_closing_lead"],
        period_s=1 / info["fps"],
    ))
    if info["fps"] != 20 or info["total_episodes"] != 55:
        raise ValueError("Unexpected dataset FPS or episode count")
    eval_eps = set(range(info["total_episodes"] - math.ceil(info["total_episodes"] * .1), info["total_episodes"]))
    files = sorted((dataset / "data").rglob("*.parquet"))
    videos = sorted((dataset / "videos/observation.images.iphone").rglob("*.mp4"))
    if len(files) != 1 or len(videos) != 1:
        raise ValueError("Unexpected dataset layout")
    rows = pq.read_table(files[0], columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"]).to_pylist()
    if len(rows) != info["total_frames"]:
        raise ValueError("Parquet frame count mismatch")
    args.output.mkdir(parents=True)
    csv_path = args.output / "actions.csv"
    headings = ["episode_index", "frame_index", "timestamp_s", "chunk_start_frame", "chunk_offset", "chunk_refreshed",
                "inference_ms", "cycle_work_ms", "guard_latched", "reasons"]
    for prefix in ("measured", "teacher", "predicted", "limited"):
        headings += [f"{prefix}.{name}" for name in NAMES]
    counts, negative, negative_by_episode, raw_min, raw_max, deltas = 0, [], {}, np.full(6, np.inf), np.full(6, -np.inf), []
    refresh_ms, all_ms, work_ms, previous_predicted, previous_limited, previous_ep = [], [], [], None, None, None
    chunk_start, latch_frames, warnings = None, {}, {}
    with csv_path.open("w", newline="") as output, av.open(str(videos[0])) as video, torch.inference_mode():
        writer = csv.DictWriter(output, fieldnames=headings)
        writer.writeheader()
        decoded = 0
        frame_iter = video.decode(video=0)
        for row in rows:
            cycle_start = time.perf_counter()
            frame = next(frame_iter, None)
            if frame is None:
                raise ValueError("MP4 ended before Parquet")
            decoded += 1
            ep, idx = int(row["episode_index"]), int(row["frame_index"])
            if ep not in eval_eps or (args.max_frames_per_episode and idx >= args.max_frames_per_episode):
                continue
            state = np.asarray(row["observation.state"], dtype=np.float32)
            teacher = np.asarray(row["action"], dtype=np.float32)
            if ep != previous_ep:
                policy.reset()
                guard.reset_for_new_episode(state)
                previous_predicted = previous_limited = None
                chunk_start = idx
                previous_ep = ep
            # LeRobot select_action refreshes a 50-action queue only when empty.
            refreshed = len(policy._action_queue) == 0
            if refreshed:
                chunk_start = idx
            obs = prepare_observation_for_inference({
                "observation.state": state,
                "observation.images.iphone": frame.to_ndarray(format="rgb24"),
            }, torch.device(args.device))
            start = time.perf_counter()
            normalized = policy.select_action(pre(obs))
            predicted = post(normalized).detach().cpu().numpy().reshape(-1)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if predicted.shape != (6,) or not np.isfinite(predicted).all():
                raise ValueError("Invalid model action")
            limited, reasons = guard.propose(predicted, state)
            cycle_work_ms = (time.perf_counter() - cycle_start) * 1000
            if limited is None and ep not in latch_frames:
                latch_frames[ep] = idx
            for reason in reasons:
                warnings[reason] = warnings.get(reason, 0) + 1
            item = dict(episode_index=ep, frame_index=idx, timestamp_s=row["timestamp"],
                        chunk_start_frame=chunk_start, chunk_offset=idx - chunk_start,
                        chunk_refreshed=int(refreshed), inference_ms=round(elapsed_ms, 3),
                        cycle_work_ms=round(cycle_work_ms, 3),
                        guard_latched=int(limited is None), reasons=";".join(reasons))
            for prefix, values in (("measured", state), ("teacher", teacher), ("predicted", predicted)):
                item.update({f"{prefix}.{n}": float(values[i]) for i, n in enumerate(NAMES)})
            if limited is not None:
                item.update({f"limited.{n}": float(limited[i]) for i, n in enumerate(NAMES)})
            writer.writerow(item)
            if previous_predicted is not None:
                deltas.append(np.abs(predicted - previous_predicted))
            previous_predicted = predicted
            if limited is not None:
                previous_limited = limited
            counts += 1
            raw_min = np.minimum(raw_min, predicted)
            raw_max = np.maximum(raw_max, predicted)
            all_ms.append(elapsed_ms)
            work_ms.append(cycle_work_ms)
            if refreshed:
                refresh_ms.append(elapsed_ms)
            if predicted[5] < 0:
                negative.append(float(predicted[5]))
                negative_by_episode[ep] = negative_by_episode.get(ep, 0) + 1
        if decoded != len(rows) or next(frame_iter, None) is not None:
            raise ValueError("MP4/Parquet frame count mismatch")
    summary = {
        "dataset": str(dataset), "checkpoint": str(checkpoint), "limits": str(args.limits_json.resolve()),
        "frames": counts, "evaluated_episodes": sorted(eval_eps), "fps": info["fps"],
        "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
        "chunk_update_period_s_nominal": cfg.n_action_steps / info["fps"],
        "chunk_refresh_count": len(refresh_ms),
        "inference_ms_mean": float(np.mean(all_ms)), "inference_ms_max": float(np.max(all_ms)),
        "cycle_work_ms_mean": float(np.mean(work_ms)), "cycle_work_ms_max": float(np.max(work_ms)),
        "chunk_refresh_ms_mean": float(np.mean(refresh_ms)), "chunk_refresh_ms_max": float(np.max(refresh_ms)),
        "prediction_min": dict(zip(NAMES, raw_min.tolist())), "prediction_max": dict(zip(NAMES, raw_max.tolist())),
        "max_raw_action_delta_per_frame": (
            dict(zip(NAMES, np.max(deltas, axis=0).tolist())) if deltas else None
        ),
        "gripper_negative": {"count": len(negative), "min": min(negative, default=None),
                             "by_episode": negative_by_episode,
                             "quantiles": np.quantile(negative, [0, .1, .25, .5, .75, .9, 1]).tolist() if negative else []},
        "guard_first_latch_frame_by_episode": latch_frames,
        "reason_counts": warnings,
        "note": "Offline replay only. Recorded states are teacher-trajectory states, not responses to proposed model goals. No Action was sent. Position lag is not a force measure.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
