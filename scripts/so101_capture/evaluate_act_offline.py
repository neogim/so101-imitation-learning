"""Evaluate a local ACT checkpoint on held-out recorded episodes, without hardware."""

import argparse
import json
import math
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    parser.add_argument("--episode-ids", help="Comma-separated episode IDs; default is held-out split")
    parser.add_argument("--n-action-steps", type=int, choices=(1, 5, 10, 50), default=50)
    args = parser.parse_args()
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    if args.output.exists():
        raise FileExistsError(args.output)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.utils import prepare_observation_for_inference

    root = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = args.device
    if (config.chunk_size, config.n_action_steps, config.temporal_ensemble_coeff) != (50, 50, None):
        raise ValueError("Unexpected ACT chunk configuration")
    config.n_action_steps = args.n_action_steps  # inference only; checkpoint remains untouched
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process")
    expected_input = {"observation.images.iphone", "observation.state"}
    if set(config.input_features) != expected_input or set(config.output_features) != {"action"}:
        raise ValueError("Unexpected model features")
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    n_episodes = info["total_episodes"]
    n_eval = math.ceil(n_episodes * 0.1)
    eval_episodes = set(range(n_episodes - n_eval, n_episodes))
    if args.episode_ids is not None:
        eval_episodes = {int(x) for x in args.episode_ids.split(",")}
        if not eval_episodes or any(ep < 0 or ep >= n_episodes for ep in eval_episodes):
            raise ValueError("Episode ID outside dataset")
    rows = pq.read_table(next((root / "data").rglob("*.parquet")), columns=[
        "episode_index", "frame_index", "observation.state", "action"
    ]).to_pylist()
    video_files = sorted((root / "videos/observation.images.iphone").rglob("*.mp4"))
    if len(video_files) != 1 or len(rows) != info["total_frames"]:
        raise ValueError("Dataset frame/video layout differs from the validated source")

    names = info["features"]["action"]["names"]
    errors = []
    first_frame_errors = []
    predictions = []
    targets = []
    raw_predictions = []
    per_episode = {}
    current_episode = None
    total_video_frames = 0
    with av.open(str(video_files[0])) as video, torch.inference_mode():
        for row, frame in zip(rows, video.decode(video=0), strict=True):
            total_video_frames += 1
            ep = int(row["episode_index"])
            if ep not in eval_episodes:
                continue
            if ep != current_episode:
                policy.reset()
                current_episode = ep
                per_episode[ep] = 0
            if args.max_frames_per_episode and per_episode[ep] >= args.max_frames_per_episode:
                continue
            obs = prepare_observation_for_inference({
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                "observation.images.iphone": frame.to_ndarray(format="rgb24"),
            }, torch.device(args.device))
            action_normalized = policy.select_action(preprocessor(obs))
            action = postprocessor(action_normalized)
            pred = action.detach().cpu().numpy().reshape(-1)
            raw = action_normalized.detach().cpu().numpy().reshape(-1)
            target = np.asarray(row["action"], dtype=np.float32).reshape(-1)
            if pred.shape != (6,) or raw.shape != (6,) or target.shape != (6,):
                raise ValueError("Action shape is not six-dimensional")
            if not (np.isfinite(pred).all() and np.isfinite(raw).all() and np.isfinite(target).all()):
                raise ValueError("Nonfinite action")
            errors.append(np.abs(pred - target))
            if int(row["frame_index"]) == 0:
                first_frame_errors.append(np.abs(pred - target))
            predictions.append(pred)
            raw_predictions.append(raw)
            targets.append(target)
            per_episode[ep] += 1
    if total_video_frames != len(rows) or set(per_episode) != eval_episodes:
        raise ValueError("Validation episodes or video frames incomplete")
    err = np.stack(errors)
    pred = np.stack(predictions)
    raw = np.stack(raw_predictions)
    target = np.stack(targets)
    training_actions = np.asarray(
        [row["action"] for row in rows if int(row["episode_index"]) not in eval_episodes],
        dtype=np.float32,
    )
    training_min = training_actions.min(axis=0)
    training_max = training_actions.max(axis=0)
    result = {
        "dataset": str(root), "checkpoint": str(checkpoint), "device": args.device,
        "eval_episodes": sorted(eval_episodes), "frames_by_episode": per_episode,
        "n_action_steps": args.n_action_steps,
        "frames_evaluated": len(errors), "action_shape": [6],
        "units": ["degrees"] * 5 + ["normalized_0_100"],
        "joint_mae": {name: float(value) for name, value in zip(names, err.mean(axis=0), strict=True)},
        "first_frame_joint_mae": {name: float(value) for name, value in zip(
            names, np.stack(first_frame_errors).mean(axis=0), strict=True)},
        "prediction_min": {name: float(value) for name, value in zip(names, pred.min(axis=0), strict=True)},
        "prediction_max": {name: float(value) for name, value in zip(names, pred.max(axis=0), strict=True)},
        "target_min": {name: float(value) for name, value in zip(names, target.min(axis=0), strict=True)},
        "target_max": {name: float(value) for name, value in zip(names, target.max(axis=0), strict=True)},
        "training_action_min": {name: float(value) for name, value in zip(names, training_min, strict=True)},
        "training_action_max": {name: float(value) for name, value in zip(names, training_max, strict=True)},
        "outside_training_range_frames": {
            name: int(value) for name, value in zip(
                names, ((pred < training_min) | (pred > training_max)).sum(axis=0), strict=True
            )
        },
        "gripper_outside_0_100_frames": int(((pred[:, 5] < 0) | (pred[:, 5] > 100)).sum()),
        "normalized_prediction_first": raw[0].tolist(),
        "denormalized_prediction_first": pred[0].tolist(),
        "teacher_action_first": target[0].tolist(),
        "note": "Teacher-forced offline MAE; it does not measure closed-loop task success or hardware safety.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
