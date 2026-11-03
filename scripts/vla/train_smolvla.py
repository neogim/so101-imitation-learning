#!/usr/bin/env python3
"""Offline-only VLA training launcher. Does not connect to robot hardware."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data/so101/muhi-side-wrist-success-053-20261025-168x224"

def dataset_preflight(path: Path) -> dict:
    if not path.is_dir():
        raise ValueError(f"Dataset missing: {path}")
    info_path = path / "meta/info.json"
    if not info_path.is_file():
        raise ValueError(f"Not a LeRobot dataset; meta/info.json missing: {path}")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    for name in ("action", "observation.state"):
        if name not in features:
            raise ValueError(f"Dataset missing required feature: {name}")
    cameras = sorted(k for k in features if k.startswith("observation.images."))
    if not cameras:
        raise ValueError("Dataset has no image observations")
    return {"episodes": info.get("total_episodes"), "frames": info.get("total_frames"), "cameras": cameras,
            "state": features.get("observation.state"), "action": features.get("action")}

POLICY_NAME = "smolvla"

def policy_args(args, stats):
    expected = {"observation.images.side", "observation.images.wrist"}
    if set(stats["cameras"]) != expected:
        raise ValueError(f"Expected side/wrist cameras, got {stats['cameras']}")
    return ["--policy.path=lerobot/smolvla_base",
            f"--policy.device={args.device}",
            '--rename_map={"observation.images.side":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}']

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--repo-id", default="local/muhi_vla_success_053_2cam")
    p.add_argument("--device", default="cuda", choices=("cuda", "mps", "cpu"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true", help="Actually start offline training (default: print only)")
    args = p.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        p.error("--steps and --batch-size must be positive")
    try:
        dataset = args.dataset.expanduser().resolve(strict=True)
        stats = dataset_preflight(dataset)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    if args.output is None:
        p.error("--output is required; choose an unused output directory")
    output = args.output.expanduser().resolve()
    if output.exists():
        p.error(f"Output already exists; refusing overwrite: {output}")
    if not (shutil.which("lerobot-train") or not args.execute):
        p.error("lerobot-train not found in current environment")
    cmd = [
        "lerobot-train", f"--dataset.repo_id={args.repo_id}", f"--dataset.root={dataset}",
        "--dataset.video_backend=pyav", "--dataset.use_imagenet_stats=false",
    ]
    cmd += policy_args(args, stats)
    cmd += [f"--batch_size={args.batch_size}", f"--steps={args.steps}",
            "--num_workers=0", "--env_eval_freq=0", "--wandb.enable=false",
            "--policy.push_to_hub=false", "--save_checkpoint_to_hub=false",
            f"--output_dir={output}"]
    print(json.dumps({"policy": POLICY_NAME, "dataset": str(dataset), "dataset_info": stats,
                      "output": str(output), "mode": "TRAIN" if args.execute else "DRY_RUN"},
                     ensure_ascii=False, indent=2))
    print("Command:", shlex.join(cmd))
    if not args.execute:
        return 0
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            p.error("CUDA unavailable; use a CUDA host or explicitly select --device")
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return subprocess.call(cmd, cwd=ROOT, env=env)

if __name__ == "__main__":
    sys.exit(main())
