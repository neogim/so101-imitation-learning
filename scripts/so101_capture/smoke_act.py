"""One or two local ACT forward/backward steps; no Hub or robot access."""
import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    torch.set_num_threads(4)
    chunk = args.chunk_size
    dataset = LeRobotDataset("local/muhi_act_success_030", root=args.root,
        delta_timestamps={"action": [i / 20 for i in range(chunk)]}, video_backend="pyav")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    stats = json.loads((args.root / "meta/stats.json").read_text())
    def normalize(key, value):
        mean = torch.as_tensor(stats[key]["mean"], dtype=value.dtype)
        std = torch.as_tensor(stats[key]["std"], dtype=value.dtype).clamp_min(1e-6)
        return (value - mean) / std
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process")
    config = ACTConfig(device=args.device, chunk_size=chunk, n_action_steps=chunk,
        pretrained_backbone_weights=None, dim_model=128, n_heads=4,
        dim_feedforward=512, n_encoder_layers=2, n_vae_encoder_layers=2,
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (6,)),
                        "observation.images.iphone": PolicyFeature(FeatureType.VISUAL, (3, 168, 224))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))})
    start = time.perf_counter()
    policy = ACTPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    initialized = time.perf_counter() - start
    it = iter(loader)
    records = []
    for step in range(args.steps):
        started = time.perf_counter()
        batch = next(it)
        sample = {"observation.state": normalize("observation.state", batch["observation.state"].float()),
                  "observation.images.iphone": normalize("observation.images.iphone",
                      F.interpolate(batch["observation.images.iphone"].float(), size=(168, 224),
                                    mode="bilinear", align_corners=False)),
                  "action": normalize("action", batch["action"].float()),
                  "action_is_pad": batch["action_is_pad"]}
        sample = {key: value.to(device) for key, value in sample.items()}
        if args.device == "mps":
            torch.mps.synchronize()
        load_s = time.perf_counter() - started
        optimizer.zero_grad(set_to_none=True)
        loss, details = policy(sample)
        loss.backward()
        optimizer.step()
        if args.device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        records.append({"step": step + 1, "loss": float(loss.detach()), "load_s": load_s,
                        "total_s": elapsed, "details": details,
                        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
                        "mps_allocated_mb": torch.mps.current_allocated_memory() / (1024 * 1024)
                        if args.device == "mps" else None})
        print(json.dumps(records[-1]), flush=True)
    print(json.dumps({"device": args.device, "init_s": initialized, "frames": len(dataset),
                      "records": records}, indent=2))


if __name__ == "__main__":
    main()
