#!/usr/bin/env python3
"""Compute strict train-split stats for Cartesian ACT without writing to the dataset.

Uses all TRAIN frames for action/state, samples only TRAIN camera frames with
LeRobotDataset API, and saves a new JSON under a user-specified path.
All source/candidate Dataset files are read-only; no robot operations.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

KEYS = ("action", "observation.state")
CAMERAS = ("observation.images.side", "observation.images.wrist")
FIELDS = ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99")

def make_stats(x, *, image=False):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite [samples, channels] array")
    d = {}
    for key, v in (
        ("mean", x.mean(0)), ("std", x.std(0)),
        ("min", x.min(0)), ("max", x.max(0)),
        *[(f"q{q:02d}", np.percentile(x, q, axis=0)) for q in (1,10,50,90,99)]
    ):
        if image:
            v = v.reshape(-1,1,1)
        d[key] = np.asarray(v).tolist()
    d["count"] = [len(x)]
    return d

def compute(root: Path, output: Path, image_samples: int, seed: int):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite: {output}")
    if root in output.parents:
        raise ValueError("Output must be outside candidate dataset to avoid accidental changes")
    if not (root / "NOT_TRAINING_READY").exists():
        raise ValueError("Expected candidate NOT_TRAINING_READY marker")
    info = json.loads((root/"meta/info.json").read_text())
    split = json.loads((root/"episode_split.json").read_text())
    tr = set(map(int, split["train_episode_indices"]))
    val = set(map(int, split["validation_episode_indices"]))
    if tr & val or tr | val != set(range(53)) or len(tr) != 45 or len(val) != 8:
        raise ValueError("Unexpected train/validation split")
    if (info["total_frames"], info["features"]["action"]["shape"],
        info["features"]["observation.state"]["shape"]) != (30846,[7],[6]):
        raise ValueError("Dataset schema mismatch")

    collected = {k: [] for k in KEYS}
    n_train = 0
    for file in sorted((root/"data").rglob("*.parquet")):
        table = pq.read_table(file, columns=["episode_index", *KEYS])
        eps = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        mask = np.isin(eps, list(tr))
        n_train += int(mask.sum())
        for key in KEYS:
            vec = np.asarray(table[key].to_pylist(), dtype=np.float64)
            if vec.ndim != 2 or vec.shape[1] != (7 if key == "action" else 6):
                raise ValueError(f"Unexpected feature {key} shape in {file}")
            collected[key].append(vec[mask])
    if n_train != 26296:
        raise ValueError(f"Unexpected train frames: {n_train}")

    stats = {k: make_stats(np.concatenate(v, axis=0)) for k,v in collected.items()}

    if image_samples < 20:
        raise ValueError("Use >=20 image samples")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/muhi_cartesian_stats_trainonly", root=root, video_backend="pyav")
    if len(ds) != 30846:
        raise ValueError("LeRobot frame count mismatch")
    # Select image samples from actual Parquet positions, not uniform indices
    # across the whole Dataset. Avoid including even one validation image.
    selected = []
    offset = 0
    for file in sorted((root/"data").rglob("*.parquet")):
        table = pq.read_table(file, columns=["episode_index"])
        eps = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        selected.extend((offset+int(i)) for i in np.flatnonzero(np.isin(eps, list(tr))))
        offset += len(eps)
    if len(selected) != n_train:
        raise AssertionError("Selection mismatch")
    rng = np.random.default_rng(seed)
    image_indices = sorted(rng.choice(selected, size=min(image_samples,len(selected)), replace=False).tolist())
    buffers = {k: [] for k in CAMERAS}
    for i in image_indices:
        sample = ds[i]
        for key in CAMERAS:
            arr = sample[key].detach().cpu().numpy()
            if arr.shape != (3,168,224) or not np.isfinite(arr).all():
                raise ValueError(f"Unexpected camera tensor {key}: {arr.shape}")
            # Per image/per-channel mean, std on pixels, using a pooled
            # second moment to correctly capture within-image variation.
            buffers[key].append(arr.astype(np.float64))
    for key, stack in buffers.items():
        images = np.stack(stack)
        pixels = images.transpose(0,2,3,1).reshape(-1,3)
        # Sampled frames are all training frames; values already normalized to [0,1]
        result = make_stats(pixels, image=True)
        result["count"] = [len(images)]
        stats[key] = result
    record = {
        "data": stats,
        "audit": {
            "source_dataset": str(root),
            "train_episodes": sorted(tr),
            "validation_episodes": sorted(val),
            "train_frames": n_train,
            "sampled_image_frames": len(image_indices),
            "image_sample_seed": seed,
            "note": "Image stats from sampled TRAIN frames only. State/action stats from all TRAIN frames. No metadata modified.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    return record["audit"]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--image-samples",type=int,default=256)
    p.add_argument("--seed",type=int,default=20261102)
    a=p.parse_args()
    print(json.dumps(compute(a.dataset,a.output,a.image_samples,a.seed),indent=2))
if __name__=="__main__":
    main()
