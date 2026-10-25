#!/usr/bin/env python3
"""Prepare candidate meta/stats.json; derived Dataset only, no training.

Preserve source state/image stats (full dataset) and calculate CARTESIAN action
stats using TRAIN episodes only. This is a pragmatic compatibility smoke test;
it is NOT strict train-only normalization for all features.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

def prepare(source: Path, candidate: Path, seed: int = 20261102, validation_fraction: float = 0.15):
    source, candidate = source.resolve(), candidate.resolve()
    src_stats = source / "meta/stats.json"
    dest = candidate / "meta/stats.json"
    split_path = candidate / "episode_split.json"
    if not src_stats.is_file():
        raise FileNotFoundError(f"Source stats unavailable: {src_stats}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not (candidate / "NOT_TRAINING_READY").is_file():
        raise ValueError("Expected NOT_TRAINING_READY marker")
    if dest.exists():
        raise FileExistsError(f"Will not overwrite stats: {dest}")
    original_info = json.loads((source / "meta/info.json").read_text())
    info = json.loads((candidate / "meta/info.json").read_text())
    if (original_info["total_episodes"] != 53 or info["total_episodes"] != 53 or
        info["total_frames"] != 30846 or
        info["features"]["action"]["shape"] != [7] or
        original_info["features"]["action"]["shape"] != [6]):
        raise ValueError("Unexpected Dataset schema")
    split_source = "candidate"
    if split_path.is_file():
        split = json.loads(split_path.read_text())
    elif (source / "episode_split.json").is_file():
        split = json.loads((source / "episode_split.json").read_text())
        split_source = "original_dataset"
    else:
        shuffled = list(range(53))
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, round(53 * validation_fraction))
        val_ids = sorted(shuffled[:n_val])
        split = {
            "seed": seed, "validation_fraction": validation_fraction,
            "train_episode_indices": [i for i in range(53) if i not in set(val_ids)],
            "validation_episode_indices": val_ids,
            "rule": "deterministic episode-level shuffle; no frame crosses splits",
        }
        split_source = "new_deterministic_20261102"
    # Only persist after both split and Dataset metadata checks pass below.
    train = list(map(int, split["train_episode_indices"]))
    valid = list(map(int, split["validation_episode_indices"]))
    if not train or not valid or set(train) & set(valid) or set(train) | set(valid) != set(range(53)):
        raise ValueError("Invalid train/validation split")
    if not split_path.exists():
        split_path.write_text(json.dumps(split, indent=2) + "\\n")
    stats = json.loads(src_stats.read_text())
    if "observation.state" not in stats:
        raise ValueError("Source state stats missing")
    for name in ("observation.images.side", "observation.images.wrist"):
        if name not in stats:
            raise ValueError(f"Source camera stats missing: {name}")
    values = []
    for file in sorted((candidate / "data").rglob("*.parquet")):
        table = pq.read_table(file, columns=["episode_index", "action"])
        eps = np.asarray(table["episode_index"].to_pylist(), dtype=int)
        v = np.asarray(table["action"].to_pylist(), dtype=float)
        if v.ndim != 2 or v.shape[1] != 7 or not np.isfinite(v).all():
            raise ValueError(f"Invalid Cartesian action vectors in {file}")
        values.append(v[np.isin(eps, train)])
    if not values:
        raise ValueError("No action data")
    x = np.concatenate(values, axis=0)
    if len(x) < 2 or not (x.std(axis=0) > 0).all():
        raise ValueError("Not enough varying train observations")
    stats["action"] = {
        "mean": x.mean(axis=0).tolist(),
        "std": x.std(axis=0).tolist(),
        "min": x.min(axis=0).tolist(),
        "max": x.max(axis=0).tolist(),
        "count": [len(x)],
    }
    for q in (1,10,50,90,99):
        stats["action"][f"q{q:02d}"] = np.percentile(x, q, axis=0).tolist()
    # Verify that legacy action stats are not accidentally preserved.
    for name in ("mean", "std", "min", "max"):
        if len(stats["action"][name]) != 7:
            raise AssertionError("Wrong action stats dimensionality")
    dest.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    report = {
        "split_source": split_source,
        "train_episodes": train,
        "validation_episodes": valid,
        "train_action_frames": len(x),
        "action_stats_shape": [7],
        "source_non_action_stats": "copied_from_full_dataset_for_smoke_test",
        "status": "compatibility_test_only",
        "training_ready": False,
    }
    (candidate / "cartesian_stats_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    a = p.parse_args()
    print(json.dumps(prepare(a.source,a.candidate,a.seed,a.validation_fraction), indent=2))
if __name__ == "__main__":
    main()
