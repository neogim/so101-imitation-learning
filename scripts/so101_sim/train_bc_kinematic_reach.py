#!/usr/bin/env python3
"""Minimal PyTorch behavior cloning of normalized DLS actions (9D -> 5D).

BC-only study; not a SAC initialization and not a physical controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True,
                   help="dls_transitions.npz from generate_dls_bc_dataset.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    args = p.parse_args()
    if args.out.exists():
        p.error("Output directory already exists; refusing to overwrite")
    if not 0 < args.validation_fraction < 1 or args.epochs <= 0 or args.batch_size <= 0:
        p.error("Invalid split, epochs or batch size")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    with np.load(args.dataset, allow_pickle=False) as saved:
        x = np.asarray(saved["observations"], dtype=np.float32)
        y = np.asarray(saved["actions"], dtype=np.float32)
        episode_ids = np.asarray(saved["episode_ids"], dtype=np.int32)
    if x.ndim != 2 or x.shape[1] != 9 or y.shape != (len(x), 5):
        raise ValueError(f"Expected observations (N,9) and actions (N,5), got {x.shape}, {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.max(np.abs(y)) > 1.00001:
        raise ValueError("Nonfinite or unnormalized expert data")
    rng = np.random.default_rng(args.seed)
    episodes = np.unique(episode_ids)
    if len(episodes) < 2:
        raise ValueError("Need at least two episodes for episode-wise validation")
    rng.shuffle(episodes)
    cutoff = max(1, min(len(episodes)-1, int(len(episodes) * (1-args.validation_fraction))))
    train_mask = np.isin(episode_ids, episodes[:cutoff])
    val_mask = ~train_mask
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Empty training or validation split")
    # Normalize by TRAINING transitions only; use same transform at inference.
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0).clip(min=1e-5)
    x = np.clip((x-mean)/std, -20, 20).astype(np.float32)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = nn.Sequential(nn.Linear(9, 64), nn.ReLU(),
                          nn.Linear(64, 64), nn.ReLU(),
                          nn.Linear(64, 5), nn.Tanh())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    data = TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask]))
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    val_x, val_y = torch.from_numpy(x[val_mask]), torch.from_numpy(y[val_mask])
    history = []
    best_val = float("inf")
    args.out.mkdir(parents=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y))
        item = {"epoch": epoch+1, "train_action_mse": train_loss_sum / train_mask.sum(),
                "validation_action_mse": val_loss}
        history.append(item)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out / "bc_best_state_dict.pt")
        if epoch == 0 or epoch == args.epochs-1 or (epoch+1) % 10 == 0:
            print(f"epoch {epoch+1:3d}: train={item['train_action_mse']:.5f} val={val_loss:.5f}")
    np.savez_compressed(args.out / "input_normalization.npz", mean=mean, std=std)
    summary = {
        "status": "BC_SUPERVISED_ACTION_FIT_ONLY_NOT_ROLLOUT_EVALUATED",
        "seed": args.seed, "epochs": args.epochs,
        "train_episodes": int(cutoff),
        "validation_episodes": int(len(episodes)-cutoff),
        "train_transitions": int(train_mask.sum()),
        "validation_transitions": int(val_mask.sum()),
        "best_validation_action_mse": best_val,
        "architecture": "9-64-64-5, ReLU, tanh",
        "note": "Low action MSE does not imply closed-loop Reach success; evaluate BC in a fresh environment.",
        "history": history,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("Saved BC weights and normalization to", args.out)


if __name__ == "__main__":
    main()
