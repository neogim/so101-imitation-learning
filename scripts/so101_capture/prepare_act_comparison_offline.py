#!/usr/bin/env python3
"""Prepare a reproducible 6D-vs-7D ACT comparison manifest. Offline only.

Validates that both dataset variants contain the same 53 episodes, frames, and
observations (by non-action Parquet columns), and freezes an explicit 45/8 split.
Does not train models or change either source dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

DEFAULT_HELD_OUT=(5,8,15,18,23,31,46,51)

def inspect(root:Path,expected_action:int):
    info=json.loads((root/"meta/info.json").read_text())
    if info["total_episodes"]!=53 or info["total_frames"]!=30846:
        raise ValueError(f"Unexpected episode/frame count: {root}")
    if tuple(info["features"]["action"]["shape"])!=(expected_action,):
        raise ValueError(f"Unexpected action dimension at {root}")
    if tuple(info["features"]["observation.state"]["shape"])!=(6,):
        raise ValueError("State dimension mismatch")
    files=sorted((root/"data").rglob("*.parquet"))
    if not files:raise ValueError(f"No parquet data in {root}")
    counts=Counter()
    observation_digest=hashlib.sha256()
    keys=[]
    for path in files:
        table=pq.read_table(path)
        essential={"episode_index","frame_index","timestamp","observation.state"}
        if not essential<=set(table.column_names):
            raise ValueError(f"Missing observation/frame fields in {path}")
        for r in table.drop(["action"]).to_pylist():
            e=int(r["episode_index"]);f=int(r["frame_index"])
            counts[e]+=1
            keys.append((e,f))
            # Stable content digest, including all recorded observations but not actions.
            observation_digest.update(json.dumps(r,sort_keys=True,separators=(",",":"),default=str).encode())
            observation_digest.update(b"\n")
    if len(keys)!=30846 or len(set(keys))!=len(keys) or set(counts)!=set(range(53)):
        raise ValueError("Missing/duplicate frames or episode IDs")
    return info,dict(sorted(counts.items())),observation_digest.hexdigest(),keys

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joint-dataset",type=Path,default=Path("data/so101/muhi-side-wrist-success-053-20261025"))
    ap.add_argument("--cartesian-dataset",type=Path,default=Path("outputs/so101-cartesian-act-dataset-v2"))
    ap.add_argument("--output",type=Path,default=Path("outputs/so101-act-comparison-plan-v1"))
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.joint_dataset.resolve()==args.cartesian_dataset.resolve():raise ValueError("Datasets must differ")
    for root in (args.joint_dataset,args.cartesian_dataset):
        if not (root/"meta/info.json").exists():raise FileNotFoundError(root/"meta/info.json")
    i6,c6,h6,k6=inspect(args.joint_dataset,6)
    i7,c7,h7,k7=inspect(args.cartesian_dataset,7)
    if c6!=c7 or k6!=k7 or h6!=h7:
        raise ValueError("Datasets do not match on frame keys and non-action columns. Do not compare yet.")
    if i6["fps"]!=i7["fps"] or i6["features"]["observation.state"]!=i7["features"]["observation.state"]:
        raise ValueError("Dataset FPS / state feature mismatch")
    for name in ("observation.images.side","observation.images.wrist"):
        if i6["features"].get(name)!=i7["features"].get(name):
            raise ValueError(f"Camera input mismatch: {name}")
    val=list(DEFAULT_HELD_OUT)
    train=[e for e in range(53) if e not in val]
    report=dict(status="COMPARISON_PLAN_ONLY_NOT_TRAINED",
        train_episode_indices=train,validation_episode_indices=val,
        train_frames=sum(c6[e] for e in train),validation_frames=sum(c6[e] for e in val),
        joint_dataset=str(args.joint_dataset.resolve()),cartesian_dataset=str(args.cartesian_dataset.resolve()),
        same_frame_keys=True,same_recorded_observations=True,sha256_nonaction=h6,
        frame_counts_by_episode=c6,action_dimensions={"joint":6,"cartesian":7},
        control_variables={"policy":"ACT","images":"side+wrist","state_dim":6,
           "chunk_size":50,"n_action_steps":50,"dim_model":128,"n_heads":4,
           "dim_feedforward":512,"n_encoder_layers":2,"n_vae_encoder_layers":2,
           "batch_size":2,"learning_rate":1e-5,"optimizer":"AdamW",
           "device":"mps","steps":20000,"seed":1000},
        required_before_training=[
          "Compute normalization statistics separately for both representations using TRAIN episodes only.",
          "Exclude validation episodes from the training dataloader explicitly; do NOT assume eval_split alone uses these IDs.",
          "Verify the exact selected training episode indices in both LeRobot loaders.",
          "Compare TCP position/orientation and gripper in common physical units, not 6D vs 7D training loss.",
          "Cartesian ACT IK remains non-deployable until physical J2/J3 mapping, collision and motion limits are validated.",
          "The existing 5-step smoke checkpoint is not a comparable trained baseline." ])
    args.output.mkdir(parents=True)
    (args.output/"comparison_plan.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({k:report[k] for k in ("status","train_episode_indices","validation_episode_indices","train_frames","validation_frames","same_recorded_observations")},indent=2))
    print("Plan saved:",args.output/"comparison_plan.json")
if __name__=="__main__":main()
