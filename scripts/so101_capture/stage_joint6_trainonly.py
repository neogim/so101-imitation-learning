#!/usr/bin/env python3
"""Stage original six-joint dataset with train-only action stats and shared camera stats.

Only action stats are newly computed. Camera/state stats are reused from the
existing verified 45-train/8-heldout Cartesian train-only audit (same images/state).
Never overwrites source.
"""
from __future__ import annotations
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

KEYS=("observation.state","observation.images.side","observation.images.wrist")
def build(source:Path,split_dataset:Path,stats_file:Path,out:Path):
    for p in (source,split_dataset,stats_file):
        if not p.exists():raise FileNotFoundError(p)
    source=source.resolve();out=out.resolve()
    if out.exists():raise FileExistsError(out)
    if source==out or source in out.parents:raise ValueError("Unsafe output path")
    info=json.loads((source/"meta/info.json").read_text())
    split=json.loads((split_dataset/"episode_split.json").read_text())
    train=list(map(int,split["train_episode_indices"])); val=list(map(int,split["validation_episode_indices"]))
    assert len(train)==45 and len(val)==8 and not(set(train)&set(val))
    if info["features"]["action"]["shape"]!=[6] or info["features"]["observation.state"]["shape"]!=[6]:
        raise ValueError("Not the original 6D dataset")
    audit=json.loads(stats_file.read_text())
    prior=audit["audit"]
    if prior["train_episodes"]!=train or prior["validation_episodes"]!=val:
        raise ValueError("Mismatched split and train-only statistics")
    stats=audit["data"]
    if not all(k in stats for k in KEYS):raise ValueError("Missing shared image/state train stats")
    count=0;mean=np.zeros(6);m2=np.zeros(6)
    low=np.full(6,np.inf);high=np.full(6,-np.inf)
    for file in sorted((source/"data").rglob("*.parquet")):
        tab=pq.read_table(file,columns=["episode_index","action"])
        eps=tab["episode_index"].to_numpy()
        mask=np.isin(eps,train)
        if not mask.any():continue
        values=np.asarray(tab["action"].to_pylist(),dtype=float)[mask]
        if values.ndim!=2 or values.shape[1]!=6 or not np.isfinite(values).all():
            raise ValueError("Nonfinite/bad 6D actions")
        low=np.minimum(low,values.min(axis=0));high=np.maximum(high,values.max(axis=0))
        n=values.shape[0];batch_mean=values.mean(axis=0);batch_m2=((values-batch_mean)**2).sum(axis=0)
        delta=batch_mean-mean
        new_count=count+n
        mean+=delta*n/new_count
        m2+=batch_m2+delta*delta*count*n/new_count
        count=new_count
    if count!=prior["train_frames"] or count!=26296:raise ValueError(f"Unexpected train frames: {count}")
    std=np.sqrt(m2/count)
    std=np.maximum(std,1e-6)
    joint_stats={k:arr.tolist() for k,arr in zip(("mean","std","min","max"),(mean,std,low,high))}
    staged=json.loads((source/"meta/stats.json").read_text())
    for k in KEYS:staged[k]=copy.deepcopy(stats[k])
    staged["action"]=joint_stats
    shutil.copytree(source,out)
    (out/"meta/stats.json").write_text(json.dumps(staged,indent=2)+"\n")
    (out/"TRAIN_ONLY_JOINT6_STATS.json").write_text(json.dumps({
        "train_episode_indices":train,"validation_episode_indices":val,
        "train_frames":count,"joint_action_dims":6,
        "cameras_and_state_stats_from":str(stats_file),
        "warnings":["Dataset contains all 53 episodes; training MUST specify only train IDs",
                    "Never deploy model: IK/URDF limits and collisions unresolved"]},indent=2)+"\n")
    print(json.dumps({"output":str(out),"train_frames":count,"train_episodes":len(train),
                      "heldout_episodes":len(val),"joint_action_mean":mean.round(3).tolist(),
                      "joint_action_std":std.round(3).tolist()},indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--split-dataset",required=True,type=Path)
    p.add_argument("--train-stats",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    build(a.source,a.split_dataset,a.train_stats,a.output)
if __name__=="__main__":main()
