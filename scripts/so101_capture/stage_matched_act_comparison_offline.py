#!/usr/bin/env python3
"""Stage two isolated comparison datasets, apply train-only statistics, audit LeRobot episode selection.
No hardware access. No training runs. Never overwrites existing output paths.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import shutil
from pathlib import Path
import numpy as np

FIELDS=("action","observation.state","observation.images.side","observation.images.wrist")

def validate_stats(stats,dim):
    for key in FIELDS:
        if key not in stats:raise ValueError(f"Missing stats: {key}")
        shape=(dim,) if key=="action" else (6,) if key=="observation.state" else (3,1,1)
        for field in ("mean","std","min","max"):
            v=np.asarray(stats[key][field],dtype=float)
            if v.shape!=shape or not np.isfinite(v).all():
                raise ValueError(f"Invalid {key}/{field}: {v.shape}; expected {shape}")
            if field=="std" and np.any(v<=0):
                raise ValueError(f"Invalid std: {key}")

def audit_selection(root,train,val,expected_frames,repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset(repo_id,root=root,episodes=sorted(train),video_backend="pyav")
    eps=np.asarray(ds.hf_dataset["episode_index"],dtype=int)
    actual=set(map(int,np.unique(eps)))
    if actual!=train or actual & val or len(ds)!=expected_frames:
        raise RuntimeError(f"TRAIN DATA LEAK / mismatch: {root}: episodes={sorted(actual)}, frames={len(ds)}")
    if tuple(ds.features["action"]["shape"]) not in ((6,),(7,)):
        raise RuntimeError("Unexpected action feature shape")
    return dict(episodes=sorted(actual),frames=len(ds),action_dim=ds.features["action"]["shape"][0])

def stage(plan_path,stats_path,output_root):
    plan=json.loads(plan_path.read_text())
    stats=json.loads(stats_path.read_text())
    if plan["status"]!="COMPARISON_PLAN_ONLY_NOT_TRAINED" or not plan["same_recorded_observations"]:
        raise ValueError("Comparison plan has not passed audit")
    train=set(map(int,plan["train_episode_indices"]))
    val=set(map(int,plan["validation_episode_indices"]))
    if len(train)!=45 or len(val)!=8 or train&val or train|val!=set(range(53)):
        raise ValueError("Invalid 45/8 split")
    if stats["status"]!="TRAIN_ONLY_STATS_AUDITED_NOT_APPLIED" or set(stats["train_episode_indices"])!=train or set(stats["validation_episode_indices"])!=val or stats["train_frames"]!=plan["train_frames"]:
        raise ValueError("Stats/split disagreement")
    if not stats["image_stats_shared"]:raise ValueError("Image stats are not shared")
    jobs=[("joint6",Path(plan["joint_dataset"]).resolve(),stats["joint_stats"],6),
          ("cartesian7",Path(plan["cartesian_dataset"]).resolve(),stats["cartesian_stats"],7)]
    for label,source,values,dim in jobs:
        if not (source/"meta/info.json").is_file():raise FileNotFoundError(source)
        validate_stats(values,dim)
        info=json.loads((source/"meta/info.json").read_text())
        if info["features"]["action"]["shape"]!=[dim] or info["total_frames"]!=30846:
            raise ValueError(f"Source schema mismatch: {label}")
    output_root=output_root.resolve()
    if output_root.exists():raise FileExistsError(f"Never overwrite: {output_root}")
    if any(output_root==src or src in output_root.parents or output_root in src.parents for _,src,_,_ in jobs):
        raise ValueError("Output cannot nest with source datasets")
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["HF_DATASETS_OFFLINE"]="1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    report=dict(status="STAGING_INCOMPLETE",train_episode_indices=sorted(train),
                validation_episode_indices=sorted(val),sources={})
    try:
        output_root.mkdir(parents=True)
        for label,source,values,dim in jobs:
            target=output_root/label
            shutil.copytree(source,target,symlinks=False)
            metadata=json.loads((target/"meta/stats.json").read_text())
            for field in FIELDS:
                metadata[field]=copy.deepcopy(values[field])
            (target/"meta/stats.json").write_text(json.dumps(metadata,indent=2)+"\n")
            # A full 53-episode copy intentionally retains validation rows for offline evaluation.
            # Training MUST pass --dataset.episodes with the audited 45-ID allowlist.
            checked=audit_selection(target,train,val,plan["train_frames"],f"local/so101_compare_{label}")
            if checked["action_dim"]!=dim:raise ValueError("Incorrect action dimension")
            report["sources"][label]=dict(dataset=str(target),source=str(source),
                                        train_loader_audit=checked)
        report["status"]="STAGED_AND_TRAIN_LOADER_AUDITED"
        (output_root/"comparison_training_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    except Exception:
        if output_root.exists():
            (output_root/"STAGING_FAILED_DO_NOT_TRAIN").write_text("Staging incomplete; do not train.\n")
        raise
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("outputs/so101-act-comparison-plan-v1/comparison_plan.json"))
    p.add_argument("--stats",type=Path,default=Path("outputs/so101-act-comparison-trainonly-stats-v1.json"))
    p.add_argument("--output",type=Path,default=Path("outputs/so101-act-comparison-staged-v1"))
    args=p.parse_args()
    report=stage(args.plan,args.stats,args.output)
    print(json.dumps({"status":report["status"],
      "joint6":report["sources"]["joint6"]["train_loader_audit"],
      "cartesian7":report["sources"]["cartesian7"]["train_loader_audit"],
      "output":str(args.output)},indent=2))

if __name__=="__main__":main()
