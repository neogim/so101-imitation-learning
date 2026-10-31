#!/usr/bin/env python3
"""Stage isolated Cartesian ACT smoke-test dataset and print training command.

Never trains or changes candidate/source. Train-only stats are checked and
applied to a fresh copy; train/val episode ordering preserved in CLI.
"""
from __future__ import annotations
import argparse, copy, json, math, os, shlex, shutil
from pathlib import Path
import numpy as np

KEYS=("action","observation.state","observation.images.side","observation.images.wrist")

def stage(candidate:Path,stat_file:Path,output:Path,model_output:Path,steps:int):
    candidate,stat_file,output,model_output=map(lambda x:x.resolve(),(candidate,stat_file,output,model_output))
    if output.exists() or model_output.exists():
        raise FileExistsError("Dataset or model output already exists; refusing overwrite")
    if candidate in output.parents or candidate==output or output in model_output.parents:
        raise ValueError("Unsafe nested output directory")
    if steps<1:
        raise ValueError("Steps must be positive")
    info=json.loads((candidate/"meta/info.json").read_text())
    split=json.loads((candidate/"episode_split.json").read_text())
    audit_obj=json.loads(stat_file.read_text())
    stats=audit_obj["data"]
    audit=audit_obj["audit"]
    train=list(map(int,split["train_episode_indices"]))
    val=list(map(int,split["validation_episode_indices"]))
    if set(train)&set(val) or set(train)|set(val)!=set(range(53)) or len(train)!=45 or len(val)!=8:
        raise ValueError("Split is not the previously validated 45/8")
    if train!=audit["train_episodes"] or val!=audit["validation_episodes"] or audit["train_frames"]!=26296:
        raise ValueError("Statistics were computed for a different split")
    if info["total_frames"]!=30846 or info["features"]["action"]["shape"]!=[7]:
        raise ValueError("Wrong Dataset shape")
    for k in KEYS:
        if k not in stats:
            raise ValueError("Missing stats "+k)
        d=7 if k=="action" else 6 if k=="observation.state" else 3
        for field in ("mean","std","min","max"):
            a=np.asarray(stats[k][field],dtype=float)
            expected=(d,1,1) if k.startswith("observation.images.") else (d,)
            if a.shape!=expected or not np.isfinite(a).all():
                raise ValueError(f"Bad stats {k}/{field}: {a.shape}")
        if np.any(np.asarray(stats[k]["std"])<=0):
            raise ValueError(f"Zero/negative std {k}")
    shutil.copytree(candidate,output,symlinks=False)
    original=json.loads((output/"meta/stats.json").read_text())
    for k in KEYS:
        original[k]=copy.deepcopy(stats[k])
    (output/"meta/stats.json").write_text(json.dumps(original,indent=2)+"\n")
    (output/"TRAIN_ONLY_STATS_APPLIED.json").write_text(json.dumps({
        "source_candidate":str(candidate),"stats_source":str(stat_file),
        "train_episodes":train,"validation_episodes":val,
        "warning":"ACT 7D output needs offline IK decoder before any robot use",
    },indent=2)+"\n")
    order=train+val
    cmd=[
        "lerobot-train",
        "--dataset.repo_id=local/muhi_cartesian_act_smoke_053",
        f"--dataset.root={output}","--dataset.video_backend=pyav",
        "--dataset.episodes=["+",".join(map(str,order))+"]",
        f"--dataset.eval_split={len(val)/len(order)}",
        "--dataset.use_imagenet_stats=false",
        "--policy.type=act","--policy.device=mps","--policy.push_to_hub=false",
        "--policy.pretrained_backbone_weights=null","--policy.chunk_size=50",
        "--policy.n_action_steps=50","--policy.dim_model=128","--policy.n_heads=4",
        "--policy.dim_feedforward=512","--policy.n_encoder_layers=2",
        "--policy.n_vae_encoder_layers=2","--policy.optimizer_lr=1e-5",
        f"--steps={steps}","--batch_size=2","--num_workers=0",
        "--eval_steps=0","--env_eval_freq=0","--log_freq=1",
        "--save_freq=0","--save_checkpoint_to_hub=false",
        "--wandb.enable=false",f"--output_dir={model_output}"
    ]
    return shlex.join(cmd)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",required=True,type=Path)
    p.add_argument("--stats",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--model-output",required=True,type=Path)
    p.add_argument("--steps",type=int,default=10)
    a=p.parse_args()
    print("Staged Dataset; NOT_TRAINING_READY marker retained as a reminder: no hardware deployment.")
    print("Smoke test command (NOT executed):")
    print(stage(a.candidate,a.stats,a.output,a.model_output,a.steps))

if __name__=="__main__":main()
