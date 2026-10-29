#!/usr/bin/env python3
"""Print paired ACT 20k training recipes ONLY after audited split/stats staging.
Does not run training or connect to hardware.
"""
import argparse,json,shlex
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit",type=Path,default=Path("outputs/so101-act-comparison-staged-v1/comparison_training_audit.json"))
    p.add_argument("--steps",type=int,default=20000)
    p.add_argument("--output-prefix",type=Path,default=Path("outputs/so101-act-comparison-20k"))
    args=p.parse_args()
    if args.steps<1:raise ValueError("steps must be positive")
    audit=json.loads(args.audit.read_text())
    if audit["status"]!="STAGED_AND_TRAIN_LOADER_AUDITED":raise RuntimeError("Staging audit not passed")
    train=audit["train_episode_indices"];val=audit["validation_episode_indices"]
    if len(train)!=45 or len(val)!=8 or set(train)&set(val):
        raise RuntimeError("Unexpected split")
    for label,dim in (("joint6",6),("cartesian7",7)):
        s=audit["sources"][label]
        if s["train_loader_audit"]["frames"]!=26296 or s["train_loader_audit"]["episodes"]!=train or s["train_loader_audit"]["action_dim"]!=dim:
            raise RuntimeError(f"Loader audit invalid for {label}")
        dest=Path(str(args.output_prefix)+"-"+label)
        if dest.exists():raise FileExistsError(f"Would overwrite training output: {dest}")
        cmd=["lerobot-train",
          f"--dataset.repo_id=local/so101_compare_{label}",
          f"--dataset.root={s['dataset']}","--dataset.video_backend=pyav",
          "--dataset.episodes=["+",".join(map(str,train))+"]",
          "--dataset.eval_split=0","--dataset.use_imagenet_stats=false",
          "--policy.type=act","--policy.device=mps","--policy.push_to_hub=false",
          "--policy.pretrained_backbone_weights=null",
          "--policy.chunk_size=50","--policy.n_action_steps=50",
          "--policy.dim_model=128","--policy.n_heads=4",
          "--policy.dim_feedforward=512","--policy.n_encoder_layers=2",
          "--policy.n_vae_encoder_layers=2","--policy.optimizer_lr=1e-5",
          f"--steps={args.steps}","--batch_size=2","--num_workers=0",
          "--eval_steps=0","--env_eval_freq=0","--log_freq=50",
          "--save_freq=1000","--save_checkpoint_to_hub=false",
          "--seed=1000","--wandb.enable=false",f"--output_dir={dest}"]
        print(f"\n# {label}: explicit train IDs only; evaluation later on eight held-out episodes")
        print(shlex.join(cmd))
    print("\nCommands PRINTED ONLY. Check --dataset.episodes behavior with a short training run before 20k steps.")
if __name__=="__main__":main()
