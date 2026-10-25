#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h:h}
cd "$project_root"
output_root=${1:?Pass a new output directory}
steps=${2:-20000}
eval_steps=1000
save_freq=1000
log_freq=50
if (( steps < 1000 )); then
  eval_steps=0
  save_freq=0
  log_freq=1
fi
dataset_root="$project_root/data/so101/muhi-act-success-190-20261022-168x224"
selection="$project_root/outputs/so101-act-190-selection-20261022/summary.json"

export PYTHONPATH="$project_root/.local/act-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
export HF_HOME=/tmp/so101-act-hf190

python_bin=${PYTHON_BIN:-python}
train_bin=${LEROBOT_TRAIN_BIN:-lerobot-train}
test -d "$dataset_root"
test ! -e "$output_root"
"$python_bin" -c 'import accelerate, psutil, torch; assert accelerate.__version__ == "1.15.0" and psutil.__version__ == "7.2.2" and torch.backends.mps.is_available()'
eval_split=$("$python_bin" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["adopted"]==190 and x["train"]+x["eval"]==190; print(x["eval_split"])' "$selection")

exec "$train_bin" \
  --dataset.repo_id=local/muhi_act_success_190_168x224 \
  --dataset.root="$dataset_root" \
  --dataset.video_backend=pyav \
  --dataset.eval_split="$eval_split" \
  --dataset.use_imagenet_stats=false \
  --policy.type=act \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.dim_model=128 \
  --policy.n_heads=4 \
  --policy.dim_feedforward=512 \
  --policy.n_encoder_layers=2 \
  --policy.n_vae_encoder_layers=2 \
  --policy.optimizer_lr=1e-5 \
  --steps="$steps" \
  --batch_size=2 \
  --num_workers=0 \
  --eval_steps="$eval_steps" \
  --env_eval_freq=0 \
  --log_freq="$log_freq" \
  --save_freq="$save_freq" \
  --save_checkpoint_to_hub=false \
  --wandb.enable=false \
  --output_dir="$output_root"
