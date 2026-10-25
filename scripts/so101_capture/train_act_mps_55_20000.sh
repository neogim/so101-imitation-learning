#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h:h}
cd "$project_root"

export PYTHONPATH="$project_root/.local/act-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled

python_bin=${PYTHON_BIN:-python}
train_bin=${LEROBOT_TRAIN_BIN:-lerobot-train}
dataset_root="$project_root/data/so101/muhi-act-success-055-168x224"
output_root="$project_root/outputs/so101-act-muhi-055-step20000"

test -d "$dataset_root"
test ! -e "$output_root"
"$python_bin" -c 'import accelerate, psutil, torch; assert accelerate.__version__ == "1.15.0" and psutil.__version__ == "7.2.2" and torch.backends.mps.is_available()'

exec "$train_bin" \
  --dataset.repo_id=local/muhi_act_success_055_168x224 \
  --dataset.root="$dataset_root" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0.1 \
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
  --steps=20000 \
  --batch_size=2 \
  --num_workers=0 \
  --eval_steps=1000 \
  --env_eval_freq=0 \
  --log_freq=50 \
  --save_freq=1000 \
  --save_checkpoint_to_hub=false \
  --wandb.enable=false \
  --output_dir="$output_root"
