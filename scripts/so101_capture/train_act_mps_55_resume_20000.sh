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
checkpoint_root="$project_root/outputs/so101-act-muhi-055-step20000/checkpoints/014000"
config_path="$checkpoint_root/pretrained_model/train_config.json"
output_root="$project_root/outputs/so101-act-muhi-055-step20000-resume14000"

test -f "$config_path"
test ! -e "$output_root"
"$python_bin" -c 'import accelerate, psutil, torch; assert accelerate.__version__ == "1.15.0" and psutil.__version__ == "7.2.2" and torch.backends.mps.is_available()'

exec "$train_bin" \
  --config_path="$config_path" \
  --resume=true \
  --steps=20000 \
  --eval_steps=1000 \
  --save_freq=1000 \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --save_checkpoint_to_hub=false \
  --wandb.enable=false \
  --output_dir="$output_root"
