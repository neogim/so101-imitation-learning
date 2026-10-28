#!/usr/bin/env bash
# Offline five-step LeRobot ACT Cartesian-action smoke test on Apple Silicon.
# Does not open robot ports and does not produce a deployable policy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET="${1:-$ROOT/outputs/so101-cartesian-act-dataset-v2}"
OUT="${2:-$ROOT/outputs/so101-cartesian-act-smoke-v1}"
STEPS="${3:-5}"

if [[ ! -f "$DATASET/meta/info.json" ]]; then
  echo "Missing dataset: $DATASET/meta/info.json" >&2; exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite output directory: $OUT" >&2; exit 2
fi
if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || (( STEPS > 20 )); then
  echo "Steps must be an integer in 1..20" >&2; exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/.local/act-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import json
from pathlib import Path
import os
import torch
assert torch.backends.mps.is_available(), "MPS not available"
root=Path(os.environ.get("DATASET_PATH","outputs/so101-cartesian-act-dataset-v2"))
print("MPS available",torch.backends.mps.is_available())
PY

echo "ACTION SPACE: exploratory 7D Cartesian deltas; NO hardware replay."
lerobot-train \
  --dataset.repo_id=local/muhi_cartesian_act_experimental \
  --dataset.root="$DATASET" \
  --dataset.video_backend=pyav --dataset.eval_split=0.1 \
  --dataset.use_imagenet_stats=false \
  --policy.type=act --policy.device=mps --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --policy.dim_model=128 --policy.n_heads=4 --policy.dim_feedforward=512 \
  --policy.n_encoder_layers=2 --policy.n_vae_encoder_layers=2 \
  --policy.optimizer_lr=1e-5 --steps="$STEPS" --batch_size=2 \
  --num_workers=0 --eval_steps=1000 --env_eval_freq=0 \
  --log_freq=1 --save_freq=1000 --save_checkpoint_to_hub=false \
  --wandb.enable=false \
  --output_dir="$OUT"

echo "Smoke training command exited successfully. Model is NOT deployable."
