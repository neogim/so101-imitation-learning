#!/usr/bin/env bash
# Sequential SO-101 ACT image domain randomization training.
# Offline training only; no motor / camera I/O.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/.local/act-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
STEPS="${STEPS:-20000}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
BASE="$ROOT/outputs/so101-act-comparison-dr-${RUN_ID}"
JOINT="$ROOT/outputs/so101-act-comparison-staged-v1/joint6"
CART="$ROOT/outputs/so101-act-comparison-staged-v1/cartesian7"
TRAIN_BIN="${LEROBOT_TRAIN_BIN:-lerobot-train}"
PY="${PYTHON_BIN:-python}"

if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid STEPS: $STEPS" >&2; exit 2; fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then echo "DRY_RUN must be 0 or 1" >&2; exit 2; fi
command -v "$TRAIN_BIN" >/dev/null || { echo "Missing lerobot-train in current environment" >&2; exit 2; }
command -v "$PY" >/dev/null || { echo "Missing Python" >&2; exit 2; }
for ds in "$JOINT" "$CART"; do
  test -f "$ds/meta/info.json" || { echo "Missing dataset: $ds" >&2; exit 2; }
done
test ! -e "$BASE-joint6" && test ! -e "$BASE-cartesian7" || { echo "Output directory exists: $BASE" >&2; exit 2; }

# Fail closed if this editable LeRobot revision has no augmentation interface.
"$PY" - <<'PY'
from dataclasses import fields
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.transforms import ImageTransformsConfig
names={f.name for f in fields(ImageTransformsConfig)}
assert {"enable","max_num_transforms"}.issubset(names),names
cfg=ImageTransformsConfig(enable=True,max_num_transforms=2)
assert cfg.enable and cfg.max_num_transforms==2
print("DR config validated:",cfg)
PY

COMMON=(
  --dataset.video_backend=pyav
  --dataset.eval_split=0.1
  --dataset.use_imagenet_stats=false
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.max_num_transforms=2
  --policy.type=act
  --policy.device=mps
  --policy.push_to_hub=false
  --policy.pretrained_backbone_weights=null
  --policy.chunk_size=50
  --policy.n_action_steps=50
  --policy.dim_model=128
  --policy.n_heads=4
  --policy.dim_feedforward=512
  --policy.n_encoder_layers=2
  --policy.n_vae_encoder_layers=2
  --policy.optimizer_lr=1e-5
  "--steps=$STEPS"
  --batch_size=2
  --num_workers=0
  --eval_steps=1000
  --env_eval_freq=0
  --log_freq=50
  --save_freq=1000
  --save_checkpoint_to_hub=false
  --wandb.enable=false
)
run_one(){
  local label="$1" dataset="$2" out="$3"
  local -a cmd=("$TRAIN_BIN" "--dataset.repo_id=local/so101_${label}_dr" "--dataset.root=$dataset" "${COMMON[@]}" "--output_dir=$out")
  echo "=== $label: $STEPS steps; target $out ==="
  printf ' %q' "${cmd[@]}"; echo
  if [[ "$DRY_RUN" == 1 ]]; then return 0; fi
  # tee preserves exit status under pipefail. Stop if first model fails.
  "${cmd[@]}" 2>&1 | tee "$BASE-${label}.log"
}
run_one joint6 "$JOINT" "$BASE-joint6"
run_one cartesian7 "$CART" "$BASE-cartesian7"
echo "Sequential training completed: $BASE"
