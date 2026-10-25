#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/cameras.local.yaml"
DATA_PARENT="${SO101_DATA_PARENT:-$HOME/workspace/so101-data}"

cd "$REPO"

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda initialization unavailable" >&2
  exit 1
fi
conda activate lerobot
export PYTHONPATH="$REPO/.local/camera-deps:$REPO/.local/act-deps:${PYTHONPATH:-}"

[[ -f "$CONFIG" ]] || { echo "ERROR: missing local camera config: $CONFIG" >&2; exit 1; }
[[ -f "$SCRIPT_DIR/record_episode.py" ]] || { echo "ERROR: record script missing" >&2; exit 1; }
[[ -d "$DATA_PARENT" ]] || { echo "ERROR: data parent missing: $DATA_PARENT" >&2; exit 1; }

# Validate the exact production camera profile without opening OBS or a camera.
python -c '
import sys
from multi_camera import capture_image_shape, load_config
cfg = load_config(sys.argv[1], "side,wrist")
side, wrist = cfg["active_cameras"]["side"], cfg["active_cameras"]["wrist"]
assert capture_image_shape(cfg) == (240, 320, 3), "capture must be 320x240"
assert side["provider"] == "obs_source", "side must use OBS"
assert side.get("jpeg_quality", 90) == 90, "side JPEG quality must be 90"
assert wrist["provider"] == "avfoundation_unique_id", "wrist must use AVFoundation"
' "$CONFIG"

# --help does not connect to either arm.
HELP="$(python "$SCRIPT_DIR/record_episode.py" --help)"
for arg in --rotary-slew --gripper-slew --j6-max-closing-lead --enable-cameras; do
  [[ "$HELP" == *"$arg"* ]] || { echo "ERROR: missing CLI option $arg" >&2; exit 1; }
done

printf '\nBefore proceeding, confirm:\n'
printf '  [ ] OBS is running; iPhone6_Fixed image is updating.\n'
printf '  [ ] OBS preview is ENABLED (keep the live view visible).\n'
printf '  [ ] OBS InnoMaker_Wrist source is disabled (AVFoundation owns the camera).\n'
printf '  [ ] cameras.local.yaml selects side OBS quality 90 and wrist direct.\n'
printf '  [ ] Work area clear; arm/cables supported; independent power stop ready.\n'
printf '  [ ] Leader/Follower poses are close enough for safe alignment.\n\n'
read -r -p 'Type START to enable robot operation and recording: ' CONFIRM
if [[ "$CONFIRM" != START ]]; then
  echo "Cancelled; no robot connection initiated."
  exit 0
fi

ROOT="$DATA_PARENT/muhi-side-wrist-production-30-$(date +%Y%m%d-%H%M%S)"
[[ ! -e "$ROOT" ]] || { echo "ERROR: output already exists: $ROOT" >&2; exit 1; }
printf '\nOutput: %s\n' "$ROOT"

python "$SCRIPT_DIR/record_episode.py" \
  --root "$ROOT" \
  --seconds 40 \
  --num-episodes 30 \
  --min-seconds 1 \
  --countdown 3 \
  --alignment-timeout 60 \
  --alignment-tolerance 2.5 \
  --camera-config "$CONFIG" \
  --enable-cameras side,wrist \
  --slew 30 \
  --rotary-slew 40 \
  --gripper-slew 30 \
  --j6-max-closing-lead 3 \
  --task "Pick up the Muhi bottle and place it into the target container." \
  --execute

printf '\nRecording process finished. Check labels and recordings at: %s\n' "$ROOT"
