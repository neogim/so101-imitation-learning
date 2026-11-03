#!/usr/bin/env bash
# SO-101 Muhi position/rotation augmentation capture launcher.
# Defaults to a read-only plan. Use --check for preflight or --execute for motion.
set -euo pipefail

# ============================================================
# Select EXACTLY ONE condition by removing its leading '#'.
# Baseline center already has 53 accepted demonstrations;
# add center recordings only if camera/layout conditions changed.
# ============================================================
CONFIG="left-3cm";          NUM_EPISODES=5
# CONFIG="right-3cm";       NUM_EPISODES=5
# CONFIG="front-3cm";       NUM_EPISODES=5
# CONFIG="back-3cm";        NUM_EPISODES=5
# CONFIG="rotate-plus30";   NUM_EPISODES=5
# CONFIG="rotate-minus30";  NUM_EPISODES=5
# CONFIG="rotate-plus60";   NUM_EPISODES=5
# CONFIG="rotate-minus60";  NUM_EPISODES=5
# CONFIG="center";          NUM_EPISODES=5 # optional; not needed by default

usage() {
  cat <<'USAGE'
Usage: bash scripts/so101_capture/record_muhi_augmentation.sh [--dry-run|--check|--execute]

  --dry-run   Show the selected condition and command; no device access (default)
  --check     Run record_episode.py preflight (USB/calibration/destination; no motion)
  --execute   Explicitly approve and launch real SO-101 teleoperation/capture

Edit the CONFIG line at the top of this script to choose a different setup.
Collect each condition separately and record the actual Muhi position/angle.
USAGE
}

MODE="dry-run"
case "${1:-}" in
  ""|--dry-run) MODE="dry-run" ;;
  --check) MODE="check" ;;
  --execute) MODE="execute" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Error: pass only one option." >&2
  exit 2
fi

case "$CONFIG" in
  center|left-3cm|right-3cm|front-3cm|back-3cm|rotate-plus30|rotate-minus30|rotate-plus60|rotate-minus60) ;;
  *) echo "Error: unsupported CONFIG=$CONFIG" >&2; exit 2 ;;
esac
if ! [[ "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: NUM_EPISODES must be a positive integer" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/.local/camera-deps:$REPO_ROOT/.local/act-deps:${PYTHONPATH:-}"

BASE="$HOME/workspace/so101-data/muhi-side-wrist-augmentation"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="$BASE/$CONFIG/$RUN_STAMP"

ARGS=(
  scripts/so101_capture/record_episode.py
  --root "$ROOT"
  --seconds 40
  --num-episodes "$NUM_EPISODES"
  --min-seconds 1
  --countdown 3
  --alignment-timeout 60
  --alignment-tolerance 2.5
  --camera-config scripts/so101_capture/cameras.local.yaml
  --enable-cameras side,wrist
  --slew 30
  --rotary-slew 40
  --gripper-slew 30
  --j6-max-closing-lead 3
  --task "Pick up the Muhi bottle and place it into the target container."
)

echo "Mode: $MODE"
echo "Condition: $CONFIG"
echo "Episodes: $NUM_EPISODES"
echo "Destination: $ROOT"
printf 'Command: python'
printf ' %q' "${ARGS[@]}"
if [[ "$MODE" == "execute" ]]; then printf ' --execute'; fi
printf '\n'

if [[ "$MODE" == "dry-run" ]]; then
  echo "Plan only: no directories created, hardware accessed, or robot moved."
  exit 0
fi

if [[ ! -f scripts/so101_capture/cameras.local.yaml ]]; then
  echo "Error: missing scripts/so101_capture/cameras.local.yaml" >&2
  exit 1
fi
if [[ ! -d "$CONDA_PREFIX" && "${CONDA_PREFIX:-}" != "" ]]; then
  echo "Error: CONDA_PREFIX is not an existing directory." >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1; then
  echo "Error: python is not available (activate the lerobot environment)." >&2
  exit 1
fi

# record_episode.py requires the destination's immediate parent to exist.
mkdir -p "$BASE/$CONFIG"
if [[ -e "$ROOT" ]]; then
  echo "Error: destination already exists. Choose a new timestamp/run." >&2
  exit 1
fi

if [[ "$MODE" == "check" ]]; then
  exec python "${ARGS[@]}"
fi

# --execute needs interactive terminal both here and in record_episode.py.
if [[ ! -t 0 || ! -t 1 ]]; then
  echo "Error: --execute requires an interactive terminal." >&2
  exit 1
fi
echo
echo "SAFETY: Secure Follower and workspace. Check camera latency, USB ports,"
echo "Leader/Follower initial poses, cable clearance, and how to stop motion."
echo "The original script does not disable torque on exit; support the arm."
echo "Recording end (e) is NOT motion stop; use the documented safe reset."
read -r -p "Type EXECUTE to confirm real robot motion: " approval
if [[ "$approval" != "EXECUTE" ]]; then
  echo "Cancelled without starting the capture process."
  exit 1
fi
exec python "${ARGS[@]}" --execute
