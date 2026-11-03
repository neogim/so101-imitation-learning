# SO-101 SmolVLA — offline training (experimental)

This branch starts from `fix/isolate-multicam-episode-encoding` commit `d32d613` and preserves the existing ACT, calibration and safety controls. **No hardware/VLA servo execution is implemented.** Dataset capture can continue separately.

Use a dedicated Conda environment rather than upgrading the working `lerobot` environment. Install a LeRobot version supporting SmolVLA (and its `smolvla` extra), then verify `lerobot-train --help`. Hugging Face's pretrained model may require network access and substantial memory. M1/MPS is not performance-validated; CUDA is preferable for training.

```bash
# Print command and validate metadata (does not load model or touch robot)
python scripts/vla/train_smolvla.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \
  --output outputs/smolvla-muhi53-001

# Run training only after environment + dataset mapping verification:
python scripts/vla/train_smolvla.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \
  --output outputs/smolvla-muhi53-001 --device cuda --execute
```

Important: Pretrained `lerobot/smolvla_base` expects camera inputs `observation.images.camera1/camera2`. This recipe maps `side→camera1`, `wrist→camera2` using LeRobot `--rename_map`. Check compatibility with your checked-out LeRobot version before executing. `--steps=1000` is only a starter run, not a validated full recipe.

Before any future live control, validate data shapes, action units (5 joints in degrees + gripper 0–100), frame latency, safe-action gating, tracking error 8° and cycle-gap 250ms protections. This work intentionally exposes **no** live-robot command.
