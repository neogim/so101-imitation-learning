# SO-101 imitation learning

Independent SO-101 data capture, ACT training, and guarded inference tools for Apple Silicon. Dataset files, checkpoints, experiment outputs, camera identifiers, credentials, and robot calibration stay local and are not published.

## Environment

Prerequisites:

- Apple Silicon macOS with MPS available
- Conda environment `lerobot`
- LeRobot installed (the development setup uses an editable checkout at `~/workspace/lerobot`)
- SO-101 calibration already created by LeRobot

From this repository root:

```sh
conda activate lerobot
python -c 'import torch; assert torch.backends.mps.is_available()'
python -m pip install --no-deps --target .local/camera-deps \
  -r scripts/so101_capture/requirements-multicam.txt
cp .env.example .env
cp scripts/so101_capture/cameras.example.yaml \
  scripts/so101_capture/cameras.local.yaml
```

Set the local OBS WebSocket password in `.env` and camera-specific identifiers in `cameras.local.yaml`. Both files are ignored. The scripts discover the repository root from their own location. Optional isolated dependencies are read from `.local/camera-deps` and `.local/act-deps`.

## Dataset capture

Keep raw sessions outside this repository (by default under `~/workspace/so101-data`). Review the safety and capture procedure before connecting hardware:

- `scripts/so101_capture/RECORDING.md`
- `scripts/so101_capture/MULTI_CAMERA.md`
- `scripts/so101_capture/COMMUNICATION_SAFETY.md`

Preview cameras without opening robot serial ports:

```sh
python scripts/so101_capture/preview_cameras.py \
  --camera-config scripts/so101_capture/cameras.local.yaml \
  --enable-cameras side,wrist --output /tmp/so101-preview
```

For hardware recording commands, follow `scripts/so101_capture/RECORDING.md`. Never replace or regenerate calibration as part of capture.

## ACT training

Place the prepared local dataset below `data/so101/`; this directory is ignored. Training scripts derive dataset and output paths from the repository root and use the active Conda environment. `PYTHON_BIN` and `LEROBOT_TRAIN_BIN` may override the executables.

The current 53-episode, two-camera dataset is expected at:

```text
data/so101/muhi-side-wrist-success-053-20261025-168x224
```

The preserved training recipes are under `scripts/so101_capture/train_act_mps_*.sh`. For the current 53-episode, two-camera baseline, run the equivalent 20,000-step recipe below (choose a new output path):

```sh
export PYTHONPATH="$PWD/.local/act-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
lerobot-train \
  --dataset.repo_id=local/muhi_act_success_053_2cam_168x224 \
  --dataset.root="$PWD/data/so101/muhi-side-wrist-success-053-20261025-168x224" \
  --dataset.video_backend=pyav --dataset.eval_split=0.1 \
  --dataset.use_imagenet_stats=false \
  --policy.type=act --policy.device=mps --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --policy.dim_model=128 --policy.n_heads=4 --policy.dim_feedforward=512 \
  --policy.n_encoder_layers=2 --policy.n_vae_encoder_layers=2 \
  --policy.optimizer_lr=1e-5 --steps=20000 --batch_size=2 \
  --num_workers=0 --eval_steps=1000 --env_eval_freq=0 \
  --log_freq=50 --save_freq=1000 --save_checkpoint_to_hub=false \
  --wandb.enable=false \
  --output_dir="$PWD/outputs/so101-act-muhi53-NEW-RUN"
```

Outputs and checkpoints are written below ignored `outputs/` paths. Always choose a new output directory.

## Offline inference and guarded hardware inference

Load and evaluate a checkpoint without hardware:

```sh
python scripts/so101_capture/evaluate_act_190_offline.py \
  --dataset data/so101/REPLACE_WITH_DATASET \
  --checkpoint outputs/REPLACE_WITH_RUN/checkpoints/020000/pretrained_model \
  --output /tmp/so101-act-eval
```

Observation-only and motion commands are documented in `scripts/so101_capture/ACT_HANDOFF_STAGE.md`. Supply checkpoint paths explicitly for continuous ACT. Before any real run, confirm camera freshness, calibration, joint limits, Goal/Present synchronization, and the selected Action Chunk mode. Do not treat the synthetic smoke checkpoint as deployable.

## Tests

These checks do not intentionally connect to hardware:

```sh
python -m compileall -q scripts/so101_capture
python -m unittest discover -s scripts/so101_capture -p 'test_*.py'
```

Some imports require the `lerobot` Conda environment and local optional dependencies. Dataset and checkpoint verification are local-only because those artifacts are excluded from Git.

## Publication policy

Publish source code and reproduction instructions only. Do not commit `.env`, `*.local.yaml`, calibration, raw/processed datasets, checkpoints, logs, validation dumps, or experiment media. The original RoboOps teaching repository is not required at runtime and is not redistributed here.
