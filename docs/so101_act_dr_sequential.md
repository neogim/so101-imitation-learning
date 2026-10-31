# Sequential image DR experiments: Joint6 then Cartesian7

Both jobs train a **new** ACT checkpoint from scratch on the existing local 53-episode comparison datasets. The original 20k baselines are never overwritten. No robot connection is made.

- Stage 1: `outputs/so101-act-comparison-staged-v1/joint6`
- Stage 2: `outputs/so101-act-comparison-staged-v1/cartesian7`
- Same ACT architecture and 20k recipe as the repository README. `--dataset.image_transforms.enable=true` and `max_num_transforms=2` activate LeRobot's **default** random image transforms (check the local revision's printed config; this is not a promise of particular per-transform strengths). Camera image augmentation does not modify numeric actions.
- The `dataset.eval_split=0.1` uses LeRobot's built-in split; **it is not guaranteed to reproduce the previously reported fixed 45/8 episode holdout**. For a fair final comparison, evaluate new models on the original eight held-out episodes via the existing evaluation scripts and verify no train/eval leakage.
- Requires the local LeRobot checkout in the `lerobot` environment, two datasets with `meta/info.json`, an active `lerobot-train` command and MPS. No hardware.

```bash
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
git pull --ff-only origin feature/cartesian-action-offline-export
conda activate lerobot
# Prints both training commands, checks LeRobot DR configuration and local datasets.
DRY_RUN=1 bash scripts/so101_capture/train_act_dr_sequential.sh
# If preflight succeeds, start sequential training. Keep charger connected.
caffeinate -i bash scripts/so101_capture/train_act_dr_sequential.sh
```

Optional short smoke (does create new training outputs): `STEPS=10 bash scripts/so101_capture/train_act_dr_sequential.sh`. Beware `--save_freq=1000`, so a 10-step test may not create a checkpoint.

The runner aborts on failed preflight/first training command, uses fresh timestamped output folders and `tee` logs, and refuses overwrite of existing output/log paths. `DRY_RUN=1` only prints commands after preflight. The launcher does not resume interrupted jobs; do **not** restart expecting automatic continuation. The timestamped output prefix is printed at start. Training may exceed a 3h20m absence window and may slow under M1 MacBook Air thermal load. `caffeinate -i` only prevents idle sleep during the foreground command; keep Mac plugged into power and avoid closing its lid.

Important: Before claiming an accuracy improvement, run evaluation with **unaugmented** validation images on identical episode/frame indices and report both Joint6 and Cartesian7 metrics; check the actual local augmentation default parameters. Do not compare a training loss on randomly augmented frames to the prior holdout MAE as if they were equivalent.
