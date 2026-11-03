# SO-101 TurboVLA — offline training (experimental)

This branch starts from `fix/isolate-multicam-episode-encoding` commit `d32d613` and preserves the existing ACT, calibration and safety controls. **No hardware/VLA servo execution is implemented.** Data collection is unaffected.

Third-party policy: https://github.com/ravediamond/lerobot-policy-turbovla-so101 . Verify its license, supported LeRobot/Transformers/Python versions and plugin registration in a **separate environment**. Do not `pip install` into the working robot-capture environment. The default DINOv3 backbone may require gated Hugging Face access. For a first offline smoke test, choose a public backbone only after checking the plugin version.

```bash
python scripts/vla/train_turbovla.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \
  --output outputs/turbovla-muhi53-001

# Execute only in a verified isolated environment
python scripts/vla/train_turbovla.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \
  --output outputs/turbovla-muhi53-001 --device cuda --execute
```

The launcher does not install dependencies, download weights itself, modify the dataset, or access serial ports. It uses LeRobot's `turbovla_so101` plugin and a starter 1000-step training run; quality and generalization are **not** established. Camera handling, dataset task metadata and image feature shapes must be verified with the plugin.

No online inference adapter exists in this branch. Before any live control, validate action units and timing, retain the existing 8° tracking-error/250ms cycle-gap stops, and require separate explicit hardware authorization.
