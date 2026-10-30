# SO-101 ACT: Joint6 / Cartesian7 with and without image DR

The script evaluates **the same 160 clean input frames** (20 frames each from episodes 5,8,15,18,23,31,46,51) across four final 20k checkpoints. It does not connect to a robot. It creates a separate `summary.json` and `predictions.csv` folder without touching baseline results.

```bash
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
conda activate lerobot
git pull --ff-only origin feature/cartesian-action-offline-export
python scripts/so101_capture/evaluate_act_dr_four_models_offline.py \
  --joint-dr outputs/so101-act-comparison-dr-20261030-184458-joint6/checkpoints/020000/pretrained_model \
  --cart-dr outputs/so101-act-comparison-dr-20261030-184458-cartesian7/checkpoints/020000/pretrained_model \
  --output outputs/so101-four-models-clean-eval-v1
```

Interpretation:

- Compare `joint6_baseline` vs `joint6_dr` on joint/gripper MAEs.
- Compare `cartesian7_baseline` vs `cartesian7_dr` on XYZ Cartesian displacement mm, rotation radians, and gripper errors.
- Do NOT compare the raw ACT loss between Joint6 and Cartesian7 (different output dimensions and normalized targets).
- **Data leakage warning:** The recent training launcher uses `dataset.eval_split=0.1`, and the original 8-episode holdout may have been present in the DR training data. Thus this 160-frame number is a **paired diagnostic**, not a scientifically valid unseen-test comparison until the actual training split provenance is independently verified. If the fixed eight episodes were trained on, retrain using a provably disjoint split or select entirely unseen episodes. The script deliberately reports `held_out_status=UNVERIFIED`.
- This script evaluates unaugmented images. Robustness against shifted brightness, contrast or camera geometry requires a separate controlled perturbation test. No real robot or physics accuracy is established.
