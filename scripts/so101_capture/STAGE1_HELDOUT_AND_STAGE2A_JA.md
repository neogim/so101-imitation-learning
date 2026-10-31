# SO-101 6D ACT: heldout validation and Stage 2A decision plan (2026-10-27)

## Context
- 53 successful original demonstrations, 30,846 frames (20 fps)
- Fixed training 45 episodes (26,296 frames); heldout 8 episodes (4,550 frames)
- Heldout IDs: 5, 8, 15, 18, 23, 31, 46, 51
- 6D joint ACT trained 20k, checkpoints 10k / 16k / 20k
- Previous 160-sample evaluation: mean J1–J5 MAE = 3.8713 / 2.8424 / 2.8099; J6 MAE = 2.5821 / 1.5101 / 2.0586, respectively.
- These numbers are *calibrated action units*, not validated physical joint degrees. No hardware success is inferred.

## Part A: continuous J6 time series (same 160 samples as initial baseline)
First plot the existing sparse results for a quick diagnostic. **20-frame stride is one second** and misses grasp onset timing; do not infer precise closing delay from these plots.

```bash
python scripts/so101_capture/analyze_joint6_validation.py \
 --eval-dir outputs/act_joint6_eval_010000 --label 10k \
 --eval-dir outputs/act_joint6_eval_016000 --label 16k \
 --eval-dir outputs/act_joint6_eval_020000 --label 20k \
 --output outputs/act_joint6_temporal_sparse \
 --threshold 25
```

The output has `episode_XX_J6.png` for eight episodes and `summary.json` with teacher/prediction agreement at J6 > 25. **Threshold is a proxy only**. Check motor/gripper calibration to establish whether increasing J6 is opening or closing; J6 > 25 counts by themselves are not grasp-event recall.

## Part B: all 4,550 heldout frames

This evaluates at every recorded frame, still in **teacher-forced, fresh chunk, first-action-only** mode. Each frame is independent observation from successful expert trajectory, not a predicted closed-loop rollout.
Do not overwrite the existing sparse evaluation output.

```bash
conda activate lerobot
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export

for step in 010000 016000 020000; do
  python scripts/so101_capture/evaluate_joint6_act_offline.py \
    --dataset data/so101/muhi-side-wrist-joint6-trainonly-053 \
    --split-dataset data/so101/muhi-side-wrist-cartesian-smoke-053 \
    --checkpoint outputs/act_joint6_stage1_20k/checkpoints/$step/pretrained_model \
    --output outputs/act_joint6_eval_full_$step \
    --stride 1 \
    --per-episode 10000
done

python scripts/so101_capture/analyze_joint6_validation.py \
 --eval-dir outputs/act_joint6_eval_full_010000 --label 10k \
 --eval-dir outputs/act_joint6_eval_full_016000 --label 16k \
 --eval-dir outputs/act_joint6_eval_full_020000 --label 20k \
 --output outputs/act_joint6_temporal_full \
 --threshold 25
```

Outputs: one `predictions.csv` and `metrics.json` per checkpoint, and one J6 overlay plot per heldout episode. Check all evaluation sample counts equal 4,550. If they do not, stop and inspect split / episode/frame ranges.

## Stage 2A: proposed grasp-focused fine-tuning (do NOT execute before review)
1. Establish true J6 opening vs closing mapping and find stable grasp onset in **train 45 only**. Validate a subset manually against synchronized side/wrist video; J6 values alone can be ambiguous.
2. Create a *sampling manifest* of frame indices/episode IDs without copying or changing camera/action alignment. Put sampled windows near: open approach, grasp onset, secure grasp/lift, release into container. Sample start frames that keep ACT's `chunk_size=50` future actions in the original episode.
3. Maintain a replay share of original train demonstrations to limit catastrophic forgetting. A starting experiment is 50% baseline/replay windows, 50% grasp-focused windows, followed by a ratio ablation. These are hypotheses, not known optima.
4. Begin **from either 16k or 20k**, whichever wins full-frame J6 event diagnostics and arm metrics. Do not automatically select 20k because its training loss is lower.
5. Start low-LR full-policy experiments at `1e-6` with checkpoints after 2k and 5k further steps; isolate the output directory and preserve optimizer/scheduler configuration. Verify LeRobot 0.6.2 resume/from-pretrained syntax before any actual fine-tuning. Merely setting a pretrained path is not always equivalent to resuming optimizer state.
6. Compare *the same* eight heldout episodes, and separately evaluate temporal J6 event timing and arm J1–J5 MAE/RMSE. Check for overfitting and failure on untrained initial object placements.
7. Do not perform any robot playback without collision checks, verified joint calibration, workspace clearance and a supervised emergency-stop arrangement. Offline teacher-forced prediction is not a hardware safety test.

## Acceptance / selection
- Prefer model with lower heldout J6 errors and more correctly timed grasp/release, subject to acceptable J1–J5 errors.
- No single numerical threshold has yet been justified for acceptable arm or gripper error.
- If even 16k/20k predictions do not show usable event dynamics in the full sequence, inspect data alignment, action sign conventions and training sample selection before collecting more data or fine-tuning.
- Document the measured results into the existing Google Sheets workbook and final course report after execution.
