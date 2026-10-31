# SO-101: staged 120-frame validation before 648-frame DP2 (Mac, no Codex)

**Status:** GitHub code prepared; **not run on the user's Mac**. No robot hardware or ACT data is used. The previous independent-seed run at frames 565–595 yielded 6/6 for both ++ and --.

## Why staging?
The previous DP selector hard-coded frames 565–595. It now accepts `--frame-start` and `--frame-end`. First test an unseen contiguous train interval **frames 0–119** under ++ and --, then assess quality before authorizing a separate whole-episode 0–647 run. The criterion is per-hypothesis/per-direction full coverage and the prior six research thresholds; numerical success does not certify motor safety or physical sign/zero mapping.

## 1. Pull/test/verify

```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python -m unittest discover -s scripts/so101_capture -p 'test_generate_independent_ik_candidates_offline.py' -v
python -m unittest discover -s scripts/so101_capture -p 'test_select_ik_trajectory_dp_offline.py' -v
python -m py_compile scripts/so101_capture/generate_independent_ik_candidates_offline.py scripts/so101_capture/select_ik_trajectory_dp_offline.py scripts/so101_capture/summarize_dp_gate_offline.py
test -f "$FOLLOWER_CALIBRATION_JSON" && echo "Calibration path OK"
```

If the last line does not print, set the variable to the **existing private follower JSON path**. Do not paste its contents.

## 2. 120-frame independent candidate generation

The following is computation-heavy (2 hypotheses × 6 directions × 120 frames × deterministic seeds). Leave all full outputs under ignored `outputs/`. Use a different `--output` directory if one exists. Do not blindly delete old outputs.

```bash
python scripts/so101_capture/generate_independent_ik_candidates_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --split-dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --mappings 'shoulder+_elbow+' 'shoulder-_elbow-' \
  --output outputs/so101-ik-independent-000-119-stage-v1 \
  --episode 0 --frame-start 0 --frame-end 119 \
  --offset-mm 3 --feasible-residual-mm 2 --max-nfev 100 \
  > outputs/so101-ik-independent-000-119-stage-v1-console.log
```

The redirected console log is compact; do not commit it.

## 3. DP2 selection, same dataset and private calibration

```bash
python scripts/so101_capture/select_ik_trajectory_dp_offline.py \
  --candidates outputs/so101-ik-independent-000-119-stage-v1/per_seed.json \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --output outputs/so101-ik-dp-000-119-stage-v1 \
  --episode 0 --frame-start 0 --frame-end 119 \
  > outputs/so101-ik-dp-000-119-stage-v1-console.log
```

## 4. Small summary to share in chat

```bash
python scripts/so101_capture/summarize_dp_gate_offline.py \
  --summary outputs/so101-ik-dp-000-119-stage-v1/summary.json \
  --mappings 'shoulder+_elbow+' 'shoulder-_elbow-' \
  --output outputs/so101-ik-dp-000-119-stage-v1/compact_gate.json
```

Paste only this output (or error) into ChatGPT. It reports 12 direction outcomes and `FULL_NUMERICAL_GATE: PASS/STOP`.

## 5. Acceptance, STOP conditions, and next step

The quality gate on EACH of 120 contiguous frames across 6 directions and each hypothesis is: position residual ≤2mm, recorded motor deviation ≤15°, per-frame max jump ≤5°, speed ≤100°/s, acceleration ≤2000°/s², tool orientation difference ≤15°, no missing candidate layers. These are numeric study thresholds, **not** certified physical robot safety limits. No physical limit inference from normalized servo calibration.

- If STOP: **do not launch 648**. Send compact summary. Diagnose which directions/frames fail, then review candidate coverage, joint bounds, and branch transitions.
- If PASS: do not yet run 648 automatically. Share compact summary first; the assistant will prepare the final scoped run/checkpoint and review whether full-episode feasibility and computational burden justify it.
- The selector's DP2 works over complete contiguous layers. A small 120-frame study is not itself proof that a 648-frame DP path exists.
- The 31-frame proven interval 565–595 is not part of this 120-frame check; this is an independent temporal stress test.

**Limitations:** all tested sign mappings and zero offsets remain hypothetical; cameras are not used, and the user's later camera/base shifts do not enter these URDF-world calculations. Collision, torque, physical TCP, tool orientation constraints, and ACT image/Action consistency are unverified. Do not move hardware or generate training data.
