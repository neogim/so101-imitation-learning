# SO-101 finite-candidate IK trajectory DP review

## Decision

Second-order dynamic programming (DP2) is the best of the three selectors for the cached finite
candidate set. It converts `++` from 1/6 passing directions under recomputed greedy selection to
6/6, and reduces its worst jump/velocity/acceleration below the study thresholds. For `--`, all
three methods pass 6/6; DP2 gives the lowest worst acceleration. `-+` and `+-` cannot obtain full
31-frame paths because residual≤2 mm candidate layers are missing.

This is **CONDITIONAL_NUMERICAL_SENSITIVITY** only. DP did not solve new IK problems and is optimal
only within the cached candidates, whose `previous_witness` seeds were influenced by the earlier
greedy run. The result does not identify physical shoulder/elbow signs, validate safety, authorize
hardware playback, or create ACT training data.

## Input and candidate graph

- Candidate source: `outputs/so101-ik-multistart-565-595-20261028-v2/per_seed.json`
- Train-split Episode 0, frames 565–595, consecutive 20 Hz timestamps.
- Four hypothetical `b=0` shoulder/elbow sign mappings and six URDF-world ±3 mm directions.
- Only candidates already marked feasible with position residual ≤2 mm were admitted.
- Near-identical motor vectors were deduplicated within a frame at `1e-6` degrees; lowest residual
  then deterministic seed/vector ordering broke ties.
- Missing layers were never filled. DP ran independently on maximal contiguous feasible segments
  and never teleported across a gap.

The same candidate pool and costs were used for all selectors. Normalized cost terms were:

- recorded-Action fidelity: mean squared motor deviation / 15°;
- first difference: mean squared motor velocity / 100°/s;
- second difference (DP2 only): mean squared acceleration / 2,000°/s²;
- tool orientation diagnostic penalty: weight 0.05, normalized by 15°;
- active-bound penalty: 0.02 per active joint.

These are study scales and weights, not manufacturer safety limits. Raw metrics, not cost alone,
determine the quality gate.

## Three-method comparison

Each “pass” count is out of six directions and requires 31/31 coverage, residual≤2 mm,
deviation≤15°, jump≤5°, velocity≤100°/s, acceleration≤2,000°/s², and tool ΔR≤15°.

| mapping | method | covered targets /186 | passing directions /6 | max residual mm | max deviation ° | max jump ° | max velocity °/s | max acceleration °/s² | max tool ΔR ° |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ++ | Greedy | 186 | 1 | 0.604 | 45.793 | 6.413 | 128.266 | 2,859.975 | 11.439 |
| ++ | DP1 | 186 | 4 | 0.511 | 7.071 | 6.850 | 136.989 | 2,285.855 | 4.393 |
| ++ | DP2 | 186 | **6** | 0.511 | 9.253 | **4.500** | **89.995** | **1,661.350** | 4.393 |
| -+ | Greedy | 149 | 1 | 1.969 | 90.731 | 68.419 | 1,368.350 | 34,358.942 | 149.679 |
| -+ | DP1 | 149 | 1 | 1.969 | 90.731 | 68.396 | 1,367.892 | 34,358.910 | 149.679 |
| -+ | DP2 | 149 | 1 | 1.969 | 91.467 | 70.771 | 1,415.379 | 28,307.321 | 69.112 |
| +- | Greedy | 46 | 0 | 1.562 | 4.104 | 3.203 | 64.070 | 1,062.342 | 2.617 |
| +- | DP1 | 46 | 0 | 1.562 | 4.104 | 2.881 | 57.617 | 1,062.342 | 2.617 |
| +- | DP2 | 46 | 0 | 1.562 | 4.105 | 2.806 | 56.114 | 1,062.342 | 2.620 |
| -- | Greedy | 186 | **6** | 0.363 | 9.894 | 2.073 | 41.466 | 984.697 | 2.918 |
| -- | DP1 | 186 | **6** | 0.341 | 6.254 | 2.215 | 44.294 | 1,305.905 | 2.311 |
| -- | DP2 | 186 | **6** | 0.341 | 6.254 | 2.215 | 44.294 | **811.461** | 2.311 |

No selected path leaves the cached LeRobot calibration interval. That interval is not a certified
physical safety limit.

### Direction gates

- `++`: Greedy passes x− only; DP1 passes x+, y−, z+, z−; DP2 passes all six.
- `--`: all three methods pass all six.
- `-+`: only z− passes, while x− and z+ contain missing layers.
- `+-`: no method passes because every direction lacks most late-frame layers.

Coverage is the controlling failure for `-+`/`+-`; their smooth segment metrics do not repair the
missing frames. The worst metrics shown above are calculated within contiguous segments only.

## Frames 571→572

The recorded shoulder Action at frame 572 is −100.923°, beyond the hypothetical URDF ±100.000°
bound but within the cached motor interval. DP selects from already bounded candidates; it does not
widen or reinterpret the URDF.

Maximum 571→572 joint jump across the six directions:

| mapping | Greedy | DP1 | DP2 |
|---|---:|---:|---:|
| ++ | 4.097° | 3.367° | 3.367° |
| -+ | 2.413° | 2.006° | 2.005° |
| +- | 2.771° | 2.771° | 2.771° |
| -- | 1.602° | 1.497° | 1.497° |

For `++` DP2, examples include:

- x+: `elbow_+2deg` at f571 (shoulder −99.999°) → `shoulder_+2deg` at f572
  (shoulder −96.632°), 3.367° shoulder jump;
- y+: `shoulder_-5deg` at f571 (−97.575°) → `elbow_-2deg` at f572 (−98.018°),
  1.437° maximum joint jump;
- z+: `shoulder_-5deg` at f571 (−98.772°) → `elbow_-5deg` at f572 (−99.995°),
  1.517° maximum joint jump.

For `--` DP2, all six transitions are ≤1.497°. Selected seed names can change across the boundary,
but the chosen motor vectors remain continuous. A seed label is provenance, not an IK branch proof.

The largest DP2 `++` jump occurs elsewhere and is 4.500°; its largest velocity is 89.995°/s and
largest acceleration 1,661.350°/s². This directly addresses the previous greedy discontinuity
within the finite cached pool.

## Interpretation and limitations

DP2 demonstrates that the previous greedy discontinuity was partly a branch-selection problem,
not an unavoidable property of all cached witnesses. It does **not** show that every feasible IK
branch was sampled, because previous-witness candidates were generated along a greedy-dependent
history. The recomputed greedy comparator also differs from the historic greedy selector because
it applies the new common cost and deduplication rules.

Position residuals and orientation are internal to a hypothetical URDF world frame. Tool
orientation remains unconstrained by IK. Collision, self-collision, swept volume, torque, contact,
gripper behavior, measured TCP accuracy, physical sign/zero calibration, and hardware safety are
not evaluated.

## Recommendation

Use DP2, not per-frame greedy selection, for any further **offline numerical** study of the cached
candidate set. The result is strong enough to justify studying a larger or less greedy-dependent
candidate lattice for `++` and `--`, but not to choose between those physical mappings or generate
ACT examples. Keep `-+`/`+-` stopped until their missing feasible layers are addressed by a
separately preregistered IK search; DP cannot create missing nodes.

Do not expand to 648 frames in this task. A future expansion should first regenerate candidates
with seed families independent of a single greedy history and apply the same DP2 gate.

## Reproduction and tests

```bash
python scripts/so101_capture/select_ik_trajectory_dp_offline.py \
  --candidates outputs/so101-ik-multistart-565-595-20261028-v2/per_seed.json \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --output outputs/so101-ik-trajectory-dp-20261028 --episode 0
```

Six tests cover candidate deduplication/tie-breaking, missing-layer segmentation, DP1 versus
greedy, DP2 versus brute force, timestamp-aware acceleration, and quality-gate failure. Python
compilation passed. Detailed selected paths remain gitignored under `outputs/`.
