# SO-101 matched regularized mapping IK — priorities 1 and 2

## Decision

**STOP_AT_31.** The predeclared all-hypothesis gate failed, so the experiment was not expanded to
648 frames. Regularization substantially improved solver stability, but it did not make all four
hypothetical mappings pass position/deviation, temporal, orientation, and bound-activity checks.

This is **CONDITIONAL_NUMERICAL_SENSITIVITY**, not evidence for physical shoulder/elbow sign or
zero, not a safety result, and not authorization to generate ACT data or replay motion. No hardware
was accessed and no dataset, calibration, prior output, or URDF was modified.

## Matched experiment

- Train-split episode 0; consecutive frames 565–595 inclusive; 31 frames at median 0.05 s.
- Four fixed `b=0` sign hypotheses: `++`, `-+`, `+-`, `--` for shoulder/elbow.
- Six hypothetical URDF-world TCP translations: X/Y/Z ±3 mm.
- Same FK, bounded least-squares solver, URDF limits, maximum 80 evaluations, and per-frame seed
  (recorded Action transformed to URDF coordinates and clipped) for both methods.
- Unregularized residual: TCP position in millimetres.
- Regularized residual: TCP position plus motor-coordinate priors
  `0.05*(candidate-recorded)` and, after the first frame,
  `0.01*(candidate-previous_candidate)` in degrees.
- Candidate angles were transformed back to motor coordinates before deviation and derivatives.
- Numeric screen: optimizer success, TCP position residual ≤2 mm, motor deviation ≤15°.

Each method/mapping comprises 186 rows (31 frames × 6 directions). Position residual was
independently recomputed after the inverse mapping; maximum disagreement with the solver-reported
value was exactly 0 mm.

## Aggregate comparison

| signs | method | success /186 | screen /186 | max residual mm | max deviation ° | max jump ° | max velocity °/s | max acceleration °/s² | max tool ΔR ° |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ++ | unregularized | 18 | 15 | 0.713 | 24.045 | 26.477 | 529.543 | 17,840.893 | 24.569 |
| ++ | regularized | 186 | 186 | 0.063 | 6.392 | 3.694 | 73.871 | 2,312.496 | 1.626 |
| -+ | unregularized | 146 | 90 | 4.114 | 92.065 | 158.750 | 3,175.046 | 114,498.186 | 149.679 |
| -+ | regularized | 186 | 138 | 4.114 | 37.420 | 57.807 | 1,156.105 | 25,978.833 | 42.986 |
| +- | unregularized | 159 | 16 | 10.140 | 25.310 | 37.668 | 753.346 | 14,900.746 | 25.203 |
| +- | regularized | 186 | 46 | 10.140 | 5.624 | 4.001 | 80.010 | 1,645.788 | 3.527 |
| -- | unregularized | 29 | 20 | 0.410 | 37.437 | 54.394 | 1,087.856 | 36,885.923 | 37.189 |
| -- | regularized | 186 | 186 | 0.012 | 4.812 | 2.066 | 41.330 | 1,048.594 | 1.948 |

Regularization removes optimizer failures in this interval and strongly reduces branch jumps in
three hypotheses. It cannot overcome target/URDF-bound incompatibility in `-+` and `+-`, and it
does not identify the true physical mapping. A lower residual is internal to that hypothetical
coordinate system.

## Peak provenance (regularized)

| signs | residual peak | deviation peak | jump/velocity peak | acceleration peak | orientation peak |
|---|---|---|---|---|---|
| ++ | z− f595, 0.063 mm | z− f595 shoulder, 6.392° | x+ f572 shoulder, 3.694° / 73.871°/s | x+ f572 shoulder, 2,312.496°/s² | x− f577, 1.626° |
| -+ | z+ f590, 4.114 mm | y− f594 wrist roll, 37.420° | y+ f594 pan, 57.807° / 1,156.105°/s | y+ f594 pan, 25,978.833°/s² | y− f595, 42.986° |
| +- | z− f595, 10.140 mm | z− f583 wrist flex, 5.624° | x+ f572 wrist flex, 4.001° / 80.010°/s | x+ f572 wrist flex, 1,645.788°/s² | z− f581, 3.527° |
| -- | x+ f574, 0.012 mm | z− f595 elbow, 4.812° | x− f571 elbow, 2.066° / 41.330°/s | z− f572 elbow, 1,048.594°/s² | x− f581, 1.948° |

Tool orientation is unconstrained by IK; ΔR is the SO(3) geodesic angle between recorded-mapping
FK and candidate FK in the same hypothetical URDF world frame.

## Gate and original-action reference

Original Action peaks within this interval were 2.0° one-frame jump (shoulder, f566), 40.001°/s
(shoulder, f566), and 676.903°/s² (elbow, f573). Study thresholds were fixed before expansion:

- jump ≤ max(5°, 2× original peak) = 5°;
- speed ≤ max(100°/s, 2× original peak) = 100°/s;
- acceleration ≤ max(2,000°/s², 2× original peak) = 2,000°/s²;
- unconstrained tool orientation change ≤15°;
- every row optimizer-successful and inside the 2 mm / 15° numeric screen;
- seed clipping must equal recorded hypothetical-URDF violations, with no additional bound-active
  candidate rows.

| signs | gate result | failed checks |
|---|---|---|
| ++ | FAIL | acceleration; one additional bound-active row |
| -+ | FAIL | screen, jump, speed, acceleration, orientation; additional bound activity |
| +- | FAIL | screen; one additional bound-active row |
| -- | FAIL | one additional bound-active row |

All mappings have 24 recorded frames outside their transformed hypothetical URDF bounds. This
produces 24 explained seed clips per direction (144 per method/mapping). Regularized candidates
have 145 bound-active rows per mapping: one more than the explained 144, so even `--` is not a
clean GO. No candidate exceeded the separate cached LeRobot motor calibration interval.

These thresholds are study diagnostics derived from the recorded interval, not certified SO-101
limits or physical safety criteria.

## Frame 572

The recorded shoulder motor Action is −100.923°, outside the transformed ±100.000° URDF bound
under either shoulder sign but within the cached ±105.231° motor calibration interval.

Regularized results across six directions:

| signs | passes at f572 | max residual mm | max deviation ° | max tool ΔR ° | shoulder candidate range ° | bound-active directions |
|---|---:|---:|---:|---:|---:|---:|
| ++ | 6 | 0.051 | 4.979 | 1.488 | −100.000 … −96.186 | 6 |
| -+ | 4 | 3.465 | 2.508 | 2.531 | approximately −100.000 | 6 |
| +- | 3 | 5.094 | 3.352 | 2.249 | exactly −100.000 | 6 |
| -- | 6 | 0.010 | 2.709 | 1.637 | −100.000 … −99.935 | 6 |

Frame 572 is also the regularized acceleration peak for `++` and `--`, and the jump/speed peak for
`++` and `+-`. The behavior is dominated by the hypothetical URDF shoulder boundary; it is not
proof of a physical limit crossing.

## Stop decision and proposed next numerical fixes

The all-candidate gate is false; no 648-frame run was performed. Root-cause hypotheses are:

1. Recorded shoulder Actions are already outside the hypothetical URDF bound in this interval.
2. Position-only IK is redundant and orientation-unconstrained, allowing different branches.
3. Some sign hypotheses make ±3 mm targets unreachable or poorly conditioned near bounds.
4. Regularization improves continuity but cannot remove geometric infeasibility.

If further offline numerical work is desired, predeclare a bound-aware target feasibility check,
multi-seed branch audit, and sensitivity grid for regularization weights. Do not widen URDF limits,
select a physical sign from these metrics, create ACT examples with unchanged images, or replay
these trajectories.

## Reproduction and tests

```bash
python scripts/so101_capture/evaluate_mapping_hypotheses_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --split-dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --output outputs/so101-mapping-ik-regularized-priority12-20261028-v3 \
  --episode 0 --frame-start 565 --frame-end 595 --offset-mm 3 --max-nfev 80 \
  --recorded-weight 0.05 --previous-weight 0.01
```

Six offline unit tests cover all sign transforms, sign-aware bounds, motor-coordinate
regularization, timestamp derivatives, peak frame/joint reporting, and deterministic synthetic IK.
Python compilation also passed. Full per-frame traces remain gitignored under `outputs/`.

Independent physical sign/offset evidence remains missing. Collision, torque, contact, gripper,
camera alignment, measured TCP, and hardware safety were not evaluated. The URDF loader's neutral
self-collision warnings were recorded but collision checking was not part of this solver.
