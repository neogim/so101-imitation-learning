# SO-101 ±3 mm IK feasibility multistart audit

## Decision

The finite multistart audit found residual≤2 mm witnesses for every target under `++` and `--`,
but not under `-+` or `+-`. Even where witnesses exist, selected branches can be temporally
discontinuous. Final decision: **STOP after the 31-frame interval; do not expand to 648 frames**.

This is **CONDITIONAL_NUMERICAL_SENSITIVITY**. It does not identify the physical shoulder/elbow
signs or offsets, prove physical TCP accuracy, prove global infeasibility, or authorize hardware
playback or ACT training data. No robot, register, torque, source dataset, calibration, or URDF was
modified.

## Method

Episode 0 is train-only in the fixed split. Action timestamps are consecutive at 20 Hz. For every
mapping, frame, and ±3 mm URDF-world X/Y/Z target, bounded position-only IK used deterministic
starts:

- recorded Action mapped to URDF coordinates and clipped to current URDF bounds;
- previous frame's selected witness when available;
- shoulder motor coordinate ±2° and ±5°;
- elbow motor coordinate ±2° and ±5°.

Clipping and duplicate removal make the actual seed count smaller near a boundary. Frame 572 has
six unique starts per case. The maximum budget is 100 function evaluations per start.

A **feasible witness** means final position residual ≤2 mm inside solver URDF bounds, irrespective
of the optimizer's termination flag. `success=False` can still leave a valid residual witness;
conversely `success=True` does not guarantee residual≤2 mm. If no start found a witness, the result
is **NOT FOUND / inconclusive**, never proof that the target is globally unreachable.

The candidate motor angles were mapped back to URDF coordinates and FK was recomputed. Maximum
round-trip residual disagreement was below `1.5e-13` mm.

## Focus run: frames 570–574

The five-frame run was completed first. At frame 572:

| signs | x+ | x− | y+ | y− | z+ | z− |
|---|---:|---:|---:|---:|---:|---:|
| ++ | 0.0003 | <0.0001 | 0.0019 | <0.0001 | <0.0001 | <0.0001 |
| -+ | <0.0001 | **2.0422 NOT FOUND** | 0.3727 | 1.2199 | **3.4648 NOT FOUND** | <0.0001 |
| +- | **5.0942 NOT FOUND** | 0.2118 | **3.7508 NOT FOUND** | 1.5479 | 1.2855 | **3.9882 NOT FOUND** |
| -- | <0.0001 | <0.0001 | <0.0001 | <0.0001 | <0.0001 | <0.0001 |

Values are best residuals in millimetres. All six unique starts at frame 572 agreed on the
feasible/not-found classification in each case. Finite sampling still limits the inference.

## Expanded 31-frame results

The same predeclared starts were then run on frames 565–595. Each mapping has 186 targets.

| signs | seed attempts | optimizer successes | targets with witness | NOT FOUND | seed/branch-sensitive targets | max selected residual mm |
|---|---:|---:|---:|---:|---:|---:|
| ++ | 1,188 | 400 | 186 | 0 | 186 | 0.476 |
| -+ | 1,188 | 946 | 149 | 37 | 51 | 4.114 |
| +- | 1,188 | 1,005 | 46 | 140 | 43 | 10.140 |
| -- | 1,188 | 425 | 186 | 0 | 184 | 0.171 |

Direction breakdown, shown as witness/31:

| signs | x+ | x− | y+ | y− | z+ | z− |
|---|---:|---:|---:|---:|---:|---:|
| ++ | 31 | 31 | 31 | 31 | 31 | 31 |
| -+ | 31 | 18 | 31 | 31 | 7 | 31 |
| +- | 7 | 9 | 7 | 8 | 8 | 7 |
| -- | 31 | 31 | 31 | 31 | 31 | 31 |

Initial values did **not** change the binary feasibility classification for any of the 744
mapping/direction/frame targets: within each target, either all tested starts found ≤2 mm or none
did. They did change the returned solution branch and joint deviation substantially: >5° branch
spread occurred in 186 `++`, 51 `-+`, 43 `+-`, and 184 `--` targets. Optimizer termination flags
also varied strongly, demonstrating why termination status alone is not a feasibility criterion.

## Trajectory quality of selected witnesses

For frames with witnesses, selection favored continuity to the previous selected solution, then
recorded-angle deviation and residual. For NOT FOUND frames, it used the lowest residual. This is
a deterministic audit path, not a globally optimal trajectory.

| signs | max deviation ° | max jump ° | max velocity °/s | max acceleration °/s² | max unconstrained tool ΔR ° |
|---|---:|---:|---:|---:|---:|
| ++ | 52.899 | 19.601 | 392.010 | 7,868.456 | 25.730 |
| -+ | 91.384 | 82.128 | 1,642.527 | 34,358.942 | 149.679 |
| +- | 5.664 | 3.716 | 74.314 | 1,789.031 | 3.710 |
| -- | 49.146 | 12.039 | 240.784 | 7,014.824 | 35.853 |

Thus `++` and `--` establish target-space witnesses but fail the earlier study diagnostics of 5°
jump, 100°/s, 2,000°/s², and 15° tool-orientation change. `+-` is temporally smoother only because
most targets have no ≤2 mm witness. No selected candidate exceeded the cached LeRobot calibration
degree interval, which is not a certified physical safety bound.

## Frame 572 and shoulder limit projection

The recorded shoulder Action is −100.923°. With shoulder `+`, its hypothetical URDF value is
−100.923°; with shoulder `−`, it is +100.923°. Both lie outside the URDF ±100.000° declaration but
inside the cached ±105.231° LeRobot calibration interval. The recorded seed is therefore projected
to approximately ∓100.000° before every constrained solve.

- `++`: all directions have witnesses. The best y+ witness contacts `elbow_flex`; the other best
  witnesses need not remain at the shoulder boundary.
- `-+`: x− and z+ remain NOT FOUND; their best solutions contact both shoulder and elbow bounds.
- `+-`: x+, y+, and z− remain NOT FOUND; best solutions contact both shoulder and elbow bounds.
- `--`: all directions have witnesses without an active bound in the best-residual solutions.

This demonstrates projection of an outside-limit *reference FK pose* into the constrained model.
It does not demonstrate a measured physical joint-limit crossing. Targets differ across mapping
hypotheses, so cross-hypothesis residual ranking is not a same-target physical comparison.

## Prior 145 final-bound contacts versus 144 seed clips

The prior matched regularized run had 24 recorded outside-URDF frames (572–595) per direction.
Across six directions this gives exactly 144 clipped recorded seeds per mapping. The 145th final
bound-active row was a target-induced contact at frame 571, whose recorded Action was still inside
the hypothetical bound:

| mapping | direction | extra frame | active final joint |
|---|---|---:|---|
| ++ | x+ | 571 | shoulder_lift |
| -+ | z+ | 571 | elbow_flex |
| +- | x+ | 571 | shoulder_lift |
| -- | z− | 571 | shoulder_lift |

Seed clipping counts input projection; final bound activity describes the optimizer's returned
solution. They need not match, and the extra contact is not itself a bug or physical-limit event.

## Stop decision and recommendation

The 31-frame gate is `STOP`: not every target has a witness, and the selected witness paths fail
temporal thresholds. No 648-frame run was performed.

Recommended next actions remain offline:

1. Add a trajectory-level optimizer or dynamic-programming branch selector over the already found
   witness set instead of choosing each frame greedily.
2. Precheck bound-aware target feasibility and Jacobian conditioning, especially for `-+` x−/z+
   and all `+-` directions near frames 572–595.
3. If further searching is justified, add a preregistered broader branch seed set or global method;
   continue to label failure as NOT FOUND unless a certificate is available.
4. Do not widen URDF limits, infer the physical mapping from these numerics, create ACT examples
   with unchanged images, or perform hardware playback.

## Reproduction and verification

```bash
python scripts/so101_capture/audit_mapping_ik_feasibility_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --split-dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --output outputs/so101-ik-multistart-565-595-20261028-v2 \
  --episode 0 --frame-start 565 --frame-end 595 \
  --offset-mm 3 --feasible-residual-mm 2 --max-nfev 100
```

Six offline tests cover deterministic bounded seeds, previous-witness clipping, residual versus
termination semantics, peak provenance, sign round-trip, and the expansion stop rule. Python
compilation passed. Per-seed details remain gitignored under `outputs/`.

Orientation is unconstrained. Collision, self-collision, swept volume, torque, contact, camera
alignment, measured TCP accuracy, and robot safety were not evaluated. The URDF loader emitted
neutral self-collision warnings, but collision checking was not used in IK.
