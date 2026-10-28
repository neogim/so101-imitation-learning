# SO-101 lightweight visual and conditional IK review

## Decision

- Visual result: **VISUAL_SCREENING — INCONCLUSIVE**. No mapping showed a gross contradiction
  strong enough to reject it, and no mapping is confirmed.
- Numerical result: **CONDITIONAL_NUMERICAL_SENSITIVITY — UNSTABLE SMOKE TEST**. The requested
  565–595 interval completed, but the position-only IK was highly hypothesis/seed/bound dependent.
  The evaluation was therefore not expanded to all 648 frames.
- Current ±3 mm augmentation is **not ready to continue in this configuration**. This is not a
  statement that ±3 mm is physically unsafe or that any sign hypothesis is true/false. No ACT
  training examples or synchronized synthetic observations were generated.

Everything was offline. There was no robot connection, register access, motor command, torque
change, calibration, teleoperation, hardware IK, or source dataset/URDF modification.

## Provenance and split check

Episode 0 is in the fixed train split and absent from the validation split. Dataset timestamps are
20 Hz and frame indices are consecutive. The side-video episode start maps frame 0 to video frame
0, so frames 114/252/338/451/508/625 were compared with their same-index measured
`observation.state`. Action was used only for the conditional IK experiment, as required.

Local visual artifacts (not committed):

- `outputs/so101-ik-visual-sanity-20261028/`
- six labeled mosaics plus `index.json`

Local numerical artifacts (not committed):

- `outputs/so101-mapping-ik-smoke-20261028-v2/`
- aggregate `summary.json` and per-frame diagnostic JSON

## Visual screening

Each mosaic places the recorded side frame beside four URDF skeleton renders:

1. shoulder +, elbow +
2. shoulder −, elbow +
3. shoulder +, elbow −
4. shoulder −, elbow −

All use `b=0`; other arm joints retain the prior identity assumption. The virtual camera is fixed
(elevation 20°, azimuth −55°) for reproducibility but is **not calibrated or registered** to the
physical side camera. The render includes URDF joint origins and axes; link meshes and the real
camera projection are not pixel-aligned.

Observed screening result:

- The real frames clearly show the arm changing between the extended/folded recorded poses.
- The four hypotheses produce visibly different shoulder/elbow articulation in the URDF render.
- Perspective, base orientation, link occlusion, and the missing camera-to-URDF transform prevent
  a reliable correspondence between the photographed bend direction and rendered axes.
- None of the four can be rejected solely as an obvious impossible articulation across all six
  frames. A visually plausible render does not prove sign or zero.

No physical angle was extracted from pixels and no overlay/pixel score was used to rank mappings.

## Conditional ±3 mm IK method

For each mapping, recorded motor Action was transformed as `qU = s*qM` for shoulder/elbow,
FK produced a hypothetical URDF-world TCP position, and six targets at ±3 mm in X/Y/Z were solved
with bounded position-only least squares. Candidate URDF angles were converted back with
`qM = s*qU` before motor-coordinate deviation, velocity, and acceleration were calculated.
Tool orientation was unconstrained.

The smoke interval is frames 565–595 inclusive (31 frames). Each mapping has 186 solves
(31 frames × 6 directions). “Screen pass” means optimizer success, residual ≤2 mm, and maximum
motor-coordinate deviation ≤15°. It is a numerical diagnostic, not an acceptance or safety test.

URDF bounds were transformed separately for each mapping. The cached LeRobot degree intervals
were also reported separately: shoulder_lift ±105.231° and elbow_flex ±97.363°. These calibration
intervals are observed motor ranges, not certified physical limits.

## Smoke-test results

| shoulder/elbow signs | optimizer successes / 186 | screen passes / 186 | max residual (mm) | max motor deviation (°) | max velocity (°/s) | max acceleration (°/s²) | clipped seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| + / + | 87 | 2 | 0.251 | 66.393 | 842.238 | 16,500.484 | 0 |
| − / + | 145 | 90 | 4.114 | 91.466 | 1,535.641 | 30,712.892 | 102 |
| + / − | 153 | 8 | 10.140 | 67.585 | 1,451.688 | 56,398.187 | 124 |
| − / − | 149 | 8 | 0.315 | 54.163 | 326.884 | 6,548.754 | 0 |

All mappings have 24 recorded frames outside their *hypothetical* transformed URDF bounds and
zero returned candidates outside solver bounds. That does not describe physical limit crossing.
The reported rates are finite differences of counterfactual motor-coordinate waypoints, not motor
telemetry or certified dynamics.

The large deviations, very large finite-difference rates, optimizer-success variation, and
direction-dependent pass counts show that unconstrained-orientation position IK is selecting very
different redundant solutions. Lower residual or higher success therefore cannot identify the
physical sign mapping.

## Frame 572 boundary review

Recorded shoulder Action is −100.923°. It maps to hypothetical URDF −100.923° for shoulder `+`
and +100.923° for shoulder `−`, outside the symmetric ±100.000° URDF declaration in either case,
while remaining inside the cached ±105.231° motor calibration interval.

Across six directions, returned shoulder motor coordinates were:

| signs | candidate range (°) | maximum hypothetical residual at frame 572 (mm) |
|---|---:|---:|
| + / + | −96.816 … −64.793 | 0.251 |
| − / + | −100.000 … −99.808 | 3.465 |
| + / − | exactly −100.000 | 5.094 |
| − / − | −88.525 … −86.633 | <0.000001 |

These differences are solver/bound behavior in four hypothetical coordinate systems, not evidence
that the physical shoulder crossed a limit.

## Why full-episode expansion stopped

The preregistered condition was to expand only if the initial interval was stable. It was not:
the best/worst screen totals differ by 45×, two hypotheses repeatedly depend on clipped prior
seeds, and all mappings contain >54° candidate deviations. Running 648 frames would multiply an
already underdetermined result without resolving its cause.

A future offline-only study could predeclare joint regularization to recorded Action, compare
multiple seeds, constrain solution branch continuity, and report both unregularized and regularized
results. That would still be conditional numerical sensitivity, not training data or physical
mapping validation.

## Reproduction

Visual mosaics:

```bash
python scripts/so101_capture/visual_mapping_sanity_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --output outputs/so101-ik-visual-sanity-20261028 \
  --episode 0 --frames 114 252 338 451 508 625
```

Numerical smoke test:

```bash
python scripts/so101_capture/evaluate_mapping_hypotheses_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --split-dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --output outputs/so101-mapping-ik-smoke-20261028-v2 \
  --episode 0 --frame-start 565 --frame-end 595 --offset-mm 3 --max-nfev 80
```

The calibration environment variable points to the private local Follower cache. Its path and
contents are not committed.

## Tests and limitations

- Three unit tests cover all four forward/inverse maps, sign-correct transformed URDF bounds, and
  timestamp-based motor-coordinate derivatives.
- Both scripts passed Python bytecode compilation.
- Generated PNG/JSON media diagnostics remain gitignored under `outputs/`.

This review does not evaluate camera calibration, physical TCP, tool orientation, collision,
self-collision, contact, torque, gripper limits, swept paths, certified joint limits, or hardware
replay. The Placo URDF loader emitted neutral-pose collision warnings; collision checking was not
part of the position-only solver. Shoulder/elbow physical sign and zero remain **UNVERIFIED**.
