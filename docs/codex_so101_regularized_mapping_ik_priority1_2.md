# Codex execution: priority 1 + 2, matched regularized IK and temporal quality

## Objective
Run an **offline conditional numerical** comparison of four hypothetical shoulder/elbow sign mappings using identical position-only IK regularization settings, then assess temporal continuity and tool orientation. This is an exploratory test of solver stability, **NOT** identification of physical sign/zero mapping or proof of safe data augmentation.

## Baseline provenance
- Prior unregularized report: `reports/so101_joint_mapping/lightweight_visual_ik_review.md`, commit `83ff16935b3125e49a6d18eafb1bead86391e0cf`.
- Earlier identity-mapping regularized 648-frame study used `recorded_weight=0.05`, `previous_weight=0.01`, but is **not strictly apples-to-apples** with the newer mapping hypothesis code. Reproduce baseline within the same implementation.
- Existing code: `scripts/so101_capture/evaluate_mapping_hypotheses_offline.py` and `evaluate_fullframe_ik_dynamics.py`; inspect actual methods and units before modification.

## Non-negotiable limits
- **No hardware:** do not connect, actuate, change torque, read registers, change calibration, execute ACT or IK on robot, or generate training images/datasets.
- Do not change original URDF, datasets, prior experiment outputs, historical baselines, or physical limits. Keep raw per-frame outputs under gitignored `outputs/`.
- Episode 0 must be train-split; frame/time arrays must be consecutive and valid. Check 20 FPS provenance.
- Mapping candidates: shoulder/elbow signs (++,-+,+-,--), fixed offset b=0; explicitly hypothetical. Keep all other joints as identity assumptions.
- Use exact same solver, FK, position-only residual scaling, URDF bounds, max function evaluations, and warm start policy between hypotheses. Report any solver failures and bound clipping.
- Numerics of TF/TCP/orientation are in **hypothetical URDF world frame**, not measured physical coordinates. Report velocity and acceleration in transformed-back motor coordinates; compare to original motor Action derivatives (not telemetry).

## Priority 1: matched experiment
1. Add a **regularized** mode to the mapping-hypothesis offline script or a separate script. Implement residual terms as [TCP position mm, `recorded_weight * (q_motor - recorded_motor_deg)`, `previous_weight * (q_motor - preceding_candidate_motor_deg)`], or document an exactly equivalent degree-domain residual. Previous-candidate regularization is absent on first frame. Convert URDF candidate back into motor degrees before both priors; do not accidentally penalize the wrong sign.
2. Use `recorded_weight=0.05`, `previous_weight=0.01` and `offset_mm=3`. Keep an unregularized mode **with identical code path** for controls, ideally common initial seed and no continuation differences except explicitly documented experimental variable.
3. On frames **565–595 inclusive (31)** run four mappings × six ±XYZ directions. For each scenario include per-direction and aggregate: optimizer successes, screen passes (success; residual ≤2mm; motor joint deviation ≤15°), position residual distribution, joint deviation, URDF bound activity/clip, calibrated motor interval checks; max candidate motor velocity/acceleration; frame+joint of each peak.
4. Ensure the screen pass rule is explicitly a *numeric diagnostic*, excluding orientation/collision/physical safety. Check positions are in the same frame and timestamps used in derivatives; no silent first-row NaN warnings.
5. Add tests for sign-aware forward/back transforms, sign-aware regularization, bounds, derivative peaks, and a small deterministic IK synthetic case.

## Priority 2: temporal and orientation quality
1. Frame-wise inspect joint maximum deviation, 1-frame jump, finite-difference velocity and acceleration relative to original Action. Flag frame 572 and other outliers, **do not silently accept a 100% numeric pass as a smooth trajectory**.
2. Using full URDF FK transforms, calculate baseline and candidate tool orientation difference (SO(3) geodesic angle, degrees), with peak frame and direction. Orientation is unconstrained during IK; this is a diagnostic, not a pose-IK constraint.
3. Recompute hypothetical forward TCP position residual after mapping candidate motor angles back through the mapping, and compare to solver output. Check calibration interval deviations separately, not as a physical safety test.
4. Define an **explicit stop/go gate** before extending to full 648 frames:
   - all candidate rows optimizer-successful and within numeric position/deviation screen limits;
   - no unexpectedly large jump, speed, accel or tool-angle excursion relative to original; define quantitative *study diagnostic* thresholds from previous original Action distributions and explain them (not safety limits);
   - no unexplained clipping, branch switching, or loss of continuity.
   - if fail: **STOP** at 31 frames, report root-cause hypotheses and proposed fixes; do not expand automatically.
5. If pass, run 648 train frames with same diagnostics and an audit trail of gate decision. Any passed variant remains a *conditional numerical* result, not verified physical mapping or validated ACT augmentation.

## Reporting and verification
- Save small sanitized `reports/so101_joint_mapping/regularized_mapping_ik_comparison.md` and `.json`, stating run status, exact commands, per hypothesis and ± direction metrics, peak frames/joints, comparisons with matched unregularized control, orientation drift and stop/go decision.
- Store full per-frame traces and any images in ignored `outputs/`; don't commit them. Existing LeRobot calibration content is private: do not commit.
- Run offline unit tests + Python compilation. Review staged diff carefully for private user paths, credentials, calibration, raw data; commit/push only code/tests/sanitized aggregate report to `feature/cartesian-action-offline-export` (never force).
- Clearly state that independent physical sign and offset evidence is **still missing**; none of these numerical metrics can select the true mapping.
- Report commit SHA, executed tests, gate, primary findings, and any blockers. If computation is not run, explicitly report NOT RUN and do not invent results.

## Decision target
Decide whether it is worthwhile to continue numerical ±3 mm trajectory research. **Do not create ACT training examples by changing actions while leaving images unchanged.** Physical playback is out of scope.
