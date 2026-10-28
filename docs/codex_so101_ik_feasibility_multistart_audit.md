# Codex task: SO-101 ±3 mm target feasibility and multistart branch audit

## Goal and scope
The prior **matched regularized** 31-frame study (commit `3446651f5cac7cdfe778e98c65625291d39f0e47`, `reports/so101_joint_mapping/regularized_mapping_ik_comparison.md`) was STOP_AT_31. Establish whether the ±3 mm targets near frame 572 are **numerically feasible inside the hypothetical URDF limits**, or whether a local solver / starting seed / solution branch fails to find valid solutions. **Do not try to choose the physical shoulder/elbow mapping from numerical quality.**

Offline only: never connect hardware, command motors, change torque, read registers, change calibration, operate ACT, modify recorded dataset, alter URDF limits, or generate ACT training samples. No images/video or private calibration source in Git.

## Experiment plan
1. Read previous regularized report, JSON and scripts; preserve prior baselines. Verify Episode 0 belongs to the existing fixed train split, timestamp validity and 20 Hz. Read Action (not State) for IK numeric targets. Analyze frames 565–595 inclusive, especially 570–574 and 572.
2. Four explicit **hypothetical** mappings for shoulder/elbow signs: ++, -+, +-, -- with offset b=0. All other joints remain the previous identity assumption. Note that physical signs and offsets are unverified.
3. Define each target consistently by applying mapping to the recorded Action, running FK, and adding ±3 mm along URDF-world X/Y/Z. Report that this target changes with hypothesis: cross-hypothesis residual is **not** a same-target physically meaningful comparison.
4. First perform a **feasibility-oriented multistart position-only constrained IK** for each candidate × six directions × each frame, using reproducible starts:
   - recorded-angle seed clipped to current URDF limits;
   - previous-frame candidate/solution seed (when available);
   - small deterministic perturbations of recorded angle within limits (e.g. ±2° and ±5° on selected relevant joints);
   - optional finite set of configurations spanning other feasible branches, strictly within URDF limits.
   Use controlled random seed if any stochastic starts. Keep identical budget and candidates across mappings as far as meaningful. Record **per-seed optimizer success, final TCP position residual, active bound joints, solution branch and deviation from recorded Action**. Distinguish valid feasible witness (numeric residual below 2 mm) from proof of global infeasibility (NOT possible with finite multistart search). If no witness: mark 'NOT FOUND / inconclusive', never 'proven unreachable'.
5. Distinguish **target-space reachability** from **trajectory-quality gate**. A feasible position-only solution can still violate motor deviation ≤15°, velocity, acceleration, tool orientation, or branch continuity. Compute orientation geodesic SO(3) difference as previous report did.
6. Investigate **frame 572**: source shoulder Action -100.923°, hypothetical URDF shoulder ±100°, cached LeRobot calibration ±105.231° (not certified physical stop). Explicitly compare clipped recorded seed, target at initial recorded pose outside URDF bounds, and best in-bounds solutions. Record whether the optimizer is effectively projecting an outside-limits reference pose into URDF-constrained space. Never widen URDF limits.
7. Bound checks: separate (a) initial recorded Action outside hypothetical URDF bounds, (b) clipped seed, (c) final candidate at boundary, (d) LeRobot normalization/calibration range. Explain the '145 bound-active vs 144 explained seed clips' discrepancy from prior report at the per-frame/axis/joint level; do not assume every bound-active candidate is a bug.
8. **Optional** finite predeclared regularization sensitivity grid (e.g. recorded weight 0.02/0.05/0.10, previous weight 0/0.01/0.03) only if multistart feasibility study indicates informative improvement, and time budget permits. Clearly separate it from feasibility and do not tune until results appear successful. Do not change physical or URDF limits.
9. Create summary with ± direction breakdown and peak frame/joint, especially 572. If no clear trustworthy result from smoke interval, **STOP at 31** and do not run all 648. 648 only if numeric quality criteria are predefined and pass, and user did not authorize any physical work.

## Technical integrity
- Use objective units: position residual in mm; motor-coordinate angular deviation in degrees after inverse mapping; velocity °/s, acceleration °/s² based on actual timestamps.
- Explain distinction between optimizer's `success` termination flag and actual residual-based feasibility.
- Verify FK consistency by retransforming motor solution to URDF and recomputing position; include tests for sign-aware conversion, boundary handling, deterministic multistart seeds, numerical residual and stop rule.
- All numeric outputs labeled CONDITIONAL_NUMERICAL_SENSITIVITY; no physical TCP accuracy or actual frame alignment/physical calibration is demonstrated. Cameras and the base's slight displacement do not enter this internal URDF/FK numeric evaluation.
- Disclose that orientation, collision, self-collision, swept volume, motor torque, and robot safety are not constrained.
- Any solver false negatives or sampling constraints limit inference; no finite multistart scheme certifies unreachable targets.

## Deliverables and completion
- New offline-only script(s) such as `scripts/so101_capture/audit_mapping_ik_feasibility_offline.py`, plus tests; reuse existing utilities, don't silently change previous result definitions.
- Sanitized, compact `reports/so101_joint_mapping/ik_feasibility_multistart_audit.md` and `.json` describing method, exact runs, aggregate/per-hypothesis/per-direction metrics, best witnesses or 'NOT FOUND', boundary frame/joint, trace to prior 145/144 case, and a recommendation (stay offline, adjust solver setup, or stop).
- Detailed per-seed/per-frame data under ignored `outputs/` only; ensure no source data or private calibration JSON in git.
- Run tests/compilation, review diff, commit and push on `feature/cartesian-action-offline-export`. Return commit SHA, summary, explicit whether frame 572 had feasible witnesses for each of 6 directions, and blockers.
- If computing exceeds available resources, do the highest-information frames 570–574 across all hypotheses/directions first, summarize clearly and distinguish NOT RUN from NOT FOUND. Continue to 31 frames if feasible.
