# Codex task — zero-purchase SO-101 visual sanity check + conditional IK sensitivity

## Background
The user does not own inclinometers, fiducials, or a calibrated external camera and does not want metrology-grade validation. Reuse **existing** side camera recording, URDF and CAD. We need useful engineering screening, not fabricated verification. The prior 648-frame IK assumed LeRobot normalized motor degrees = URDF joint angles; five physical mappings remain unverified.

## Hard safety and data constraints
- **Offline only**: no device initialization, motor commands, torque, homing, register read, teleoperation, ACT execution, new training data, or modifications to original datasets/calibration/URDF.
- Report as **VISUAL_SCREENING** or **CONDITIONAL_NUMERICAL_SENSITIVITY**, never calibrated/validated physical TCP or physically verified sign/offset.
- Only read pre-existing recordings, and choose **train-split episodes only** for any analysis used to decide augmentations. Episode 0 train-split status should be checked.
- Keep images, rendered composites, per-frame dumps in gitignored `outputs/`; push only small sanitized Markdown/JSON and scripts/tests. Do not commit screenshots, video, raw sensor data, private paths, calibration JSON.

## Part 1 — visual sanity check from existing video
1. Inspect the current repo and prior reports `reports/so101_joint_mapping/phase2_readiness.md`, `reports/so101_joint_mapping/README.md`, `scripts/so101_capture/evaluate_fullframe_ik_dynamics.py` and existing URDF visualizations under gitignored outputs.
2. Use pre-existing side camera frames 338, 625, and a few intermediate frames (e.g. 114, 252, 451, 508) where useful. Verify episode/frame-to-video indexing, resolution and image provenance; do not confuse state with action.
3. Render URDF for **explicit candidate mappings** for shoulder_lift and elbow_flex, and retain other joints at the previously assumed identity map for a *conditional* comparison. Set a reproducible virtual camera placement, document when **not camera-calibrated**; user will inspect side-by-side mosaics locally.
4. If feasible, produce frame-specific side-by-side PNGs with **labels** for measured recorded State vs hypothetical URDF pose and mapping assumption. Generate local files `outputs/so101-ik-visual-sanity-<date>/` and make a succinct local index. Do not extract physical joint angles from pixels or rank fit by uncalibrated pixel overlays.
5. Qualitative pass/fail allowed only for gross visual contradictions: flipped elbow bend, obvious wrong articulation or unrealistic motion compared with synchronized video. Report ambiguity from viewpoint, joint origins, occlusion, and candidate camera placement. **No claim that a good visual resemblance proves q_URDF mapping.**

## Part 2 — numerical conditional mapping sensitivity
1. Compare a small, predeclared set of **hypothetical mappings**, including identity for both shoulder/elbow and sign-flip for each separately and both together; at minimum s in {+1,-1}, b=0 for those joints. If trying nonzero offsets, choose transparent finite grid such as ±5°, document these are stress-test scenarios, **not estimates**.
2. For each mapping, transform recorded motor **Action** to hypothetical URDF joint angles, run FK and ±3mm position-only IK offline using same settings, then convert candidate IK back to *motor angle coordinates* before reporting maximum motor-coordinate deviation, speed/acceleration. Show both physical-URDF bounds under that hypothesis and LeRobot calibrated degree interval separately. Never mark an arbitrary mapping 'true' based on lower internal FK error, optimizer convergence, or smoothness.
3. Report candidate comparison: 6 directions, optimizer success, position residual in **hypothetical URDF world frame**, joint continuity, candidate clipping vs URDF bounds, differences from baseline. Include direct caveat that **tool position orientation is unconstrained**, camera alignment and collision/torque/gripper limits are not evaluated. When high computational cost, first do a 30-frame contiguous smoke test around 565–595 and only expand to all 648 when robust.
4. Recheck the 572 peak near the URDF shoulder_lift boundary for each scenario, but do not interpret constraint crossing as measured physical limit crossing.
5. Build unit tests for forward/inverse candidate angle conversion and for correct use of coordinate-specific joint limits; execute offline. DO NOT modify the original baseline code to silently reinterpret historical results.

## Deliverables
- `scripts/so101_capture/visual_mapping_sanity_offline.py` and/or `scripts/so101_capture/evaluate_mapping_hypotheses_offline.py` with offline-only CLI (use filenames appropriate to actual repo).
- `reports/so101_joint_mapping/lightweight_visual_ik_review.md` and `lightweight_visual_ik_review.json`: methods, hypotheses, results, checks, evidence and limitations; explicitly state local path of image mosaics without committing media.
- Tested commands reproducible on this Mac and precise local output paths.
- Review staged diff and push to `feature/cartesian-action-offline-export` (no force, no secrets). Report commit SHA, quantitative results, visual findings (or limitations), tests, unresolved calibration.
- **If the current environment cannot render or calculate, write runnable, tested offline scripts and mark experiments NOT RUN — never invent metrics.**

## Decision
Aim to select or reject obvious geometric contradictions and decide whether ±3mm numerical dataset augmentation is worth continuing as an **explicitly exploratory offline experiment**, while deferring precision calibration until necessary. Never authorize playback on a physical SO-101.
