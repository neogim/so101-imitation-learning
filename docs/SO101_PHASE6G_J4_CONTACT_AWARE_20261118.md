# Phase 6G: J4>=0 contact-aware placement search (2026-11-18)

## Why
With the repaired SO-101 jaw collision geometry (12x3 convex patches), the historical `middle` candidate grasped Muhi with sustained bilateral contact (500/500), although the bottle moved 27.55 mm. An alternative `middle_j4_nonnegative` with J4≈0° reached the nominal fixed anchor to 4.32 mm, **but 0/500 bilateral contacts**: it looked too far away. Anchor-position error alone is not a correct measure of actual jaw-to-bottle placement.

## Implementation
Changed `scripts/so101_sim/phase6g_fixed_workspace_grasp_depth.py` on branch `feature/so101-dynamic-arm-control`. When the nonreverse candidate is enabled, explore **18 alternate gripper-local 3D anchor positions** (3x3x2) with bounded IK, always J4>=0° and with Muhi unchanged in WORLD coordinates. For each solution measure:
- fixed-jaw to Muhi gap at open J6,
- moving-jaw to Muhi gap at open J6,
- both distances again at the closed J6 target.
Choose the geometry-feasible IK solution minimizing a heuristic for small fixed gap and opposing moving contact, while rejecting >0.5 mm open-jaw penetration and IK error beyond `--max-ik-error-mm`.

The search uses MuJoCo's actual collision meshes and produces:
- `j4_nonnegative_contact_search.csv`: all 18 candidates, scores, gaps, IK errors, wrist angle
- `j4_nonnegative_selected_pose.json`: selected candidate with source diagnostics
- `middle_j4_nonnegative/grasp.mp4`, `comparison.csv`: dynamic validation, if preflight passes

`middle` remains a baseline. **No changes to friction, collision meshes, camera, target bottle position, or physical robot.** The mode `--no-j4-contact-search` restores the old single-anchor nonnegative J4 IK behavior.

## Reproducibility
[New dedicated Colab notebook](https://drive.google.com/file/d/1yKiUzdvzZ_wxht2iYL5m7tacbO5ZMEXg/view) (source is a standard .ipynb): run cells top to bottom, save results to Drive, inspect the 18-row search table and both videos. Original notebook remains available.

## Interpretation
This is a **candidate search**, not a proven grasp fix. The geometric score does not directly ensure normal directions pass through the cylinder center or that both fingertips touch after MuJoCo dynamics, so judge the video and contact trace before iterating. The next iteration, if needed, should use actual contact-point elevation/radial offset and contact normals. Friction remains fixed until after geometry is solved.

Rollback: `backup/phase6g-before-j4-contact-aware-search-20261118`.
