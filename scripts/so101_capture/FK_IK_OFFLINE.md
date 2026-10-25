# FK/IK investigation (offline only)

This feature branch adds a **read-only audit**, not a robot controller or a working IK solver.
It does not install ROS 2, MoveIt, Placo, or alter the current ACT/capture pipeline.

## How to run on the Mac

After checking out this branch, copy the Codex-generated `grasp_pose_summary.csv`
from its temporary location if it still exists, or use the attached CSV:

```bash
conda activate lerobot
python scripts/so101_capture/analyze_fk_grasp_csv.py \\
  --csv /path/to/grasp_pose_summary.csv
python -m unittest discover -s scripts/so101_capture \\
  -p 'test_analyze_fk_grasp_csv.py' -v
```

An optional `--json-out /path/to/new-report.json` writes a **new** report and
refuses to overwrite an existing file.

The CSV is **not** committed. It is a 53-episode snapshot produced by an
independent NumPy FK analysis. The pose is a heuristic from gripper action
transitions, not confirmed contact. This audit checks the geometric spread and
phase ordering, not URDF agreement, physical TCP calibration, collision
clearance, motion safety, or closed-loop success.

## Before any hardware work

1. Verify the upstream SO-101 calibrated URDF matches measured zero
   positions and the physical end-effector transform; do not add homing offsets
   twice.
2. Establish conservative joint planning bounds, accounting for differences
   between URDF limits and follower calibration, without widening safety limits.
3. Validate Placo/LeRobot FK against an independent FK at known configurations.
4. Test numerical IK via FK residuals, joint limit checks and multistart
   convergence; avoid enforcing an unreachable full 6D pose on a 5-DOF arm.
5. Run IK **shadow mode** with no Goal writes; validate switching near contact,
   gripper-open readiness, camera observations and fault handling.
6. Keep existing communication and ActionGuard pathways intact. No automatic
   ACT-to-IK handoff until grasp confirmation is reliable.

Synthetic IK joint trajectories cannot be paired with unchanged side/wrist
images and treated as additional ACT episodes.
