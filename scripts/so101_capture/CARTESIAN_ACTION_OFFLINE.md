# Cartesian action target export (offline / experimental)

This experiment derives a **seven-component Cartesian training target** from
the existing 53-episode two-camera LeRobot Dataset. It does **not** create
additional demonstration episodes and does **not** train a policy.

The output is a new Parquet sidecar, with one row per original Dataset frame.
Each row shares its original `episode_index`, `frame_index` and nominal
`timestamp`. Because the FK calculations use the original state/action pair,
the side and wrist image observations remain associated with the same sampled
frame. The original videos are **not** copied into the sidecar.

Target fields:

- `delta_tcp_x_m_base`, `delta_tcp_y_m_base`, `delta_tcp_z_m_base`
- `delta_rotvec_x_rad_base`, `delta_rotvec_y_rad_base`, `delta_rotvec_z_rad_base`
- `gripper_goal_normalized_0_100`

Important: `delta_tcp` means **FK(action[t]) - FK(state[t])**. The original
`action[t]` is an absolute joint goal, NOT a measured future state. The
rotation vector is the log of `R_action * R_state.T` in the base frame.
The target is neither a measured frame-to-frame displacement nor a velocity.

## Prepare (Mac, no robot connected)

Use the installed LeRobot and Placo packages; verify local import before
running the full dataset. The upstream SO-101 calibrated URDF is an **external
input** and must be reviewed against the actual follower calibration.
Do not modify the existing robot calibration.

```bash
conda activate lerobot
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
git fetch origin
git switch feature/cartesian-action-offline-export

python -m unittest discover -s scripts/so101_capture \\
  -p 'test_export_cartesian_action_sidecar.py' -v

python -c 'from lerobot.model.kinematics import RobotKinematics; import placo; print("FK dependencies OK")'

python scripts/so101_capture/export_cartesian_action_sidecar.py \\
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \\
  --urdf /absolute/path/to/so101_new_calib.urdf \\
  --output data/analysis/so101_cartesian_action_53 \\
  --expected-episodes 53
```

An unavailable Placo dependency or missing URDF produces an error; the
script deliberately does not auto-install packages or fetch models. Output
folder must not exist, and must lie outside the source Dataset. No source
Dataset or checkpoint is changed.

## Validation gates before any policy training

1. Compare several FK outputs with the independent FK audit and actual known
   robot poses, including TCP frame orientation and coordinate convention.
2. Confirm the URDF joint names are in the state column order; joint indices
   J1-J5 are in calibrated degrees and J6 is the gripper 0-100.
3. Inspect target distribution (units, magnitudes and outliers), especially
   grasp/open transitions. Check if the predicted physical goal is within
   planning limits and reachable.
4. Build a **separate** LeRobot-compatible derived Dataset with explicit 7D
   Action schema and reproducible image/frame correspondence; the current
   sidecar by itself is not accepted by the baseline ACT training recipe.
5. Retrain ACT for this output representation, validate normalization,
   inference time and action chunk semantics.
6. Implement IK decoding and FK residual checks in offline or shadow mode,
   then independent safety review before any hardware command. A 5-DOF
   SO-101 arm cannot generally realize arbitrary 6D rotation commands.
7. Never bypass `ActionGuard`, joint limits, tracking checks, or existing
   emergency/communication safety paths.

Synthetic IK motions must not be paired with unchanged images and called
new demonstrations. This branch creates no augmented episodes.
