# Shared SAC/DLS collision-constrained local joint shield (2026-11-14)

## Why the same safety constraint belongs to both

Neither a vanilla SAC policy nor a Jacobian damped least-squares (DLS) controller guarantees tabletop collision avoidance. SAC maximizes its learned value/reward and can exploit an environment that allows penetration; DLS solves a local TCP task-space step but can freely pass through obstacles that are not encoded. A kinematic trajectory happening not to collide is **not a safety guarantee**.

The authoritative constraint for the simulated tabletop is the **signed minimum moving-robot-geometry/table distance** (currently >=3mm), with URDF joint limits and swept-path checking. A TCP Z-floor alone can miss contact by a finger, arm link or Wrist camera fixture. For real hardware, validate calibration, sensor faults, joint velocity/torque, braking distance, and independent emergency stop measures; 3mm is merely an experiment parameter.

## Mathematical formulation

Given `q=[J1,J2,J3,J4,J5]`, task target `p_target` and the controller's proposal `q_proposal`, select locally valid `q_safe`:

```
min_q  w_p ||p_TCP(q)-p_target||^2
     + w_o E_grasp_orientation(q)
     + w_a ||q-q_proposal||^2
subject to  q_min <= q <= q_max,
            min_geom_signed_distance(robot(q), table) >= 0.003m,
            same clearance for sampled intermediate configurations.
```

- `q`: candidate joint vector (radians); `q_safe`: adopted joint vector; `q_proposal`: unmodified next q from SAC or DLS.
- `p_TCP(q)`: TCP position via MuJoCo FK; `p_target`: Muhi/cup target; `E_grasp_orientation`: application-specific grasp orientation deviation (not implemented yet, not quietly assumed).
- `w_p,w_o,w_a`: task / grasp orientation / action deviation weighting, with careful attention to units and scaling.
- `q_min,q_max`: imported URDF joint limits; `min_geom_signed_distance`: clearance of **all configured moving link collision shapes**, including fixed and moving fingers, against tabletop.

These formulae describe the desired constrained optimization. **The code below is a finite local candidate shield and NOT a solver for the general constrained nonlinear program.**

## Implementation

- `scripts/so101_sim/coordinated_joint_guard.py` exports `joint_candidates` and `coordinate_step`. It proposes bounded single-axis and multi-axis J2/J3/J4 variations (both angle signs); each candidate is evaluated by the existing **swept geometry** guard, and only feasible ones are compared. Score is a normalized TCP task cost plus a small deviation penalty from the controller's proposal. The shield returns the safest preferred local candidate, or stops when none is feasible. It does **not** hard-code 'J2 open => J3 open' and can alter J4.
- `scripts/so101_sim/compare_dls_sac_table_guard.py`: both DLS and SAC now default to `--guard-mode coordinated`, but `--guard-mode backtrack` remains to compare previous behavior. Their updated `q` is the actually accepted safe q on the next iteration; the original trained checkpoint is not modified. Reports `coordinated_steps` in JSON and statuses in CSV and 4-panel MP4. The home stage also uses the safety checker. This is **kinematic**, not torque-driven physics.
- `scripts/so101_sim/test_coordinated_joint_guard.py`: synthetic test proves the same guard can replace a collision-bound J2 movement with a safe coordinated J3 action and refuses to teleport out of an unsafe initial state.

## Critical blocker

The user's actual saved Episode 4 has initial tabletop collision **before policy inference**:
- `gripper_link` geom 70: −9.3366mm
- `moving_jaw_so101_v1_link` geom 81: −8.0379mm
- recorded joint angles J2 −104.459° and J3 +98.984° violate imported URDF limits.

This shield deliberately refuses to run from that start; do not bypass with silent table translation, joint teleport, zeroed contacts or swept-path skipping. First validate the table/base calibration and finger collision meshes, and establish a shared collision-free valid starting pose. A successful static pose from J2/J3 scan is **not proof of a collision-free transition from the penetrating pose**.

## After scene correction, compare using same SAC checkpoint

```bash
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
python -m unittest discover -s scripts/so101_sim -p 'test_coordinated_joint_guard.py' -v
SAC_MODEL="outputs/so101-sac-experiment-c2-precision10-train-20261111/sac_reward10_success10mm.zip"
OUT="outputs/so101-guarded-coupled-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/compare_dls_sac_table_guard.py \
  --sac-model "$SAC_MODEL" --guard-mode coordinated --clearance-mm 3 --out "$OUT"
```

**At present this command is expected to reject the known unsafe Episode 4 initial pose** (and persist `preflight_diagnostic.json`), so do not rerun without changing the geometry source for a grounded reason. CI synthetic tests do not demonstrate Mac Episode 4 trajectory success, successful Muhi grasp, or physical contact correctness. To train a new SAC to learn this behavior intrinsically, separately enforce identical constraints in the RL environment transition, design collision penalty/observation and evaluate with unseen scenes. A shield is useful regardless of retraining.
