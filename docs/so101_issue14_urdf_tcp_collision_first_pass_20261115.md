# Issue #14 first-pass: URDF / TCP / joint mapping / collision audit (2026-11-15)

**Scope**: Static GitHub repository inspection only. No Mac MuJoCo run, servo commands, physical pose measurement, new physics tuning, collision force proof, or camera image measurement. This is an evidence inventory, not a diagnosis of the exact cause of physics divergence.

## Observed in checked-in source (2026-11-15)

| Topic | Repo evidence | Finding | Status |
|---|---|---|---|
| Joint mapping | `docs/so101_joint_mapping_collision_followup_20261028.md` | Hypothesized motor→URDF sign vector `(+,-,-,+,+)` gave numeric optimization success; real physical zero/sign was **not validated** | OPEN; do not interpret optimization PASS as correct real geometry |
| URDF joint topology | `data/urdf/so101/so101_new_calib.urdf` | 6 revolute joints all defined with local axis `0 0 1`; axes operate in rotated joint frames | STATIC VERIFIED; real mapping OPEN |
| Full TCP pose | Same URDF, `gripper_frame_joint` | `xyz=(-0.0079,-0.000218121,-0.0981274)`, `rpy=(0,3.14159,0)` relative to `gripper_link` | STATIC VERIFIED |
| Bench TCP site | `physical_grasp_table_bench.py` → `gripper_spec.add_site(name=TIP_SITE, pos=list(GRIPPER_FRAME_LOCAL_POS), ...)` | Bench site sets position **but not orientation**, so its local frame is identity; URDF `gripper_frame_link` orientation is ~180° about Y. **Critical for any 6D TCP frame comparison**, not necessarily for current XYZ-only near-contact heuristic | STATIC FRAME MISMATCH IDENTIFIED |
| Jaw collision | URDF `visual`/`collision` often reference same STL; `install_fixed_surface_proxy_bench` and `install_moving_surface_proxy_bench` disable original mesh collision in favor of many BOXes | The rendered and solver geometry *may* differ even when URDF filenames match. Need same-pose overlays and inner-face distances. No final validity claim | OPEN |
| Neutral self contacts | `docs/so101_joint_mapping_collision_followup_20261028.md` and `audit_urdf_collision_pairs_readonly.py` | Six reported parent-child link pairs; adjacent-link warning may not mean physical interference | OPEN |
| Wrist camera | `wrist_camera_sim.py`, preservation contract | Installed in `physical_grasp_table_bench.py` using provisional `ruler_15_35_35`. The model includes collision geoms; three camera views plus two side views are expected | CODE PRESENT; rendering and true calibration unverified |
| Dynamic controls | `physical_grasp_table_bench.py` | Legacy per-step kinematic J1–J5 writes and abrupt `--physical-arm-hold` PD switching are documented. Equality-constraint experiment PR #13 is not validated. | OPEN; avoid tuning before geometric gates |

## Critical distinction

The **bench's 6D TCP site orientation mismatch is a verified source-code fact**. It does NOT prove that this mismatch caused the giant simulated impact: the current bench mainly uses site *XYZ* and the physics break was correlated with PD activation. The direction/pose of an imagined TCP must not be used as a substitute for actual visual and collision finger geometry.

## Existing read-only audits (do not create yet another diagnosis stack)

These two scripts do **not** move hardware or change URDF and already exist in the repository:

```bash
conda activate so101-real2sim
git fetch origin
git switch feature/so101-constraint-grasp-rebuild
git pull --ff-only origin feature/so101-constraint-grasp-rebuild

OUT="outputs/so101-urdf-static-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

python scripts/so101_capture/audit_physical_mapping_urdf_readonly.py \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --output "$OUT/joint_mapping_metadata.json"

python scripts/so101_capture/audit_urdf_collision_pairs_readonly.py \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --output "$OUT/collision_pairs_metadata.json"

open "$OUT"
```

The first command omits optional `--calibration` intentionally; it cannot verify the true motor↔URDF mapping without the user's local calibration data. These scripts inventory metadata, **not** mesh penetration or dynamics.

### After static metadata

1. Reuse existing successful `data/so101/muhi-side-wrist-success-053-20261025` episode 0 (local dataset path from runbook); determine whether the exact successful J1–J6 grasp state is extractable. The GitHub source tree does not establish that this dataset is available locally today.
2. Compare same-joint-angle positions **and orientations** of real vs simulated robot, with the correct calibration, before changing physical forces.
3. Run only ONE same-pose jaw visible-STL/compiled-collider overlay, retaining the photographed wrist camera rig, then diagnose the measured mismatch. Avoid adding a new approximation without evidence.
4. After G0/G1/G2 pass, begin Issue #15 grasp-only physical bench, not `--physical-arm-hold` re-run.

References: [Issue #14](https://github.com/neogim/so101-imitation-learning/issues/14), [Notion learning plan](https://app.notion.com/p/41985e16f4dc8108bf65eae86540d52a), [source runbook](https://github.com/neogim/so101-imitation-learning/blob/feature/so101-constraint-grasp-rebuild/docs/so101_camera_fk_geometric_calibration_runbook.md).
