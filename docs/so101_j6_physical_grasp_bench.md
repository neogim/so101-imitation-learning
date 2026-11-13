# Phase 1.3 / Step 1 — SO-101 J6 physical grasp bench

## Scope

This is a **new**, isolated, simulation-only, exploratory MuJoCo test, intentionally NOT the saved Episode 4 tabletop scene and NOT the DLS/SAC comparison. We use:

- The **existing SO-101 URDF** joint `gripper` (J6), `gripper_link` and `moving_jaw_so101_v1_link` collision meshes.
- The **already implemented physical Muhi** in `physical_workcell_sim.install_dynamic_muhi` (provisional 50 g, radius 17.5 mm, length 97 mm; these are not verified material/measurements).
- Physical contact, friction and gravity, plus a bounded J6 PD torque controller. J1–J5 are kept kinematically immobile throughout.
- A **temporary virtual setup fixture**: while the jaw closes, the Muhi pose is held in a collision-free search-generated location. Then the fixture is released and only MuJoCo gravity/jaw contact can keep the object in place.

**This is NOT a pickup from the real/saved table.** The fixture-assisted pre-grasp placement explicitly differs from a real picking approach. There is no weld or object pose correction after release. No SAC/ACT/DLS is executed.

## File

`scripts/so101_sim/physical_grasp_j6_bench.py` (separate from `physical_workcell_sim.py`, `compare_dls_sac_table_guard.py` and the URDF).

The script scans jaw-local 3D positions at URDF J6 open limit; requires **positive collision gap** to each jaw and no other robot interpenetration at the initial position, and prefers objects near both jaws. If no valid candidate is found it writes `summary.json` with `NO_COLLISION_FREE_TWO_JAW_START` rather than silently inventing a grasp. This can indicate the provisional 35 mm Muhi cylinder does not fit the SO-101 jaws, source URDF meshes are too conservative, or the search grid is too narrow. **Do not auto-resize Muhi, widen the gripper limits or disable collisions merely to pass.**

During the simulated fixture + release, write `grasp_trace.csv` with joint angle/torque, jaw contacts, signed distances and Muhi xyz/speed/displacement. Report `GRASP_BENCH_PASS` only if bilateral contact persists over >=25% of post-release frames (at least five), final jaw contact is bilateral, final displacement under 10mm, and no deep setup fixture penetration (<-2mm) occurs. These are provisional software criteria, not a physical calibration or a real-world grasp success guarantee.

## Mac smoke run (one command)

```bash
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
OUT="outputs/so101-j6-physical-grasp-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/physical_grasp_j6_bench.py \
  --out "$OUT" --video
open "$OUT"
```

Expected file paths if a collision-free setup is found: `summary.json`, `grasp_trace.csv`, `grasp_side_views.mp4` (if ffmpeg exists), and closeup frame PNGs. If the setup search fails, **only `summary.json`** is expected and that is a valuable result: avoid retrying without inspecting its reason. The view cameras are two free-camera closeups, not calibrated Side/Wrist; they are labeled accordingly.

## Evaluate before expanding

1. Does the setup find a physically nonpenetrating object position near both fingers? If not, inspect the **URDF jaw mesh, Muhi dimensions and search region**, not full-scene physics.
2. Are initial geometry gaps positive? Does closure produce contact with both *fixed and moving* jaw? Is there setup-induced deep overlap?
3. After the fixture is released, does contact alone retain Muhi against gravity? Does it slip/fall? This determines whether to tune grip force/friction or shape.
4. Only if Step 1 succeeds: add a **table** and approach path, then a dynamically controlled arm Lift → Place, preserving the existing swept collision guard. Reusing the safe Episode 4 table with a physics-enabled arm also requires resolving real-versus-collision geometry mismatch.

## Status

Implementation prepared in GitHub; **Mac MuJoCo run pending**. CI checks do not imply frictional grasp success. Do not report Pick/Lift/Place success until backed by physics traces and video.

## Existing reference results

- [Physical grasp execution plan in Notion](https://app.notion.com/p/41785e16f4dc81e0acd7d28fef1be470)
- [Previous safe kinematic Reach experiment, video/CSV/graph](https://app.notion.com/p/41785e16f4dc8113a69dfe1abfe14df8)
- [Existing Muhi drop/push physics workcell](https://github.com/neogim/so101-imitation-learning/blob/feature/so101-sim-data-augmentation/scripts/so101_sim/physical_workcell_sim.py)
