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

## Follow-up trial 2026-11-14: J5 -90° + independent Muhi placement

The user's second Mac test `so101-j6-grasp-fixed-20261114-012551.zip` **did not pass**:
- The new candidate finder sampled 7,514 placements, predicted 174 J6 jaw crossings, and selected a placement (local x=+48mm, y=-30mm, z=-74mm, Muhi axis `across_x`).
- Physical trace: fixed-jaw contacts **0**, despite moving-jaw contacts; post-release bilateral contacts **0/500**; deepest penetration **-2.188mm**, final Muhi displacement **4,605mm** (gravity fall).
- This shows a predicted moving-jaw crossing was NOT sufficient. The fixed jaw never actually supported the object. Do not count geometric predicted collision as a physical successful grasp.

**Change:** Isolated test default J5 is now **-90 degrees about the positive URDF `wrist_roll` axis** (right-hand convention; apparent clockwise depends on viewing direction). The Muhi stays upright along **world Z**, NOT rotated along with J5. Search independently repositions the upright object relative to the new jaw geometry. The script's `--j5-deg` can test another signed angle; it rejects out-of-limit values. Existing table/URDF/world pose data and checkpoints remain unchanged.

Inspect `summary.json`: `j5_wrist_roll_deg`, `muhi_orientation`, `setup_search`, `status`. If no valid placement is found, that is a useful failure: do **not** widen the mesh or silently place the object penetrating a jaw. New dynamic results are pending, so neither bilateral contact nor a 3D physical hold is claimed.

The same command uses -90° by default; explicit form is `--j5-deg -90`. Cameras remain uncalibrated bench closeups. 

## Third run: contact geometry clips through the object, corrected the fixture (2026-11-14)

The attached third Mac run `so101-j6-rotated-20261114-013219.zip` recorded:
- J5 **-90°**, Muhi upright in world Z, white follower.
- 3,971 placement candidates; selected gap 0.5596mm to fixed jaw and 4.2873mm to moving jaw. Predicted moving-jaw overlap at J6 angle ~1.265 rad was **NOT** a physical grasp.
- The old virtual fixture **reset object qpos and qvel to a constant every time step during J6 closure**. This prevented Muhi from being pushed towards the opposing finger, causing a maximum measured contact penetration **-26.852mm**. Fixed and moving contact together occurred only **4/500** post-release steps, final displacement **4,516.7mm**. This is a FAIL, not a valid representation of an end effector passing safely through an object.
- The fact that the moving jaw had contact pairs throughout the held phase DOES NOT mean it was applying physically valid gripping force to a movable object. The forced reset dominated contact response.

**Fix:** remove all per-step object position/velocity resets during MuJoCo integration. While the **setup interval** lasts, apply **bounded compliant force** (spring-to-setup-position plus damping and limited anti-gravity, max 1.5N) to the dynamic Muhi body. At fixture release, applied force becomes exactly zero. This is still a **temporary assisted placement experiment**, not a table pickup or real-world hardware controller. J1–J5 remain kinematically locked and J6 receives bounded PD torque. This preserves MuJoCo's ability to translate/rotate the object in response to collisions in the setup phase.

Success still requires post-release two-sided contacts and stable object position. Significant interpenetration remains an explicit failure. The next run will determine whether the fixed jaw provides support or if jaw collision geometry needs further work. Do not increase physical plausibility claims based on CI or the predicted geometric sweep.
