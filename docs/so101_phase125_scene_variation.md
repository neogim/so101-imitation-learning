# SO-101 Phase 1.25 — Scene variation and IK-data generation

**Status 2026-11-08:** Work started; scene-variation previews implemented. **No teacher trajectories, IK, contact physics, ACT-ready episodes or Sim2Real success have been generated or validated yet.**

## Scope and dependencies

- Preserve existing physical SO-101 mounting at table edge and real Side/Wrist cameras.
- Preserve all original LeRobot recordings. No robot or camera devices opened by this PoC.
- Start from measured Muhi (97×35 mm), cup (78 mm diameter × 83 mm high), curved tabletop, Episode 4 scene and the **previously saved cup alignment**. Default fixed yaw deltas are **0°, +30°, -30°, +60°, -60°, +90°, -90°**; the first seven scenes exercise these exact angles with unchanged XY positions. The remaining 13 scenes exercise XY/cup perturbations and mixed layouts.
- Scene transformation is in the **SO-101 base XY plane**, rotating Muhi about its own center and translating the cup without rotating it.
- Baseline and independently varied placements are generated with deterministic RNG seed. No unrestricted exploration around robot collisions.
- Scene previews are **not** ACT training datasets; there are no labeled correct actions until separate IK and evaluation steps pass.

## First runnable implementation

After updating to `feature/so101-sim-data-augmentation`, in the same `so101-real2sim` Python environment:

```bash
DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
OUT="outputs/so101-phase125-scenes-$(date +%Y%m%d-%H%M%S)"

python scripts/so101_sim/generate_scene_variations.py \
  --dataset-root "$DATASET" --episode 4 \
  --extrinsics outputs/so101-eye-to-hand-id1-20261107/result.json \
  --intrinsics assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json \
  --object-layout-json outputs/real2sim/episode4-objects-20261108-004859.json \
  --cup-pose-json outputs/real2sim/episode4-cup-gui-20261108-131429.json \
  --table-extra-left-cm 4 --table-extra-front-cm 5 \
  --table-base-x-cm 12 --table-base-y-cm 7 \
  --table-yaw-delta-deg -9.375 \
  --count 20 --seed 101 --muhi-range-mm 20 \
  --cup-range-mm 20 --yaw-range-deg 30 \
  --output "$OUT"

open "$OUT/contact_sheet.png"
```

Outputs: `scene_000.png` through `scene_019.png`, `contact_sheet.png`, and `manifest.json`. Manifest records seed, source dataset/pose paths, per-scene Muhi/Cup base XY, delta yaw, and **NOT_GENERATED / NOT_RUN / NOT_TESTED** status. All scene output is outside the source dataset.

## Subsequent work, not implemented in this first commit

1. Extract a reference demonstration's **TCP world trajectory** with MuJoCo FK at recorded sample times.
2. Segment Pick / Lift / Place; change Pick waypoints relative to the moved Muhi and Place waypoints relative to the moved cup; interpolate transport, using bounded workspace and appropriate gripper schedule.
3. Solve regularized IK for *each transformed waypoint*, starting near logged joint angles; reject discontinuity, joint limit and known collision risks. Require source robot geometry validation rather than fitting offsets to video.
4. Replay the new action sequence with strict geometry checks, then add clearly labeled idealized grasp following and a *separate* physics/contact validation step.
5. Export synchronized ACT dataset only after image-observation/Joint6 action units and time base match; compare real-only training against real+synthetic, holding out positions/yaws.

## Next-stage acceptance criteria

- [x] Deterministic baseline + 19 varied layouts and static Side preview renderer
- [x] Mutating Muhi/cup geoms leaves table/camera/robot untouched; unit tests
- [ ] Run on the local Episode 4 dataset and inspect generated sheet (CI uses synthetic data, not user's local episodes)
- [ ] Transformed TCP trajectory, IK validation and rejection manifest
- [ ] Kinematic Pick/Lift/Place replay and geometric success checks
- [ ] Optional physical grasp test and ACT-ready export, **only after** preceding criteria

## Main integration status (2026-11-08)

Phase 1 prerequisite PRs are **merged into main** in dependency order:

- [PR #11](https://github.com/neogim/so101-imitation-learning/pull/11): multicamera recording safety changes (merged).
- [PR #8](https://github.com/neogim/so101-imitation-learning/pull/8): Real2Sim MuJoCo MVP (merged).
- [PR #9](https://github.com/neogim/so101-imitation-learning/pull/9): ArUco/camera, measured workcell and GUI (merged).
- [PR #12](https://github.com/neogim/so101-imitation-learning/pull/12): this Phase 1.25 development, **still Draft / unmerged**, now targeting main.

The working branch was created on the former PR #9 head, but GitHub now correctly shows only this Phase 1.25 document, generator and unit-test files in the diff against main. Do not merge #12 before IK trajectory generation and real-data smoke inspection.
