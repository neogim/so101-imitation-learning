# SO-101 J2/J3 coupling vs tabletop, grounded in SAC J5 ablation video (2026-11-13)

## Observed video

User-provided `so101-sac-j5-ablation(2).mp4`: 44.17 s, 1280x720, 30fps, two panels: left normal SAC, right *same checkpoint with J5 action set to zero*. This is **NOT a DLS-vs-SAC video**. The first frame is a shared, visually aligned home that violates URDF J2/J3 limits. Around 8s, both elbows/wrists swing down with different configurations, and the gripper visually intersects or is very close to the orange tabletop in the right panel; the normal SAC left panel points toward Muhi. This is visualization-only; no actual contact response. The visual appearance cannot alone determine whether a specific J2 sign or J3 direction is always safe. The gripper/finger underside position depends on *all* joints plus table-to-base transform and attachment geometry.

The independently saved `preflight_diagnostic.json` from the SAME Real-to-Sim Episode-4 layout establishes initial static signed gaps: gripper link mesh ID 70 −9.3366mm; moving jaw ID 81 −8.0379mm. This pre-existing overlap is not caused only by SAC action, because it exists at the clipped initial pose before any SAC inference. The source visual pose has −10.1980mm. J2 −104.46° and J3 +98.98° exceed imported URDF ranges before clipping.

## Correct control strategy

Avoid a hard-coded `J2 opens -> J3 opens` rule, which can fail for different poses/angle conventions. Instead, enforce coupled joint feasibility:

```text
q = [J1,J2,J3,J4,J5]
SAC or DLS proposes q_desired
Find a nearby q_safe minimizing ||q_safe - q_desired||
subject to q_min <= q_safe <= q_max;
           min signed distance of all moving links & gripper fingers to table >= 3mm;
           each interpolated pose is safe;
           optionally other objects, self-collision and arm speed limits.
If none exists: stop or invoke a true detour planner, NEVER tunnel.
```

The existing `TableCollisionGuard.filter_step` performs simple action scaling, not such constrained optimization. It **cannot rescue an already penetrating initial pose**. The next physics run must not start from that pose; first validate a geometric baseline, then plan a collision-free transition from a verified safe initial state.

## Implemented independent J2/J3 diagnostic

[Source code](../scripts/so101_sim/diagnose_j2_j3_table_clearance.py) samples a **grid of J2 and J3 angles** within ±30° around the *URDF-clipped* Episode-4 start. J1/J4/J5 are held fixed. For each pair it computes minimum signed distance of moving robot collision geoms (including both fingers and wrist rig) to saved tabletop, plus TCP z. It exports a CSV grid, heatmap PNG and JSON `j2_to_safe_j3_lookup` showing a safe J3 angle per proposed J2 *if one exists*. It does not run SAC, alter scene, move objects, simulate time, or guarantee that a safe state can be reached from the unsafe original start. Positive clearance applies to sampled static geometry only, not continuous motion.

Mac:
```bash
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
OUT="outputs/so101-j2-j3-clearance-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/diagnose_j2_j3_table_clearance.py --out "$OUT" --samples 25 --span-deg 30 --clearance-mm 3
open "$OUT/j2_j3_clearance_map.png"
cat "$OUT/summary.json"
```

Only one run is needed. Depending on the result, distinguish a feasible J2/J3 compensated pose from cases where the saved scene/table calibration or collision geometry must first be corrected. A static safe pair is **not** enough to drive the robot: it can be disconnected from the unsafe start by intermediate collisions. Once a verified safe start exists, augment both DLS and SAC with the same coordinated collision-constrained safety projection/trajectory planner and compare them. Add physics only after the kinematic path and initial state pass clearance checks.

Do not claim the new map is measured until the user runs the script with the local source data. The test suite only tests synthetic sampling and lookup.
