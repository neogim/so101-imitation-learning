# Table clearance before physical DLS/SAC (2026-11-13)

## Decision: collision-aware kinematic path first, dynamics later

The 2026-11-13 physical tracking experiment **did not reproduce the original visual initial pose** and penetrated scene geometry. Its error gate was intentionally triggered on the user Mac:

- J2 shoulder_lift: −104.46° versus URDF lower limit −100.00° (4.46° violation).
- J3 elbow_flex: +98.98° versus URDF upper limit +96.83° (2.15° violation).
- J1/J4/J5 within limits.
- Therefore the previous physics rollout would force an initial jump. Do not bypass the original comparison validation by silently editing its joint targets.

**Do not repeat that failed physics experiment**. Before resuming physical `mj_step` control, establish a *new* explicitly documented common in-limit starting pose and avoid the tabletop in the **kinematic** joint trajectories. This is a prerequisite, not a substitute for physically valid PD.

## New implementation

- `scripts/so101_sim/table_collision_guard.py`: evaluates the signed surface-to-surface gap between **moving robot geoms** (including wrist rig when present) and the saved tabletop (optional cup), using MuJoCo `mj_geomDistance`. This does not need physics integration or globally enabling collision flags, so normal visual scene is preserved.
- Each proposed DLS/SAC joint step is checked at intermediate configurations, max `--check-resolution-rad 0.01`. Backtrack to 1/2, 1/4, ... if necessary. If blocked, stop rather than penetrate. **It cannot actively search a detour** or guarantee a grasp.
- `scripts/so101_sim/compare_dls_sac_table_guard.py`: creates *fresh* DLS and SAC kinematic rollouts from the **same** clipped joint vector (records J2/J3 adjustment), with the same 3mm table clearance. SAC observes the **filtered actual q** on the next iteration; it is not a postprocessing of a stale open-loop SAC trajectory. DLS recomputes the Jacobian at the filtered current pose. The same filter applies to both, including the return-home stage.
- Saves `guarded_rollout.csv`, `guarded_tcp_error.png`, `summary.json`, and a 2×2 `dls_sac_table_guard_side_wrist.mp4` showing original overhead camera + Wrist for both controllers. The base visual scene and recorded Episode-4 object locations remain unchanged.
- If the common in-limit start is *already* in contact with the table, **preflight fails** with named geom pair and signed gap. That is a diagnostic indicating source geometry or pose alignment needs repair; do not disable collision checking to get a movie. Background/visual geoms are not collision-checked.
- Important: URDF mesh convex hulls (not raw concave STL surfaces) can produce false-positive collisions. Table clearance is conservative, and **arm self-collision is not yet filtered**. If false-positive geometry blocks progress, investigate named geom pairs rather than widening an exclusion list without evidence.

## Recommended Mac command

Find your existing **9D-observation / 5D-action SAC checkpoint**, which was used in the original DLS/SAC comparison. This is *not* the `comparison_rollout.csv` file. Use the same checkpoint as the earlier original run:

```bash
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation

python -m unittest discover -s scripts/so101_sim -p 'test_table_collision_guard.py' -v

# Inspect locally saved SAC model files; select the ORIGINAL 9D/5D model.
find outputs -type f -name '*.zip' | grep -Ei 'sac|reach|policy'

# Replace with the checkpoint path you used in the previous comparison.
SAC_MODEL="path/to/your/saved/sac_model.zip"
OUT="outputs/so101-dls-sac-table-guard-$(date +%Y%m%d-%H%M%S)"

if python scripts/so101_sim/compare_dls_sac_table_guard.py \
  --sac-model "$SAC_MODEL" \
  --clearance-mm 3 \
  --out "$OUT"; then
  if [ -f "$OUT/dls_sac_table_guard_side_wrist.mp4" ]; then
    open "$OUT/dls_sac_table_guard_side_wrist.mp4"
  fi
  open "$OUT/guarded_tcp_error.png"
  cat "$OUT/summary.json"
else
  echo "TABLE GUARD PREFLIGHT OR PLANNING FAILED; use named geom diagnostic."
fi
```

## Acceptance gate before physics

1. The shared initial pose, including J2/J3 changes, is explicitly recorded; do not claim the *previous* out-of-limit visual pose was preserved.
2. The preflight starting configuration has >=3mm signed clearance to the saved table according to tested MuJoCo collision geoms; otherwise stop with the contact pair. Report if convex-hull geometry produces conservative results.
3. All accepted **interpolated** motions preserve the clearance margin for moving links and wrist fixture. No quiet or sudden teleport through a table.
4. DLS and SAC show the target points, stage labels, meaningful camera framing and separate Wrist images in a comparison MP4, just like the kinematic baseline. Record success and blocked steps separately.
5. The *physical* DLS/SAC experiment should only be revisited after the above and self-collision model/PD validation. Kinematic collision-avoidance by itself does not prevent physical momentum-induced penetration, so the physics run needs its own integration and contact checks.

This keeps the original kinematic and physical experiment scripts intact, with a separate opt-in guarded experiment, reducing regressions.
