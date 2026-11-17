# Phase 6G: SO-101 jaw collision correction (simulation only)

## Motivation
The real SO-101 photo (2026-11-18 discussion) shows tapered inner jaw faces, small retaining steps and 3D-print texture. In the existing Phase 6G middle case the moving jaw makes contact first and the Muhi slides >100 mm in XY. The original simulation compiled full jaw STLs into convex collision geometries, potentially bridging concavities. This is a hypothesis, not proof that convexity alone caused the slip.

## Design
- **Visual:** retain the original official SO-101 URDF visual CAD STL references, with screws/holes/steps and the original joint transforms. Do not invent geometric changes based on a single photo.
- **Collision:** Phase 6G defaults to `--jaw-collision-mode axial_convex_slabs`. Split original fixed and moving jaw STL meshes into 12 axial slabs each; each slab becomes an individual convex collision geom with the **same URDF collision origin** as the original mesh. This better approximates longitudinal curvature and steps than a single whole-mesh convex hull. It is not exact tooth-level collision or deformable plastic.
- Source URDF/STLs and real-robot configs are never changed. The generated temporary URDF, STLs, and `jaw_collision_build.json` are saved in `<OUT>/collision_urdf/`.
- Each jaw's original visual geometry stays independently visible while its collision geometry becomes compound convex.
- For baseline A/B, pass `--jaw-collision-mode original`. For an alternative number of slices pass `--jaw-collision-slabs 12` (supported: 2–32).
- Phase6C and other callers retain `original` mode by default; only Phase6G opts into the split mode.

## Suggested Colab check
Run notebook sections 5–8 again, check the new Git SHA and the generated `collision_urdf/jaw_collision_build.json`.
Compare the middle video, contact-force graph, and XY travel to the recorded baseline 105.319 mm. Retain the raw original baseline run.
A successful compile/run and smaller XY drift are prerequisites before asserting improved grasping.
If compilation fails or collision gets worse, run the original mode for A/B, then adjust the collision splitting strategy instead of tuning physical friction/IK.

## Rollback snapshot
Branch `backup/phase6g-before-jaw-collision-20261118` points to commit `de85629390809ac27fc98f4f2386d4e1e41f0639`.

**Git tag note:** GitHub connector in this session does not expose create-tag functionality; the backup is a branch, not a git tag. A true annotated tag can be added from a local authorized Git environment:

```bash
git fetch origin feature/so101-dynamic-arm-control
git tag -a so101-phase6g-jaw-collision-20261118 -m "SO-101 jaw collision compound convex milestone" origin/feature/so101-dynamic-arm-control
git push origin so101-phase6g-jaw-collision-20261118
```
