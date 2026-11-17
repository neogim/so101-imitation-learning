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
git tag -a so101-phase6g-jaw-collision-20261118 -m "SO-101 jaw collision compound convex milestone" acd383d585cecdcdb4fa8b28b479d360a67fd2e5
git push origin so101-phase6g-jaw-collision-20261118
```

## 2026-11-18: Root aperture improvement (follow-up)

The reference photo shows an obvious opening at the inner/root side of the two gripper fingers. Original 12 axial slabs improved the Phase 6G middle case displacement from 105.3 mm to 66.148 mm, but this is not stable grasping.

- Updated default Phase 6G mode to `--jaw-collision-mode axial_cross_convex_patches`: 12 longitudinal slices x **3 transverse bands** per jaw. The second local PCA axis reduces convex bridging of concave jaw sections; **this is an approximation, not a dimensional reproduction of the photo**.
- The unrelated STS3215 actuator housing geometry on `gripper_link` is no longer considered a *finger-to-bottle contact geom* in split modes. It remains visible and mechanically attached. Previous `jaw_mesh_audit.json` contained one extra fixed geom beside the 12 split jaw geoms.
- New automatic Phase 6G diagnostics at the **middle IK pose**, without moving the real robot:
  - `jaw_root_gap_profile.json`, `jaw_root_gap_profile.png`: MuJoCo geom-to-geom signed separation for thicker/root end only and whole finger, across nine J6 opening angles.
  - `jaw_root_section_comparison.png`: 2D projections of original CAD triangles and split collision surfaces, independently for fixed and moving jaws. These local projections are not a calibrated shared-frame section nor a measured gap.
  - `jaw_root_gap_error.txt` if the probe fails; Phase 6G physics can still run.
- Backward compatibility: old 12 axial slices are `--jaw-collision-mode axial_convex_slabs`; full original model is `--jaw-collision-mode original`. Do not claim these comparisons differ **only** in segmentation: the new filtering of unrelated actuator housing affects both split modes.
- Original URDF and visual STLs are unchanged. All changes are simulation-only.
- **Before this change** backup branch: `backup/phase6g-12slabs-before-root-gap-20261118`.

### How to confirm
Run the existing Colab notebook's sections 5 through 8 to refresh Git SHA and execute new code. Inspect `jaw_root_gap_profile.png` and `jaw_root_section_comparison.png` as well as `middle/grasp.mp4` and `comparison.csv`. The expected baseline for the immediately previous run was middle XY max displacement **66.148316 mm**, 384 bilateral hold steps; these numbers are not predicted new results.

Root is identified via the thicker end of the CAD mesh as a *heuristic*; a positive pairwise root clearance does not prove all sections have sufficient clearance for Muhi. If the gap remains bridged or the result worsens, retain the baseline and refine partitioning before friction tuning. The current integration has **not yet been executed in Colab**.
