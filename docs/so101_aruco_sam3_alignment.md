# SAM3 tabletop alignment on the existing ArUco Real2Sim branch

This script extends `feature/aruco-calibration-capture-20261107`. **Do not use
the early `real_to_sim.py --render` example output or the placeholder
`workcell.example.json` as visual references**: they use an uncalibrated
camera, different lighting, and a different table geometry.

The alignment loads the **same camera extrinsics/intrinsics, studio fill
light, white robot preview, and approximate table** used by
`compare_real_sim_motion.py`. Only the existing tabletop and its four legs
are moved in the *compiled* MuJoCo scene. The source calibration, datasets,
robot/camera positions and original scene code remain unchanged.

The `r2s_preview_table` is a simulated 65×60 cm *desk*; it is not a
calibrated tray. A SAM3 mask must refer to the corresponding **real desktop**,
not the transparent cup, the white Muhi bottle, or a separate box.

## Prerequisites

- Active Conda env `so101-real2sim` with Python **3.11** (e.g. 3.11.18)
- Existing original extrinsics and 640×480 intrinsics JSON files from
  the **successful previous comparison run**
- One original Side frame resized to 640×480, not the two-panel screenshot
- One reviewed same-size binary tabletop mask OR actual local SAM3 weights

The prior video code uses `compare_real_sim_motion.py motion --extrinsics
... --intrinsics ...`. Retrieve the **same actual paths from shell history
or run configuration**; do not substitute guesses. The reviewed image should
depict the same physical setup as those parameters. If using SAM3 weights,
use an existing model location, **not** `/path/to/sam3.pt`.

## Run (reviewed binary mask, avoids installing SAM3)

```bash
conda activate so101-real2sim
python --version
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107
python -m unittest scripts/so101_sim/test_align_previous_scene_sam3.py -v

# Replace these paths with actual existing files.
REAL="/actual/path/to/side_frame_640x480.png"
MASK="/actual/path/to/reviewed_table_mask_640x480.png"
EXTR="/actual/path/to/previous_extrinsics.json"
INTR="/actual/path/to/previous_intrinsics_640x480.json"
OUT="outputs/real2sim/aruco-sam3-$(date +%Y%m%d-%H%M%S)"

python scripts/so101_sim/align_previous_scene_sam3.py \
  --real-image "$REAL" --real-mask "$MASK" \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --output "$OUT"
open "$OUT/before.png" "$OUT/after.png"
cat "$OUT/report.json"
```

Alternatively omit `--real-mask` and give
`--sam3-model /ACTUAL/path/to/sam3.pt --bbox X1 Y1 X2 Y2`
(using image-specific 640×480 pixel coordinates).

Output `report.json` has added camera-relative table offsets:
`left_m` (positive = screen-left) and `front_m` (positive = toward the
camera, negative = away). These are applied as **3D XY translations** in
MuJoCo, not by scaling pixels; apparent size changes according to the
camera's perspective. Reuse the suggested
`table_extra_left_cm`/`table_extra_front_cm` only after inspecting
overlays. The current optimizer does **not** calibrate table dimensions,
camera intrinsics, or table yaw; it cannot repair a wrong underlying camera.

Overlay colors: green matched, red real-only, blue simulation-only.
This is a mask-based visual candidate, **not** verified metric pose or
permission to move real robot hardware. If SAM3 mask is unreliable, fix
the image/mask first rather than adding extra optimization passes.

Note: the alignment does not yet copy the episode's dynamic arm pose or
its per-episode Muhi/cup annotations into the segmentation scene. It
isolates tabletop geometry. Where occlusions dominate, use a cleaner
Side frame and inspect the mask rather than interpreting IoU as accuracy.


## Re-render the previous movie using the optimized offsets

The original `compare_real_sim_motion.py motion` CLI now accepts the
suggested table offsets. **Keep all the other arguments identical to the
successful previous video**, including any object-layout JSON and timing
settings. For example, append these two flags with numbers from the report:

```bash
  --table-extra-left-cm REPORTED_LEFT_CM \
  --table-extra-front-cm REPORTED_FRONT_CM
```

The reported values already include the old baseline (4 cm left, 5 cm front).
Do not add them to the old values again. Negative front is **away from camera**.
The default values are unchanged so old invocations render identically.


## 2026-11-08: Corrected axis semantics and why the first SAM3 fit was not enough

**Important correction:** The first optimizer used camera-relative left/front
directions, which translated base X and Y together. It scored **IoU only**,
despite the 640×480 rendered table still appearing oversized and with a
mismatched upper edge. Its previous 'suggested' offsets are NOT a validated
3D alignment. Do not repeatedly change `--table-extra-front-cm` and
assume that means base X.

The revised tool searches **SO-101 base frame** coordinates. By default
`--axis xy` adjusts only X and Y, holds Z and yaw fixed, and moves only
`r2s_preview_table` plus its four legs. `--axis x`, `--axis y`, and
`--axis z` explicitly limit the motion to one base axis (or `xyz`
if height really must be included).

Existing table size, tilt/yaw, camera, Muhi and cup remain unchanged.
Perspective-driven size changes follow automatically from 3D projection.
The tool reports not only IoU but **real/sim bounding rectangles,
per-edge residuals, area ratio and centroid shift**. It warns if
area differs by >10% or any bounding edge differs by >15 pixels.
The rectangles are *image envelopes*, not physical 3D corners.

Run using the SAM3 mask already saved from the **same Episode 4 Side image**:

```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107
python -m unittest scripts/so101_sim/test_align_previous_scene_sam3.py -v

REAL="outputs/real2sim/episode4_side_640x480.png"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"

# Reuse the latest EXISTING Episode 4 SAM3 mask without another 3.2GB inference.
MASK="$(find outputs/real2sim -type f -path '*/sam3-episode4-*/real_mask.png' -print | sort | tail -n 1)"
test -n "$MASK" && test -f "$MASK" || { echo "Episode 4 SAM3 mask not found"; false; }

OUT="outputs/real2sim/episode4-base-xy-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/align_previous_scene_sam3.py \
  --real-image "$REAL" --real-mask "$MASK" \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --axis xy --output "$OUT"

open "$OUT/before.png" "$OUT/after.png"
cat "$OUT/report.json"
```

Copy **only** the reported `recommended_video_flags` into the prior video
command, while preserving `--object-layout-json` and any other existing
episode-specific options. The old camera-relative flags are fixed at
their original defaults `--table-extra-left-cm 4 --table-extra-front-cm 5`,
then the new `--table-base-{x,y,z}-cm` offsets are added. Do not combine
new base offsets with old SAM3's `8 / 10.5` values or double-apply the shift.

A lower mask IoU after moving only base X is not evidence of an error in
the coordinate transform: the model may have wrong geometry, yaw, camera
intrinsics, perspective or mismatched real/Sim occlusions. SAM3 yields a
**2D segmentation**, not a metrically valid world pose. If the warning
persists, inspect the actual visible tabletop **four sides/corners** and
the camera/object model rather than continuing blind translations.

No local user IDs, usernames or absolute home paths should be included
in repository files.


The revised optimizer maximizes an **image-space diagnostic objective**:
mask IoU minus penalties for excess apparent area and the projected
bounding-envelope discrepancy. This is still approximate: occluded mask
regions, camera parameter errors and mismatched 3D table dimensions can
prevent precise overlap. Do not treat the score as independently measured
table pose. The report explicitly warns on residual area/edge discrepancies.

For a **single-axis diagnostic** use `--axis x` or `--axis y`; the other
SO-101 base coordinates will remain fixed. Do not attempt to correct a
wrong table yaw by moving the table in X or Y.


## 2026-11-08: Match Episode 4 props and diagnose Muhi orientation

The first alignment version used the default uncalibrated **Muhi/cup
positions and neutral robot pose**, even when the comparison movie had
used Episode 4's three-click annotations. This made the real/sim
occlusion patterns incomparable and explains why improved table-mask
IoU did not improve Muhi orientation.

Re-run using the **same first-frame pose and object annotations**.
Reuse the saved SAM3 mask, and specify the actual Episode 4 files:

```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
REAL="outputs/real2sim/episode4_side_640x480.png"
OBJECTS="outputs/real2sim/episode4-objects-20261108-004859.json"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
MASK="outputs/real2sim/sam3-episode4-20261108-003728/real_mask.png"
OUT="outputs/real2sim/episode4-aligned-scene-$(date +%Y%m%d-%H%M%S)"

python scripts/so101_sim/align_previous_scene_sam3.py \
  --real-image "$REAL" --real-mask "$MASK" \
  --dataset-root "$DATASET" --episode 4 \
  --object-layout-json "$OBJECTS" \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --axis xy --edge-x-min 440 --edge-x-max 600 \
  --output "$OUT"

open "$OUT/before.png" "$OUT/after.png"
cat "$OUT/report.json"
```

The top-edge strip 440..600px is merely a **working diagnostic
region**, not an automatically verified unoccluded segment. Inspect
the real mask before relying on its error. A single image cannot
guarantee metric 3D correspondence.

To inspect whether the earlier three clicks actually yield the same
body/cap direction in the MuJoCo scene, generate an offline movie using
the unchanged historic table offsets (4cm left/5cm front):

```bash
MOVIE="outputs/real2sim/episode4-muhi-axis-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/compare_real_sim_motion.py motion \
  --dataset-root "$DATASET" --episode 4 --stride 5 --max-frames 60 \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --object-layout-json "$OBJECTS" \
  --output "$MOVIE"

open "$MOVIE/muhi_axis_reprojection.png"
cat "$MOVIE/report.json"
```

The `muhi_axis_reprojection.png` draws **green** actual click
body/cap endpoints and **red** projected model endpoints. The
`muhi_axis_reprojection` object in the report includes per-endpoint
pixel errors. This does not change the bottle's physical pose or
simulate a grasp. To make Muhi move with the gripper in a movie,
`--attach-frame` / `--release-frame` must be provided using
observed frame indices, never invented.

## 2026-11-08: Measured tabletop dimensions

The tabletop was measured as **width 800mm × depth 510mm**. The
`r2s_preview_table` default geometry now uses **[0.400, 0.255, 0.020]m
MuJoCo half-sizes** (40mm thickness remains an estimate). The local X
side is treated as width and local Y as depth; the current yaw is still
a provisional 15 degrees. The legs follow the new tabletop footprint.

**Do not interpret the visible part of the tabletop as its full width.**
The Side camera view is cropped/occluded. It is not necessary or valid
to click an invisible left/right endpoint and call it a measured
correspondence. To add manual point constraints later, mark only
physically visible points on the same identified edge, and use the
known metric dimensions to project the model's hidden edge.
A 2-point visible edge segment provides an image-space line constraint,
not an independently determined 6DoF pose.

The previous +12cm/+2.25cm SAM3 translation was fitted to the old
650×600mm tabletop and is invalid for the corrected dimensions.
Re-run from the **baseline offsets** (4cm camera-left, 5cm camera-front)
using the saved Episode 4 SAM3 mask and matching Episode 4 pose/object JSON.
Keep real camera, Muhi and cup unchanged.

After pulling the updated branch, use the concrete Episode 4 command in
the preceding section and inspect its new report. **Do not reapply
the old +12cm adjustment automatically.**


## Reference photo / table coordinates (ACT report slide 3)

Reference: https://docs.google.com/presentation/d/1aMca_m5pZ96-GOnCLb1dOHfdw3RP1GL6/edit?slide=id.p3#slide=id.p3

Slide 3 contains a labeled, near-overhead view of the physical tabletop:
- iPhone 6 Side camera mounted on **table-left**
- SO-101 robot based on **table-right**, Wrist camera attached
- transparent target cup toward **far/top-middle** of tabletop
- Muhi bottle toward **near/bottom-middle** of tabletop
- the diagram explicitly defines **TABLE coordinates**: +X to the image
  right and +Y to the image bottom (Z can be defined as tabletop normal)
- user's measured table footprint: **800 mm horizontally, 510 mm vertically**

IMPORTANT: Those are **table-local axes** in the schematic, **not** the
SO-101 base XYZ axes and **not** the Side-camera optical axes. Their transform
is unknown until the tabletop pose is fitted. Do not treat right/down in the
slide as a guaranteed sign for base X/Y.

The overhead slide establishes object layout, not calibrated perspective or
exact metric object locations. It should be used as a constraint/check when
fitting tabletop yaw, base XY and camera projection. The Side recording crops
or occludes part of the tabletop's horizontal extent: don't invent coordinates
for invisible corners. Compare only visible, physically corresponding edge
segments, then project the remaining 800×510mm geometry.

Do not move the camera, robot, Muhi or cup in response to tabletop mask loss;
the scene-fitting target is tabletop geometry. Existing 3-click Muhi/cup
annotations are a separate constraint and should remain stable.


## Physical reference source checklist (2026-11-08)

Reference folder (user-supplied):
https://drive.google.com/drive/folders/1TtqoxJLrBC0lT_ElRJmlF0PiJtv9n4bo

In addition to the ACT report slide 3, use these original images before
further tabletop geometry adjustments:
- `画像/写真_真上.jpg`:
  https://drive.google.com/file/d/1U_ohHzj3vlByrPHsxFn9DwATpMa-Hyz-/view
- `画像/写真_斜め.jpg`:
  https://drive.google.com/file/d/1uTNq-oSG_FewtH-4xaY5xt4vrpRskqTs/view
- `画像/写真_横.jpg`:
  https://drive.google.com/file/d/10ibc9nm1_DUIOoXyS9CK2kOjPI_-5HxJ/view

Observed: the tabletop has **rounded corners** (current MuJoCo box has sharp
corners), the camera sits on the left, the SO-101 base on the right, and a
second lower board is visible in angled photography. Therefore:
- A silhouette mismatch around the corners is not necessarily XYZ error.
- Do not fit the lower shelf as if it were the top work surface.
- Do not equate the cropped visible Side-camera span with 800mm width.
- Top/angled/side photos are reference geometry, not necessarily synchronized
  Episode 4 frames; do not use their raw pixels as that frame's corners.
- **Fixed measurements**: tabletop 800x510mm. **Provisional**: thickness
  40mm, yaw 15deg, tabletop 3D origin and camera extrinsics.
- Match the *actual visible far edge* and yaw before simply widening the
  XYZ search when X hits its 12cm bound.
- Respect three-click Muhi/cup annotations independently from table pose.

Most recent measured-table fit report (Episode 4, supplied 2026-11-08):
IoU 0.6740 -> 0.8123; apparent area ratio 1.0033 -> 1.0228;
visible far-edge mean absolute error 65.8 -> 31.1 px. X offset hit
+12cm search bound, and right image border nearly matched; this is
**NOT_ALIGNED**, not evidence of calibrated depth. Old 650x600mm fit
numbers must not be mixed with the corrected 800x510mm model.


## 2026-11-08: Joint XY + tabletop Yaw fitting with measured 800×510mm

The earlier position-only solver could not correct a slanted far edge:
it held tabletop yaw at 15°. The latest script adds `--fit-yaw` to fit
the tabletop rigid pose around its center in SO-101 base XY and Z-yaw,
while keeping the measured tabletop dimensions fixed. It rotates four
legs consistently and does not move the camera, SO-101, Muhi or cup.
Use only the **visible upper edge strip** of the SAM3 real mask.
Image-cropped/hidden corners are not invented or annotated.

Recommended existing Episode 4 invocation (repository-relative paths):

```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
REAL="outputs/real2sim/episode4_side_640x480.png"
MASK="outputs/real2sim/sam3-episode4-20261108-003728/real_mask.png"
OBJECTS="outputs/real2sim/episode4-objects-20261108-004859.json"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
OUT="outputs/real2sim/table-xy-yaw-$(date +%Y%m%d-%H%M%S)"

python scripts/so101_sim/align_previous_scene_sam3.py \
  --real-image "$REAL" --real-mask "$MASK" \
  --dataset-root "$DATASET" --episode 4 \
  --object-layout-json "$OBJECTS" \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --axis xy --fit-yaw --max-yaw-delta-deg 30 \
  --edge-x-min 440 --edge-x-max 600 \
  --output "$OUT"

open "$OUT/before.png" "$OUT/after.png"
cat "$OUT/report.json"
```

Inspect `upper_edge_after`, `area_ratio_sim_over_real`,
`iou_after`, `search_bound_hit`, and `warnings`.
`recommended_video_flags` now includes `--table-yaw-delta-deg`.
When creating a comparison video, use **all** reported table flags
with the same `--object-layout-json`; don't stack the previous fit's
+12cm or the earlier 8/10.5cm table offsets.

The existing 3-click annotations are for **Muhi body/cap and cup**, not
table corners. This step does **not** require new clicks and does not
claim a perfect homography or metric 3D calibration. If the top edge
still disagrees, the next meaningful input is a reviewed, visible
physical edge reference with known correspondence—not inferred hidden
corners, and not unbounded XY search.


## 2026-11-08: Cup height correction from upright ruler photographs

The user supplied new photos of the Muhi standing vertically in the
transparent cup with a metric tape. Corrected inference: Muhi length
roughly 100mm, **cup maximum height about 90mm**, not 120mm as
previously assumed from less informative top/side photos. Cup outside
diameter remains approximately 100mm.

The simple MuJoCo cup model's nominal wall height is now **90mm**
(wall from z=2mm to z=92mm, including the 2mm base offset).
This is a **photo-based estimate**, not a precise caliper measurement;
the cup's real asymmetric cut-out is still **not** modeled, so some
silhouette differences are expected. No modification to table size,
table XY/Yaw, camera calibration or three-click object layout.

To see the change, rebuild/re-render using the existing scene CLI
after pulling the established branch; old rendered videos will not
change. The table SAM3 optimizer uses the same geometry module, so
its occlusion comparison can also change slightly and should not be
treated as identical to an earlier report.


## 2026-11-08: SO-101 MUST be clamped at the table edge

Critical physical constraint from the actual workcell: **SO-101 is
fixed to the edge of the real 800×510mm tabletop**, not floating
outside it. The first SAM3 optimizers freely translated the tabletop
without constraining its position relative to the robot, causing the
same unrealistic apparent scale and robot/table separation to persist.

The updated alignment adds an explicit edge constraint. For each
table XY/Yaw candidate, the tabletop+legs are shifted so a selected
table edge passes through the designated SO-101 base-frame anchor.
The camera, URDF, Muhi and cup stay fixed. The axis of the edge
(`x+`, `x-`, `y+`, `y-`) is **table local**, not a Side-frame axis.
Use `auto` to choose the nearest edge from the initial provisional
table pose, then review `report.json.robot_anchor_edge`. This
automatic choice is a **hypothesis**, not a verified physical clamp edge.

IMPORTANT: SO-101 base-frame origin may not coincide with the clamp
contact point. The defaults `--robot-anchor-x-cm 0`,
`--robot-anchor-y-cm 0`, `--robot-edge-inset-cm 0` are **diagnostic
assumptions**, not measurements. If the real clamp point relative to
the robot base is known, supply those offsets. There is no hardware
movement or automatic arm recalibration. Do not claim metric accuracy.

Use the existing Episode 4 input paths from above and run:

```bash
OUT="outputs/real2sim/table-robot-edge-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/align_previous_scene_sam3.py \
  --real-image "$REAL" --real-mask "$MASK" \
  --dataset-root "$DATASET" --episode 4 \
  --object-layout-json "$OBJECTS" \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --axis xy --fit-yaw --anchor-robot-edge auto \
  --edge-x-min 440 --edge-x-max 600 --output "$OUT"

open "$OUT/before.png" "$OUT/after.png"
cat "$OUT/report.json"
```

The new `report.json.recommended_video_flags` includes the selected
edge and clamp-anchor assumptions, in addition to XY and yaw. For
the comparison movie, pass all those flags **together**. If the
selected edge is physically wrong, replace `auto` with the correct
table-local edge after inspecting the top-view photo. Never move
the robot or change camera intrinsics to make an edge look correct.

Note: a 2D realistic picture (or image-to-illustration conversion)
does NOT recover physical depth or coordinates from a single photo.
This rigid edge constraint plus known dimensions is a geometric
assumption to reduce ambiguity, not exact real-to-sim calibration.

## 2026-11-08: Controlled axis diagnosis after failed auto-edge fitting

Do NOT use `--anchor-robot-edge auto` again. Its selected `y+`
was inferred from an already-wrong simulated workcell, and the
assumption SO-101 base origin = actual clamp point was not measured.
The last fit was rejected: IoU **0.4237** (previous XY/Yaw fit
**0.8185**), projected tabletop area ratio **0.659**, upper-edge
residual **215.5px**, yaw **40.625°**.

The model now rejects `auto` rather than silently inventing an edge.
A validated clamp reference would be needed to apply an explicit
`x+/x-/y+/y-` constraint; do not assume which edge in SO-101 base
coordinates from an overhead photograph alone.

For a controlled experiment, preserve the previous visually better
base-X +11cm and yaw=12.5° (delta=-2.5°), and modify ONLY base Y.
Do not mix the failed +7/+9cm offsets with these. Example:

```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
REAL="outputs/real2sim/episode4_side_640x480.png"
MASK="outputs/real2sim/sam3-episode4-20261108-003728/real_mask.png"
OBJECTS="outputs/real2sim/episode4-objects-20261108-004859.json"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
OUT="outputs/real2sim/table-base-y-only-$(date +%Y%m%d-%H%M%S)"

python scripts/so101_sim/align_previous_scene_sam3.py \
  --real-image "$REAL" --real-mask "$MASK" \
  --dataset-root "$DATASET" --episode 4 \
  --object-layout-json "$OBJECTS" \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --axis y --max-delta-m 0.06 \
  --initial-base-x-cm 11 --initial-base-y-cm 6.25 \
  --initial-base-z-cm 0 --initial-yaw-delta-deg -2.5 \
  --anchor-robot-edge off \
  --edge-x-min 440 --edge-x-max 600 --output "$OUT"

open "$OUT/before.png" "$OUT/after.png"
cat "$OUT/report.json"
```

The report's `base_axis_projection_delta_uv_px` shows how 5cm
increments of **robot-base X** and **robot-base Y** project into Side
image pixels. The camera can mix these directions. This is a
diagnostic, not confirmation that the tabletop's long side runs along
robot-base X or Y.

Both score and image-space comparison now start from the **same
explicit initial table pose**. If the final candidate has worse IoU
than that baseline, the report marks regression and omits video
flags. If the fit is not acceptable, stop iterating and locate the
actual robot clamp contact point and desk side in a fixed camera frame,
or use a marker/known physical correspondence. Real2Sim needs a
physically defined transform chain:
`T_base_table`, `T_base_camera`, tabletop dimensions, camera K.
A SAM3 2D segmentation mask does not by itself recover these metric
3D transforms. See OpenCV's solvePnP/reference projection and MuJoCo
frame definitions.

## Non-mutating SO-101 clamp/tabletop inspection (2026-11-08)

Before any more SAM3 optimization, inspect the robot-table relationship
already stored in the MuJoCo model. The table is known to measure
800x510mm, and the robot is physically clamped on its edge, but the
actual position of the clamp contact relative to the URDF base origin
has **not** been measured. Therefore do not use guessed edge constraints
or optimize translation and yaw blindly.

A new offline command produces a **top-down base XY diagram** and
numerical distances from the SO-101 base origin to each tabletop edge.
It performs no hardware actions and does not alter the scene:

```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

OUT="outputs/real2sim/mount-diagnostic-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/inspect_table_robot_mount.py \
  --extrinsics outputs/so101-eye-to-hand-id1-20261107/result.json \
  --intrinsics assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json \
  --table-base-x-cm 11 --table-base-y-cm 6.25 \
  --table-yaw-delta-deg -2.5 \
  --output "$OUT"

open "$OUT/table_robot_topview.png"
cat "$OUT/table_robot_geometry.json"
```

The plot is in **SO-101 base XY**, not camera pixel coordinates,
and shows the robot URDF base origin, *not* a measured clamp point.
`edge_signed_inward_distance_m` contains signed orthogonal
distances to the four **infinite** tabletop edge lines; the plot
also shows finite edge segments. A distance near zero on an infinite
edge line does not establish clamp contact when projected outside the
segment. The script does not automatically label a physical edge as
the correct mounting edge.

Use previous overhead/angled photos as qualitative reference. If the
clamp contact point's offset in base XY cannot be deduced from existing
images and URDF, a single orthogonal photo with a ruler by the clamp
is preferable to another round of SAM3 mask fitting. Do not invent
an anchor location or change the known 800x510mm dimensions.


## 2026-11-08: Curved front edge in the real tabletop

Photo `IMG_5424.jpeg` confirms the real desk's front outline gently bows
outward, and the front corners are rounded. MuJoCo's current work surface
is a straight-sided rectangular box. Do **not** force camera/base pose
to align the curved image-right edge in the recorded Side view.

The SAM3 alignment CLI adds `--ignore-right-px N` to drop the rightmost
N Side-image columns from the **objective only**. It does not crop images,
alter the recorded masks, or change table model dimensions/camera/robot
pose. The report still includes full-image IoU and bounding rectangles.
For an initial 640px Side diagnostic, try `--ignore-right-px 80`, and
set the upper-edge strip fully inside that retained area, e.g.
`--edge-x-min 400 --edge-x-max 550`. Both are **provisional image
regions** and should be visually confirmed, not interpreted as physical
corner correspondences.

A high full-image IoU is not required to declare the visible straight
far edge satisfactory if remaining mismatch is due to genuine tabletop
curvature. If the far edge is occluded in the selected strip, move the
strip rather than inventing hidden points. Robot-table mounting must
remain an independent fixed-location constraint.


## 2026-11-08: Measured curved tabletop depth (supersedes 510mm rectangle)

New physical measurement provided by the user:
- tabletop width: 800mm
- depth at center: **550mm**
- depth at edge: **470mm**
- center-to-edge difference: **80mm**
- back edge appears approximately straight; the front edge bows outward.

Thus the previous 800×510mm *rectangular* MuJoCo geometry is
**not a faithful model of the physical tabletop**, and its apparent
perimeter differences cannot safely be interpreted as camera pose or XY/Yaw
error. The single 510mm depth is not a global physical depth.
The exact cross-section curve, left-versus-right edge symmetry,
measurement reference points, thickness and orientation are not independently
verified. Do NOT silently adopt a precise parabola or new mesh.

Next implementation should use measured tabletop edge constraints
(center 550mm, edge 470mm, width 800mm), with an explicitly approximate
smooth profile, or omit the curved front perimeter from optimization and
fit only the visible straight rear edge. Confirm both end depths if exact
geometry is required. Keep the camera, 3-click Muhi/cup and robot pose fixed.
Existing 800×510mm rectangle remains in code until the geometry change is
implemented and separately tested.


## 2026-11-08: Measured curved desk geometry (supersedes rectangular approximation)

The user confirmed **rear-to-front depths measured from the same straight rear edge**:
left end **470mm**, center **505mm**, right end **470mm**; width **800mm**.
Earlier "510mm depth everywhere" was a rectangular approximation. The
new physical footprint keeps the rear edge fixed and approximates the
front edge with
`D(x) = 505mm - 35mm * (x / 400mm)^2`,
where x is table-local lateral position and `|x| <= 400mm`.
This interpolates three measured positions; detailed curvature and corner
radii were **not measured**.

For compatibility with the established camera/robot/Yaw code, the
MuJoCo tabletop now consists of the original named **BOX**
`r2s_preview_table` covering the shared 470mm rear portion plus
32 narrow BOX strips representing the extra curved 0-35mm front
projection. All pieces share the same brown appearance, transform
together with the four legs and are included in the SAM3 tabletop
segmentation mask. The visual contour is piecewise approximate rather
than a perfect smooth mesh. The bottom shelf/corners are not fitted.

This replaces a trial single-mesh approach that changed the
compiled geom orientation through MuJoCo mesh principal-axis
normalization. The stable BOX implementation passed the Real2Sim and
ArUco CI tests before this final parameter/documentation revision.

**Run again after pulling** the main established branch. The old
alignment and rendered videos used a different table footprint and
should **not** be used to validate this one. Recommended command:
```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
REAL="outputs/real2sim/episode4_side_640x480.png"
MASK="outputs/real2sim/sam3-episode4-20261108-003728/real_mask.png"
OBJECTS="outputs/real2sim/episode4-objects-20261108-004859.json"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
OUT="outputs/real2sim/table-measured-curve-$(date +%Y%m%d-%H%M%S)"

python scripts/so101_sim/align_previous_scene_sam3.py \
  --real-image "$REAL" --real-mask "$MASK" \
  --dataset-root "$DATASET" --episode 4 \
  --object-layout-json "$OBJECTS" \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --axis xy --fit-yaw --max-yaw-delta-deg 30 \
  --edge-x-min 400 --edge-x-max 550 \
  --ignore-right-px 80 --output "$OUT"

open "$OUT/before.png" "$OUT/after.png"
cat "$OUT/report.json"
```
The right-edge exclusion is optional and provisional: its purpose is
to avoid scoring rounded corners and areas of uncertain correspondence,
**not** to hide a poorly matched straight far edge. Full-image report
metrics remain available. Do not reapply previous +12cm offsets; only
use the new report's proposed flags after reviewing the overlay.


## Episode 4: prioritize cup ground-contact localization (2026-11-08)

The table fit with measured curved front improved full silhouette IoU from
0.674 to 0.813, but X reached the +12cm search bound and the far tabletop
edge was still 26.3 pixels off (report: NOT_ALIGNED). Do not treat that
table pose as calibrated or apply it automatically to object geometry.

**Root issue in the cup's old three-click placement:** the 3rd click marks
an unspecified *visual cup center*, while the code intersected that
camera ray at base Z=55mm, an unverified assumption. This can shift the
3D cup position relative to the robot/table. A table-only SAM3 objective
cannot repair it. The new one-click refinement preserves the two Muhi
endpoints/yaw and original cup-center pixel, but adds an explicit cup
**tabletop contact-center** pixel.

```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
ORIGINAL="outputs/real2sim/episode4-objects-20261108-004859.json"
CUP_JSON="outputs/real2sim/episode4-cup-contact-$(date +%Y%m%d-%H%M%S).json"

python scripts/so101_sim/refine_cup_contact.py \
  --dataset-root "$DATASET" --episode 4 \
  --source-json "$ORIGINAL" --output "$CUP_JSON"
```

Click **once** on the middle of the cup's *bottom footprint where it
contacts the tabletop* in the recorded Episode 4 Side frame. Do NOT
click the rim or visual center. If the contact is invisible, cancel
rather than invent a point. The new JSON stores `cup_contact_px`
separately and leaves Muhi annotations untouched.

When using this new JSON, **pass**
`--cup-click-height-cm 0.1` (base Z=1mm in the current approximate
scene) to both image comparison and SAM3 scripts, e.g.:

```bash
OUT="outputs/real2sim/cup-check-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/compare_real_sim_motion.py motion \
  --dataset-root "$DATASET" --episode 4 \
  --stride 5 --max-frames 1 \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --object-layout-json "$CUP_JSON" \
  --cup-click-height-cm 0.1 \
  --output "$OUT"
open "$OUT/real_vs_sim_episode.mp4"
cat "$OUT/report.json"
```

This is an **independent cup localization correction**. Keep the table
defaults for the first diagnostic. The cup wall is still a simplified
90mm high transparent approximation, without its real asymmetric
cutout. A single contact pixel does not determine its orientation or
full 6DoF pose; compare actual first-frame projection before accepting
the new position. Do not use overhead phone photos as if they came
from the calibrated Side camera.


## 2026-11-08: Real / Sim cup-position editor (recommended next step)

The prior SAM3 table-only alignment overfitted a partially occluded,
rounded tabletop silhouette. The cup center had been back-projected from
a *single image point at an arbitrary height*. Do not try more invented
bottom contact clicks. Instead edit the MuJoCo cup pose visually with
the fixed camera, the existing three-click Muhi annotation and recorded
Episode 4 robot joints. New script:

`scripts/so101_sim/align_cup_gui.py`

- Left: **recorded 640x480 Real**, middle: **640x480 MuJoCo Sim**,
  right: **semi-transparent overlay** of both.
- Sliders: base-frame X / Y (±300mm) and Z (-50..+100mm),
  plus overlay alpha. They move **only the cup's 17 preview geoms**.
- Each slider update re-renders the same virtual perspective camera:
  depth motion changes projected size **naturally**, unlike resizing
  a 2D overlay. SO-101 base X is not necessarily camera depth.
- The adjusted pose is expressed as an additive displacement from
  the *unchanged legacy three-click layout* and saved as standalone
  JSON; moving other objects/URDF or actual hardware is forbidden.
- Saved pose specifies Episode, input annotations and table offset
  settings for reproducible replay. No local username or absolute path.

```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
OBJECTS="outputs/real2sim/episode4-objects-20261108-004859.json"
CUP_POSE="outputs/real2sim/episode4-cup-gui-$(date +%Y%m%d-%H%M%S).json"

python scripts/so101_sim/align_cup_gui.py \
  --dataset-root "$DATASET" --episode 4 \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --object-layout-json "$OBJECTS" \
  --table-extra-left-cm 4 --table-extra-front-cm 5 \
  --table-base-x-cm 12 --table-base-y-cm 7 \
  --table-yaw-delta-deg -9.375 \
  --output "$CUP_POSE"
```

Drag the cup base-XYZ sliders until Real and Sim appear aligned; check
both real/sim and overlay. Then click **SAVE JSON** before closing.
The existing table pose is a *visual candidate*, not validated 3D
calibration; changing the table changes this alignment reference.

Re-render the episode with the **same** table flags and original
annotation JSON, adding the saved cup pose. Note that the GUI does
not include SAM3: use SAM3 masks later as **an optional fit score**
for the selected cup, not as the authority for unverified 3D geometry.

```bash
VIDEO_OUT="outputs/real2sim/episode4-cup-gui-video-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/compare_real_sim_motion.py motion \
  --dataset-root "$DATASET" --episode 4 --stride 5 --max-frames 60 \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --object-layout-json "$OBJECTS" \
  --cup-pose-json "$CUP_POSE" \
  --table-extra-left-cm 4 --table-extra-front-cm 5 \
  --table-base-x-cm 12 --table-base-y-cm 7 \
  --table-yaw-delta-deg -9.375 \
  --output "$VIDEO_OUT"
open "$VIDEO_OUT/real_vs_sim_episode.mp4"
```

If Matplotlib launches without an interactive window on macOS,
choose a native GUI backend (e.g. `MPLBACKEND=MacOSX`) rather than
installing unrelated GPU/Quest tooling.

Meta Quest 3 is *not* a shortcut to metric SO-101 workcell calibration.
MuJoCo Studio itself supports a native macOS app (.dmg) as of 2026;
the GUI in this repository uses the established MuJoCo Python bindings
and Matplotlib instead, to compare the exact recorded Episode 4 frame.


## Cup diameter/height GUI controls (after episode4 GUI alignment)

The saved prior cup adjustment from the user's Episode 4 report is
`[-32,-63,-1]mm` in SO-101 base X/Y/Z, stored at repository-relative
`outputs/real2sim/episode4-cup-gui-20261108-131429.json`.
The visual comparison suggested that the *projected cup width was
too large and its maximum height slightly too low* even after this
position adjustment. This does NOT by itself prove physical dimensions
were measured incorrectly: perspective and the simplified geometry
may contribute. Edit the geometry in 3D and compare the same image.

The GUI now includes **cup diameter (60–130mm)** and **cup height
(60–140mm)** sliders, in addition to XYZ and overlay alpha. The
height adjustment anchors the cup's bottom and moves the top; diameter
scales the 16 walls radially and updates the base cylinder radius.
No 2D image-scale manipulation. All cup geometry parts are updated
rigidly in translation while the robot, Muhi, camera and table remain
fixed. The simplified cup still does not model the real rim cutout.

To keep the **previous GUI position** instead of starting from zero,
pass `--initial-pose-json` and choose a NEW output path:
```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
OBJECTS="outputs/real2sim/episode4-objects-20261108-004859.json"
OLD_CUP="outputs/real2sim/episode4-cup-gui-20261108-131429.json"
NEW_CUP="outputs/real2sim/episode4-cup-size-$(date +%Y%m%d-%H%M%S).json"

python scripts/so101_sim/align_cup_gui.py \
  --dataset-root "$DATASET" --episode 4 \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --object-layout-json "$OBJECTS" \
  --initial-pose-json "$OLD_CUP" \
  --table-extra-left-cm 4 --table-extra-front-cm 5 \
  --table-base-x-cm 12 --table-base-y-cm 7 \
  --table-yaw-delta-deg -9.375 \
  --output "$NEW_CUP"
```

Click SAVE JSON to retain both size and location, plus an aligned PNG.
Existing video replay accepts these same files through
`--cup-pose-json "$NEW_CUP"` with the same episode and table flags.
Older GUI JSON without size fields defaults to diameter 100mm and
height 90mm. Keep these sizes as candidates until confirmed by actual
measurements and reprojection, not as physical ground truth.


## 2026-11-08: User-measured object dimensions supersede visual fit estimates

The user measured these physical object dimensions (not photographic
estimates):
- **Transparent cup outside width/diameter: 78 mm; maximum height: 83 mm**
- **Muhi bottle total length: 97 mm; diameter/width: 35 mm**

These measurements supersede previous model defaults of cup 100x90mm
and Muhi 100x38mm. The MuJoCo scene builder uses the measured
dimensions; Muhi retains its 65% body / 35% cap length ratio and the
existing position/orientation from the original three-click layout.
The cup's full-height segmented walls still omit the physical notch
and remain an approximation.

The GUI now displays **fixed measured dimensions**, without
editable diameter/height sliders. Only cup base XYZ and overlay alpha
remain adjustable. Existing GUI pose files are read **for XYZ only**;
older hand-tuned size fields are ignored and replaced with the user's
measurements. The comparison-video renderer also always uses the
same measured cup dimensions, even for older JSON data. This avoids
silent geometry drift between GUI and rendered video.

Reopen from the previous location edit, with a fresh output filename:
```bash
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
OBJECTS="outputs/real2sim/episode4-objects-20261108-004859.json"
OLD_CUP="outputs/real2sim/episode4-cup-gui-20261108-131429.json"
NEW_CUP="outputs/real2sim/episode4-measured-objects-$(date +%Y%m%d-%H%M%S).json"

python scripts/so101_sim/align_cup_gui.py \
  --dataset-root "$DATASET" --episode 4 \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --object-layout-json "$OBJECTS" \
  --initial-pose-json "$OLD_CUP" \
  --table-extra-left-cm 4 --table-extra-front-cm 5 \
  --table-base-x-cm 12 --table-base-y-cm 7 \
  --table-yaw-delta-deg -9.375 \
  --output "$NEW_CUP"
```

The previously fitted cup base XYZ offsets (-32,-63,-1mm) are loaded,
but those remain **visual estimates**, not physical measurements.
Click SAVE JSON for a new pose file and comparison PNG. The original
three-click image point and tentative camera calibration are unchanged.
Do not tune physical geometry to hide perspective/extrinsic errors.


## 2026-11-08: SO-101 five-joint pose inspection GUI (frame 0 / 24 / 40 / 52)

The earlier 15-second overlay shows an arm silhouette mismatch even
at frame 0, and the mismatch persists at frames 24, 40, and 52. The
source motion renderer assumes all five joint signs +1 and offsets 0deg.
This is *unverified*; do not immediately blame ArUco or re-do desk
fitting. A new **offline diagnostic** Matplotlib GUI
`scripts/so101_sim/compare_real_sim_robot_overlay_gui.py` keeps
previous camera, measured table, Muhi, cup dimensions and **saved cup
XYZ pose** frozen. It exposes five temporary joint-offset sliders
(±25deg, 0.25deg increments), frame selector and alpha blending.
The same values persist across all four frames. No robot control,
no new data capture, no SAM3/Quest setup. The initial pose overlay
is only a *visual candidate*, not a mechanical calibration.

Use the exact already saved objects and cup pose (repository-relative
paths) to avoid silently discarding previous placement work:

```bash
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
conda activate so101-real2sim
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only origin feature/aruco-calibration-capture-20261107

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
EXTR="outputs/so101-eye-to-hand-id1-20261107/result.json"
INTR="assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
OBJECTS="outputs/real2sim/episode4-objects-20261108-004859.json"
CUP_POSE="outputs/real2sim/episode4-cup-gui-20261108-131429.json"
JOINTS="outputs/real2sim/episode4-robot-joints-$(date +%Y%m%d-%H%M%S).json"

python scripts/so101_sim/compare_real_sim_robot_overlay_gui.py \
  --dataset-root "$DATASET" --episode 4 \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --object-layout-json "$OBJECTS" --cup-pose-json "$CUP_POSE" \
  --table-extra-left-cm 4 --table-extra-front-cm 5 \
  --table-base-x-cm 12 --table-base-y-cm 7 \
  --table-yaw-delta-deg -9.375 \
  --output "$JOINTS"
```

Compare **frame 0** first. Adjust J1-J5 only as diagnostic
candidates; check **24, 40, 52 with unchanged offsets** before clicking
SAVE JSON + PNG. Save exports the offsets and a single complete four-row,
three-column Real / Sim / Overlay contact sheet
(`"$JOINTS"` and `"${JOINTS%.json}.png"`).
If the initial frame fits but the later frames do not, stop tuning
static joint offsets and investigate frame timestamps / servo mapping
instead of forcing scene geometry to match.

Re-render with the **same** scene and candidate offsets:

```bash
VIDEO_OUT="outputs/real2sim/episode4-robot-pose-video-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/compare_real_sim_motion.py motion \
  --dataset-root "$DATASET" --episode 4 --stride 5 --max-frames 60 \
  --extrinsics "$EXTR" --intrinsics "$INTR" \
  --object-layout-json "$OBJECTS" --cup-pose-json "$CUP_POSE" \
  --robot-joint-offsets-json "$JOINTS" \
  --table-extra-left-cm 4 --table-extra-front-cm 5 \
  --table-base-x-cm 12 --table-base-y-cm 7 \
  --table-yaw-delta-deg -9.375 --output "$VIDEO_OUT"

open "$VIDEO_OUT/real_vs_sim_episode.mp4"
```

The replay script validates episode, object annotations, cup pose,
and table settings against the saved offset JSON to prevent mixing
calibration states. Do **not** apply this file directly to SO-101 motors;
it is only a rendering diagnostic. If the GUI cannot open interactively
on macOS, run with a native Matplotlib backend such as
`MPLBACKEND=MacOSX python ...` after verifying the environment supports
it.

**Frame-index mapping correction:** The earlier movie was rendered at
`--stride 5` (20fps source => 4fps video). The UI labels
`frame 0 / 24 / 40 / 52` are **movie frame positions**, corresponding
to the dataset's **source frames 0 / 120 / 200 / 260** and elapsed times
**0 / 6 / 10 / 13 seconds**. The GUI now defaults to
`--comparison-stride 5` and explicitly maps these indices, so the
inspected robot pose is synchronized to the actual prior comparison
video. Do not treat these labels as raw dataset indices.
