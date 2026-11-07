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
