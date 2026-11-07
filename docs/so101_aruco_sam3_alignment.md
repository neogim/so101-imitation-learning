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
