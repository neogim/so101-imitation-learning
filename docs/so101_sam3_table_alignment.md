# SO-101: SAM3 / MuJoCo table silhouette alignment (offline)

This is an **optional, uncalibrated** refinement of `feature/mujoco-real-to-sim`.
It **does not drive the robot**, change camera positions, rewrite raw episodes, or
modify the source workcell JSON.

## What changes with distance?

The optimizer moves `r2s_table` in MuJoCo world **X/Y meters**, retains its
physical dimensions and fixed Side-camera pose, then renders a fresh segmentation
mask using MuJoCo's perspective camera. Thus moving the table **farther along the
camera's viewing direction reduces its apparent pixel area**; moving it nearer
increases it. The actual farther direction depends on camera orientation:
**do not assume +X is always farther**. No 2-D scaling shortcut is used.

## Important: identify the same physical object

Current workcell `r2s_table` is a *solid tabletop primitive*, **not a separate
tray with calibrated walls**. Segment the corresponding tabletop region in the
real Side-camera image. If the intended target is a separate box/tray, first add
a correctly named/positioned MuJoCo tray mesh or geom, then adapt the target
geometry in the alignment script. Matching tabletop against a tray mask will
produce meaningless optimization.

SAM3 text masks can be ambiguous and can merge objects. Prefer a box prompt
around the tabletop, inspect `real_sam3_mask.png`, and reject poor masks.
Camera intrinsics/extrinsics, joint mapping and geometry are not established
by matching one silhouette; scale and depth remain potentially confounded.

## Installation / execution on the local Mac (outside LeRobot hardware env)

In the existing `so101-real2sim` Conda environment, install the optional
inference dependencies when using direct SAM3 rather than a precomputed mask:

```bash
python -m pip install numpy pillow "ultralytics>=8.3"
# Obtain sam3.pt in the supported way for your environment; confirm version.
```

Capture a **single source Side image** (not the combined real-vs-sim MP4 frame).
The image must correspond to the existing camera setup and workcell and should
show the tabletop. No camera activation is performed by the script.

```bash
python scripts/so101_sim/align_sam3_table.py \
  --real-image /path/to/side_frame.png \
  --bbox 30 20 280 210 \
  --sam3-model /path/to/sam3.pt \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --config scripts/so101_sim/workcell.example.json \
  --output outputs/real2sim/sam3-align-001
```

The bbox is just an **example**, not measured coordinates. Replace with the
tabletop bbox in your original image; the image and generated SAM3 mask must
have the same dimensions. If your SAM3 pipeline already generated a binary
mask, pass `--real-mask /path/to/reviewed_mask.png` (and no model required).
Foreground is white (>127); background is black.

A unique output directory is mandatory. Outputs:
- `real_sam3_mask.png`, `sim_table_mask.png`
- `overlay_before.png` and `overlay_after.png` (green overlap; red real-only; blue sim-only)
- `report.json` with IoU, apparent area ratio and centroid discrepancy
- `workcell.candidate.json` with modified table XY, still `calibrated=false`

**Review both overlays before using the candidate config.** If mask quality is
poor, fix the prompt/bbox or take a clearer existing frame instead of adding
complex downstream corrections. If the table is outside the simulated camera
view, fix the camera calibration/scaffold before optimizing. Position-only
optimization cannot fix incorrect camera focal length, yaw/pitch, geometry,
or a mismatched object identity.

Tests (no robot connection):
```bash
python -m unittest scripts/so101_sim/test_align_sam3_table.py -v
```
