# SO-101 Side camera calibration checkerboard

| File | Intended use |
| --- | --- |
| [checkerboard_A4_7x5_inner_corners_20mm.pdf](checkerboard_A4_7x5_inner_corners_20mm.pdf) | **Print this PDF** on A4 at **100% / Actual size**. Disable Fit to page and borderless scaling. |
| [checkerboard_A4_7x5_inner_corners_20mm_preview.png](checkerboard_A4_7x5_inner_corners_20mm_preview.png) | **Preview only.** PNG print scale depends on DPI/printer settings; do not assume 1:1 dimensions. |

- 8 columns x 6 rows of 20 mm squares; **7 x 5 internal corners**.
- Board physical grid = 160 x 120 mm.
- A 100 mm measurement reference line is printed below the grid. Check it with a ruler after printing.
- Measure actual square side length with a ruler/caliper. Pass the *measured* square size via `--square-mm`, not necessarily 20 if the print scale was wrong.
- Mount the print flat on rigid cardboard/board; avoid wrinkles, curves and reflections.
- Use **iPhone → same OBS Virtual Camera scene/transforms → LeRobot Side** path as the dataset. Keep the actual Side camera fixed; move/tilt the printed board into 12–20 varied positions.
- The existing 53 recorded demos are **not** to be modified or rerecorded.
- The ACT Side images are 224x168 (width x height). Checkerboard corner detection may be unreliable at that low resolution; preferably calibrate at higher resolution and apply measured crop/resize transformations to the final camera matrix.

Example once calibration images have been saved:

~~~bash
conda activate so101-real2sim
IMAGES="outputs/real2sim/calibration/side_chessboard_images"
OUT="outputs/real2sim/side-intrinsics-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/calibrate_side_obs_intrinsics.py \
  --images "$IMAGES" --columns 7 --rows 5 --square-mm 20 \
  --output "$OUT"
open "$OUT"
~~~

**Important:** This flat checkerboard can calibrate camera *intrinsics*, but it does **not** supply the independently measured *noncoplanar 3D-to-2D landmarks* required by the current `fit_side_camera_pnp.py` extrinsics script. For that next step, acquire measured 3D landmarks in the SO-101/MuJoCo frame; do not invent dimensions/poses from the rendered pictures.

See [Side camera PnP calibration procedure](../../docs/so101_side_camera_pnp_calibration.md).
