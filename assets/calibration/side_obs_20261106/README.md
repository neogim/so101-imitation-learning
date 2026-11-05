# Side camera OBS checkerboard calibration candidates (2026-11-06)

**Images selected:** 16 / 21 input screenshots. OpenCV `findChessboardCornersSB(...,(7,5))` detected all 35 internal corners in each selected crop. The rejected screenshots are listed in `selection_manifest.json` but omitted to minimize repo size.

**Processing:** Source images are 1280×480 screenshots with black side margins. Only the central **640×480** image area (`x=320..959`) is preserved. Cropped PNG data are losslessly re-encoded; the full original screenshot hash (SHA256) is in the manifest. Beware: this crop is a convenience derived from the screenshots, not necessarily the same geometrical transform as LeRobot's 224×168 `observation.images.side` data.

**Camera:** Side = iPhone → OBS Virtual Camera; Wrist = Innomaker U20CAM. This is *only* fixed Side camera intrinsics evidence. Actual phone mounting and OBS transform used for the original 53 demos must still match.

**Calibration preview:** A fit over these 16 candidates gave RMS ≈ 0.58 pixels, but the high-order lens distortion coefficients were unstable. **Do not treat the obtained `fx/fy/cx/cy` or distortion parameters as validated**; board flatness, viewpoint coverage, source resizing and lens model need review.

**Security:** Screenshots may contain details of the desk / room. Review before committing a public repo.

**Example evaluation** (use actual measured square size):

~~~bash
OUT="outputs/real2sim/side-intrinsics-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/calibrate_side_obs_intrinsics.py \
  --images assets/calibration/side_obs_20261106/images \
  --columns 7 --rows 5 --square-mm 20 \
  --output "$OUT"
open "$OUT"
~~~

These are calibration **input images**, not a validated external camera pose. For Side camera extrinsics, independent world-frame 3D measurements are still needed.
