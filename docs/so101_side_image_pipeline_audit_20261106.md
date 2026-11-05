# Side camera image transformation verification (2026-11-06)

## Verified from code and user dataset metadata

- Side physical source: iPhone → OBS Virtual Camera. `scripts/so101_capture/obs_camera.py` configures **640×480 RGB** and does not crop or resize internally.
- Original 53-episode dataset `muhi-side-wrist-success-053-20261025`: Side **320×240**.
- Resized dataset `muhi-side-wrist-success-053-20261025-168x224`: Side **224×168**.
- `scripts/so101_capture/resize_act_dataset.py` explicitly resizes the source frame to `(224,168)` with `PIL.Image.Resampling.BILINEAR`. This is a direct conversion from the recorded 320×240 frames, not a direct export from OBS.
- The **actual 640×480 to recorded 320×240** image path has not been independently established, despite the example OBS adapter configuration.
- The 640×480 checkerboard images are centered crops of 1280×480 desktop screenshots. They cannot be assumed geometrically equivalent to the captured OBS Side data until this imaging path is verified.

## Intrinsic scaling under explicit no-crop/no-rotation assumption

If the recorded 320×240 image is an unaltered half-size downscale of the **same 640×480 field of view** as the calibration images, and the 224×168 image is the PIL resize of that 320×240 image, then the total image scale equals **0.35** (both axes).

Using the exploratory pinhole candidate from the 16 selected 640×480 checkerboard crops:
```
fx=540.1684754
fy=539.1891253
cx=317.3535479
cy=231.5487659
```
The approximate scale-only candidate for 224×168 would be:
```
fx≈189.0590, fy≈188.7162, cx≈111.0737, cy≈81.0421
```
**NOT validated.** Pixel-center conventions can introduce small shifts (e.g., 0.5-pixel corrections). These numerical conversions do not prove physical camera alignment and are not automatically written to the workcell config.

## Next evidence

1. Review the exact camera processing in `scripts/so101_capture/multi_camera.py`, `capture_session.py` and recorded video source configuration.
2. Compare same-frame original and downscaled Side MP4s, accounting for H.264 encoder loss.
3. Confirm the OBS scene's active iPhone source, filter, crop and transform at recording time (not just today's settings).
4. Only then generate a candidate camera intrinsics file in the ACT image coordinate system.
5. Side camera extrinsics require measured 3D-to-2D correspondences in the robot/workcell frame; intrinsics alone do not solve camera pose.

Avoid blindly taking source-lineage.json absolute paths into Git: they can expose a local username. Use relative paths in any committed report.
