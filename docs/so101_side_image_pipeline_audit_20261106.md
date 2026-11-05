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

## 追加確認：マルチカメラ収録での変換箇所

\`scripts/so101_capture/capture_session.py\` が \`--camera-config\` を受けると、\`multi_camera.capture_image_shape(camera_cfg)\` の結果が実際の保存解像度となる。 \`multi_camera.py\` の \`capture_image_shape()\` は640×480、448×336、320×240に対応する。

また \`multi_camera.py\` のAVFoundationフレーム取得部分（\`_fetch()\`）では、入力解像度と \`image_shape\` が異なれば **Pillow BILINEAR** で指定サイズに縮小している。つまり、収録元のSide 320×240が、640×480入力からリサイズされた可能性はあるが、コードだけでは入力サイズも、その当時の \`cameras.local.yaml\` の設定も確定できない。

**次のローカル確認（設定ファイルの秘密を公開しない）：**

~~~bash
# 画面やローカルIP、機器シリアルを含む設定ファイルを貼り付けず、
# 画像サイズだけ抽出して表示する
python - <<'PY'
from pathlib import Path
import yaml
for path in sorted(Path("scripts/so101_capture").glob("cameras*.local.yaml")):
    raw = yaml.safe_load(path.read_text()) or {}
    size = raw.get("capture", {})
    print(path.name, "capture.width =", size.get("width"),
          "capture.height =", size.get("height"))
PY

# 収録時の情報が残る場合はローカルのsource_lineage.jsonも確認するが、
# 絶対パスが含まれるため、原文をGitへコミットしない
~~~

**Note:** \`obs_camera.py\` はOBS仮想カメラの単独経路（640×480）。マルチカメラ録画で必ず通るとは限らないため、同一視しない。

## 比較対象

- 内部パラメータの640×480推定は「スクリーンショットの中央640×480」に基づく。
- 実際の録画が別のOBS crop、フィルタ、リサイズ経路を通ったなら、同じ画素変換のKとはならない。
- \`cx,cy\` はピクセル中心定義により \`c'=(c+0.5)s-0.5\` の補正が必要になる場合がある。値の単純拡大縮小は概算。


## 2026-11-06 local capture configuration verification

The user inspected three gitignored camera configuration files locally, without exposing private device identifiers. All three currently specify capture.width=320 and capture.height=240: cameras.local.yaml, cameras.obs-wrist-compare.local.yaml, cameras.side-jpeg70.local.yaml. This agrees with the 320x240 source LeRobot video and supported multi_camera.capture_image_shape setting. Historical October 25 capture configuration, OBS crop, and source frame dimensions are still not independently verified.

For pure 640x480 to 224x168 full-field resizing (s=0.35), exploratory pinhole candidate yields fx~189.059, fy~188.716. Principal point with simple scaling is cx~111.074 cy~81.042; pixel-center-based mapping (c+0.5)*s-0.5 gives cx~110.749 cy~80.717. This is conditional, NOT physically validated, and should not be promoted to calibrated extrinsics/intrinsics.

Next required measurement: six or more independently measured noncoplanar 3D landmark positions in SO-101 base/workcell coordinates and matching Side-camera 2D pixels in the same unchanged imaging pipeline. Do not infer real 3D points from illustrative MuJoCo world geometry.
