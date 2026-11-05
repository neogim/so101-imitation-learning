# 2026-11-06 Side OBS calibration image model comparison

**Status: exploratory numerical calibration, NOT production-calibrated.**

## Inputs

- 16 selected 640×480 checkerboard screenshot crops, tracked in [assets/calibration/side_obs_20261106](../assets/calibration/side_obs_20261106/README.md)
- 7×5 **inner corners**, nominal square side = 20 mm. Check physical print scale and flatness.
- Assumed central 640×480 screenshot crop (black side bars removed). **NOT yet proven identical to the image delivered to LeRobot by OBS.** Recorded ACT Side size = 224×168.
- The fixed camera was an **iPhone through OBS Virtual Camera**; Wrist uses **Innomaker U20CAM**.

## Results on 16 images

All sixteen images detected by \`cv2.findChessboardCornersSB\`. Calibration is based on OpenCV with default calibration and iterative PnP in leave-one-image-out diagnostics. The held-out board pose is estimated from the held-out frame; this does **not** verify physical camera-to-robot extrinsics.

| model | training RMS (px) | leave-one-image-out average per-view RMS (px) | fx | fy | cx | cy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pinhole, D=0 | 0.315 | 0.312 | 540.2 | 539.2 | 317.3 | 231.5 |
| radial k1 only | 0.314 | 0.311 | 540.3 | 539.6 | 316.5 | 231.5 |
| radial k1/k2 | 0.313 | 0.311 | 542.1 | 541.1 | 316.8 | 232.3 |
| full five coefficients | 0.311 | 0.315 | 540.4 | 539.7 | 321.0 | 224.0 |

5-coefficient distortion \`[k1,k2,p1,p2,k3]\` ≈ \`[0.097,-1.739,-0.006,0.004,10.337]\`; this illustrates unstable/unphysical-looking overfit. Tiny reduction in training RMS is not evidence this more complex lens model is better. **No final model selected**; pinhole or k1-only is a reasonable *intermediate diagnostic candidate*.

## Reproduce on Mac

~~~bash
conda activate so101-real2sim
git switch feature/mujoco-real-to-sim
git pull --ff-only
python -m pip install "opencv-python-headless>=4.8"
OUT="outputs/real2sim/side-intrinsics-compare-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/compare_side_intrinsics_models.py \
  --images assets/calibration/side_obs_20261106/images \
  --columns 7 --rows 5 --square-mm 20 \
  --output "$OUT"
open "$OUT"
~~~

Outputs: model_comparison.json, intrinsics_XXX_candidate.json. Never substitute these directly for measured robot-to-camera extrinsics.

## Next validation gates

1. Determine whether screenshot's central 640×480 pixels correspond to exactly the OBS output, or were cropped/scaled/rotated. Check OBS source selection: iPhone, **not FaceTime**.
2. Determine the exact OBS → LeRobot conversion to ACT Side 224×168. If it is pure uniform resize from 640×480, scaling is \`s_x=224/640=0.35\` and \`s_y=168/480=0.35\`, with camera matrix \`K' = diag(s_x,s_y,1) K\`, e.g. pinhole fx ≈ 189.1, fy ≈ 188.7, cx ≈ 111.1, cy ≈ 81.0. **These numeric examples are CONDITIONAL, not validated transformation settings.** Additional crop or rotation invalidates simple scaling. Distortion coefficients are independent of pixel scaling in normalized camera coordinates when only resize is applied.
3. Collect independent noncoplanar points or a measured calibration target frame in the SO-101/MuJoCo base frame; only then run Side PnP extrinsic calibration.
4. Compare rendered Side views over multiple frames; do not use attractive image similarity to assert verified joint mapping. Wrist is an independent later task.

## Confirmed repo capture pipeline (code evidence)

- \`scripts/so101_capture/MULTI_CAMERA.md\` documents that OBS screenshot capture supports **640×480**, **448×336**, and **320×240**, with actual capture size determined by the selected camera YAML settings.
- \`scripts/so101_capture/resize_act_dataset.py\` uses **Pillow bilinear resize to (224,168)**. It records \`source_shapes\` in dataset provenance for generated outputs.
- Therefore, the pure-resize rule above is applicable **only after verifying** the recording's actual source shape and the exact correspondence between the 640×480 screenshot crop and OBS's program/source output. Neither dataset name nor screenshot dimensions alone proves that.
- If source=640×480 and screenshot pixels match its RGB source, a preliminary pinhole vertical field-of-view estimate is **~48°**, compared to the current uncalibrated 60° MuJoCo Side setting. MuJoCo \`fovy\` by itself cannot model an off-center principal point or lens distortion, so simulated image registration may still differ. Do not silently mark \`calibrated=true\`.

Source-size review (read-only):

~~~bash
python - <<'PY'
from pathlib import Path
import json
for p in sorted(Path("data/so101").glob("*/meta/info.json")):
    try:
        m=json.loads(p.read_text())
        cams={k:v.get("shape") for k,v in m.get("features",{}).items() if "images.side" in k}
        if cams:
            print(p.parent.parent.name, cams)
            if "source_shapes" in m:
                print("source_shapes:",m["source_shapes"])
    except Exception as exc:
        print(p, type(exc).__name__, str(exc))
PY
~~~

If the dataset's \`source_shapes\` appears in another provenance file, inspect the resize run's manifest and original dataset metadata. Do not infer recording size from the resized dataset alone.
