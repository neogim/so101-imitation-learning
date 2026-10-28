# SO-101 calibration / URDF offline audit

## Scope and safety

This audit read the 53-episode, 30,846-frame dataset, cached **Follower** calibration JSON,
SO-101 URDF, and the installed LeRobot 0.6.2 source. It did not import robot classes,
open a serial port, initialize motors, read servo registers, or write source data. The cached
calibration contents and absolute local paths are intentionally not included here.

## Angle definitions and evidence

- Dataset `action` and `observation.state` are named joint positions, but their numeric unit is
  LeRobot `MotorNormMode.DEGREES`: `(encoder_value - (range_min+range_max)/2) * 360/4095`.
  This is normalized **motor degrees**, not independently proven URDF joint-space degrees.
- In LeRobot 0.6.2, Follower calibration first centers the encoder by a homing offset, then
  interactively records the range swept by hand. Therefore `range_min/max` are observed
  operational calibration endpoints, not certified hard mechanical limits.
- `drive_mode` is 0 for the inspected joints. In the LeRobot 0.6.2 `DEGREES` branch it does not
  alter sign, and degree normalization/unnormalization does not clip to `range_min/max`.
- The URDF defines both `shoulder_lift` and `elbow_flex` around axis `[0,0,1]` in their respective
  joint frames. Their origins differ, so matching scalar degree values alone cannot prove the
  physical zero or sign transformation.

## Results

| joint | URDF range (deg) | cached calibration degree range | Action min/max | State min/max |
|---|---:|---:|---:|---:|
| shoulder_lift | -100.000043 .. 100.000043 | -105.230769 .. 105.230769 | -103.824173 .. 40.967033 | -103.472527 .. 40.879120 |
| elbow_flex | -96.829867 .. 96.829867 | -97.362637 .. 97.362637 | -42.417583 .. 97.450546 | -39.428570 .. 97.010986 |

Counts outside each bound at tolerances 0/0.1/0.2/0.5 degrees are in `summary.json`.
The earlier Action counts reproduce at strict tolerance: shoulder_lift outside URDF =
8203; elbow_flex outside URDF =
9204; elbow_flex outside the inferred
calibration range = 612.

The 612 elbow count is a floating-point/endpoint artifact plus a very small real overshoot:
606 Actions serialize as 97.362640 degrees, only about 0.000003 degrees above the computed
97.362637 endpoint, while 6 Actions are one encoder quantum higher at 97.450546 degrees
(about 0.087909 degrees over). Consequently the count is zero with a 0.1-degree tolerance.
No State sample exceeds the inferred elbow calibration endpoint. At the strict Action exceedance
frames, State remains below Action, consistent with commanded goal versus measured position lag;
it does not establish that the servo crossed its calibrated endpoint.

Strict elbow exceedance frame ranges (covering all 612 indices): `ep6:54-95, ep8:514-520, ep9:515-519, ep10:498-510, ep11:498-512, ep13:511-624, ep14:0-79, ep14:499-510, ep15:476-588, ep16:0-64, ep16:460-472, ep20:543-554, ep21:522-527, ep22:451-451, ep22:453-458, ep23:453-454, ep26:549-549, ep27:536-545, ep28:409-414, ep30:469-471, ep32:396-396, ep34:449-457, ep38:413-421, ep40:461-471, ep41:496-499, ep42:437-445, ep43:483-495, ep44:428-435, ep45:432-438, ep46:496-500, ep48:442-449, ep51:394-395`.
`peaks.csv` contains only the 6 materially distinct (>0.01 degree) rows.

The other joints are summarized in `summary.json`; none has Action or State outside its cached
calibration degree span. Shoulder pan, wrist flex, and wrist roll also remain inside URDF limits.

## Conclusions and unresolved items

The data proves a convention mismatch risk: URDF joint angles and LeRobot normalized motor
degrees have been compared numerically without an independently measured zero/sign mapping.
It does **not** justify widening URDF or physical limits. The shoulder_lift Action minimum is
inside the cached swept calibration span but outside the URDF lower bound. The elbow values over
the URDF upper bound are likewise not proof of physical URDF-angle travel.

Unresolved: whether this cached file matched the firmware registers during recording, and the
physical URDF zero/sign correspondence. Resolving either requires a separately authorized,
read-only register snapshot or a controlled physical pose measurement; neither was attempted.

## Reproduction

```bash
python scripts/so101_capture/audit_so101_calibration_consistency.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --output reports/so101_calibration_audit
```

The environment variable must point to the local cached Follower calibration; its private path
and contents are not committed. The command refuses to overwrite an existing output directory.
