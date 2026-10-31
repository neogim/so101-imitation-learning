# SO-101 LeRobot motor-degree to URDF joint-angle mapping — phase 1

## Result

No joint's mapping can be physically verified from the available offline evidence. In
particular, both `shoulder_lift` and `elbow_flex` remain:

```text
q_URDF = (+1 or -1) * q_LeRobot + unknown_offset
```

The sign and offset are therefore **unverified**, not zero and not inferred from range overlap.
No robot, serial port, motor register, calibration routine, ACT policy, or hardware IK was used.

## Evidence matrix

| Joint | LeRobot motor convention | URDF axis/origin | Independent sign | Independent zero | Mapping status |
|---|---|---|---|---|---|
| shoulder_pan | known | known | absent | absent | **UNVERIFIED** |
| shoulder_lift | known | known | absent | absent | **UNVERIFIED** |
| elbow_flex | known | known | absent | absent | **UNVERIFIED** |
| wrist_flex | known | known | absent | absent | **UNVERIFIED** |
| wrist_roll | known | known | absent | absent | **UNVERIFIED** |

“Known” here means the software convention or URDF declaration is known. It does not mean the
two coordinate systems have been tied together physically.

## LeRobot conversion evidence

The inspected installed distribution is LeRobot 0.6.2.

- `lerobot/motors/motors_bus.py:775-797`, `set_half_turn_homings`: the pose chosen by the
  operator is made the encoder half-turn. This is a motor calibration center, not a URDF zero.
- `lerobot/motors/motors_bus.py:803-852`, `record_ranges_of_motion`: range endpoints are the
  extrema observed while a user moves the arm by hand.
- `lerobot/motors/motors_bus.py:854-881`, `_normalize`: `DEGREES` is
  `(value - (range_min+range_max)/2) * 360/(resolution-1)`.
- `lerobot/motors/motors_bus.py:883-911`, `_unnormalize`: the inverse uses the same midpoint.
  Unlike percentage modes, the degree branch does not apply `drive_mode` or clamp the value.
- `lerobot/robots/so_follower/so_follower.py:131-153`, `calibrate`: the operator selects the
  middle pose; the saved Follower entries set `drive_mode=0`.
- `lerobot/motors/feetech/feetech.py:247-266`, `read_calibration`, and lines 278 onward,
  `_get_half_turn_homings`: homing offset affects the servo's reported encoder coordinate.
  It does not encode the URDF joint-frame zero or axis direction.

Thus the source establishes normalized **motor degrees**, but contains no transform tying those
degrees to this URDF's joint coordinates.

## URDF evidence and visual zero

All five arm joints declare local axis `[0, 0, 1]`, but each axis is expressed after its own
joint-origin rotation. Relevant declarations are in `data/urdf/so101/so101_new_calib.urdf`:

| Joint | Parent → child | origin xyz | origin rpy | limits (degrees) |
|---|---|---|---|---:|
| shoulder_pan | base_link → shoulder_link | 0.0388353, ~0, 0.0624 | π, ~0, −π | −109.9999…109.9999 |
| shoulder_lift | shoulder_link → upper_arm_link | −0.0303992, −0.0182778, −0.0542 | −π/2, −π/2, 0 | −100.0000…100.0000 |
| elbow_flex | upper_arm_link → lower_arm_link | −0.11257, −0.028, ~0 | ~0, ~0, π/2 | −96.8299…96.8299 |
| wrist_flex | lower_arm_link → wrist_link | −0.1349, 0.0052, ~0 | ~0, ~0, −π/2 | −94.9998…94.9998 |
| wrist_roll | wrist_link → gripper_link | ~0, −0.0611, 0.0181 | π/2, 0.0486795, π | −157.2110…162.7893 |

URDF lines 353-439 contain these joints. The links' visual elements additionally have their own
mesh origins/rotations earlier in the file. Those transforms determine how CAD geometry appears
at `q_URDF=0`; they are not motor calibration offsets. A visual comparison must apply the full
link/joint transform chain, camera intrinsics/extrinsics, and identifiable landmarks.

## Recorded traces and images

Episode 0 has 648 frames at 20 FPS. Frames 565/570/575/580 show `shoulder_lift` State decreasing
from −82.1099° to −101.1868° while `elbow_flex` increases from 79.2527° to 96.4835°.
This is useful trace provenance, but the simultaneous anticorrelated motion cannot identify either
joint's URDF sign or offset.

Existing side-camera comparisons include a relatively separated shoulder pair (frames 355→374,
shoulder −27.1648°, with smaller cross-joint motion) and elbow pair (405→494, elbow +29.8022°,
with shoulder also moving −16.8791°). The images visibly confirm that the recorded arm pose
changes. They do not supply calibrated projection, camera extrinsics, labeled joint centers, link
orientation measurements, or a verified `q_URDF` reference. No numeric angle was inferred from
pixels. Closest-to-zero recorded motor frames are likewise not proof of URDF zero.

## Mapping candidates and fitting diagnostic

For every joint both `sign=+1` and `sign=-1` remain candidates and offset remains unknown.
`scripts/so101_capture/fit_joint_mapping_offline.py` accepts independently supplied CSV rows,
fits the offset under both fixed-sign hypotheses, and reports residual, maximum error, and
leave-one-out RMSE. It flags fewer than two distinct motor angles or fewer than two independent
observations. It never labels a numerical fit physically validated.

```bash
python scripts/so101_capture/fit_joint_mapping_offline.py \
  --evidence measured_mapping_evidence.csv \
  --output mapping_fit.json
```

`mapping_evidence_template.csv` is header-only to avoid presenting synthetic data as evidence.

## Prior 648-frame IK/FK result

The earlier Episode 0 work processed 648 recorded frames while directly treating LeRobot
normalized motor degrees as URDF joint coordinates. Any reported TCP/FK/IK error is therefore an
**internally consistent numerical FK/IK error only**, not a validated physical TCP error. It cannot
be used as independent evidence for sign or zero, and no playback was performed here.

## Evidence still required

A motionless, externally measured pose can constrain offset if it provides an independently known
URDF joint angle for each motor value. It cannot determine sign by itself. Sign needs at least two
distinct, independently measured poses per joint (with other joints fixed or fully measured), or
an equivalent calibrated multi-view/fixture measurement. Camera intrinsics/extrinsics and visible
link landmarks are required if images are used. A read-only servo-register snapshot can confirm
which calibration was active, but cannot by itself validate physical URDF zero or sign.

## Tests

The diagnostic has synthetic tests for positive sign, negative sign, nonzero offsets, an outlier,
degenerate repeated angles, non-independent synthetic provenance, and invalid joint names.
