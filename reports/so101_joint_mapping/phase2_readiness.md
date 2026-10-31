# SO-101 joint mapping Phase 2 — Stage A readiness

## Outcome

Stage A was completed offline. Existing data contains useful, synchronized, stationary pose
candidates for `shoulder_lift` and `elbow_flex`, but contains **zero independently measured
parent–child joint angles**. Neither joint's sign nor zero offset can be verified from the existing
recordings. Status remains **UNVERIFIED**.

No robot connection, live register read, motor command, torque change, calibration, manual pose
change, ACT execution, or hardware IK was performed. No image, video, raw dataset, device
identifier, or calibration content is included in this report.

## Static evidence inventory

Files consulted, using repository-relative paths only:

- `data/so101/muhi-side-wrist-success-053-20261025/data/chunk-000/file-000.parquet`:
  measured LeRobot State and Action, 53 episodes / 30,846 frames.
- `data/so101/muhi-side-wrist-success-053-20261025-168x224/videos/observation.images.side/chunk-000/file-000.mp4`
  and the corresponding `observation.images.wrist` video: synchronized stored views. Images were
  inspected locally and are not copied into the report.
- `outputs/so101-joint-zero-direction-ep0-20261027-v2/`: prior offline frame synchronization,
  selected pose images, direction pairs, and URDF rendering. This output was read only.
- `data/urdf/so101/so101_new_calib.urdf`: joint parent/child, origins, axes, limits, and visual
  transforms.
- `data/urdf/so101/assets/`: 31 CAD artifacts (`.stl` and `.part`), including shoulder/upper-arm,
  lower-arm, motor-holder, and servo geometry.
- Installed LeRobot 0.6.2 source: `MotorNormMode.DEGREES`, motor midpoint normalization, homing,
  Follower calibration, and Feetech offset handling, as cited in the Phase 1 report.
- Repository search for camera intrinsics/extrinsics, fiducials, ArUco/AprilTag targets,
  protractors, inclinometers, and joint measurement fixtures. No applicable measurement artifact
  was found. `scripts/so101_capture/fixtures/muhi_pilot_tail.json` is a software test fixture, not
  a physical angle fixture.

The camera diagnostic files establish stream identity, sampling rate, and approximate cross-camera
timing only. They do not contain lens calibration or camera-to-robot extrinsics suitable for
recovering link orientation.

## Candidate stationary poses

Episode 0 was checked at 20 FPS. A candidate required a 10-frame (0.5 s) window in which every
arm-joint State span was at most 0.5°. Two well-separated, visually usable candidates are:

| Candidate | frame / window | shoulder_lift State | elbow_flex State | max arm-joint span in window | stored views |
|---|---|---:|---:|---:|---|
| A | 338 / 334–343 | +38.769° | −19.560° | 0.000° | side + wrist |
| B | 625 / 621–630 | −101.978° | +96.484° | 0.000° | side + wrist |

The motor-angle separation is approximately 140.747° for the shoulder and 116.044° for the elbow,
well above the Phase 2 target of 15°. This makes the frames good *candidates* for later measurement,
not independent correspondences: the only numeric angles presently associated with them are the
LeRobot States whose mapping is under investigation.

Other stationary windows exist, including around frames 114, 252, 311, 436, 451, 466, and 508.
They may provide redundancy after an independent image/fixture reference is defined. Action was
not substituted for measured State.

## Evidence checklist

| Joint / candidate set | same-pose LeRobot State available? | stationary and separated? | parent–child physical angle independently measurable now? | URDF mathematical zero defined? | URDF zero registered to physical/image reference? | qualifying correspondences |
|---|---|---|---|---|---|---:|
| shoulder_lift A/B | yes | yes | **no** | yes | **no** | 0 |
| elbow_flex A/B | yes | yes | **no** | yes | **no** | 0 |

The URDF defines `shoulder_link → upper_arm_link` and `upper_arm_link → lower_arm_link`, each with
local axis `[0,0,1]` and a joint-origin transform. Link visual meshes have additional origins and
rotations. Therefore a line seen on a mesh is not automatically the URDF joint coordinate or zero.

## Why existing imagery and CAD are insufficient

The side view visibly distinguishes the two poses and exposes much of the arm, but it lacks:

- calibrated camera intrinsics, distortion, and camera-to-base extrinsics;
- a known scale or a measured projection plane;
- identifiable reference axes fixed to both parent and child links;
- a surveyed relationship between visible CAD surfaces and URDF link frames;
- uncertainty estimates and independently recorded physical angles.

The wrist camera is mounted on the moving chain. Without its camera-to-link transform and an
external calibrated reference, it cannot serve as an inertial reference for shoulder or elbow
rotation. A single uncalibrated perspective view also mixes out-of-plane rotation with apparent
2D angle. CAD/URDF can predict an image only after the missing camera and physical registration
transforms are known; fitting those transforms using the same unknown joint angles would be
circular evidence.

Consequently the images prove pose change and synchronization, not `q_URDF`. Range overlap,
closest-to-zero frames, URDF limits, and internally consistent FK/IK are also not independent
zero/sign evidence.

## Next safe measurement procedure (requires separate authorization)

No part of this procedure was performed in Stage A.

1. Have a human approve the physical risk assessment. Secure the base on a stable table, clear the
   reach and pinch envelope, and support links so loss of torque cannot cause a fall. Do not remove
   power while the arm bears an unsupported load.
2. Define measurable parent and child references. For `shoulder_lift`, reference the
   `shoulder_link` and `upper_arm_link`; for `elbow_flex`, reference `upper_arm_link` and
   `lower_arm_link`. Use non-invasive fiducials/straightedges or a surveyed jig whose orientation
   relative to the corresponding URDF link frames is documented from CAD and checked physically.
3. Calibrate the measurement system before using it: either a fixed camera with intrinsics,
   distortion, camera-to-base extrinsics, scale, and visible link fiducials, or two independently
   referenced digital inclinometers/angle fixtures measuring child relative to parent. Record the
   method and angular uncertainty.
4. For each stationary pose, obtain the physical relative angle and the measured LeRobot
   `observation.state` for that exact pose and timestamp. A live read-only observation or any new
   connection is a separate authorization; do not use Action as the angle. Record observation ID,
   reference frames, uncertainty, and whether any other joint changed.
5. Acquire at least two poses per joint separated by at least 15° and preferably three or more.
   Any manual repositioning, torque change, or power-state change requires a separate safety plan
   and explicit authorization; this report does not prescribe or authorize motion.
6. Keep the detailed evidence log private. Put only sanitized correspondences into a local copy of
   `mapping_evidence_template.csv`, run `fit_joint_mapping_offline.py`, and compare both signs,
   offset stability, residuals, and leave-one-out RMSE against the declared measurement uncertainty.
7. A human must review geometry and safety before marking a mapping verified. A register snapshot
   can confirm active calibration but cannot establish physical URDF zero or sign.

A single motionless independently measured pose can constrain an offset under an assumed sign, but
cannot identify the sign. Two or more distinct independent poses are required.

## Stage A decision

- `shoulder_lift`: **NOT READY TO FIT / UNVERIFIED** — zero qualifying correspondences.
- `elbow_flex`: **NOT READY TO FIT / UNVERIFIED** — zero qualifying correspondences.
- Existing candidate imagery is adequate for planning a measurement setup, not for angle recovery.
- Proceed to Stage B only after separate human authorization and safety review.
