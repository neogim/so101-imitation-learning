# SO-101 preflight: real vs sim (2026-11-15)

## Data and transforms
- [ ] Identify existing successful episode/video; do **not** ask to recollect if already available.
- [ ] Verify J1–J6 naming/order, direction, zero offsets and measured real state, with explicit source.
- [ ] Check full TCP frame transform: `gripper_frame_joint` position **and** RPY (0,pi,0).
- [ ] For one real successful pose, check sim link/world frames and fingertip contact patches; separate calibration/measurement error from dynamics.

## Collision modeling
- [ ] Confirm STL units, origins, transforms, orientation and scaling.
- [ ] Overlay visible STL, compiled MuJoCo collision hull, current BOX proxies at same timestep and pose.
- [ ] Check moving jaw sweep J6 open/mid/closed; compare actual inner grasp surfaces, not arbitrary vertex distances.
- [ ] Check self-collision exclusions, collision bits and that wrist bracket/camera are not accidentally omitted.
- [ ] Inspect initial penetration among fingers, Muhi, table and base/shoulder.

## Dynamics and control
- [ ] Check mass/inertia plausibility, limits, contact parameters, real J6 servo command-vs-response.
- [ ] Separate J1–J5 **fixed** and J6 physical closure in an isolated simulation.
- [ ] Require sustained positive forces on both jaws; table support alone is not grasp.
- [ ] Lift 5mm and verify actual object/table separation; do not teleport Muhi.
- [ ] Restore dynamic full-arm position control only after standalone grasp is valid.
- [ ] Treat equality constraint experiment in PR #13 as *unvalidated*; no merge by inference.

## Required camera/video contract
- [ ] Preserve wrist camera rig `ruler_15_35_35`, optical mounts and collision proxies.
- [ ] 2 side + 3 wrist, synchronized 1440x780 combined MP4, plus individual MP4s.
- [ ] Report camera pose assumptions (uncalibrated), frames and FFmpeg statuses.
- [ ] Preserve old outputs/ZIPs, URDF and visible STL.

## Decision rule
One hypothesis per experiment. After ~2 unsuccessful variants, revisit model and canonical MuJoCo usage instead of stacking guards and tuning.
