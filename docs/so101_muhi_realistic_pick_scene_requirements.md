# SO-101: realistic Muhi Pick start-state requirements (2026-11-14)

## Why the current tabletop bench is NOT a realistic Pick

The current `physical_grasp_table_bench.py` invokes
`physical_grasp_j6_bench.select_bench_pose()`, which **moves the free
Muhi bottle to a computed pose in gripper-local coordinates** before
simulation. Its purpose is a narrow contact experiment, not approach
planning from an independently placed object.

Measured from `so101-muhi-contact-first-20261114-170857.zip`:

- Object pose selected at MuJoCo world center
  `(0.34723450, -0.03525764, 0.21464014) m`.
- Table top = `0.16564014 m`, upright Muhi radius 17.5 mm,
  height 97 mm (provisional geometry).
- Object gripper-local center = `(36,-18,-54) mm`.
- Initial fixed-jaw clearance = **2.003 mm**,
  moving-jaw clearance = **4.772 mm**.
- Nearest fixed-jaw collision witness is **44.402 mm away from the
  configured TCP**; previous acceptable threshold = 45 mm.
- J1 contact-first trial stopped at time 4.154 s with
  `J1_PRECONTACT_NOT_ESTABLISHED`, after 14 non-consecutive fixed-jaw
  contact samples, no J6 closure, no lift.
- The root-contact concern is credible based on rendered video and
  witness distance, but a precise collision-mesh subregion classification
  has not been established.

## Required architecture for the next realistic Pick exercise

1. Define **robot home and independent object/table world poses** as
   configuration. Object pose is not chosen using closeness to the
   gripper. Support the actual photographed episode pose when
   calibrated; otherwise label test poses synthetic and provisional.
2. At t=0, ensure robot–object and robot–table clearances are above
   documented minimum thresholds; require object support by the table.
3. **Reach**: plan/collision-check a trajectory using J1–J5 from home
   to a pre-grasp waypoint. Keep J6 open. Reject plans that hit the
   table, object, mount, or arm root.
4. **Pre-grasp alignment**: adjust TCP position and J5 according to
   bottle direction and jaw opening. The jaw-to-object closest witness
   must be within a deliberate fingertip zone (start with ≤25 mm from
   the chosen TCP reference, calibrated against real meshes), NOT a
   45-mm root-adjacent tolerance.
5. **Contact-first**: perform bounded dynamic or impedance-controlled
   approach to confirm contact on the fixed fingertip, not just any
   gripper_link collision. Abort on contact with palm/root or excess
   force, rather than falsely passing pregrasp.
6. **Close**: only when appropriate fixed-tip contact is confirmed
   for multiple solver steps, close J6 with force and motion caps.
7. **Lift**: require maintained bilateral contact and lift clearance,
   then try a short lift. Always report stage-specific failures.
8. Record side and wrist camera, object displacement, active contact
   geom IDs, witness points relative to TCP, force, gap and angles.
   Use non-invasive collision pair filtering only after validation.

## Present state and next implementation step

No real-photo coordinate calibration or robust home-to-pregrasp IK path
is yet wired into the current contact bench. Do **not** claim the scene
already starts separated from the arm or that this commit implements
full Reach-to-Pick. The current tabletop bench should be renamed or
described as a **near-grasp contact diagnostic**.

Suggested immediate code work: separate `place_object_world()` from
`plan_robot_reach()`, define a fixed world-pose test fixture, and add
preflight checks for initial scene clearance and fingertip witness
location. If a pose is infeasible, report it explicitly instead of
teleporting the bottle near the claw to make progress.
