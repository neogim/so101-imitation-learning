# SO-101 Real-to-Sim rebuild plan (2026-11-15)

Source: [Notion learning plan](https://app.notion.com/p/41985e16f4dc8108bf65eae86540d52a). Reference baseline: actual successful SO-101 grasp of Muhi. Keep original URDF/STLs and existing experiment outputs.

## Decision
1. Do not merge [draft PR #13](https://github.com/neogim/so101-imitation-learning/pull/13) as a demonstrated solution. The equality-constraint mode is an untested diagnostic alternative.
2. Compare **one real successful pose** first: J1–J6 positions/zero offsets, TCP translation AND RPY, Muhi world pose, object dimensions, gripper closure direction.
3. Audit collision geometry at identical joint pose: visual STL, raw convex collider, fixed/moving proxy boxes, object. Do not claim visual penetration from solver box distance alone.
4. Build a **separate fixed-J1–J5, dynamic-J6** minimal grasp bench before introducing dynamic arm hold. Avoid both per-step J1–J5 qpos writes *during contact* and abrupt physical-PD activation.
5. Only after stable bilateral solver forces, run physical 5-mm Lift, then full arm position-actuator hold, then Reach/MoveIt 2/ACT/SAC.

## Baseline vs. unproven
- Confirmed historical traces: zero effective jaw contact forces despite generated candidates; J6 commanded target speed differed from actual speed; physical-PD arm hold caused abrupt large joint deviation; visual STL vs collision BOX gap mismatch.
- Unproven: root cause of jaw 0 N; true real-to-sim joint calibration; real Servo tracking; equality-constraint mode behavior on Mac.
- Model files: `data/urdf/so101/so101_new_calib.urdf` includes `gripper_frame_joint` RPY=(0,pi,0), not merely XYZ. Keep this transform in 6D verification.

## Mandatory cameras and reporting
- Existing `scripts/so101_sim/wrist_camera_sim.py` photographed bracket, collision, and provisional 15/35/35 mm geometry must remain.
- 2 side and 3 wrist views; 1440×780 combined MP4 + individual MP4, synchronized; not a PNG-only substitute.
- Real physical contact: require sustained **positive force on both jaws**, then object/table separation and 5 mm rise. No synthetic grasp or object teleport.
- Preserve all historical files and original branch. Sim-only study mode does not justify claims about hardware safety.

## MoveIt/GPU
MoveIt 2 covers IK, path planning and collision checking, **not** contact-force-based dynamic grasp. MuJoCo remains suitable for the grasp. MoveIt 2 / ROS 2 does not require a new NVIDIA GPU. Consider GPU later for many parallel simulations, Isaac Lab, VLA/ACT training.

## Gates
| Gate | Evidence | Stop condition for advancement |
|---|---|---|
| G0 | known successful real pose and data provenance | no comparable pose => do not tune physics |
| G1 | FK, axes, angle sign, full TCP transform | sim/real disagreement => repair geometry/calibration |
| G2 | collision overlay and finger inner contact patches | visual/collision disagreement => repair collider |
| G3 | isolated fixed-arm dynamic J6 grasp | no bilateral positive force => do not begin Reach |
| G4 | 5-mm physically held lift | no stable Lift => no ACT/SAC physical claims |
| G5 | physical J1–J5 actuator hold | abrupt initial deviation => repair control model |
| G6 | Home/Pregrasp planned Reach | plan and execution independently checked |

No arbitrary 0.5-degree early abort. However numerical divergence and physically implausible motion are failures, not success. Reassess architecture after two repeated unsuccessful variants.
