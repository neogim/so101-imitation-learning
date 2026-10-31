# SO-101 Phase 2 Stage B — joint-angle measurement fixture design (Codex task)

## Objective and boundary
Design (not perform) an independent physical angle measurement for shoulder_lift and elbow_flex, to establish q_URDF = s*q_LeRobot + b. **Never claim mappings verified without measurements.**

Offline-only. DO NOT connect to the robot, send servo commands, toggle torque/power, recalibrate, manually reposition the robot, or alter limits/datasets. Do not fabricate evidence.

## Design requirements
1. Inspect the actual URDF joint origins/axes, CAD/meshes and link geometry for shoulder_link ↔ upper_arm_link and upper_arm_link ↔ lower_arm_link. Identify physical planar surfaces, fiducial attachment points, and the transformation from each measured surface/line to its URDF link frame. Account for mesh origin rpy and joint origin rpy. If ambiguous, explicitly mark registration unknown.
2. Compare two methods: (a) two-link reference straightedges + digital inclinometers (with parent/child referenced readings), (b) fixed camera with known intrinsics and fiducials plus camera-to-base geometry. Specify observability for rotations around local joint Z, gravity alignment problems (an inclinometer is not a yaw sensor), out-of-plane projection error, cost/effort, and repeatability.
3. Make an annotated *textual* geometry specification (can include Mermaid diagrams) describing where independent references attach and which angle is measured. Do not claim actual CAD dimensions unless derived from checked artifacts. Do not generate fictitious photos.
4. Define a concrete static-mapping measurement worksheet and uncertainty budget: pose ID, motor State (not commanded Action), measured relative parent/child angle, reference zero relationship, methods, timestamps, uncertainty, sign inference, angular separation. Include at least three distinct poses per joint in a prospective plan, with actual pose selection deferred until authorized safe measurement. Do not target known recorded extreme postures for physical replication.
5. Analyze the sign/offset identifiability mathematically; detail how to transform angle readings to q_URDF and avoid circular validation. Joint reference markers must register to URDF frame by an independently established CAD/physical reference, not by fitting marker alignment from same unknown motor angles.
6. Recommend one practical method for home SO-101 with explicit reasons and gating criteria. Assess static support, pinch hazards, loss of torque, cable routing. No movement or power changes authorized at this stage.
7. Put sanitised deliverables in reports/so101_joint_mapping/: stage_b_fixture_design.md, stage_b_measurement_template.csv (header only), stage_b_decision.json. Include uncertainty and known/unknown flags; no private paths/calibration JSON/raw images/device IDs.
8. Add tests only if introducing executable calculations. Review staged diff, commit and push feature/cartesian-action-offline-export (no force).
9. State blockers and precisely which **human-approved** physical steps would be needed next. The engineering design is not permission to operate the robot.

## Existing evidence
Read reports/so101_joint_mapping/phase2_readiness.md and phase 2 protocol first. Phase A identified stationary historical frames 338/625 but no independent physical joint angles. These frames are references, not instructions to reproduce extremes.
