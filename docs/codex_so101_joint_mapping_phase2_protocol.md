# SO-101 joint angle mapping: physical evidence protocol (Phase 2)

**State: PREPARATION ONLY — no independent measurements collected; all mappings UNVERIFIED.**

Purpose: collect independently measured correspondences for `q_URDF = s*q_LeRobot + b`, initially shoulder_lift and elbow_flex. The existing Phase 1 script `scripts/so101_capture/fit_joint_mapping_offline.py` can evaluate supplied evidence, but cannot manufacture it.

## Safety gate (before touching the robot)

- **Do not run ACT, IK, motor commands, calibration, homing, torque changes, register writes or auto-teleoperation as part of this protocol.** Do not ask Codex to connect hardware.
- Stage the arm on a stable, secured table with clear reach, supported links and no pinch hazards. Keep people and cables outside the moving envelope. Do not disconnect servo power while it bears an unsupported load; the arm may fall. Follow manufacturer safety guidance for any later manual repositioning.
- Initial inspection is with the arm **stationary**. Use existing static photographs and stored LeRobot observations first. Treat changing pose, releasing torque, manual repositioning, powering motors, or reading live registers as separately authorized and separately risk-assessed operations.
- Do not equate calibration min/max to certified physical limits, or shift URDF limits based only on a fit.

## What qualifies as independent angle evidence?

1. A timestamped, identifiable stationary pose with known `lerobot_deg` from an observation collected for *that same pose*; never substitute a different episode frame or a commanded target `action`.
2. An independently measured **physical relative rotation of the child link with respect to its parent**, in a defined reference convention, converted to URDF joint coordinate after accounting for the URDF origin transform and linkage geometry. Use a measured jig or fiducial/protractor referenced to both parent and child; image-only appearance does not establish this without calibrated projection.
3. At least two distinct poses of the *same joint* with adequate angular separation (target >=15° where physically safe; no commanded motion prescribed here). Three or more independent observations are preferable. Record measurement uncertainty and whether other joints changed.
4. A provenance record specifying observation_id, reference frame, measurement method, operator measurement uncertainty, and whether mapping sign/zero is independently fixed. The current fitting CSV has only five columns; put supporting details in a separate local evidence log and do not upload it unless sanitized.

## Acquisition stages

**Stage A (now, no robot use)**: Codex inventories existing side/wrist videos/photos and dataset frames, LeRobot source, URDF CAD/mesh orientation, and existing measurement fixtures. Select candidate stationary, well-separated poses. Create an evidence checklist marking `lerobot_deg available?`, `relative parent-child physical angle independently measurable?`, `URDF zero/reference defined?`. Do not claim that images alone provide measured joint angles. Produce a sanitized report.

**Stage B (separate authorization)**: Arrange static support and independent measurement of two or more safe, independently specified poses per prioritized joint, with a read-only observation at exactly the measured pose if available. No motor writes. Recording a photograph alone, without measured link geometry/coordinate reference, is not enough for an absolute offset.

**Stage C (offline only)**: Fill `reports/so101_joint_mapping/mapping_evidence_template.csv` *locally as a copy outside the repository* with evidence:
`joint,lerobot_deg,urdf_deg,observation_id,measurement_source`.
Run:
```bash
python scripts/so101_capture/fit_joint_mapping_offline.py \
  --evidence /path/to/private_measured_mapping_evidence.csv \
  --output outputs/mapping_fit_$(date +%Y%m%d_%H%M%S).json
```
Compare both fixed-sign hypotheses (+1, -1), offset, RMSE, and leave-one-out RMSE; confirm geometric reference and uncertainties before declaring any mapping verified.

## Acceptance gates

- Data provenance: paired independent measurement and *measured State* in the same stationary pose; timestamps validated.
- Identifiability: >=2 distinct independent poses **per joint**, preferably >=3, with displacement large compared to quantization and measurement uncertainty; other joints controlled or accounted for.
- Geometry: correct parent/child link reference frames and URDF origin and axis; distinguish servo output rotation, linkage angle and joint rotation.
- Quantitative: chosen sign must outperform alternative by a margin larger than measurement uncertainties; offset stable across poses; residual/LOO RMSE within predeclared independent measurement uncertainty. **No universal numerical pass threshold without measurement specification.**
- Review: safety/physical limitations separately reviewed by a human. The fitting script never marks results as physically validated.
- If any gate fails: status remains UNVERIFIED. **Do not replay prior IK trajectories on hardware.**

## Codex Stage A deliverables

- `reports/so101_joint_mapping/phase2_readiness.md`: static read-only evidence inventory, whether 2 independent correspondences exist for each priority joint, measured/unknown reference frame, uncertainties and blockers, exact local files consulted without private absolute paths.
- `reports/so101_joint_mapping/phase2_readiness.json`: simple sanitized machine-readable statuses, no pictures/raw recordings/calibration content/private paths.
- Optional test improvements to `fit_joint_mapping_offline.py`: reject duplicated observation IDs when used as independent evidence, warn when joint separation is below claimed measurement resolution, and ensure explicit non-verified output. Run offline tests.
- Commit and push only sanitized code/docs/report to `feature/cartesian-action-offline-export`, without hardware use, without force push; report commit SHA and blockers.

**Do not create fictional measured evidence for progress.** This is a readiness audit and protocol, not a completed physical mapping validation.
