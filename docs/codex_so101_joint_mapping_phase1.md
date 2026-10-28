# Codex task: establish LeRobot motor-degree → URDF joint-angle mapping (phase 1)

## Goal
Determine whether each SO-101 Follower LeRobot normalized motor degree value q_M corresponds to the URDF joint coordinate q_U via q_U = s*q_M + b, where s is +1 or -1 and b is an offset in degrees. **Do not assume s=+1 or b=0.** Investigate each of five arm joints; prioritize shoulder_lift and elbow_flex.

## Safety and scope
- PHASE 1 IS READ-ONLY AND OFFLINE. Do not connect to serial, import modules that instantiate a robot, read hardware registers, torque motors, run calibration, send movement commands, or manipulate URDF/calibration/dataset contents.
- Reading cached files, dataset Parquet, images/videos that already exist on disk, installed source, and URDF is allowed.
- No auto-execution of ACT or IK on hardware. No URDF limit expansion.
- If images or external ground truth cannot establish sign and zero, declare the mapping **unverified**; never guess it from numeric degree overlap alone.

## Tasks
1. Confirm source versions and actual conversion code for LeRobot MotorNormMode.DEGREES, including homing_offset and drive_mode treatment. Cite file paths + function names + line numbers in the local report.
2. Parse URDF per joint: parent/child, joint origin xyz/rpy, axis xyz in the joint frame, limits, and mesh transforms that affect the visual zero. Document the distinction between a URDF joint frame and a motor's calibrated center.
3. Extract paired Action/State traces and cameras for **selected existing recorded frames** near 565–580 and at well-separated shoulder/elbow values. If existing images are accessible, use them only for offline consistency checks, and state if projection/camera calibration prevents exact numeric pose validation. Do not invent link angles from pixels.
4. Propose mapping candidates s=±1; b as *unknown*, rather than infer b from dataset range. If evidence permits, fit s/b to an independent measured geometric landmark or verified posed configuration, not to the URDF joint limits themselves. Report uncertainty and provenance.
5. Build a reproducible diagnostic that can take **externally supplied independent joint-angle correspondences** (CSV with joint, lerobot_deg, urdf_deg, observation_id, measurement_source), fit both sign hypotheses and offset, compute residuals and leave-one-out error, and flag insufficient sample diversity (<2 distinct angles) or insufficient independent references. The program must not silently label a fit physically validated.
6. Add tests with synthetic +sign, -sign, offset, outliers, degenerate data. Run tests locally; no robot access.
7. Create sanitised results in `reports/so101_joint_mapping/` with README.md and summary.json. May add `mapping_evidence_template.csv` containing only headers and synthetic/example rows clearly marked synthetic (better headers only). Document what exact physical evidence is still needed and whether a motionless, externally measured pose suffices.
8. Explicitly revisit the prior 648-frame IK results: they were computed treating normalized motor degrees as URDF joint coordinates, which is an assumption. Mark their TCP error as **internally consistent numerical FK error only**, not validated physical TCP error. No hardware playback.
9. Review staged diff for credentials, usernames, absolute paths, private calibration JSON, videos, raw datasets; commit only scripts/tests/sanitised reports. Push to `feature/cartesian-action-offline-export` without force, and report commit SHA + evidence and blockers.

## Deliverables
- A short clear evidence matrix (joint; motor convention known? URDF axis/origin known? independent zero/sign verified? mapping status).
- A safe fitting/validation script with tests.
- Summary highlighting which mappings are *verified / hypothesized / unknown*.
- Next check requiring user authorization, if needed. Do not present read-only motor-register access as equivalent to physical joint zero validation.
