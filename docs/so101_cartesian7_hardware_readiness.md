# SO-101 Cartesian7 → real-arm validation gate (2026-10-30)

**Status: NO-GO for Cartesian7 motor commands.** These checks are offline and never move hardware.

## 1. Verified against LeRobot upstream (not automatically the locally installed version)

- SOFollower constructs J1–J5 as `MotorNormMode.DEGREES` when `config.use_degrees=True`, otherwise `MotorNormMode.RANGE_M100_100`; gripper uses `RANGE_0_100`.
- Current upstream `SOFollowerRobotConfig.use_degrees` default is `True`, **but the trained dataset's actual mode and LeRobot revision must be verified**.
- Motor bus normalization (current upstream):
  - body `RANGE_M100_100`: `200 * clamp(raw,min,max)/(max-min) - 100` after subtracting min; inverted if `drive_mode=1`.
  - body `DEGREES`: `(raw-(range_min+range_max)/2)*360/4095` (STS3215 nominal resolution), NOT an absolute URDF angle; `homing_offset` already applied in servo readout.
  - gripper `RANGE_0_100`: `100*(raw-range_min)/(range_max-range_min)` (inverted if drive mode).
  - normalization may clip raw values to calibration bounds in some modes. A value outside [-100,+100] is a hint about mode/data path, not evidence the servo exceeded its mechanical limits.
- Sources: https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py ; https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/config_so_follower.py ; https://github.com/huggingface/lerobot/blob/main/src/lerobot/motors/motors_bus.py

## 2. What existing results mean

The existing Cartesian7 labels were constructed with `deg2rad(observation.state[:5])` and `deg2rad(action[:5])` plus the assumed URDF joint axes/sign/zero. Thus the training target is *internally* consistent with that chosen mathematical model, but it is NOT independently verified as a physical TCP displacement. An offline Cartesian7 vs Joint6 TCP error comparison remains a comparison **within that hypothesis**. The numerical IK fit, tiny FK residuals and zero URDF violations do not authorize real motor commands.

Existing neutral model self-collision warnings are diagnostics. They neither prove real physical collisions nor certify the swept path; meshes, SRDF/ACM adjacent-link exclusion, padding and tool/workspace scene must be validated.


## 2a. Measured Follower J2 calibration cross-check (supplied JSON and recorded range)

Follower `shoulder_lift`: `range_min=776`, `range_max=3170`, `homing_offset=-1361`, `drive_mode=0`. If `use_degrees=True`, current LeRobot STS3215 `DEGREES` formula yields calibrated display limits ±105.230769°. Reported Episode 5 J2 State interval [−103.0329666°,−99.4285736°] inversely maps to **approximately 801 through 842 encoder counts** (using midpoint 1973 and 4095 counts/revolution), **entirely within [776,3170]**. This is stronger evidence that the recorded values are in the **degrees** mode than in normalized [-100,100] mode. In the latter case a value below −100 is outside its expected representable range. Still verify the exact installed configuration and whether any processing changed the recorded values.

This result **does not mean URDF ±100° limit is wrong or safe to expand**: calibration's measured motor range and URDF mechanical/kinematic limits are independent, and the relation to URDF origin/sign has not been validated. Physical collision checking remains missing.

## 3. Read-only audit

The user's actual follower calibration must be supplied via local `--follower`; it is intentionally **not committed** to GitHub. Do not use Leader calibration for Follower.

Choose the mode matching the *actual training recording configuration*, not a guessed mode:

```bash
python scripts/so101_capture/audit_so101_calibration_readiness.py \
  --follower /path/to/my_follower_arm.json \
  --predictions outputs/so101-cartesian7-continuous-pred-v1/predictions.csv \
  --body-mode degrees \
  --output outputs/so101-calibration-readiness-degrees
```

To evaluate alternative hypothesis use a **different** output directory and `--body-mode range_m100_100`. This is not evidence of which config was used.

## 4. Mandatory gates before actuating the robot

1. Pin and inspect installed LeRobot revision, `SOFollowerRobotConfig.use_degrees` and recorded dataset feature units. Ensure the follower calibration file matches the one used for collection. Verify that the motor hardware calibration matches the file with torque and operating mode procedures approved by the operator.
2. Independently validate joint sign, offset and units with **known observed physical postures** (at least two non-collinear arm configurations), NOT by adjusting offsets until the URDF passes numeric limits. Validate J2 especially; the recorded −103.03 value is not sufficient to infer a 3.03° offset.
3. Rebuild physical FK-grounded Cartesian labels and retrain or re-evaluate as required if the original URDF transform is wrong. DO NOT silently remap old Cartesian7 predictions and call them physical targets.
4. Validate collision geometry against an actual known arm configuration and correctly exclude only expected adjacent/link-pair geometry. Use a real robot collision checker and test all candidate points **and continuous/interpolated segments** against self-collision and workcell obstacles. A static XML inventory is not collision detection.
5. Add deterministic hard limits, max joint step/velocity/acceleration, tracking-error watchdog, E-stop behavior, camera timeouts, and a deliberate operator-enabled hardware gate. Test the same signed and timed trajectory offline before any low-speed robot motion.
6. With workspace clear, tethered emergency stop and follower secured, do *observe-only* first; then carefully bounded single-joint tests under supervision, then no-object short trial, then short Cartesian7 model trials, and only then repeated pick-and-place Joint6 versus Cartesian7 at matched conditions. Never use the current generated IK candidate JSON directly as executable commands.

## 5. Remaining evidence requests

For GitHub-only follow-up: add static tests, pin actual dependencies, and evaluate any *committed* URDF. Local data/hardware required for the actual calibration verification and dynamic collisions. The next machine-side information needed is read-only `use_degrees`, installed LeRobot revision and model's `max_relative_target`, not a motor operation.

## 6. Local verification supplied 2026-10-30

The installed local LeRobot repo was reported at `~/workspace/lerobot/src/lerobot`. Introspection gave `SOFollowerRobotConfig.use_degrees` dataclass default `True` (instantiating without `port` raises TypeError; field inspection avoids hardware and port). Repo searches confirm `use_degrees=True` explicitly in `scripts/so101_capture/record_episode.py`, `record_episode_backup.py`, `observe_act_read_only.py`, `first_motion_test.py`, and other acquisition scripts. The default plus these explicit settings make degree-mode likely for recorded datasets; capture provenance still needs verification for exactly which entrypoint generated the 53 episodes.

Local audit with genuine Follower calibration `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json` on 100 consecutive episode 5 states returned zero inverse-normalized encoder violations for all body joints:
- J1 inferred 1714..1991, calibrated 733..3468
- J2 inferred 801..842, calibrated 776..3170
- J3 inferred 3084..3113, calibrated 907..3122
- J4 inferred 2827..2941, calibrated 863..3250
- J5 inferred 879..881, calibrated 0..4095

Unit test run: `python -m unittest discover -s tests -p test_so101_calibration_readiness.py -v` passed all 5 tests. **This confirms only internal numeric normalization consistency**, not the physical kinematic mapping. J2 is merely near its calibrated raw lower endpoint (25-count minimum gap for the sampled frames), so **future motion directions require endpoint margin checks**, not a blind offset. Never extend URDF bounds to silence a limit warning.

Status of items: (1) conversion implementation & degrees setting STRONGLY SUPPORTED, actual capture entrypoint still to verify; (2) 100-frame calibrated range numerical check DONE; (3) physical URDF sign/zero and TCP registration OPEN; (4) actual collision geometry overlap and swept-path evaluation OPEN.

