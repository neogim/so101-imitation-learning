# SO-101 ACT: first-motion readiness (no autonomous drive performed)

The pinned policy is the 55-demo `018000` checkpoint. `observe_act_read_only.py`
supports `--n-action-steps 1`. In installed LeRobot 0.6.2, `select_action()`
predicts 50 actions, takes `actions[:, :1]`, queues one, returns that first
action, and repeats at the next observation. The other 49 are discarded. The
checkpoint has `temporal_ensemble_coeff=None`; no temporal ensembling is used.
Ensembling instead predicts every cycle and combines actions intended for the
same future time with exponentially weighted older predictions.

## Findings before any write

- Read-only Follower + OBS index 1 observations completed at 20 Hz for 5 seconds
  with queue length 50 and again with queue length 1. All six `Torque_Enable`
  values were 0. Thus these observations do not demonstrate powered tracking.
- Queue length 50 produced a `shoulder_lift` candidate jump of 46.49 degrees
  at frame 49→50. Queue length 1 had at most 0.25 degrees of raw change in a
  static scene. Moving-scene continuity is unverified.
- The Action guard enforces calibrated joint endpoints, 0–100 gripper range,
  30 units/s per-axis goal slew (1.5 units per 50 ms), tracking error >8,
  communication-failure latch, and stale-state rejection. Gripper closing lead
  remains `None`; no unmeasured value has been chosen.
- A synthetic 100-cycle ACT loop with 2 ms fake Goal_Position writes completed
  under 50 ms after a no-write warmup. Injected write timeout did not retry;
  injected 60 ms write caused a stop after the first write. These do **not**
  measure a real servo write. Feetech `sync_write` has no status acknowledgment;
  the next position read is needed to assess actual following.

## Safe staging, only after explicit approval for motor writes

1. Remove Muhi and the container. Arrange physical support for the arm and a
   clear path for the chosen joint. Assign one operator to the keyboard and a
   physical power-stop method. Confirm power removal may release the arm and
   anything held by the gripper.
2. Recheck USB identity, six calibration joints, calibration hash, current
   `Present_Position`, existing `Goal_Position`, `Torque_Enable`, and servo
   supply/temperature. Do not auto-calibrate or run `Robot.connect()` here:
   LeRobot's normal connection calls `configure()` and writes registers.
3. With the arm supported, use a separately approved, previously validated
   initialization procedure that writes current positions as goals **before**
   enabling torque. Observe the arm throughout. The first-motion script does
   not enable torque and refuses if any axis is off, so the current state of
   six disabled axes cannot proceed to motion.
4. Re-read all six positions and goals; the script requires goal–position
   difference ≤1 unit on every axis, all torques enabled, and <0.5 unit pose
   drift during a 0.5-second no-write period. It also requires the chosen
   target to be inside existing calibrated limits.
5. First run `first_motion_test.py` on a synthetic bus. For the approved
   hardware trial, choose one collision-free **rotary** joint and a signed
   displacement no greater than 1 degree. No object or gripper motion. The
   script holds that small target for half the test and returns toward the
   baseline in the second half, at 20 Hz. It measures state read, torque read,
   guard, Goal_Position write, full cycle, and following. The test stops if
   tracking error exceeds 8, the joint fails to move measurably, a read/write
   fails, a cycle exceeds 50 ms, the loop gap exceeds 75 ms, or the operator
   presses `q`/Ctrl-C. There is no retry or automatic rearm.
6. On any stop, **no new Goal_Position is sent in cleanup**. The motor may
   continue toward or hold its last target. Support the arm before removing
   power; physical stopping and holding are not guaranteed. Inspect the
   diagnostic JSON and do not resume automatically.
7. Only after the independent small-motion trial passes should an ACT-driven
   no-object trial be designed and separately approved. The current
   `first_motion_test.py` does not run ACT or a Pick & Place task.

Synthetic command:

```sh
conda activate lerobot
python scripts/so101_capture/first_motion_test.py \
  --mode synthetic --joint wrist_flex.pos --delta-deg 1 \
  --output outputs/first-motion-synthetic-NEW
```

Hardware mode requires separate approval, all four hardware flags, an
interactive terminal, and a typed arming phrase. The signed joint/direction
and unused output path must be chosen after inspecting the current pose.

## Calibrating a gripper closing-lead limit later

No numerical closing lead is set. It is a different quantity from the hard
0–100 gripper range. The gripper closes toward **smaller** normalized values.

1. Under a separately approved supervised teleoperation trial, measure at
   every 20 Hz cycle: requested gripper position, previous sent goal, current
   measured position, timestamps, and the sign of motion. Record free-space
   closing at several slow speeds, repeated from different starting positions.
   Log `Present_Current`, `Present_Load`, voltage and temperature at a tested
   lower rate; these are diagnostics, not calibrated force readings.
2. Compute closing lead as `measured - previously_sent_goal` only while
   closing. Quantify its distribution, delayed samples, and worst normal
   free-motion lag. Check the whole-cycle timing impact of extra reads. Do not
   infer safe grasp force from position error alone.
3. With support and an easily released object, repeat gently while recording
   the same signals and video. Compare contact, free motion and an intentionally
   aborted obstruction. Any approach to the unchanged tracking limit 8 or
   abnormal current/temperature is a stop, not a reason to widen the limit.
4. Choose a lead bound only from those measurements with margin for noise and
   delay; review false contact detections. Test the bound first in replay, then
   in a separately approved no-object run, then under supervision with the
   object. Holding a goal can still maintain force; opening must remain
   operator-controlled and verified. No automatic measured-position reset.

## Remaining limits

The first real Goal_Position write duration, actual motor response, sustained
20 Hz under powered motion, contact load, and safe gripper lead are unmeasured.
Do not treat the no-write timing test or a successful synthetic test as an
authorization to run autonomous Pick & Place.
