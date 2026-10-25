# ACT startup using the proven recording initialization order

Status: **synthetic validation only**. `ACTStartup.run()` explicitly rejects a
`LeRobotStartupDevice`; no hardware runner or executable ACT control loop is
enabled by this change. Do not run the withdrawn
`sync_goal_to_present.py --mode hardware` command.

`act_startup.py` keeps the recording path's order: both existing calibrations
and initial Leader/Follower pose are checked; Leader is configured; Follower
torque is disabled; all six fresh raw Present values are broadcast to Goal;
Follower is configured; torque is enabled; the existing
`record_episode.synchronize_positions()` aligns to a stationary Leader. It
uses LeRobot's bus methods and existing alignment routine; the recording
program itself is unchanged.

## Checkpoints and stop behavior

Raw Goal, Present, Torque_Enable and Status are read for all six axes at:

1. Immediately before the broadcast Goal write, after torque-off.
2. Immediately after the Goal write.
3. Before configuration, following a 20 ms no-write observation interval.
4. After register configuration and before explicit torque-on.
5. After explicit torque-on.
6. Before alignment.
7. Before the first ACT Action.

Each checkpoint requires all six expected torque values, Status=0, complete
register data, matching existing calibration and in-range Present raw values.
The pre-alignment checks also require no unexpected raw position change and
Goal readback matching the fresh pre-write Present. Any failure latches the
startup fault and prevents the next stage. No retry, Goal write or torque
change occurs during fault cleanup. Serial reads are sequential, so a
checkpoint is not an atomic six-axis sample.

When `diagnostics_path` points to a **new** JSON file, each checkpoint is
saved atomically as soon as its four six-axis raw-register reads finish, then
marked `ok` or `failed` after validation. The file also contains the current
stage, latched fault, last validated raw snapshot and a single best-effort
read-only fault snapshot. If that last read fails, `current_state` is
`unknown`; the last confirmed snapshot is **historical**, not proof of the
stopped servo state. Existing files are never overwritten. Log writing must
succeed before the state machine continues; it is part of the startup gate.
Synthetic example logs are in `validation/act_startup_synthetic/`:
`goal_torque_fault.json` shows Torque=1 immediately after Goal and no
configure checkpoint; `partial_torque_fault.json` shows a partly enabled
six-axis readback after a simulated explicit enable failure.

The local LeRobot `SO101Follower.configure()` uses `torque_disabled()`;
its `finally` block attempts to enable torque even if a configuration write
fails. The ACT adapter therefore invokes the *same LeRobot bus configuration
operations* in the same order, checks Torque and Status after each write, and
performs explicit torque-on only after a clean configuration checkpoint. This
is a deliberate difference from the recording implementation. The adapter is
for already-open devices and has no CLI. It has not been validated on hardware.

Before any ACT Action, the adapter reads all six raw registers and calibrated
Goal/Present values, checks calibration and communication, and refuses a
Goal/Present difference above the existing 8-unit tracking limit. The existing
`ActionGuard` is initialized with the existing 30 units/s per-axis slew limit,
8-unit tracking limit, calibrated joint limits and gripper 0–100 range. The
communication fault must remain latched in the eventual control loop; this
module does not itself send inference Actions.

## Independent hardware trial stages

Each stage requires its **own explicit approval** after reviewing the prior
stage's log. Do not treat an approval for one stage as approval for the next.
The synthetic `run()` method traverses all phases only with a fake device;
the future hardware runner must stop at each stage boundary.

| Stage | Entry and execution condition | Stop condition and output | Approval scope |
| --- | --- | --- | --- |
| 0: read only | Arm supported, workspace clear, power cutoff identified, correct USB port and existing calibration. Use only `inspect_first_motion.py` to read six-axis raw Goal/Present/Torque/Status and calibration hash. | Any missing response, Status != 0, unexpected torque, motion, or calibration mismatch stops. Save a new read-only JSON file; do not write Goal. | Explicit approval for **connection and reads only**. |
| 1: six-axis Goal broadcast | Only after reviewing Stage 0. Arm supported throughout. Recheck six-axis Torque=0, Status=0, fresh Present inside calibration; compare Goal before write. Permit exactly one existing LeRobot `sync_write('Goal_Position', fresh_present, normalize=False, num_retry=0)`. See `GOAL_BROADCAST_STAGE1.md`. | Read all four raw registers immediately and at 20, 100 and 500 ms targets. Any torque=1, position change, missing/incorrect Goal, or communication fault stops before configure. Save the fault state; no retry or cleanup write. | New approval that names **one broadcast Goal write and possible motor activation**. |
| 2: configuration/torque | Only after Stage 1 shows all six torque=0 and Goal≈Present, in the same controlled powered session. Recheck raw state and calibration. Apply the existing LeRobot register configuration operations with per-write Torque/Status monitoring; then explicitly enable torque and read back every axis. | Any error or partial torque state stops before Alignment. Record the six-axis state. A power cycle invalidates prior SRAM Goal and requires a fresh Stage 0 and review. | New approval for **configuration writes and explicit torque enable**. |
| 3: at most 1° on one rotary joint | Only after Stage 2, a supported arm, all six torque=1, Status=0, fresh Goal/Present and successful bounded Alignment. Choose a clear, in-range direction; monitor readback and the existing 8-unit tracking limit. | Any communication fault, tracking error, unexpected motion or timing overrun stops new Action. Record actual state; do not automatically torque off or write a new Goal in cleanup. | Separate approval specifying **joint, direction and ≤1°**. ACT inference remains disabled. |

Stage 0 was performed on 2026-10-21 and saved to
`outputs/so101-readonly-before-act-001.json`: all six Torque=0 and Status=0,
all Present values within their calibrated raw ranges, and all stored Goals=0.
This does **not** authorize Stage 1. The Stage-1 code and synthetic tests are
prepared; its command and independent approval conditions are in
`GOAL_BROADCAST_STAGE1.md`. The Stage-0 read-only command used was:

```sh
conda activate lerobot
python scripts/so101_capture/inspect_first_motion.py \
  --output outputs/so101-readonly-before-act-001.json
```

The displayed Stage-0 output path is already in use. Do not run that command
again with the same path. Stage 1 must obtain fresh Present values immediately
before any Goal write; it never uses the Stage-0 JSON as the target.

## Physical safety limits

- Support the arm and clear the workspace. Identify the physical servo-power
  cutoff. Confirm the Leader and Follower ports and calibration hashes.
- Review the currently unverified possibility that a Goal write itself enables
  torque. Treat the broadcast write as a potential motion event, even when
  Torque_Enable readback was 0 immediately before it.
- Arrange an operator-supervised trial of the checkpointed startup *without*
  ACT Action sending. Check all six Goal/Present/Torque/Status values at every
  boundary, especially immediately and 20 ms after the broadcast write.
- If any read, write, torque or position check fails, stop software control.
  **Stopping new software commands does not turn the motor off.** A servo may
  retain holding torque or continue toward its last Goal. If power is cut, the
  arm or held object may fall. Keep the arm supported, keep people clear, and
  use the identified physical servo-power cutoff only while maintaining that
  support. Software cannot guarantee physical holding or stopping.
- Only after reviewing that trace should ACT Action sending be considered.
  The gripper closing-lead threshold remains unset until measured safely.

Synthetic validation command:

```sh
conda activate lerobot
python -m unittest discover -s scripts/so101_capture -p test_act_startup.py -v
```
