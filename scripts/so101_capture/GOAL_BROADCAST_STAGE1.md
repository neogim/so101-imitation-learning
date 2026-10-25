# Stage 1: one six-axis Goal broadcast (prepared, NOT executed)

**Current authorization covers code and synthetic tests only.** The Follower
was not connected for this stage. The old single-axis
`sync_goal_to_present.py --mode hardware` remains disabled. This Stage-1 probe
has no path to `configure()`, torque enable, Alignment, 1° movement, or ACT
inference.

The 55-demo recorder used LeRobot's `follower.bus.sync_write('Goal_Position',
current)` after a normalized Present read. The probe uses the **same LeRobot
broadcast sync_write implementation**, but reads raw Present values and passes
`normalize=False, num_retry=0` so each transmitted Goal register value is
exactly the freshly read raw Present. A SYNC_WRITE has no servo status ACK;
the program verifies all six Goal registers by subsequent reads. A returned
SDK success code alone is not proof that all six motors accepted the packet.

## Gates and one-shot behavior

1. A new output directory is created exclusively. The event log is flushed to
   disk before the broadcast and after each observation; existing outputs are
   not overwritten.
2. Open only the Follower bus, without `Robot.connect()` or `configure()`.
   Check the existing calibration. Read six-axis raw Goal, Present, Torque and
   Status twice. Require complete integer values, Torque=0, Status=0, and
   Present inside each calibrated raw range. Between the two reads, reject
   unexpected movement or a changed stored Goal.
3. Build six Goal values from the **second, fresh Present read**. Reject if the
   oldest part of that sample is more than 100 ms old before write. The prior
   read-only JSON is evidence for review, never a write target.
4. Make **one** broadcast `sync_write('Goal_Position', target,
   normalize=False, num_retry=0)` call. Treat any timeout as ambiguous
   delivery; never retry or restore the old Goal.
5. Read all four registers for all six axes immediately, then at targets of
   20, 100 and 500 ms after write return. Require every Goal raw value to equal
   the sent target exactly, Torque=0, Status=0 and no more than 5 raw ticks of
   Present change from the pre-write sample. Record actual read times; reads
   are sequential rather than an atomic snapshot.
6. End the process after readback. A failure stops new writes and permits only
   one best-effort diagnostic read. The serial port is closed without a torque
   command. Neither successful readback nor a later recovered read advances
   to configuration.

## Operator procedure for a future, separately approved trial

Before seeking approval, check that all other robot programs are stopped and
the output directory is unused. Support the whole arm, clear the workspace,
remove grasped objects, identify the **servo power** cutoff (not merely USB),
and have an operator ready to keep the arm supported if power is cut. Power-on
and read-only connection themselves must be treated as potentially hazardous
because the stored Goal may be far from Present if torque unexpectedly enables.

The separately authorized Stage-1 command would be:

```sh
conda activate lerobot
cd /path/to/so101-imitation-learning
python scripts/so101_capture/probe_goal_broadcast.py \
  --mode hardware \
  --output outputs/so101-goal-broadcast-stage1-001 \
  --hardware-goal-broadcast-approved \
  --ack-arm-supported \
  --ack-clear-workspace \
  --ack-power-stop-ready
```

It also requires the exact interactive phrase `ONE BROADCAST GOAL WRITE`.
**Do not run this command under the present authorization.** A separate
explicit approval must identify the single broadcast Goal write and the
possibility of unexpected torque activation. It does not authorize Stage 2.

If any axis turns torque ON, moves unexpectedly, has a Goal mismatch, reports
Status != 0 or loses communication, stop software control. The operator should
keep the arm supported and clear of people, then use the identified physical
servo-power cutoff if necessary while maintaining support. Software stop
does **not** remove holding torque or guarantee that the arm stops; power-off
can let the arm or a held object fall. Do not issue cleanup Goal or torque
commands, and do not proceed to configure or Alignment. Review `events.jsonl`
and `result.json` before deciding anything further.

The 2026-10-20 single-axis Goal write was followed by unexpected torque ON.
Its cause remains unknown; the broadcast write could have the same effect.

Synthetic-only check:

```sh
python -m unittest discover -s scripts/so101_capture -p test_probe_goal_broadcast.py -v
python scripts/so101_capture/probe_goal_broadcast.py \
  --mode synthetic --output outputs/so101-goal-broadcast-synthetic-001
```
