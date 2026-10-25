# Next SO-101 hardware step after the 2026-10-20 Goal write

The old six-axis `sync_goal_to_present.py --mode hardware` procedure is
withdrawn. Its first Goal_Position write was followed within milliseconds by
`shoulder_pan` Torque_Enable changing from 0 to 1. The operator cut Follower
servo power while supporting the arm. Do not run the old command again.

## Present authorization boundary

No Follower power-on, Goal write, torque command, teleoperation, or ACT Action
will be initiated by Codex under the former "Goal sync with torque OFF"
authorization. The next Goal write must be treated as a **possible torque-on
and motion event**. A new explicit operator approval is required.

## Proposed staged procedure for review, not executable yet

1. Support the whole arm and remove grasped objects. Clear the workspace and
   identify the servo supply's physical power switch. USB disconnection is not
   the power-stop procedure. A second person should be ready to cut servo power
   while keeping the arm supported.
2. With the arm supported, power the servo bus and perform **read-only** checks
   of all six Torque_Enable, Status, Operating_Mode, Angular_Resolution,
   Goal_Position and Present_Position raw registers, plus calibration match.
   If any axis is powered unexpectedly, any Status is nonzero, or the arm moves,
   do not proceed to a Goal write.
3. The prepared `probe_goal_torque_one_axis.py` is a separate, interactive
   program that permits **at most one named servo Goal_Position write per
   invocation**. It must read
   that servo's fresh Present raw value, verify it is inside its calibrated
   range, and write only that same raw value. It must not enable/disable torque,
   retry an ambiguous write, write a second joint, or run ACT.
4. The program reads back Goal, Present, Torque and Status at 0, 20, 100 and
   500 ms after the write. The trial is a **torque-activation probe** even if no
   explicit Torque_Enable write occurs. If torque becomes 1, treat that as the
   expected possibility, not a successful torque-OFF synchronization. Any
   motion, read failure, Status change or mismatched Goal ends software work;
   the supported operator decides whether to cut servo power.
5. Do not advance to other axes or to the one-degree motion test based on one
   successful readback. Review the individual trace and decide the next step
   with fresh approval. The shoulder and elbow axes are load bearing and need
   particular support.

This design cannot make a Goal write physically harmless. Software stop means
no *new* command; it cannot remove holding torque or guarantee no motion.
