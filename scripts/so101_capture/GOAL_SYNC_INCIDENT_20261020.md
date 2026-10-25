# Goal sync incident, 2026-10-20

This is a read-only analysis of `outputs/so101-goal-sync-hardware-001/`.
Do not repeat its hardware command. The hardware branch of
`sync_goal_to_present.py` is disabled, including direct `HardwareGoalIO.open()`.

The preflight read showed all six `Torque_Enable=0`, `Status=0`, position mode 0,
and angular resolution 1. `shoulder_pan` had `Goal_Position=0` and
`Present_Position=1481` in decoded raw register units. The script sent exactly
one per-servo `Goal_Position=1481` write to that motor with `normalize=False`
and `num_retry=0`. The write returned after 0.356 ms. At 1.435 ms after the
write, its immediate readback was Goal=1481, Present=1481, Torque=0, Status=0.
At 6.850 ms after the write, the next all-axis read showed shoulder_pan
Torque=1; all other torque values remained 0. Every Present value was unchanged.
The script stopped before any second Goal write. A later read-only probe again
found shoulder_pan Goal=Present=1481 and Torque=1, with the other five Goals=0
and Torques=0. The operator then cut Follower servo power while supporting the arm.

The installed LeRobot `bus.write("Goal_Position", ...)` path sends one write to
the Goal register and does not explicitly write `Torque_Enable`. The observed
delayed torque change is therefore a hardware/firmware or external-bus behavior,
but the exact cause cannot be proved from this trace. The Feetech register
manual distinguishes the two registers; it does not guarantee that Goal writes
preserve torque state under every firmware and protection condition. The
initial Status=0 gives no evidence of an active reported protection fault.

The trial falsifies the assumption that a Goal write while Torque=0 leaves
Torque=0 on this setup. No further Goal sync under that assumption is safe.
Do not re-enable the old command or automatically write Torque=0 as a cleanup:
torque loss can let the arm fall. Any revised procedure must explicitly handle
the possibility of immediate torque activation and must receive separate
operator approval. Existing datasets, checkpoint, LeRobot source and
calibration are untouched.
