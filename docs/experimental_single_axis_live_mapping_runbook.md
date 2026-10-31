# Experimental single-axis SO-101 live mapping: operator runbook

## What changed
`scripts/so101_capture/experimental_single_axis_live_mapping.py` is a **new experimental hardware controller** and NOT yet bench-validated with physical servos. Its default mode is a pure Python simulation (no LeRobot or serial import). Hardware mode connects to Leader/Follower, checks calibrations and existing torque-off state, checks all six joint initial pose differences <=1.5 degrees (gripper normalized units), raw calibration bounds and selected-axis margin >=12 ticks and enough for +/-1degree, writes a one-shot all-axis Goal=Present in raw ticks, verifies readback, then **enables Follower torque**. This moment may move the arm. It allows only J2 OR J3 to follow Leader delta, at most +/-1 degree, max .15degree per ~50ms control cycle, with an 8s time window. All other goals held at initial positions. Writes CSV. It does not run ACT and does not recalibrate motors.

## STOP / residual risks
- This software has not been physically tested. No guarantee against falling, collision, communication timeout, or mechanical damage. Work only with an operator physically present, secured base, supported arm and independently available power cutoff.
- When startup/loop fails the code **does not automatically disable torque**, because this may collapse the arm under gravity. Conversely torque may already be lost in a bus fault. Support the arm before power removal.
- The URDF limits are not physical hard stops. J2=-102.242deg and J3=+96.571deg in previous readout; a small raw calibrated margin does NOT prove adequate collision clearance. Manually check the entire potential motion path before enabling.
- The first torque ON can induce unwanted movement. If base/support/emergency cutoff/cable condition not verified, do not run hardware mode.
- This is an exploratory diagnostic, not ready for autonomous operation or production.

## Simulation (no device opened)
```bash
conda activate lerobot
git pull --ff-only origin feature/cartesian-action-offline-export
python -m py_compile scripts/so101_capture/experimental_single_axis_live_mapping.py
python scripts/so101_capture/experimental_single_axis_live_mapping.py --axis shoulder_lift
python scripts/so101_capture/experimental_single_axis_live_mapping.py --axis elbow_flex
```

## Experimental hardware operation (only after local code inspection and physical checks)
```bash
python scripts/so101_capture/experimental_single_axis_live_mapping.py \
  --axis shoulder_lift --hardware \
  --approval 'I HAVE SECURED THE ARM AND VERIFIED EMERGENCY STOP' \
  --output outputs/so101-live-j2-mapping-v1.csv
```
Keep Leader entirely still when torque is enabled, then shift **only the selected Leader axis** slightly and slowly within +/-1 degree. It is not a wide-range teleop. If any unexpected behavior occurs, immediately use independent physical power-off / emergency-stop and secure the arm. Do not run a second robot script simultaneously. Review the CSV and controller logs before considering J3. If a startup check refuses, do not bypass it; investigate the actual calibrated endpoint and support.

**Do not use J3 command until its clearance from the calibrated endpoint is verified:** prior J3 angle is near upper URDF boundary and likely near calibrated end. No guarantee its permitted direction is known.

## Interpretation
A delta in LeRobot J2/J3 matched between Leader and Follower only proves command tracking in LeRobot units. To determine URDF sign, observe which way the physical link rotates and compare that to URDF positive joint axis using the FK. To determine offset, compare a known physical reference posture. CSV alone cannot determine zero offset.
