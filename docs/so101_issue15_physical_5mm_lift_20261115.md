# Issue #15 — physical 5mm Lift with J6 grasp preserved (2026-11-15)

## Objective

Validate that the previously observed **J6-first tabletop bilateral grasp** can hold the Muhi while the *robot assembly* moves vertically by 5mm. The Muhi remains a completely free MuJoCo body: gravity, friction and finger contacts determine whether it follows. No Muhi `qpos` or pose is overwritten during simulation.

### Important experimental boundary

This is **not** restored J1–J5 servo/IK motion. J1–J5 remain absent from the compiled model because the previously tested control change caused oscillation. Instead, the complete SO-101 assembly including the wrist camera rig is temporarily attached to a MuJoCo **mocap base**. An explicit 5mm smoothstep motion is commanded on the mocap base *only after real force contacts are seen on both fingers*. This is a standard kinematic fixture/retention study, **not** proof of an actuated robot pickup. The source URDF, STL, wrist geometry and Muhi freejoint are unchanged.

### Existing known tabletop result (preserve)

Previously observed `so101-fixed-arm-contact-20261115-171628.zip`:
- Fixed-jaw initial gap **1.87794mm**, J6 moving contact first at **2.386s**, fixed-jaw contact at **2.442s**
- 780 steps of simultaneous bilateral force; peaks fixed **1.55197N**, moving **1.64582N**
- Arm joints all removed; no force spikes. No Lift existed.

### New Lift sequence
1. Start with the **same** 1–2mm fixed-jaw clearance and J6 position servo (same gains/limit).
2. Count consecutive 2ms frames with **both** jaw forces >=0.02N. After 150 (0.3s), record bottle center Z and begin 5mm lift.
3. Move **mocap base** vertically using smoothstep over 1.0s. Hold at 5mm for remaining simulation duration, while Muhi remains a free body.
4. Log each step: sim time, phase, commanded arm-base Z / 5mm, bottle center Z rise relative to lift start, approximate bottle bottom gap to tabletop, bottle-table normal force, both jaw forces, J6 actual/target, object speed, penetration.
5. **Provisional success:** arm commanded 5mm, object rose >=4mm, approximate table gap >=1mm, and for final 0.3s both jaw forces persist while table normal force <0.01N. This alone is not proof of real SO-101 pick or exact geometric clearance.

Videos: original `table_contact_views.mp4` (1440x780, 3 wrist native 480x390 and 2 side native 720x390), three individual wrist MP4s. `--render-every 25 --fps 20` matches 2ms physics step to 20fps real sim time.

### Single Mac run

```bash
conda activate so101-real2sim
git fetch origin
git switch feature/so101-constraint-grasp-rebuild
git pull --ff-only origin feature/so101-constraint-grasp-rebuild

python -m py_compile scripts/so101_sim/physical_grasp_fixed_arm_bench.py

OUT="outputs/so101-5mm-lift-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/physical_grasp_fixed_arm_bench.py \
  --duration-s 5.5 \
  --j5-deg -90 \
  --j6-speed-deg-s 35 \
  --lift-mm 5 \
  --lift-ramp-s 1.0 \
  --bilateral-hold-steps 150 \
  --render-every 25 \
  --fps 20 \
  --out "$OUT"
open "$OUT"
```

If the runtime exits with an error, send the stack trace; API/camera issues should be corrected without changing successful contact parameters. GitHub commit does **not** demonstrate MuJoCo runtime correctness: this environment cannot run the user's Mac physics scene.

### Decision after one run
- **Lift success and physical Muhi rise** -> transition to a separate *actual J1–J5 actuator* implementation as a new task, avoiding abrupt PD handoff.
- **Arm rises but bottle stays on table/drops** -> inspect jaw friction, contact normals, orientation, J6 holding force (not camera calibration).
- **Bilateral contact absent** -> do not start lift; preserve current working grasp and diagnose regression.
- **Excessive collision force/solver blowup** -> stop calling it a grasp and isolate physics.

This task explicitly does not include camera recalibration, MoveIt 2, GPU shopping, or ACT/SAC training. 
