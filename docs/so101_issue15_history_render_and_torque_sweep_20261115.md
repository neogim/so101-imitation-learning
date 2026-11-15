# Issue #15: Git-history rendering recovery and J6 torque batch (2026-11-15)

## Why
The previous 5-view video had already worked in scripts/so101_sim/physical_grasp_table_bench.py. Rewriting the renderer in each new diagnostic bench introduced visual regressions and replaced required mounted wrist perspectives. This restores that existing code path rather than inventing another.

## Verified historical renderer configuration
- One 480x360 MuJoCo Renderer, 2 side free cameras at azimuths 55/150°, elevation -20°, distance .32m aimed at target.
- Original 1440x780 compositor: side panels at (240,30)/(720,30); mounted wrist TCP/shelf/retracted cameras at (0,420)/(480,420)/(960,420).
- Original MjvOption geom group visibility: group3 bracket collider hidden, group5 jaw proxy hidden; group4 pale backdrop enabled in wrist views, disabled in side views.
- Original detailed visual STL unchanged; no virtual camera substituted or overlaid onto mounted camera outputs.
Important: The historical renderer configuration does not itself guarantee the current mounted wrist optical pose shows the finger tips. Do not misrepresent this.

## User's 0.12Nm evidence
Archive so101-lift-012-20261115-180423.zip: base +5.0mm, bottle max rise +0.0184mm, final bottle change -0.0490mm, fixed peak 2.2959N, moving peak 2.2705N; LIFT_NOT_CONFIRMED.

## One-command batch
New scripts/so101_sim/sweep_fixed_arm_lift_torque.py automatically runs identical J6-first + 5mm-Lift conditions, without rendering/PNG/MP4 during each test via --no-video, then saves torque_comparison.csv and sweep_summary.json. Option --render-best reruns only the selected condition with legacy 5-view video. Selection by actual rise does not mean a pass.

```bash
conda activate so101-real2sim
git pull --ff-only origin feature/so101-constraint-grasp-rebuild
python -m py_compile scripts/so101_sim/physical_grasp_fixed_arm_bench.py scripts/so101_sim/sweep_fixed_arm_lift_torque.py
OUT="outputs/so101-lift-sweep-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/sweep_fixed_arm_lift_torque.py --torques 0.12 0.14 0.16 0.18 0.20 --duration-s 4.5 --render-best --out "$OUT"
cat "$OUT/torque_comparison.csv"
open "$OUT"
```

Duration 4.5s uses historically observed lift start 2.74s and 1.0s ramp, leaving roughly 0.76s for observation. A different torque can change timing; check lift start/command in trial logs before ruling a case out. No GPU or physical SO-101 control is used.

If no setting lifts the object, do NOT keep increasing torque. Next check tangential contact/friction, collision geometry and actual bottle mass. New implementation committed but not run on user's Mac; do not claim renderer/sweep verified.