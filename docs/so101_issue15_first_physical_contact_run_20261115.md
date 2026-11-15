# Issue #15 — one-run MuJoCo contact isolation (2026-11-15)

## Single objective (do not drift)
**Reproduce or eliminate the abnormal force / arm-pose explosion when the fixed finger approaches/touches Muhi.** The specific former failure involved an abrupt switch from prescribed J1–J5 qpos/qvel state to full-arm physical PD hold, followed within milliseconds by >400 N fixed-jaw force and unstable object motion. Side camera pixel registration, wrist calibration, robot FK precision, MoveIt 2, GPU and ACT/SAC are **out of scope** here.

## Controlled change
Use only the **existing** `physical_grasp_table_bench.py` with the opt-in `--constraint-arm-hold` experimental variant, preserving the independent J6 position servo, actual physical contact, the real table, the original URDF and source STL, the already present 2 side + 3 wrist views, bracket/collision and MP4s. J1–J5 are prescribed during kinematic pregrasp; then their **integrated qpos/qvel are not overwritten** after fixture engagement. This is a **diagnostic equality constraint**, not a final servo design, not a perfectly rigid fixture, and not a demonstrated grasp success.

**MuJoCo documentation**: for one-joint equality, `q-qpos0=eq_data[0]`; the code's `model.eq_data[eqid,0]=armhome[k]-model.qpos0[...] ` is consistent with the documentation. See https://mujoco.readthedocs.io/en/stable/computation/#equality-constraints . Fixing J1–J5 permanently is more cleanly implemented by removing/freezing the joints in a dedicated model; if this test fails due to the equality fixture itself, switch to an isolated fixed-joint bench rather than adding speculative protection thresholds.

## One Mac run (simulation only)

```bash
conda activate so101-real2sim
git fetch origin
git switch feature/so101-constraint-grasp-rebuild
git pull --ff-only origin feature/so101-constraint-grasp-rebuild
python -m py_compile scripts/so101_sim/physical_grasp_table_bench.py

OUT="outputs/so101-issue15-contact-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/physical_grasp_table_bench.py \
  --simple-grasp \
  --fixed-jaw-surface-proxy --moving-jaw-surface-proxy \
  --precise-jaw-proxy --show-jaw-collision-overlay \
  --gentle-jaw-contact \
  --constraint-arm-hold --study-observe-through-contact \
  --target-grasp-depth-min-mm 8 --target-grasp-depth-max-mm 20 \
  --j6-position-speed-deg-s 35 --j6-actual-speed-limit-deg-s 35 \
  --lift-mm 5 --duration-s 8 \
  --render-every 5 --fps 30 --video --out "$OUT"

open "$OUT"
```

The observation flag records contact instead of immediately exiting due to nonessential heuristic thresholds; **it does not change MuJoCo physics**. If numeric blow-up occurs, that is failed physics, not a successful grasp. This simulation is not a hardware-safe control prescription.

## Inspect only these first
1. `summary.json`: did the fixture engage, what status, were two-jaw forces measured, did 5-mm lift actually occur?
2. `table_grasp_trace.csv`: temporal alignment of stage change, joint pose deviation, fixed and moving jaw forces, penetration and object velocity.
3. `table_contact_views.mp4`: five simultaneous views; individual `wrist_camera.mp4`, `wrist_shelf_camera.mp4`, `wrist_retracted_camera.mp4` should be kept. Compare the exact step of force spike, not arbitrary still frames.
4. If video generation fails, keep PNGs/CSV and report the exception rather than re-running the same experiment repeatedly.

## Next decision tree (one test only)
- **Fixture activates and J1–J5 remain stable; bilateral contact works** → old sudden arm PD switch was a substantial contributor. Next independently test full-arm position control at constant pose, without tweaking camera registration.
- **Fixture stable, but fixed jaw force spikes without moving jaw force** → investigate initial intersection, actual visual-vs-collision inner-face geometry and solver contact at that step. Do not tune arbitrary force guards.
- **Fixture itself fails or arm changes violently before bilateral contact** → validate equality/activation setup or move to a simpler model with truly removed/fixed joints, instead of re-implementing camera alignment.
- **Cannot find initial collision-free placement** → diagnose source STL/proxy and geometric setup as a separate test before controller tuning.

## Evidence boundary
The code has been static-reviewed and committed to GitHub but has NOT been executed on the user's Mac from this chat. The previous historical failure log is not reproduced by this commit until the user runs it. **No claims about passing collision or Lift are made here.**
