# Phase 6A — Physical joint hold + historical Real-to-Sim offline rendering

## What is implemented

- `scripts/so101_sim/phase6_dynamic_joint_hold.py` — **object-free dynamic** J1–J5 PD position-hold from physics step one; J6 held as auxiliary. It records raw physical `joint_trace.csv` and `summary.json`, without camera or geometry proxy logic. It does not alter the source URDF or use robot hardware.
- `scripts/so101_sim/phase6_render_legacy_camera_trace.py` — **OFFLINE visual playback** reading that recorded CSV. It imports and calls the old `wrist_camera_sim.install`, old `render_wrist_camera_rollout.jaw_centers_in_wrist_view`, and the existing original `mujoco.Renderer.update_scene` sequence. It renders (1) side example from `workcell.example.json`, (2) mounted wrist optic from original `wrist_camera_sim`, and (3) the existing wrist bracket inspection optic. It does NOT invent a replacement camera that impersonates wrist footage.

The first physics simulation and the rendering are deliberately **separate processes/models**; in the rendering replay it is appropriate to set saved `qpos` frame-by-frame and call `mj_forward`. **The physics simulation never overwrites `qpos/qvel` after initialization.**

## Important scope

The original legacy wrist mount profile and original optical orientation are used by default, with no ad-hoc aim-at-TCP. The side view comes from an explicitly **uncalibrated example JSON**; exact reproduction of the user's earlier calibrated Episode0 video would require the correct experiment's camera extrinsics and scene configuration. Hence this is *reuse of the historical renderer implementation*, not a claim of identical pixels. Static Muhi is not added because Phase 6A is specifically no-object.

No prior branch is mutated. Work is only on `feature/so101-dynamic-arm-control`.

## Mac: run physical 4s pose holding

```bash
conda activate so101-real2sim
git fetch origin
git switch feature/so101-dynamic-arm-control
git pull --ff-only origin feature/so101-dynamic-arm-control

python -m py_compile \
  scripts/so101_sim/phase6_dynamic_joint_hold.py \
  scripts/so101_sim/phase6_render_legacy_camera_trace.py

OUT="outputs/so101-phase6a-hold-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/phase6_dynamic_joint_hold.py \
  --duration-s 4 --kp 18 --kd 1.8 --max-torque-nm 3 \
  --out "$OUT"
cat "$OUT/summary.json"
```

**Review first:** if the physical run errors or the max absolute joint errors/speeds look severely unstable, do not declare the holding task complete and do not move to IK.

## Mac: replay actual recorded state in historical MuJoCo camera renderer

```bash
VIS="${OUT}-legacy-render"
python scripts/so101_sim/phase6_render_legacy_camera_trace.py \
  --trace "$OUT/joint_trace.csv" \
  --out "$VIS" \
  --bracket-profile legacy \
  --stride 100 \
  --fps 5 \
  --video
open "$VIS/phase6_legacy_three_views.mp4"
```

This renderer is optional. A playback frame does not prove dynamic physical correctness; the CSV does. The side camera is a sample, **NOT** the measured physical camera extrinsic. The render does not repair or replace unresolved wrist FOV issues from Phase5.

## Next gates

1. Analyze output `summary.json` and `joint_trace.csv`. If errors/torques show unstable hold, change only PD/gain or gravity-compensation mechanisms, not cameras.
2. Once Phase6A hold is demonstrated, Phase6B generates 5mm TCP target by damped-least-squares/Jacobian or IK, and physically tracks it via J1–J5 joint actuators **without** Muhi.
3. Only then integrate Phase4 J6-first grasp for Phase6C, checking actual hold/slip rather than a peak-only rise.

**Verification:** all code is committed, but no MuJoCo Mac run has been performed from this tool session.
