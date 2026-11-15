# SO-101 Physical AI workcell: dynamic Muhi + Side/Wrist contacts

Status: **experimental implementation added 2026-11-13; local Mac/MuJoCo run pending**. Do not claim a physically successful grasp or accurate Real-to-Sim until measured.

## What is implemented

- Reuses the **existing** saved Episode 4 calibrated Side scene, opt-in wrist camera with camera bracket/PCB/lens collision proxies, and its saved transparent cup geometry.
- An **opt-in free-joint Muhi** is created in `physical_workcell_sim.install_dynamic_muhi` before scene compilation. It uses 97 mm length, 35 mm diameter and a **provisional** 50 g mass (40 g body + 10 g cap), with friction coefficients subject to validation.
- The old static Muhi preview geom is hidden and its contact mask disabled; the dynamic object takes its original saved position/yaw. Tabletop and cup bottom/walls are set collidable, with the current static 16-box wall approximation.
- `mj_step` integrates gravity, free-body friction and contacts. Mode `drop` tests gravitational settling; `push` applies a known external force to test contact/friction independent of the robot; `replay` applies limited PD joint targets from prior J5-ablation `rollout.csv`, **not** current SAC inference.
- Both cameras render frames from the same physics state. CSV and a numerical graph show Muhi displacement and contact counts; JSON lists parameters, outcomes and caveats.
- Original URDF/STL, checkpoints, recordings and earlier kinematic SAC comparisons stay unchanged. Original scene callers default to `physical_objects=False`.

## Copy-paste commands (macOS zsh; from repository root)

```bash
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
python -m unittest discover -s scripts/so101_sim -p 'test_physical_workcell_sim.py' -v
OUT="outputs/so101-physical-drop-$(date +%Y%m%d-%H%M%S)"
if python scripts/so101_sim/physical_workcell_sim.py --mode drop --steps 600 --render-every 30 --out "$OUT"; then
  open "$OUT/physics_trace.png"
  open "$OUT/side"
  open "$OUT/wrist"
  cat "$OUT/summary.json"
else
  echo "Physics drop failed; inspect the error before proceeding"
fi
```

Only if the first test is stable, test an **external force**, which is deliberately not robot grasping:

```bash
OUT_PUSH="outputs/so101-physical-push-$(date +%Y%m%d-%H%M%S)"
if python scripts/so101_sim/physical_workcell_sim.py --mode push --steps 600 --force-n 0.8 --force-steps 80 --out "$OUT_PUSH"; then
  open "$OUT_PUSH/physics_trace.png"
  cat "$OUT_PUSH/summary.json"
fi
```

Optional PD trajectory replay after inspecting collision/PD stability:

```bash
ROLL_OUT="$(find outputs -type f -name rollout.csv | grep -Ei 'j5|ablation' | sort | tail -n 1)"
if [ -f "$ROLL_OUT" ]; then
  OUT_REPLAY="outputs/so101-physical-replay-$(date +%Y%m%d-%H%M%S)"
  if python scripts/so101_sim/physical_workcell_sim.py --mode replay --rollout "$ROLL_OUT" --steps 600 --out "$OUT_REPLAY"; then
    open "$OUT_REPLAY/physics_trace.png"
    cat "$OUT_REPLAY/summary.json"
  fi
fi
```

## Acceptance criteria and open issues

1. Synthetic tests pass, without data/hardware. On the saved workcell, drop mode should rest on the tabletop rather than fall through. Record maximum penetration (MuJoCo soft contacts may permit small overlap) and pose. Compare before/after adding wrist mass.
2. No hidden static bottle should remain in physical collisions. Map anomalous named contact pairs; distinguish base–shoulder hull self-contact from workcell collisions.
3. The **provisional** camera body, mounting bracket, PCB and lens colliders should contact table, bottle, cup, and nonadjacent robot links where physically applicable, without colliding with their own parent mounting parts. Mount dimensions and cable envelope are not yet calibrated.
4. The cup is a static segmented-wall proxy, not yet a validated watertight physical transparent cylinder or deformable cup.
5. The old recorded home violates URDF limits. Physics starts with five joints clipped to model limits, **not** from the exact visual starting pose. PD replay will not produce the same SAC trajectory, and large existing base/shoulder collision hulls may cause instability.
6. Gripper physical opening/closing, stable Muhi grasp, lift/place, ACT closed-loop inference and task success criteria remain future stages. In particular, Wrist RGB rendering does not by itself guarantee that the ACT model's training camera intrinsics, feature keys, crop or time synchronization match.
7. When Mac results are received, append the **measured plot**, relevant numbers, failed cases and commit to the existing Notion experiment log; do not fabricate unrun results.

## CI failure and correction (2026-11-13)

The first CI run for the synthetic drop test failed: a dynamic Muhi fell through the synthetic table (final body Z approximately -4.815 m after 500 steps). This was an **actual failed test**, not a successful physical contact demonstration. The initial implementation enabled collision masks for the old display-only table geoms **after compilation**. The follow-up change now marks table and cup geoms collidable in `MjSpec` before `spec.compile()`, and the synthetic test creates a matching cup and applies the same preprocessing. The previously saved `kinematic` comparison continues to use the original non-collidable previews.

Do not infer the fix is successful until the new CI run and local Mac scene run pass. Keep this failure as evidence of the importance of collision-pair generation and model construction order.

## 2026-11-13 Mac Episode-4 drop audit — user-provided archive

Input: `so101-physical-drop-20261113-004308.zip`, 600 simulation steps / 1.2 s, with `summary.json`, `physics_trace.csv`, plot and 21 synchronized Side/Wrist frame pairs.

- Muhi stops on the tabletop: table contact 564/600 steps, at final body-centre z ≈12.384 mm, final displacement 25.116 mm and speed ≈0.000075 m/s. It initially dropped ~30 mm before settling. No cup contacts (0 steps).
- The **robot itself fell/collapsed** during the drop because previous `drop` and `push` did not control its joints. This is NOT a successful full arm physics simulation. Wrist rendered frames increasingly show black/partial table and a small jaw fragment, not useful task observations.
- Total minimum signed contact distance was −79.824 mm, **not** proven bottle/table penetration. Robot self-collisions contribute strongly; 547 steps with wrist contact were mostly wrist/self contact, while actual wrist/table recorded only a short transient and wrist/cup and wrist/Muhi were zero.
- Fix: `drop` and `push` now use a **kinematically held robot obstacle** while the bottle remains fully dynamic; `replay` continues to use PD and may still be unstable. This deliberately isolates free-body dynamics. Not a full dynamic-arm contact-control result and not suitable as an ACT environment yet. New output distinguishes Muhi/table distance, wrist/external distance, wrist-self and wrist-external contact steps.
- The photo-derived camera position/optical axis still needs validation against real Wrist images before ACT. Do not interpret rendered darkness as a camera hardware failure.

Run the documented drop command again with a new output directory. Compare arm silhouette, wrist camera view, body-centre z, object/table penetration and the wrist contact categories. Do not run ACT until camera geometry, meaningful image composition and physical grasp controller are validated.

## 2026-11-13: synchronized Side/Wrist MP4 output (default)

Physical workcell runs now generate `side_wrist_comparison.mp4` automatically alongside the retained PNGs, `physics_trace.csv` and `physics_trace.png`. Each frame shows Side and Wrist views from the same simulation state and includes simulation-time labels. With the default `--render-every 30` at 2ms/physics step, frames are sampled every 0.06s; the default `--video-fps 5` makes playback ~3.3x slower than simulated time for visual inspection. Use `--video-fps 10` for faster review, or `--no-video` to skip encoding. An installed `ffmpeg` is required; if missing, the script preserves PNG/CSV outputs and reports `video_status=ffmpeg_missing` in summary instead of fabricating an MP4.

Example after activation and Git pull:

```bash
OUT="outputs/so101-physical-push-video-$(date +%Y%m%d-%H%M%S)"
if python scripts/so101_sim/physical_workcell_sim.py --mode push --steps 600 --render-every 30 --video-fps 5 --out "$OUT"; then
  if [ -f "$OUT/side_wrist_comparison.mp4" ]; then
    open "$OUT/side_wrist_comparison.mp4"
  else
    echo "MP4 not generated; check ffmpeg and summary.json"
  fi
  cat "$OUT/summary.json"
fi
```

User-provided Push trial `so101-physical-push-20261113-005841.zip` reported final Muhi displacement 393.407mm, table-contact steps 504/600, cup-contact steps 0, and wrist-contact steps 0. This was an external force of 0.8N over 80 steps; do **not** call it an arm-generated push or a completed pick/place task. An independent 4.2-second slowed comparison MP4 was created from the 21 pairs of user-submitted PNGs. On Mac, the new built-in writer still needs one end-to-end validation run.
