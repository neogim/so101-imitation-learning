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
