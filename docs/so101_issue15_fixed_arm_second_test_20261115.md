# Issue #15 second experiment: true frozen arm (2026-11-15)

## Purpose

Answer **one causal question:** Does the previous J1 oscillation and abnormal unilateral contact persist when J1–J5 are **absent from the compiled MuJoCo model** instead of held by equality constraints?

This deliberately differs from the previous run's staged J1/J2/J3 approach. It is a **minimal contact isolation baseline**; it cannot by itself demonstrate that the same 21.72 N impact was fixed in identical initial geometry. Do not compare peak force as a controlled A/B until relative finger–object contact gap, J6 state, and object pose are also comparable.

## Code

`scripts/so101_sim/physical_grasp_fixed_arm_bench.py`

- Uses official `MjSpec.delete(spec.joint(name))` for J1–J5. J5 is baked into the child body's local quaternion at -90 degrees. The script **verifies compiled arm-joint count is zero**.
- Reuses existing exact source URDF/STL, the fixed/moving 2/4mm jaw BOX proxy generators, existing photographic wrist bracket and collision shapes, physical Muhi and a fixed mocap tabletop.
- Keeps J6 a *dynamic position-actuated* hinge with bounded torque; bottle has a freejoint and gravity/contact are real. No object pose reset while stepping.
- Outputs joint/force/penetration and object speed `grasp_trace.csv`, `summary.json`, the same 1440×780 combined **three wrist + two side** view MP4, plus 3 individual wrist MP4s.
- Does not pretend to reproduce the previous full-arm trajectory, real calibrated camera, actual measured mass, or successful Lift.
- `--render-every 25 --fps 20` maps 0.002×25 = 0.05 seconds per saved frame at 20fps, avoiding the previous 8s-sim/26.7s-playback mismatch.
- Early failure if jaw initial intersection or robot-table intersection is detected. This is setup validation only, not a heuristic force guard.

## Run once on Mac

```bash
conda activate so101-real2sim
git fetch origin
git switch feature/so101-constraint-grasp-rebuild
git pull --ff-only origin feature/so101-constraint-grasp-rebuild
python -m py_compile scripts/so101_sim/physical_grasp_fixed_arm_bench.py

OUT="outputs/so101-fixed-arm-contact-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/physical_grasp_fixed_arm_bench.py \
  --duration-s 4 --j5-deg -90 \
  --j6-speed-deg-s 35 --render-every 25 --fps 20 \
  --out "$OUT"
open "$OUT"
```

## Evaluation

1. `arm_joint_count == 0` confirms J1–J5 really do not exist. No J1 fluctuation is structurally possible.
2. If both jaw contact forces remain 0 → diagnosis is **contact geometry and setup**, not controller; don't declare success.
3. If fixed-jaw force spikes and the moving finger remains at 0 despite frozen joints → investigate collision proxy/initial gap, including swept J6, while avoiding controller switches.
4. If bilateral contact occurs and object stays on tabletop → inspect contact depth, forces, speed. Do not claim a Lift; it was not implemented here.
5. If model construction or camera rendering fails, capture the exception and correct the **specific API misuse**. Do not add unrelated diagnostics or repeat camera matching.

## Verification boundary

GitHub static implementation committed 2026-11-15. **MuJoCo is not installed in the connected execution environment and this code is not yet physically executed on the user's Mac.** A GitHub commit is NOT proof that a run succeeded.

Reference: [Issue #15](https://github.com/neogim/so101-imitation-learning/issues/15); [Notion results & graph](https://app.notion.com/p/41985e16f4dc8108bf65eae86540d52a).
