# SO-101: Episode 4 saved DLS vs SAC under MuJoCo physics

Implemented as an **opt-in open-loop trajectory-tracking comparison** (2026-11-13). It is intentionally separate from the prior kinematic DLS/SAC comparison and does **not** retrain the SAC checkpoint.

## What is actually being compared

The existing `compare_real2sim_muhi_cup_dls_sac.py` saves `comparison_rollout.csv` with *both* DLS and SAC 5-joint reference trajectories for the same saved Episode-4 Muhi/cup layout, the stage names (`muhi`, `cup`, `home`), and the recorded starting pose. New `compare_dls_sac_physics.py` reads exactly that file. It checks that both policy trajectories begin at identical joints and that these agree with the saved scene.

Each policy gets a separate `MjData` with identical starting pose and object placement; both share the same MuJoCo model, capped joint PD gains, robot limits, movable Muhi, collidable tabletop/cup, gripper hold, and wrist collision proxies. At each saved reference frame the simulator executes `mj_step()` repeatedly to apply torque-limited tracking. **Neither controller sees the evolving simulated observations and replans; this is not closed-loop SAC vs DLS performance**, but an assessment of whether their existing planned motions physically track and collide.

The recorded home position may be outside URDF limits. It is checked against the saved scene *before* clipping; the physics runs start from the same URDF-clipped pose, and every target is clipped in both policies. This changes the initial conditions relative to the kinematic figure and must be disclosed; never send original out-of-bounds home angles to motors.

The four-panel MP4 shows:

| Row | Left | Right |
| --- | --- | --- |
| Top | DLS Side | SAC Side |
| Bottom | DLS Wrist | SAC Wrist |

Wrist images are from the already implemented J5/J6 camera, not a separate invented view. If the camera doesn't aim at Muhi from a particular arm pose, the image will not contain the object; don't alter physical lens geometry solely to make a fake good view.

## Mac commands (repository root; no personal paths)

```bash
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
python -m unittest discover -s scripts/so101_sim -p 'test_compare_dls_sac_physics.py' -v

find outputs -type f -name comparison_rollout.csv
ROLL_OUT="$(find outputs -type f -name comparison_rollout.csv | sort | tail -n 1)"
if [ ! -f "$ROLL_OUT" ]; then
  echo "No previous Episode-4 DLS/SAC comparison_rollout.csv was found. Run the original compare script or provide its saved output."
else
  OUT="outputs/so101-dls-sac-physical-$(date +%Y%m%d-%H%M%S)"
  if python scripts/so101_sim/compare_dls_sac_physics.py --rollout "$ROLL_OUT" --out "$OUT" --substeps 10 --render-every 4 --video-fps 8; then
    if [ -f "$OUT/dls_vs_sac_physics_side_wrist.mp4" ]; then
      open "$OUT/dls_vs_sac_physics_side_wrist.mp4"
    else
      echo "MP4 missing (ffmpeg?). See summary.json and comparison_frames."
    fi
    open "$OUT/physical_tcp_error.png"
    cat "$OUT/summary.json"
  fi
fi
```

If `find` prints more than one matching CSV, select the **correct original Episode-4 DLS/SAC** path manually. The last alphabetical path is only a convenience and can be wrong. Never pass the J5 ablation `rollout.csv` (different format/experiment).

## Outputs and interpretation

- `dls_vs_sac_physics_side_wrist.mp4`: Side and Wrist, DLS and SAC at synchronized **saved-frame indexes**. Rows from one policy may finish before the other and are held at their last joint target. Frames are not time-aligned by stage transitions.
- `comparison_frames/`: four-panel PNGs, retained for detailed diagnosis.
- `physics_comparison.csv`: actual simulated TCP-to-current-goal error, J1–J5 tracking error, contact count/category, Muhi Z, time and whether a source target was clipped. This is NOT the original kinematic error curve.
- `physical_tcp_error.png`: physical tracking TCP error curves and 10 mm *position-only* threshold.
- `summary.json`: gains, control substeps, clipping, per-stage final TCP error, contact counts, video status and limitations.

This is a starting experiment, not an assertion that the physical trajectories are stable. The existing base/shoulder convex-hull false contacts may severely disturb PD, and the current wrist fixture has only photo-estimated geometry. Muhi is physically free, cup has collidable walls, but there is no validated gripping or placing. An SAC action-level closed-loop test, physical gripper grasp and collision-safe controller will follow after this diagnostic.

## 2026-11-13: User-run physics comparison failed fidelity review

Evidence: `so101-dls-sac-physical-20261113-014727.zip` (201 trajectory frames per policy, 4.02 s simulated time, 51 four-panel images, MP4, CSV, summary).

**Do not use these numbers for a DLS vs SAC performance claim**:

| Observation | DLS | SAC |
| --- | ---: | ---: |
| Maximum Euclidean five-joint tracking error | 288.14° | 302.57° |
| Muhi-stage final TCP position error | 319.88 mm | 231.67 mm |
| Cup-stage final TCP position error | 130.26 mm | 308.96 mm |
| Cumulative contact points in report | 8,967 | 6,256 |
| Wrist/external contact steps | 196 | 0 |

Both policies already had 75.12° five-joint tracking error at the *first recorded physical step*. No reliable approach/pick result exists. Rendering shows a radically altered start, horizontal collapse with contact, visual objects and wrist frame contamination. These are failure observations, not reasons to rank the algorithms.

### Confirmed regression causes

1. The original visual recorded start is **outside imported URDF limits**. The physics script silently clipped the initial `qpos` and every reference target; the previous kinematic demonstration preserved the actual saved visual pose and only returned out-of-bounds joints gradually. This immediately changed the start and invalidated faithful comparison before physics began.
2. Large base/shoulder convex-hull contacts from earlier audits were left unresolved, while a capped `3 Nm` PD controller was asked to track a new target every `10 × 0.002=0.02s`. Neither dynamic feasibility nor original target-time mapping was validated. Thus the subsequent collapse/penetration is expected, not a faithful physical version of the baseline.
3. The original side-by-side rendering included target spheres, TCP trails, step labels and a deliberately framed camera. The new output omitted these cues, changed camera choice/panel resolution, and used bare Side/Wrist views. This **visual regression is unrelated to physics**.

### Safety-first code correction

- Default behavior now **rejects** initial joint clipping with `PHYSICS BASELINE REJECTED`, showing maximum clip discrepancy. The comparison is blocked until the recorded visual initial joint angles, URDF joint zero/sign/offsets and physical joint limits are reconciled. This is intentional; do **not** silently force a fake matching pose for a more attractive animation.
- For *debug only*, `--allow-clipped-diagnostic` allows the previously mismatched start, but reports `initial_pose_audit`, labels `NON_COMPARABLE_CLIPPED_DIAGNOSTIC_ONLY`, and prominently watermarks each video panel.
- Restored reference goal markers and actual simulated TCP path in the Side panels, with TCP-error and time/stage captions. Existing Wrist view is retained; it is not an alternative to the original view.

**Still not solved:** real/Sim pose calibration, spurious self-contact, PD gain and torque feasibility, restoring the exact original stage timing/trajectory and identifying controller dynamics. Do not call the physical counterpart fully implemented or ready for ACT before these pass. The appropriate next technical action is *pose alignment and isolated initial contact model improvement*, followed by a stable physical comparison with fixed reference timings and documented convergence. Avoid another series of blind runs with changed settings.

### Useful commands

```bash
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
python -m unittest discover -s scripts/so101_sim -p 'test_compare_dls_sac_physics.py' -v
ROLL_OUT="$(find outputs -type f -name comparison_rollout.csv | sort | tail -n 1)"
OUT="outputs/so101-dls-sac-fidelity-audit-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/compare_dls_sac_physics.py --rollout "$ROLL_OUT" --out "$OUT"
```

The last command should **reject the known mismatched recorded initial pose**, and explain the reason, unless the underlying pose-mapping issue has been corrected. This rejection is a valid and desirable outcome. Only use `--allow-clipped-diagnostic` when explicitly diagnosing the invalid physics scene; it cannot establish DLS or SAC physical performance.
