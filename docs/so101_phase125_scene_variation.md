# SO-101 Phase 1.25 — Scene variation and IK-data generation

**Status 2026-11-08:** Work started; scene-variation previews implemented. **No teacher trajectories, IK, contact physics, ACT-ready episodes or Sim2Real success have been generated or validated yet.**

## Scope and dependencies

- Preserve existing physical SO-101 mounting at table edge and real Side/Wrist cameras.
- Preserve all original LeRobot recordings. No robot or camera devices opened by this PoC.
- Start from measured Muhi (97×35 mm), cup (78 mm diameter × 83 mm high), curved tabletop, Episode 4 scene and the **previously saved cup alignment**. Default fixed yaw deltas are **0°, +30°, -30°, +60°, -60°, +90°, -90°**; the first seven scenes exercise these exact angles with unchanged XY positions. The remaining 13 scenes exercise XY/cup perturbations and mixed layouts.
- Scene transformation is in the **SO-101 base XY plane**, rotating Muhi about its own center and translating the cup without rotating it.
- Baseline and independently varied placements are generated with deterministic RNG seed. No unrestricted exploration around robot collisions.
- Scene previews are **not** ACT training datasets; there are no labeled correct actions until separate IK and evaluation steps pass.

## First runnable implementation

After updating to `feature/so101-sim-data-augmentation`, in the same `so101-real2sim` Python environment:

```bash
DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
OUT="outputs/so101-phase125-scenes-$(date +%Y%m%d-%H%M%S)"

python scripts/so101_sim/generate_scene_variations.py \
  --dataset-root "$DATASET" --episode 4 \
  --extrinsics outputs/so101-eye-to-hand-id1-20261107/result.json \
  --intrinsics assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json \
  --object-layout-json outputs/real2sim/episode4-objects-20261108-004859.json \
  --cup-pose-json outputs/real2sim/episode4-cup-gui-20261108-131429.json \
  --table-extra-left-cm 4 --table-extra-front-cm 5 \
  --table-base-x-cm 12 --table-base-y-cm 7 \
  --table-yaw-delta-deg -9.375 \
  --count 20 --seed 101 --muhi-range-mm 20 \
  --cup-range-mm 20 --yaw-range-deg 30 \
  --output "$OUT"

open "$OUT/contact_sheet.png"
```

Outputs: `scene_000.png` through `scene_019.png`, `contact_sheet.png`, and `manifest.json`. Manifest records seed, source dataset/pose paths, per-scene Muhi/Cup base XY, delta yaw, and **NOT_GENERATED / NOT_RUN / NOT_TESTED** status. All scene output is outside the source dataset.

## Subsequent work, not implemented in this first commit

1. Extract a reference demonstration's **TCP world trajectory** with MuJoCo FK at recorded sample times.
2. Segment Pick / Lift / Place; change Pick waypoints relative to the moved Muhi and Place waypoints relative to the moved cup; interpolate transport, using bounded workspace and appropriate gripper schedule.
3. Solve regularized IK for *each transformed waypoint*, starting near logged joint angles; reject discontinuity, joint limit and known collision risks. Require source robot geometry validation rather than fitting offsets to video.
4. Replay the new action sequence with strict geometry checks, then add clearly labeled idealized grasp following and a *separate* physics/contact validation step.
5. Export synchronized ACT dataset only after image-observation/Joint6 action units and time base match; compare real-only training against real+synthetic, holding out positions/yaws.

## Next-stage acceptance criteria

- [x] Deterministic baseline + 19 varied layouts and static Side preview renderer
- [x] Mutating Muhi/cup geoms leaves table/camera/robot untouched; unit tests
- [ ] Run on the local Episode 4 dataset and inspect generated sheet (CI uses synthetic data, not user's local episodes)
- [ ] Transformed TCP trajectory, IK validation and rejection manifest
- [ ] Kinematic Pick/Lift/Place replay and geometric success checks
- [ ] Optional physical grasp test and ACT-ready export, **only after** preceding criteria


## Phase 1.25 Step 2: recorded TCP-proxy trajectory (implemented)

The recorded Episode 4 `observation.state` drives the SO-101 URDF
using MuJoCo FK; the script saves the **gripper_link body-frame origin**
position (m, SO-101 base) and WXYZ quaternion for each source frame.
The original **logged `action` (Joint6)**, `observation.state`,
`timestamp` and `frame_index` are stored unchanged. It does not
claim a physically measured fingertip TCP, successful grasp, or
camera accuracy. The current mapping is an **unverified**
joint sign +1 / offset 0° configuration (explicit in output metadata).

```bash
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
TCP_OUT="outputs/so101-phase125-tcp-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/extract_episode_tcp_trajectory.py \
  --dataset-root "$DATASET" --episode 4 --stride 1 --max-frames 2000 \
  --output "$TCP_OUT"
```

Artifacts: `tcp_trajectory.json` (metadata + per-frame TCP proxy, original
Joint6 state and action) and `tcp_trajectory.csv`. Source dataset is
read-only and remains unchanged.

**Not implemented:** Phase 1.25 Step 3 object-relative Pick/Lift/Place
segmentation, re-targeted IK, per-candidate collision checks and
ACT-ready action/observation/video training examples. This extraction
is a foundation, not synthetic data.

## Phase 1.25 parallel experiment: MuJoCo Reach reinforcement learning

`scripts/so101_sim/so101_reach_env.py` is a **Gymnasium** environment
using MuJoCo `mj_step` dynamics, a per-step bounded 5-joint target
increment action, PD + gravity/bias generalized-force controller,
and the gripper_link origin as its *approximate* TCP proxy. The
14-dimensional observation is joint positions (5), velocities (5),
target-to-TCP displacement (3) and distance (1). Targets are selected
by FK of sampled in-range robot joint configurations, so they are
**kinematically reachable targets**, not measured Muhi grasp targets;
collision-free paths and real servo behavior remain unverified.

Reward = negative Euclidean distance to target. Reaching 25 mm
terminates the episode; maximum length defaults to 150 control steps.
No contact grasping, physics cup/bottle dynamics or real hardware
commands are included.

```bash
# One-time optional environment dependency if not present:
python -m pip install "gymnasium>=1,<2"

# Environment smoke test (random action; not training):
python scripts/so101_sim/so101_reach_env.py --episodes 3 --seed 101

# Optional SAC training (requires separate install):
python -m pip install "stable-baselines3>=2.4,<3" tensorboard
SAC_OUT="outputs/so101-reach-sac-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/train_reach_sac.py \
  --steps 20000 --seed 101 --out "$SAC_OUT"
cat "$SAC_OUT/summary.json"
```

`summary.json` records reach rate, goal distance and episodes from a
held-out deterministic evaluation. 20,000 SAC steps are a **smoke
training budget** and may not reach adequate performance; do not
claim convergence. For a scientific comparison: fix evaluation
initial-state/goal distributions, use multiple training seeds, and
include random-policy and/or demonstration-informed baselines.

### Reach SAC baseline (2026-11-08, user-local result)

- Run directory: `outputs/so101-reach-sac-20261108-200308/`
- CPU, SAC, seed 101, 5,000 training steps, Gymnasium 1.4.0, random-policy smoke test 3 episodes.
- Final deterministic evaluation: 0/10 successful reaches within 25 mm; all 10 episodes lasted 150 steps; final TCP-proxy distance ranged ~0.157–0.437 m, mean ~0.294 m.
- EvalCallback at 5,000 steps: 0% success, mean reward -47.80 +/- 7.93, 150 steps per episode.
- Training rollout success_rate was 0.25 at 4 completed episodes and 0.0312 at 32. This is **consistent with** one success in the 32-episode rolling sample, **not proof** of when the success occurred or a statistically supported learning trend. Early 0.25 is from pre-learning/random exploration (learning_starts=1,000).
- Do not label actor_loss decrease as task improvement; no measured convergence.
- The run artifacts are on the user's Mac; not committed to GitHub.

Next minimal checks: (1) drive PD to *known joint-space FK target* without learning; (2) sample nearby FK goals and train an otherwise identical SAC run; (3) compare on matched fixed goal sets and multiple seeds before claiming a method advantage.

```bash
# Direct joint-angle controller diagnostic; no RL:
python scripts/so101_sim/diagnose_reach_pd.py \
  --episodes 10 --seed 101 \
  --out "outputs/so101-reach-pd-$(date +%Y%m%d-%H%M%S).json"

# Near-goal curriculum on the same MuJoCo model; saves a separate run.
NEAR_OUT="outputs/so101-reach-near-sac-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/train_reach_sac.py \
  --steps 5000 --seed 101 --near-goal-joint-rad 0.35 --out "$NEAR_OUT"
cat "$NEAR_OUT/summary.json"
```

Note: full-range baseline and near-goal run use **different training/evaluation goal distributions**. Their raw success rates alone are not an A/B comparison. For proper comparison, evaluate trained policies on the **same held-out goal set** and use several seeds. The near-goal sampling parameter is a starting point, not an optimized curriculum.

### Research status

- [x] Recorded Step 2 TCP-proxy / state / action / timestamp export script
- [x] Low-dimensional MuJoCo Reach Gymnasium environment with torque-based PD controller
- [x] Optional SAC entry point and 10-episode deterministic evaluation output
- [x] Synthetic code-level tests for FK trajectory and Reach resets/steps
- [ ] Run Episode 4 extraction locally and inspect FK trajectories
- [ ] Execute SAC training locally and report learning curves / reach rate
- [ ] Robot actual end-effector TCP calibration and physics validation
- [ ] Step 3 object-relative re-targeted IK for Muhi/cup variants
- [ ] Proper teacher data, gripper contact, Pick/Lift/Place success evaluation

## Main integration status (2026-11-08)

Phase 1 prerequisite PRs are **merged into main** in dependency order:

- [PR #11](https://github.com/neogim/so101-imitation-learning/pull/11): multicamera recording safety changes (merged).
- [PR #8](https://github.com/neogim/so101-imitation-learning/pull/8): Real2Sim MuJoCo MVP (merged).
- [PR #9](https://github.com/neogim/so101-imitation-learning/pull/9): ArUco/camera, measured workcell and GUI (merged).
- [PR #12](https://github.com/neogim/so101-imitation-learning/pull/12): this Phase 1.25 development, **still Draft / unmerged**, now targeting main.

The working branch was created on the former PR #9 head, but GitHub now correctly shows only this Phase 1.25 document, generator and unit-test files in the diff against main. Do not merge #12 before IK trajectory generation and real-data smoke inspection.
