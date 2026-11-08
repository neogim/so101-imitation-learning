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

### PD control failure: substep tracing and diagnosis strategy (2026-11-08)

The **user's** 10-episode known-joint-target PD oracle reached its FK TCP
goal only once (1/10; mean final distance about 0.305 m). A subsequent
near-goal SAC run (`--near-goal-joint-rad 0.35`, 5,000 steps, seed 101)
still scored 0/10 in its separate evaluation. This does **not** prove
that SAC is inherently unstable. Investigate the control/physics layer
before simply increasing learning steps.

`scripts/so101_sim/diagnose_reach_pd_trace.py` instruments the *same*
`SO101ReachEnv.step` PD controller used in the RL task via an optional
per-physics-substep callback. The normal RL path remains uninstrumented.

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
python -m pip install "matplotlib>=3.7,<4"

TRACE_OUT="outputs/so101-pd-trace-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/diagnose_reach_pd_trace.py \
  --seed 101 --episode 0 --out "$TRACE_OUT"
open "$TRACE_OUT/joint_angles.png"
open "$TRACE_OUT/joint_torques.png"
open "$TRACE_OUT/joint_velocities.png"
open "$TRACE_OUT/tcp_distance_contacts.png"
cat "$TRACE_OUT/summary.json"
```

Files: `pd_trace.csv` (physics-substep time, per-joint target/actual
angles and velocities, raw PD+bias generalized torque demand and
clipped applied torque, saturation flags, contact geom pairs, TCP distance),
`summary.json` (per-joint saturation fractions, contact-pair counts
and final errors), plus four separate PNG charts.

For contrast, re-run with `--episode 3` in a **different output
directory**; this was the successful episode in the prior 10-trial run.
Keep all other controller parameters fixed until the primary failure
mechanism is understood.

Diagnostic decision ladder (hypotheses, not confirmed causes):

1. **Frequent clipped torques**: validate units, gravity/bias forces,
   model mass/inertia and joint torque budget; only then vary a single
   parameter in Sim and compare.
2. **Oscillating joint angles without saturation**: study P/D gains,
   damping, control timestep and velocity.
3. **Movement halts with recurring contact pairs**: inspect collision
   geometry and joint limits; not all reported contact pairs are harmful.
4. **No saturation/contact, but tracking still poor**: inspect task
   time horizon, joint-name/axis mapping and FK implementation.
5. **PD tracks known joint targets reliably**: freeze controller and
   revisit the Reach SAC learning problem using matched goal evaluation
   sets and multiple seeds.

These traces are **offline simulation diagnostics**: MuJoCo generalized
force demand is not an actual servo torque measurement, contact events
are not yet categorized as unsafe collisions, and FK-reachable goals
are not guaranteed obstacle-free.

### SO-101 compiled collision model audit

The source URDF has 17 `<collision>` mesh elements across eight links
(one dummy gripper-frame link has no collision geom). On the compiled
MuJoCo collision-only model, the **first four meshes are fused into
`world`** as the fixed `base_link`; the remaining geometry is attached
to dynamic link bodies. Real compiled IDs—not a guess from XML ordering:

| MuJoCo geom | Link/body | Compiled mesh name |
| --- | --- | --- |
| 0 | world (fixed base_link) | base_motor_holder_so101_v1 |
| 1 | world (fixed base_link) | base_so101_v2 |
| 2 | world (fixed base_link) | sts3215_03a_v1 |
| 3 | world (fixed base_link) | waveshare_mounting_plate_so101_v2 |
| 4–6 | shoulder_link | sts3215_03a_v1; motor_holder_so101_base_v1; rotation_pitch_so101_v1 |
| 7–8 | upper_arm_link | sts3215_03a_v1; upper_arm_so101_v1 |
| 9–11 | lower_arm_link | under_arm_so101_v1; motor_holder_so101_wrist_v1; sts3215_03a_v1 |
| 12–13 | wrist_link | sts3215_03a_no_horn_v1; wrist_roll_pitch_so101_v2 |
| 14–15 | gripper_link | sts3215_03a_v1; wrist_roll_follower_so101_v1 |
| 16 | moving_jaw_so101_v1_link | moving_jaw_so101_v1 |

CI verified that **before any PD movement**, at zero joint angles,
MuJoCo emits 4 contact points for each of these three pairs:
`geom0-geom6`, `geom1-geom6`, `geom2-geom6`. This matches
the persistent base/shoulder contact records from the 2026-11-08 local
PD trace. It is strong evidence that the current collision model
has overlaps/contacts near the shoulder's rotation-pitch part,
but does NOT prove those are illegitimate physical penetrations
or that the contact forces are the entire cause of PD failure.

Diagnostic-only script and fixed-goal A/B comparison:

```bash
git pull --ff-only origin feature/so101-sim-data-augmentation
AUDIT_OUT="outputs/so101-collision-audit-$(date +%Y%m%d-%H%M%S).json"
python scripts/so101_sim/inspect_reach_collisions.py \
  --seed 101 --episode 0 --out "$AUDIT_OUT"
cat "$AUDIT_OUT"
```

The output maps `geom0`–`geom16` to the *compiled* body and
mesh names, logs initial contacts, and repeats the same FK-goal
PD oracle **with contacts enabled** and with all contact physics
**temporarily disabled only in the second simulation**. It makes
no changes to the URDF or the normal SAC environment.
Do NOT use the contact-disabled result to certify model realism or
safe hardware operation. If enabling/disabling contacts materially
changes torque demand or FK target convergence, inspect base/shoulder
collision meshes and adjacent-link exclusions before changing PD
gains or starting more SAC training.

### Selective base/shoulder collision ablation (diagnostic only)

The fixed URDF base and moving shoulder have three contacts at the initial
zero-angle pose: `geom0–6`, `geom1–6`, and `geom2–6`. Their compiled mesh
identities were confirmed in GitHub CI. To isolate which pair affects
tracking, `scripts/so101_sim/compare_base_shoulder_contacts.py` now
compares **seven** controlled simulations:

| Condition | Meaning |
| --- | --- |
| baseline | Original full contact model |
| reencoded_masks_control | Reassign per-geom contact bitmasks without excluding any pair |
| exclude_geom0_6 | Only base motor holder / shoulder rotation mesh excluded |
| exclude_geom1_6 | Only base body / shoulder rotation mesh excluded |
| exclude_geom2_6 | Only fixed servo / shoulder rotation mesh excluded |
| exclude_all_three | All three identified base/shoulder pairs excluded |
| all_contacts_disabled_reference | Unphysical no-contact diagnostic reference |

Every condition uses seed 101, the same Episode 0 FK target and the same
PD gains and torque limit; changes are **in-memory only**.
Specific-geom-pair exclusions use distinct contype bits and per-geom
conaffinity bitsets, because MuJoCo body excludes would hide *all* body-pair
contacts. The explicit re-encoded-mask control checks the filtering operation
itself against the original. At the initial reset all specified excluded
pairs must disappear, while contacts from other pairs are retained.

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
COMPARE_OUT="outputs/so101-base-shoulder-ablation-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/compare_base_shoulder_contacts.py \
  --seed 101 --episode 0 --out "$COMPARE_OUT"
cat "$COMPARE_OUT/pair_ablation.csv"
```

Outputs: `pair_ablation.json` with per-condition contact pairs, saturation
fractions, final errors, and `pair_ablation.csv` for a compact comparison.
Do **not** copy excluded-contacts conditions into SAC/Sim2Real training:
temporary removal may allow physically impossible intersections.

Decision:
- If one excluded pair substantially stabilizes PD tracking, inspect that
  STL mesh overlap, contact normals and clearance; retain real collisions.
- If only all-three exclusion works, investigate interactions among the
  base/shoulder mesh contact patches.
- If contacts remain or PD stays inaccurate, separate contact artifacts
  from controller gain / torque budget / inertial model issues.
- A final distance smaller than baseline but still above 25 mm is
  **partial improvement**, not a successful Reach.
- CI on Linux confirms script validity; repeat the comparison locally on
  the Mac before diagnosing numerical sensitivity.

### Base/shoulder mesh visualization and primitive collision proxy experiment

**2026-11-08 / diagnostics only.** First-phase findings:

- A compiled zero-pose mesh/contact snapshot of `geom0`, `geom1`,
  `geom2` (base) and `geom6` (shoulder) revealed three initial
  contact pairs. CI's signed MuJoCo contact distances were approximately
  **-26.4mm (0–6), -27.9mm (1–6), and -22.4mm (2–6)**, each with
  four contact points. Negative values indicate significant interpenetration
  according to MuJoCo's compiled collision geometry. This is *not*
  necessarily an equal amount of true CAD-STL physical penetration:
  MuJoCo can convexify imported collision meshes.
- `visualize_base_shoulder_contacts.py` generates a 3D plot showing
  compiled collision-mesh vertices and the exact MuJoCo contact points
  and writes signed distances to CSV. Inspect the local PNG before
  inferring which STL triangles overlap.
- `compare_collision_proxies.py` runs *in-memory* collision-primitive
  substitutions: only geom6 as scaled box/sphere/capsule; or geom0,
  geom1, geom2 and geom6 as boxes. All original contacts remain
  enabled. These are **illustrative approximations**, *not* validated
  collision shapes, and they are deliberately not installed in SAC.

#### Initial Linux CI baseline (seed 101, Episode 0, 150 steps)

| Collision condition | Final TCP error | Joint error L2 | Contact substep rate | Mean saturation |
| --- | ---: | ---: | ---: | ---: |
| Original compiled mesh | 304.68 mm | 2.791 rad | 100% | 94.4% |
| geom6 box at scale 0.65 | 223.34 mm | 2.200 rad | 100% | 74.5% |
| geom6 sphere at scale 0.65 | 224.81 mm | 3.459 rad | 100% | 76.2% |
| geom6 capsule at scale 0.65 | 225.00 mm | 3.197 rad | 100% | 76.4% |
| geoms 0,1,2,6 as boxes scale 0.65 | 225.27 mm | 3.282 rad | 100% | 76.3% |

None passed the 25mm reach threshold. Primitive sizing is based on
the compiled mesh's origin-centered vertex extents and a shrink
factor, not a physical fit. Lower error alone is **not** proof of
a better or safe collision representation. Collisions remained
active throughout all runs. Re-evaluate on the user's Mac.

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
VIS_OUT="outputs/so101-contact-visual-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/visualize_base_shoulder_contacts.py --out "$VIS_OUT"
open "$VIS_OUT/base_shoulder_contacts_3d.png"
cat "$VIS_OUT/contacts.csv"

PROXY_OUT="outputs/so101-proxy-compare-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/compare_collision_proxies.py \
  --seed 101 --episode 0 --scale 0.65 --out "$PROXY_OUT"
cat "$PROXY_OUT/proxy_comparison.csv"
```

#### Decision gate before SAC

Do **not** restart SAC on the current mesh/proxy collision model.
First check on the local Mac whether substantial zero-pose penetrations
are reproduced, and review actual base/shoulder mesh fitting.
Next build a **physically defensible** collision proxy with verified
clearance (separate visual meshes from simplified collision geoms),
and require the PD oracle to track multiple FK targets without
persistent large contact reactions and torque saturation. Only then
test RL again; track Reach success using a common goal distribution
and multiple training seeds. For time-efficient learning, defer
contact-rich Pick/Place mechanics until the small Reach controller is
stable.

### Source STL versus convex hull (2026-11-08)

Following the user's upload of the real MuJoCo contact plot to Notion, we
tested the hypothesis that importing nonconvex STL as convex collision
geometry closes up necessary base/shoulder clearances. New offline tool:

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
python -m pip install "trimesh>=4,<5"
CONVEX_OUT="outputs/so101-convexity-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/audit_stl_convexity.py --out "$CONVEX_OUT"
cat "$CONVEX_OUT/stl_convexity.json"
open "$CONVEX_OUT/geom0_stl_vs_hull.png"
open "$CONVEX_OUT/geom6_stl_vs_hull.png"
```

Source-STL geometry tests (GitHub Actions, Linux):

| geom | Part | STL watertight and valid volume | convex hull / STL volume |
| --- | --- | --- | ---: |
| 0 | base_motor_holder_so101_v1 | yes | 3.965 |
| 1 | base_so101_v2 | yes | 3.045 |
| 2 | sts3215_03a_v1 | yes | 1.109 |
| 6 | rotation_pitch_so101_v1 | yes | 2.200 |

All four are volume-valid STL meshes. The large hull-to-source ratios for
geoms 0, 1 and 6 indicate sizeable concavities/open spaces in the
original meshes that a convex envelope can fill. **This does not by
itself prove** that each MuJoCo contact is false. The STL/hull images
are in *STL-local coordinates*, not the MuJoCo world frame, and are
not overlays of the contact points. To directly diagnose phantom
contacts, compare *world-aligned* STL surface geometry against compiled
collision hull at the reported contact coordinates, respecting the URDF
origin and mesh transforms.

We have **not** modified the production collision geometry or restarted
SAC. Existing box/sphere/capsule diagnostics remained in contact
and failed the 25mm Reach success threshold. The next meaningful
engineering step is a physically fitted nonconvex-friendly collision
proxy (multiple convex parts or fitted capsules/boxes) with appropriate
clearances and verification on several FK targets.

### World-aligned STL surface vs MuJoCo contact overlay (2026-11-08)

A new **offline-only** diagnostic `overlay_stl_mujoco_contacts.py`
maps source STL triangles into the actual MuJoCo world coordinate frame
using compiler `mesh_pos/mesh_quat/mesh_scale` and `geom_xmat/geom_xpos`.
It verifies the reconstructed vertices against compiled mesh vertices
(95th percentile nearest-vertex alignment error ~1-2 **nanometers** on CI)
before trusting the overlay; a >2mm mismatch fails loudly.
The overlay and a JSON/CSV report compare each of the 12 reset contact
positions with the **nearest original STL triangle surface**.

CI result (Linux, original zero-pose contact pairs, mean over 4 contacts):

| Pair | Contact to base-side STL surface | Contact to shoulder STL surface | MuJoCo signed contact distance |
| --- | ---: | ---: | ---: |
| geom0-6 | 9.19mm | 9.58mm | -26.39mm |
| geom1-6 | **15.20mm** | **0.12mm** | -27.86mm |
| geom2-6 | 0.63mm | 4.85mm | -22.39mm |

Interpretation: e.g. geom1-6 contact locations are close to the source
shoulder surface yet far from the source base surface. This is
**consistent with** hull-induced phantom contact, but **not definitive**
because MuJoCo contact point coordinates may not lie on the nearest
surface and unsigned source distances cannot establish real
interpenetration. A real mesh-vs-mesh overlap test or geometrically
fitted collision decompositions are still needed before claiming
that the pair is a false contact or adopting a fix.

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
python -m pip install "trimesh>=4,<5" scipy matplotlib
OVERLAY_OUT="outputs/so101-world-overlay-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/overlay_stl_mujoco_contacts.py --out "$OVERLAY_OUT"
open "$OVERLAY_OUT/world_aligned_stl_contacts.png"
cat "$OVERLAY_OUT/contact_to_stl_surface.csv"
```

The diagnostic does not alter the URDF, dynamics, collision filters or
hardware; SAC remains paused until a physically defensible collision
model and stable PD tracking are established. Avoid another unconstrained
parameter sweep; review the new image and the geom1-6 source-surface
difference first.

### Step A surface overlay implementation (2026-11-08)

The new `scripts/so101_sim/step_a_surface_overlay.py` completes the
**implemented visualization/reporting portion** of the collision-model
improvement plan, without changing the URDF or simulation collision flags.

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
STEP_A_OUT="outputs/so101-surface-stepA-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/step_a_surface_overlay.py --out "$STEP_A_OUT"
open "$STEP_A_OUT/stl_vs_convex_surface_overlay.png"
open "$STEP_A_OUT/geom1_6_contact_closeup.png"
cat "$STEP_A_OUT/surface_gap_summary.csv"
```

Outputs: two 3D PNGs displaying source STL **triangle faces** and
convex-hull **triangle faces**, with all 12 MuJoCo zero-pose contact
points in a common world coordinate frame; `surface_gap_summary.csv`
and `surface_overlay_summary.json` with each contact's unsigned
distance to both source and hull surfaces, and inside/outside classification
against the convex hull planes. Two camera angles per PNG.

On Linux CI the source-STL surface distances reproduced the previous
findings. Mean distances, in mm:

| pair | raw STL: base/shoulder | convex hull surface: base/shoulder |
| --- | --- | --- |
| 0–6 | 9.19 / 9.58 | 2.42 / 11.90 |
| 1–6 | 15.20 / 0.12 | 13.92 / 8.92 |
| 2–6 | 0.63 / 4.85 | 4.98 / 11.04 |

**Interpretation guard:** a MuJoCo contact point can be **inside** a
convex hull and far from its surface. Hull surface distances are
unsigned; neither these distances nor the inside-hull flag proves
physical source-STL penetration or a safe running clearance.
The convex hull is reconstructed from the aligned source vertices;
exact MuJoCo internal contact geometry can differ. The images draw
the original triangles with a stronger translucency than the hull
triangles, but use deterministic face sampling to bound plotting time.
No signed source-mesh intersection test has been performed.

Step A's overlap-drawing and data export are implemented and CI tested.
Treat the physical interference classification as unresolved until
source mesh triangle intersections/contained volumes are checked.
**Do not treat this as authorization to modify collision checks or
restart SAC.**

### Step A final: original STL solid intersections vs convex-hull collisions

Offline, source STL solids were transformed to the same MuJoCo world
frame and evaluated using `trimesh.boolean.intersection` with
`manifold3d` for *volumetric solid* overlap. The method was unit-tested
using synthetic overlapping/non-overlapping cubes; it compares true
STL solids against their convex envelopes, rather than interpreting
unsigned contact-point-to-surface distances as clearance.

| initial zero-pose pair | original STL solid intersection (mm³) | convex hull solid intersection (mm³) |
| --- | ---: | ---: |
| geom0–geom6 | 0 | 14,631.04 |
| geom1–geom6 | 0 | 20,099.55 |
| geom2–geom6 | 1.64 | 17,528.23 |

On GitHub Actions / Linux CI (2026-11-08) the first two source-solid
intersections were empty, while all three convex-hull pairs had large
overlap. Therefore convexification contributes substantial *artificial
occupied volume* in the initial pose. The geom2–geom6 source STL
intersection was tiny (1.64mm³), though it is not exactly zero.
Surface-only touching (zero volume), the physical calibration, and
dynamic joint-range clearance remain unverified.

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
python -m pip install "manifold3d>=3,<4" "trimesh>=4,<5"
INTERSECT_OUT="outputs/so101-stl-intersection-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/audit_source_mesh_intersection.py \
  --out "$INTERSECT_OUT" --export-intersection-mesh
cat "$INTERSECT_OUT/source_mesh_intersection.csv"
```

This closes the **zero-pose volumetric collision audit** of Step A.
Next (Step B): replace overlarge single-convex collision representation
with geometry that preserves the known base-shoulder recess and
has been checked against the source STL; do not blindly turn off
self-collision. Test initial and a small set of arm poses, then compare
PD tracking before restarting SAC. Original URDF and normal training
model are untouched.

### Step B prototype (2026-11-08): one split-convex diagnostic model

A minimal, **disposable** collision prototype was created to exit the
repetitive diagnostic loop. See
`scripts/so101_sim/prototype_split_convex_collision.py`.

- Split the original STL *solids* for
  `base_motor_holder_so101_v1.stl`, `base_so101_v2.stl`,
  and `rotation_pitch_so101_v1.stl` by a 2×2×2 grid, convexifying
  each occupied volume chunk: **8 + 8 + 4 = 20** convex pieces.
- Produce a temporary generated URDF plus STL chunks under `outputs/`.
  Original source URDF/STL, joint transforms/inertias and visual meshes
  remain unchanged. The script only uses MuJoCo PD-oracle evaluation,
  not training or hardware.
- A degenerate clipped piece initially raised a test error; it is
  now skipped, and subsequent CI passed.

**Linux GitHub Actions (seed 101, episode 0, 150 control steps):**

| Metric | Original source URDF | Split-convex diagnostic URDF |
| --- | ---: | ---: |
| Initial MuJoCo contact points | 12 | **0** |
| Final TCP proxy distance | 304.68 mm | 124.77 mm |
| Mean per-joint torque saturation | 94.4% | 76.2% |
| Fraction of physics substeps with contacts | 100% | 97.3% |
| Reach success (<25 mm) | False | False |

**Decision:** Initial phantom contacts are reduced and final distance
improves, but contacts recur during motion, torque saturation remains
substantial, and the prototype fails Reach. Do NOT adopt as the
production collision model or restart SAC. These are CI/Linux
results; local Mac may differ.

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
PROXY_OUT="outputs/so101-split-convex-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/prototype_split_convex_collision.py --out "$PROXY_OUT"
cat "$PROXY_OUT/pd_comparison.csv"
```

To keep the learning project moving, **stop here rather than tuning grid
splits**. Only revisit physical collision realism when adding contact-rich
Pick/Place or when an experiment specifically requires validated self
collisions. For simpler Reach learning, consider a clearly labeled
collision-simplified **simulation-only** environment with collision
limitations recorded; never equate it to a safe Sim-to-Real model.

### Learning reset: minimal kinematic Reach baseline (2026-11-08)

To avoid unbounded collision/PD troubleshooting, there is now a separate,
short learning exercise in
`scripts/so101_sim/learn_reach_kinematic.py`.
It keeps the SO-101 URDF and MuJoCo FK, but **does not perform physics
integration**. Instead, a bounded 5-joint action directly updates joint
positions before calling `mj_forward`. There are no contact forces,
torques, PD gains, actuators, or physical hardware commands.

**Exercise:**
1. Reset to the same start and a nontrivial target TCP produced from FK.
2. Run a zero-action baseline, which should not reach the target.
3. Run a known-joint oracle: it uses the hidden joint configuration used
   to generate the FK target, so it should reach successfully. This is a
   debugging reference **not a learned policy or IK solution**.
4. Compare success (25mm tolerance) and final TCP distance from 3
   deterministic episodes; export CSV and JSON.

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
LEARN_OUT="outputs/so101-kinematic-reach-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/learn_reach_kinematic.py --out "$LEARN_OUT"
cat "$LEARN_OUT/reach_baselines.csv"
```

**Interpretation:** This is a *coordinate-and-action* lesson, not
confirmation that the previous physical PD controller is fixed.
Do not train physical SAC or use actions on the SO-101 robot based
on this benchmark alone. If the exercise passes, the next useful
learning step is an observation-based simple Reach policy before
any accurate collision/PD/Sim-to-Real modeling.

### Jacobian DLS kinematic Reach (2026-11-08)

The minimal `learn_reach_kinematic.py` exercise now compares **three**
policies with the same FK-generated TCP goals: zero action,
the hidden-known-joint oracle, and **damped least-squares Jacobian control**.

The Jacobian method uses `mj_jacBody`, current joint state and Cartesian
target error, **not the hidden target joint angles**:
`dq = J.T @ solve(J@J.T + lambda**2 I, gain * target_xyz_error)`.
It converts `dq` into bounded actions. This is model-based analytic
control, **not SAC or learned RL**.

Linux CI, seed101, 3 episodes (25mm tolerance):

| Episode | Initial (mm) | Jacobian DLS final (mm) | Steps | Success |
| --- | ---: | ---: | ---: | --- |
| 0 | 80.633 | 17.115 | 6 | True |
| 1 | 78.053 | 20.583 | 4 | True |
| 2 | 82.485 | 21.461 | 4 | True |

DLS achieves **3/3 Reach** (mean final 19.72mm).
Original zero-action remains 0/3 and known-joint oracle 3/3.
These are small deterministic educational examples only;
there is no physics stepping, self collision evaluation, PD success
or safe-to-deploy real control in this benchmark.

```bash
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
REACH_OUT="outputs/so101-jacobian-reach-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/learn_reach_kinematic.py --out "$REACH_OUT"
cat "$REACH_OUT/reach_baselines.csv"
```

Next learning step: implement an observation-driven learner in the *same*
kinematic 5D-action/9D-observation task, then compare to Jacobian DLS.
Avoid reopening mesh/PD investigations unless an experiment requires them.

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
