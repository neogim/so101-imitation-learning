# SO-101 Wrist camera in MuJoCo: optional RGB + collision prototype

Status: **implemented as an opt-in prototype (2026-11-12), NOT calibrated or validated on local MuJoCo yet**.

## Intent and scope

The photos show a 3D-printed bent white bracket, a square USB PCB with four mounting screws, a protruding lens and a cable. The rig appears fixed to the wrist-roll-following gripper assembly; it therefore attaches to `gripper_link` (J5 follower), rather than the fixed world.

Existing asset candidates:

- `data/urdf/so101/assets/wrist_camera_mount_so101_v1.stl` — measured STL bbox ~66.1 × 42.6 × 35.0 in STL coordinates, apparently **millimetres**.
- `data/urdf/so101/assets/wrist_camera_so101_v1.stl` — measured bbox ~0.032 × 0.032 × 0.0329 in STL coordinates, apparently **metres**.

**Do not import these files naively as the same unit or assume that their CAD origins match the photographed mounting pose.** The initial implementation uses transparent-to-edit proxy geometry instead of claiming an exact CAD match. Source STL/URDF files are unchanged.

## Files

- `scripts/so101_sim/wrist_camera_sim.py`: fixed body + approximated PCB, lens, L bracket; 4 named collidable primitives; fixed optical camera sensor `r2s_wrist_rgb`; provisional 25 g rig mass/inertia; optional enabling of static workcell contacts; wrist-only contact extraction.
- `scripts/so101_sim/render_wrist_camera_rollout.py`: offline rendering for a saved J5 SAC CSV trajectory, outputs RGB PNG sequence / optional MP4, contact CSV and full JSON log. Modes: `kinematic` (recorded qpos + mj_forward) or `physics` (bounded PD + mj_step; exploratory only).
- `scripts/so101_sim/test_wrist_camera_sim.py`: pure validation and MuJoCo compile/follow-joint tests.
- The existing calibrated scene builder and saved scene factory accept optional wrist config, but all prior callers default to `wrist_camera=False`.

## macOS test / render commands

From the repository root:

```bash
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation

python -m unittest discover -s scripts/so101_sim -p 'test_wrist_camera_sim.py' -v

# Locate existing output created by the earlier J5 ablation experiment:
find outputs -type f -name rollout.csv | grep -Ei 'j5|ablation'
ROLL_OUT="$(find outputs -type f -name rollout.csv | grep -Ei 'j5|ablation' | sort | tail -n 1)"
test -n "$ROLL_OUT" && test -f "$ROLL_OUT" || { echo 'J5 rollout.csv not found' >&2; exit 1; }

OUT="outputs/so101-wrist-camera-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/render_wrist_camera_rollout.py \
  --rollout "$ROLL_OUT" --policy normal --stride 4 \
  --mode kinematic --save-video --out "$OUT"

open "$OUT/wrist_camera.mp4"
open "$OUT/wrist_frames/wrist_0000.png"
cat "$OUT/summary.json"
head -n 10 "$OUT/contact_frames.csv"
```

Notes: `ffmpeg` is needed for `--save-video`. The saved Episode 4 object-layout, cup-pose, calibration files and actual LeRobot episode dataset must be in their established local locations. Do not assume these are committed to GitHub. This is not a fresh SAC inference: the video is an offline playback of a saved trajectory.

**Optional experimental physics-only follow-up** (may be unstable due to known base/shoulder mesh contacts and unknown real rig mass/inertia):

```bash
OUT_PHYS="outputs/so101-wrist-physics-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/render_wrist_camera_rollout.py \
  --rollout "$ROLL_OUT" --policy normal --stride 8 \
  --mode physics --save-video --out "$OUT_PHYS"
open "$OUT_PHYS/wrist_camera.mp4"
cat "$OUT_PHYS/summary.json"
```

## What it does NOT solve yet

- Collision geoms are simplified cuboids/cylinder, not checked against the photographed mount STL or optical calibration. The given attachment pose `(0,-0.014,0.037)` metres and FOV 75° are **hypotheses**, editable with `--wrist-mount-pos`, `--wrist-mount-rpy-deg`, and `--wrist-fovy`.
- `--mode kinematic` still teleports robot joints from the logged rollout. Contact detection cannot stop the motion.
- `--mode physics` integrates dynamic robot contact response, but **Muhi/cup are currently fixed**; they will not physically tumble/fall. This mode does not preserve a trained SAC's action semantics and is not a model comparison.
- The original URDF's base/shoulder convex-hull self-contact problem and recorded out-of-limit home pose are unresolved. No promise of collision-free or stable playback.
- USB cable routing/deformation, realistic lens calibration, and correct image orientation/crop/color remain to be verified against true Wrist video.
- ACT checkpoint inference, six-joint action mapping, grasp and place are **not implemented** by this change. They require a separate phase and model-specific verification.
- MuJoCo and LeRobot dataset were not available in the execution runtime used to author this change. **Python/MuJoCo runtime tests and actual rendered imagery are not confirmed.** Run above commands on Mac; fix any errors found before asserting completed validation.

## Acceptance checkpoints

1. Synthetic MuJoCo test compiles and camera changes world pose when J5 rotates.
2. Scene renders `wrist_camera.mp4` with image axes correctly oriented toward actual task.
3. In the contact CSV, deliberate rig-table or rig-cup geometry overlaps produce named wrist contact pairs (initial unintended overlaps must be diagnosed, not ignored).
4. On physics mode, track joint-error and contact dynamics; compare with and without the camera, then measure actual mass/pose if needed.
5. Do not connect the robot to these experimental scripts.

## 2026-11-12: user-run review and synchronized Side/Wrist fix

The user ran the first wrist-only prototype and supplied `so101-wrist-camera-20261112-220725.zip`. Observed: 48 rendered 320×240 frames, the wrist view looked sideways / toward tabletop undersides and did not consistently include the end effector. The approximate camera rig looked detached/floating; this is a **real failure of placement calibration**, not evidence of satisfactory assembly. Collision log: 48/48 frames with wrist contacts and negative distances. Unnamed opposite geoms require body-ID classification; many appear to be robot self-contacts. Do NOT call 48/48 a real object collision rate.

Changes on this branch:
- The provisional wrist optical axis is now pitched toward the gripper/tip rather than pointing approximately sideways; this is a visual hypothesis, not measured extrinsics.
- The video script captures synchronized **Side**, **Wrist**, and **Side/Wrist comparison** views from identical simulation states.
- Contact JSON now includes raw geom IDs, body names and categories (robot_self, table, object, world), even if the imported URDF geoms are unnamed.
- Physical bracket/PCB/lens geometry is still a simplified proxy. **No claim is made that a newly added shape is physically attached at the correct pose.** Use Side view to identify mounting offset and refine the proxy before assessing safety.

After the usual `conda activate so101-real2sim` and `git pull --ff-only`:

```bash
python -m unittest discover -s scripts/so101_sim -p 'test_wrist_camera_sim.py' -v
ROLL_OUT="$(find outputs -type f -name rollout.csv | grep -Ei 'j5|ablation' | sort | tail -n 1)"
test -f "$ROLL_OUT" || { echo 'Missing J5 rollout.csv' >&2; exit 1; }
OUT="outputs/so101-wrist-dualview-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/render_wrist_camera_rollout.py \
  --rollout "$ROLL_OUT" --policy normal --mode kinematic \
  --stride 4 --save-video --out "$OUT"
open "$OUT/side_wrist_comparison.mp4"
open "$OUT/side_camera.mp4"
open "$OUT/wrist_camera.mp4"
cat "$OUT/summary.json"
```

Further calibration needs a correct actual mount-to-`gripper_link` translation/rotation or visual fitting with verified STL coordinate systems. Do not tune collision margins to hide incorrect alignment. Real-object physics, collision-safe PD and ACT are not yet validated.

## 2026-11-12: corrected J5–J6 jaw-root mount (photograph-based revision)

**Correction to previous explanation:** The `gripper_link` selection was already correct. URDF: `wrist_roll` (J5) connects `wrist_link` to `gripper_link`; `gripper` (J6) connects `gripper_link` to `moving_jaw_so101_v1_link`. The issue was the **wrong local offset, disconnected visual blocks and optical direction**, not a J4 parent.

Four additional photographs show the bracket's **foot attached to the common gripper/jaw-root base**, with a white bent riser supporting a 32 mm PCB and lens looking along the jaws. Camera must not be parented to a moving finger (J6).

New provisional `gripper_link` local coordinates (metres):
- Bracket-foot anchor: `(0, 0, -0.048)`; the prior `(0, -0.014, +0.037)` raised the assembly near the upstream wrist.
- Connected foot, stem, top and PCB; lens extends from PCB in local **-Z**, and sensor at `rig z=+0.032` points mostly in gripper **-Z** (with slight tilt). This aims to include a modest amount of jaw tips at the image edge.
- Top of the bracket and bottom of the PCB proxies overlap slightly rather than float. Additional collision primitive is `r2s_wrist_bracket_foot_collision`.
- Mount position passed from the existing scene builder, standalone renderer and unit tests now shares this default. The script retains `--wrist-mount-pos` / `--wrist-mount-rpy-deg` overrides.
- Side/Wrist synchronized videos and contact-by-body/category log remain enabled. Source URDF/STL and original SAC remain untouched.

**Validation warning**: This is a photo-based starting hypothesis, not a calibrated CAD placement. Geometry is primitive proxies, not the mixed-unit STL files. Neither actual 3D rendering nor collision clearance after this patch has been verified remotely. Confirm in Side video that the bracket foot touches the J5/J6 jaw root and in Wrist video that a small portion of the end effector appears. Do not hide unexpected self-contacts by globally disabling collisions. If the attachment remains misaligned, determine the true gripper-local mount transform with the CAD or visual landmarks rather than blindly shifting positions.

For Mac test and dual video generation, use the earlier commands in this document, beginning with `conda activate so101-real2sim`.

## 2026-11-12 Mac error follow-up: duplicate geom name fixed

The first J5–J6 patch introduced a coding error: `r2s_wrist_bracket_foot_collision` was registered **twice**, because a newly inserted foot element changed the `COLLISION_NAMES` indices and the original stem was still assigned index 0. The resulting `ValueError: repeated name 'r2s_wrist_bracket_foot_collision' in geom` occurred during `spec` construction, before video generation. The subsequent 'video file does not exist' messages were consequences, NOT missing `rollout.csv`.

Correction: five add_geom calls now use unique indices `[1,0,2,3,4]` of `COLLISION_NAMES`, covering foot, stem, top, PCB and lens exactly once. `test_no_duplicate_collision_names_in_install_source` statically checks this mapping, while MuJoCo compile tests check actual construction.

Use these **zsh-friendly no-comment lines**. Open videos only after successful script exit; the `set -e` shell option is not required:

```bash
conda activate so101-real2sim
git switch feature/so101-sim-data-augmentation
git pull --ff-only origin feature/so101-sim-data-augmentation
python -m unittest discover -s scripts/so101_sim -p 'test_wrist_camera_sim.py' -v
ROLL_OUT="$(find outputs -type f -name rollout.csv | grep -Ei 'j5|ablation' | sort | tail -n 1)"
if [ ! -f "$ROLL_OUT" ]; then
  echo "J5 rollout.csv not found under outputs"; 
else
  OUT="outputs/so101-wrist-j5j6-retest-$(date +%Y%m%d-%H%M%S)"
  if python scripts/so101_sim/render_wrist_camera_rollout.py --rollout "$ROLL_OUT" --policy normal --mode kinematic --stride 4 --save-video --out "$OUT"; then
    open "$OUT/side_wrist_comparison.mp4"
    open "$OUT/side_camera.mp4"
    open "$OUT/wrist_camera.mp4"
    cat "$OUT/summary.json"
  else
    echo "Rendering failed; no videos opened."
  fi
fi
```

The `zsh: command not found: #` lines come from pasting bare `#` comment lines into an interactive zsh session with `interactivecomments` off. They are unrelated to the MuJoCo exception. No personal Mac username or absolute home paths have been recorded in this documentation.

Verification level: Github code review and static name-index inspection completed; full MuJoCo tests and resulting videos still require a Mac run.
