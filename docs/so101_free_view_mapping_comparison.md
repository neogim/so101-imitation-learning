# SO-101 joint mapping: fixed free-view comparison (offline)

The purpose of this tool is to **separate joint pose differences from uncalibrated Side/Wrist cameras**. It renders two hypotheses with precisely the same virtual free-camera viewpoint for each frame and creates a 2-panel PNG.

**Not a physical mapping proof.** The simulated free cameras are NOT aligned with the real Side/Wrist cameras. Robot and object positions are approximate; J2/J3 sign **and zero offsets** remain unverified. Comparing the render to an uncalibrated real camera is only qualitative.

## Hypotheses

- \`plus\`: J1..J5 = \`(+,+,+,+,+)\`.
- \`j2j3_minus\`: J1..J5 = \`(+,-,-,+,+)\`.
- The joint zero offset from \`--config\` is shared by both hypotheses; default offset is 0 for all joints.
- This **overrides the sign from the config only within this tool**, without modifying the config, raw dataset, URDF or hardware.
- The source is \`observation.state\` (not commanded \`action\`).

## Mac command

From the repository root and **feature/mujoco-real-to-sim**:

~~~bash
conda activate so101-real2sim
git status
git pull --ff-only
python -m unittest discover -s scripts/so101_sim -p 'test_*.py' -v

DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
OUT="outputs/real2sim/freeview-$(date +%Y%m%d-%H%M%S)"

python scripts/so101_sim/compare_free_views.py \
  --dataset-root "$DATASET" --episode 0 \
  --frames 0 80 160 240 320 \
  --views oblique opposite overhead \
  --output "$OUT"

open "$OUT"
~~~

Expected: \`frame_000000_oblique_side_by_side.png\` etc., with **left: +++++**, **right: +--++**. For each selected frame, the tool saves 3 two-panel PNGs from the SAME virtual viewpoint/scale; \`report.json\` records actual angles and URDF limit warnings for each hypothesis.

The output directory must be new and not inside the raw dataset. No robot serial/camera code is imported. Raw videos are **not** decoded by this program; real-world images can be compared separately, noting the viewpoints differ.

## If images fail

- \`ModuleNotFoundError: mujoco\`: activate \`so101-real2sim\` and check environment.
- \`Requested frame exceeds episode length\`: choose valid local frame indices.
- \`Refusing to overwrite\`: generate a new OUT path.
- Rendering GL context issues on Mac: first verify \`python -c 'import mujoco; print(mujoco.__version__)'\` and the existing \`--render\` runner. Do not change ACT environment.
- If both hypotheses look inconsistent: check joint zero and physical axis mapping; neither is necessarily correct.
- If \`shoulder_lift\` still exceeds ±100°: neither sign reversal alone nor a free camera proves a correct zero-offset. Do not expand physical limits to make the diagnostic pass.
- Contact warnings from detailed URDF geometry are not proof of real collisions.

## Interpretation and next step

1. Inspect corresponding frame and compare the 3D **fold direction / link relative orientation**, not pixel alignment against the real Side view.
2. Compare frame 0, 80, 160, 240, 320 and note which hypothesis is mechanically plausible over a range of poses.
3. Use existing static photos/videos and any independently known joint poses to check physical references; do not claim calibration from visual resemblance alone.
4. If necessary, obtain independent static-angle references and use the earlier \`docs/codex_so101_joint_mapping_phase2_protocol.md\` before claiming a physically verified sign and zero.
5. Only after resolving the arm pose, calibrate fixed Side camera, tabletop/objects, then wrist camera relative to its actual mounting link.

## Research provenance

- [PR #8](https://github.com/neogim/so101-imitation-learning/pull/8)
- [Physical joint mapping protocol](../codex_so101_joint_mapping_phase2_protocol.md)
- [Existing Real-to-Sim MVP runbook](so101_real_to_sim_mvp.md)
