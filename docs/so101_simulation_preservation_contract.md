# SO-101 simulation preservation contract (2026-11-14)

**Purpose**: Prevent feature regressions when splitting MuJoCo debug scripts, implementing Realistic Reach, or connecting Reach/Contact/Grasp/Lift. This is a required handoff checklist, not a claim that all features are validated.

## Baseline to preserve (reference)

Primary reference: `scripts/so101_sim/physical_grasp_table_bench.py`.
Reuse `scripts/so101_sim/wrist_camera_sim.py` instead of re-implementing the mount.

1. The SO-101 wrist camera is attached to `gripper_link` (J5/J6 side), with the photographed bracket model. Maintain **visual parts and collision proxies**, using `ruler_15_35_35` mounting geometry (estimated from ruler/photos, not calibrated).
2. Two outside/side inspection views, plus three wrist inspection views:
   - wrist aimed at the TCP (`r2s_wrist_rgb`)
   - wrist aligned with the physical shelf (`r2s_wrist_rgb_shelf`)
   - *virtual* retracted comparison camera (`r2s_wrist_rgb_retracted`, normally 5mm)
3. **Combined video is mandatory** when `--video` is requested: 1440×780, top row two 480×360 side views at x=240 and 720, lower row three 480×360 wrist views at x=0, 480, 960, with legible view labels.
4. Individual MP4s are **additional**, not replacements: `reach_side.mp4`, `reach_wrist.mp4`, and other individual variants. Keep `reach_combined.mp4` as the main Reach review file. For the original tabletop bench, main file is `table_contact_views.mp4`.
5. PNG start/mid/end snapshots are secondary inspection artifacts, not the video deliverable. Never report three snapshots, a PDF slideshow, or a GIF as equivalent to the requested MP4.
6. Preserve synchronized timestamps/frame counts across views; encode with ffmpeg; record filenames and statuses in `summary.json`. Fail clearly and preserve numeric data if rendering/encoding fails.
7. Preserve realistic independent WORLD-frame Muhi placement. **Never teleport the object to match a gripper TCP**; the original close-contact bench is a narrow diagnostic exception, not a general Reach test.
8. Validate approach direction relative to the robot HOME and obstacle position. For the current uncalibrated toy scene, robot HOME TCP Y≈+0.151m and Muhi Y=−0.090m, so a pregrasp on +Y side uses `--pregrasp-offset-m 0 .075 .015`, not the old `0 -.075 .015` (which put TCP behind Muhi).
9. Successful DLS TCP XYZ position tolerance does **not** establish desired fingertip pose, safe contact, stable grasp, lifting or realistic motor following. Require separate pose/contact verification before declaring Pick success.
    After Reach, compute **fixed-jaw collision geometry against Muhi**, not just TCP-to-waypoint distance. Stop at a controlled, small geometric overlap, reject contact witnesses too far from the TCP (palm/root), and require actual solver-measured fixed-jaw contact force to be sustained before reporting `CONTACT_CONFIRMED`. J6 must stay open in this phase.
10. Keep the wrist rig in collision and clearance checks. An IK route without rig is not proof of collision-free motion with the camera installed.
11. Changes to the scene, cameras or video must be reviewed against these requirements **before** requesting another Mac execution. **Do not silently remove or weaken capabilities** in successor scripts.
12. Results and plots belong in the experiment Notion page; the persistent handoff must include the camera layout, exact file names, prerequisites, and camera installation code reference.

## Required implementation checks before reporting the next Reach result

- `python -m py_compile scripts/so101_sim/realistic_muhi_reach_preflight.py`
- Run a single MuJoCo test with `--render --video --exclude-base-shoulder-contact --auto-safe-home --reach-mode staged --j4-policy prefer-fixed`, on the fixed world-frame Muhi position.
- Inspect `summary.json` for stage success, the **actual home and pregrasp world coordinates**, `video_files`, and any `video_errors`.
- Check `reach_combined.mp4`: top row 2 side views, lower row 3 wrist views; all 5 views clearly visible and actually changing with the robot motion.
- Verify near-side approach and contact geometry separately. If reaching the near-side pregrasp fails, **do not quietly restore the far-side target to make IK pass**; report clearance and joint limits.
- Record progress (and failures) with Japanese axes/legends plus actual linked MP4s in Notion.

This contract is a process safeguard; the new code has not yet been run on the user's Mac.
