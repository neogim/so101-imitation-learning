# iPhone → OBS Side camera checkerboard dataset (2026-11-06)

**Status: selected source images prepared, binary files pending local Git push.** The archive provided in the ChatGPT conversation contains the 16 selected cropped PNG inputs plus this dataset's manifest and more detailed README. Do not claim these images are committed until the follow-up local upload succeeds.

## Selection
- First ZIP: 10 accepted out of 15; second ZIP: 6 accepted out of 6
- 16/21 screenshots passed OpenCV \`findChessboardCornersSB\` for a **7×5 internal-corner** pattern
- Original 1280×480 screenshots were cropped to the central **640×480** RGB region (horizontal pixels x=320..959), removing black letterboxing. Source bytes were not altered; selected 640×480 PNG crops were losslessly re-encoded.
- Rejected timestamps: 00:08:44, 00:18:23, 00:22:23, 00:22:45, 00:23:24
- Accepted timestamps: 00:09:39, 00:17:35, 00:18:02, 00:18:44, 00:19:09, 00:19:59, 00:20:26, 00:20:46, 00:21:28, 00:21:54, 00:29:10, 00:29:27, 00:29:43, 00:29:58, 00:30:18, 00:30:40

## Local import procedure (run from repo root)

Download \`so101_side_obs_chessboard_20261106_selected.zip\` from the relevant ChatGPT conversation into \`~/Downloads/\`:

~~~bash
git switch feature/mujoco-real-to-sim
git pull --ff-only

unzip -n "$HOME/Downloads/so101_side_obs_chessboard_20261106_selected.zip" -d .

# Only the 16 selected images, manifest and README should be staged.
find assets/calibration/side_obs_20261106/images -type f -name '*.png' | wc -l
git add assets/calibration/side_obs_20261106
git status --short
git commit -m "Track 16 validated-detection Side OBS checkerboard samples"
git push origin feature/mujoco-real-to-sim
~~~

If an upstream working tree is dirty or if any generated output was staged unexpectedly, inspect carefully before committing. Do not check in raw LeRobot datasets, passwords, or any unrelated files. **Inspect room/desk background for privacy before pushing a public repository.**

## Interpretation
The provisional 16-image OpenCV fit yielded inlier chessboard detection on all 16 with **RMS ≈ 0.58 px** on 640×480 crops. However, high-order distortion was numerically unstable; **this is not an accepted camera calibration**. Confirm board flatness, camera image transformations through OBS/LeRobot, and corner coverage before fitting production intrinsics.

**Important:** this dataset documents **intrinsics calibration photographs only**, not SO-101/world-frame extrinsics. The original ACT Side training frames are **224×168**, and they may differ due to OBS/LeRobot crop/resize.
