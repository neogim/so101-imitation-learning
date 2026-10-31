# SO-101 offline calibration audit handoff (Codex → GitHub → ChatGPT)

## Objective
Investigate the LeRobot Follower calibration vs. the SO-101 URDF, especially shoulder_lift (URDF -100°, observed -103.824°) and elbow_flex (URDF +96.830°, observed +97.451°, 612 action samples beyond inferred calibration maximum). Establish exactly what is proven and what remains unverified. **Do not command, initialize, power, calibrate, or move motors.**

## Local execution instructions for Codex
1. Stay on `feature/cartesian-action-offline-export`. Inspect the current code and dataset metadata; do not guess filenames or calibration keys.
2. Read-only locate the **Follower** calibration in local config/cache. If inspecting LeRobot internals, check version (previously 0.6.2), `MotorNormMode.DEGREES`, actual conversion implementation, the homing/drive-mode meaning, and whether `range_min/max` reflect hard mechanical limits or measured operational calibration.
3. Compare shoulder_lift and elbow_flex only (and summarize the other three) across: Follower calibration JSON, URDF <joint axis/origin/limit>, LeRobot conversion implementation, and recorded `action` vs `observation.state`. For angles, distinguish joint-space degrees, normalized degrees, raw encoder ticks, and motor degrees.
4. Verify what exactly was counted in `calibration_urdf_audit.json`: 30,846 Action frames; shoulder_lift 8,203 beyond URDF; elbow_flex 9,204 beyond URDF and 612 beyond estimated LeRobot calibration range. Those figures are previous outputs, NOT authoritative if the underlying measurement convention differs. Reproduce counts with explicit tolerances (0°, 0.1°, 0.2°, 0.5°), and include which episode/frame indices exceed the calibrated elbow maximum; summarize distributions and Action-State differences. Do not rewrite source datasets.
5. If a missing calibration file prevents certainty, mark it as unresolved and document exactly which read-only path/check is required. Never silently infer active firmware calibration.
6. Add automated tests of any new analysis code. Run only non-hardware tests, document commands and status.
7. Create a **small, sanitised** report folder under `reports/so101_calibration_audit/`. Commit only:
   - `README.md`: assumptions, evidence, calibrated vs URDF ranges, threshold counts, conclusions, caveats, exact reproduction commands.
   - `summary.json`: aggregated numerical findings only, no machine usernames/absolute paths/private metadata.
   - `peaks.csv` optional: no longer than ~100 rows, containing episode/frame/joint/deviation only.
   Never commit calibration JSON contents (private equipment state), secrets, raw logs, model weights, datasets, media, or complete per-frame exports. All `data/`, `outputs/`, and `calibration/` are intentionally gitignored.
8. Commit and `git push origin feature/cartesian-action-offline-export` **only** after reviewing `git diff --cached --stat` and `git diff --cached` and verifying the staged paths have no secrets/personal paths. Avoid force push and do not touch main. If the push fails, report exact error.
9. Return only: commit SHA, pushed branch, report file relative paths, concise findings, checks run, and blockers.

## Reproducibility note
The existing `scripts/so101_capture/audit_lerobot_urdf_calibration.py` audits URDF bounds vs LeRobot dataset columns and accepts optional `--calibration`, but it does **not** prove calibration-to-URDF sign/zero transformation. Extend or add a new read-only tool if warranted. Avoid any assumption that hypothetical ±3° joint limits can be used on real hardware.

## ChatGPT follow-up
User pastes only the commit SHA (or the GitHub commit link) into the chat. ChatGPT can fetch `reports/so101_calibration_audit/README.md` and `summary.json` directly from GitHub, review evidence, and update the linked experiment Google Sheet / report without file uploads.
