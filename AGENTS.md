# SO-101 Research Agent Instructions

This document is a **project-wide, non-personal** handoff for coding assistants. Do not put names, account IDs, email addresses, device usernames, SSH hosts, IPs, tokens, secrets, absolute home-directory paths, or identifiable workstation details in this file, examples, commits, logs, or PR summaries. Use repository-relative paths and generic placeholders. Do not publish local experiment archives or dataset contents.

## Start every session

1. Read this file, `docs/so101-sim-handoff.md`, and the relevant existing scripts/tests before changing code.
2. Check the **current** GitHub branch, PR, head SHA, and recent changes. Working context as of 2026-11-12: `feature/so101-sim-data-augmentation` and Draft PR #12 (base `main`); this is not a promise that these will remain current.
3. Read the existing Notion SO-101 plan, experiment log, and MuJoCo collision/PD note linked in the handoff document. Reconcile with the current code and latest actual experiment output.
4. Distinguish measured results, implementation status, hypotheses, and proposed next steps. Never say code is tested, a PR is merged, or contact physics is implemented unless verified.

## Local execution contract (macOS)

- Use the existing conda environment before providing any Python execution command:
  ```bash
  conda activate so101-real2sim
  git switch feature/so101-sim-data-augmentation
  git pull --ff-only origin feature/so101-sim-data-augmentation
  ```
- Always show complete **copy-pasteable** macOS commands after GitHub commits, including conda activation, branch update, script invocation, generated output paths, and commands to inspect outputs.
- Prefer relative paths from the repository root. Use unique timestamped output directories. Never overwrite existing output or invent local file locations; if a run requires a locally generated file, show how to locate it or clearly mark the path as an example.
- Explicitly say when local macOS tests cannot be run from the current environment. CI passing and a Mac GUI/video test passing are different claims.

## Development and experiment rules

- Prefer direct GitHub editing, committing to the active feature branch and updating the Draft PR; do not commit directly to `main` or merge without explicit direction.
- Keep the original URDF, STL assets, LeRobot episodes, recordings, and SAC checkpoints unchanged. Do not operate physical robot hardware or send real actuator commands for an offline simulation task.
- Keep kinematic FK / `qpos` + `mj_forward` experiments separate from MuJoCo dynamic stepping / actuator control. Kinematic TCP reach is **not** proof of contact avoidance, grasp, lift, or place.
- For physical simulation, inspect collision geoms and `contype/conaffinity`, free-body and inertia definitions, friction, actuator control, joint limits, and `mj_step`. Avoid deleting all collisions as a permanent fix. Preserve visual shapes independently of collision proxies.
- Prioritize fast focused checks by default. If the user explicitly wants deep verification, thoroughly test hypotheses, instrumentation, visualization, and edge cases; do not dismiss this as over-testing.
- Record numerical configuration, seed, baseline, observations, uncertainty, exact code/commit, and output artifacts. Do not claim Sim2Real safety from MuJoCo results alone.
- Preserve the existing Notion page structure: update the plan for milestones, append evidence to the experiment log, and append collision/PD findings to the dedicated note. Never create redundant pages when an existing log fits.

## Mandatory post-commit response

Every time code is committed, report: (1) branch and commit SHA; (2) what changed; (3) checks actually run and checks not run; (4) **full** Mac copy-paste commands beginning with `conda activate so101-real2sim`, then Git pull, execution and result inspection; (5) outputs and interpretation; (6) whether Notion was updated, with a link.

## Privacy

Never use or echo a real Mac login, `/Users/<person>/...`, repository owner's personal profile details, local network addresses, cloud credentials, or actual user-specific paths. `$HOME` is acceptable when unavoidable; repo-relative paths are preferred. Review diffs for secrets, personal paths, and accidental large binaries before pushing.
