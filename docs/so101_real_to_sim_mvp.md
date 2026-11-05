# SO-101 Real-to-Sim / MuJoCo MVP

**Status: initial offline implementation.** Based on safety branch fix/isolate-multicam-episode-encoding at d32d613. No robot control, no camera capture, no original dataset modification, and no demonstrated physical Sim2Real accuracy.

## Principle: Real is the reference

The current physical Side/Wrist camera setup and previously collected two-view LeRobot episodes are preserved. Build the MuJoCo workcell around the existing real setup. Do not move cameras or re-record demos just to configure simulation.

Reuse the existing checked-in SO-101 URDF, Cartesian7 MuJoCo checker, and docs/so101_mujoco_photo_workcell_20261031.md (approximate, not measured).

Stage 1 is **kinematic state replay**, not actuator dynamics: replay stored joint angles using mj_forward, output diagnostics, optionally render virtual Side/Wrist frames and pair them with the existing real MP4 images.

## Environment

Install in a separate environment, NOT the working hardware-capture LeRobot environment.

~~~bash
conda create -n so101-real2sim python=3.11 -y
conda activate so101-real2sim
python -m pip install "mujoco>=3.3,<4" "pyarrow>=16" "pillow>=10" "av>=12"
~~~

Rendering needs a compatible OpenGL context. The PyAV dependency is only used when requesting paired real images.

## Unit tests

~~~bash
python -m unittest scripts/so101_sim/test_real_to_sim.py -v
~~~

GitHub Actions passed Python syntax and 9 tests (6 unit, 3 synthetic-data integration cases including LeRobot v3 shared-shard episode isolation) on 2026-11-05. Real captured dataset replay and Side/Wrist image rendering are not yet validated.

## Offline replay

~~~bash
DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"

OUT="outputs/real2sim/episode0-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/real_to_sim.py \
  --dataset-root "$DATASET" --episode 0 --stride 20 --max-frames 30 \
  --output "$OUT"

# Images: use a NEW output directory; no camera hardware opened.
OUT="outputs/real2sim/episode0-render-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/real_to_sim.py \
  --dataset-root "$DATASET" --episode 0 --stride 20 --max-frames 10 \
  --render --real-images --output "$OUT"
~~~

Expected output: report.json and optional frame_XXXXXX_{side,wrist}_sim.png / _real.png / _comparison.png. Output directory must not exist. Source data is read-only.

Dataset layout (automatically selected by `codebase_version`):
- **LeRobot v3.0:** `meta/info.json` has `data_path=data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`, `video_path=videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`. `meta/episodes/chunk-*/file-*.parquet` contains episode boundaries, shard indices and video timestamp offsets. **Many episodes share a file.** Select by `episode_index`, sort by local `frame_index`, and use per-camera `from_timestamp` for video decoding.
- **Legacy layout:** `data/chunk-*/episode_NNNNNN.parquet` and episode-local MP4.
- State and action must list shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos in order.

**2026-11-05 dataset confirmation:** existing `muhi-side-wrist-success-053-20261025-168x224` has `codebase_version=v3.0`, `total_episodes=53`, `fps=20`, and `data/chunk-000/file-000.parquet`. The initial script expected per-episode files and failed; the new adapter supports v3 metadata. Real local re-test remains pending. Original dataset remains read-only.

On Mac run:
~~~bash
git status
git pull --ff-only
python -m unittest discover -s scripts/so101_sim -p 'test_*.py' -v
DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
OUT="outputs/real2sim/episode0-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/real_to_sim.py --dataset-root "$DATASET" --episode 0 --stride 20 --max-frames 30 --output "$OUT"
python -m json.tool "$OUT/report.json" | head -80
~~~

If there is a failure in the new adapter, inspect metadata read-only:
~~~bash
python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
r = Path("data/so101/muhi-side-wrist-success-053-20261025-168x224")
for p in sorted((r / "meta/episodes").glob("chunk-*/file-*.parquet"))[:1]:
    print(p, pq.read_schema(p))
    print(pq.read_table(p).slice(0, 2).to_pylist())
for p in sorted((r / "data").glob("chunk-*/file-*.parquet"))[:1]:
    print(p, pq.read_schema(p))
PY
~~~


## What is deliberately NOT calibrated

| Element | Missing measurements |
| --- | --- |
| Side camera | Intrinsics, base-to-camera extrinsics, distortion, fovy |
| Wrist camera | Intrinsics, gripper-link-to-camera transform |
| SO-101 | Motor degree sign and offset relative to URDF per joint |
| Gripper | Normalized 0..100 to URDF moving-jaw joint angle; deliberately not mapped |
| Table, Muhi, cup | Precise world poses/dimensions and contact properties |
| Timing | True video/state/action skew; nominal timestamps alone do not capture this |

workcell.example.json is a **visible uncalibrated scaffold**. It is not evidence of a calibrated digital twin. The Muhi and cup are static non-colliding display geometry, not graspable bodies. mj_forward sets instantaneous joint pose; command-state differences in the report are not simulated tracking errors.

## Milestones

- [x] New isolated branch retaining previous safety code
- [x] Read-only dataset adapter and sampled Joint6 state replay
- [x] SO-101 URDF import and static illustrative workcell scaffolding
- [x] Virtual Side and Wrist cameras and optional paired PNG comparison
- [x] Hardware-free tests and synthetic Parquet/URDF integration tests: 9/9 passed in GitHub Actions (2026-11-05)
- [x] Verify MuJoCo URDF loading and kinematic replay in GitHub Actions (Linux)
- [ ] Verify offscreen Side/Wrist rendering in the user's environment
- [x] Identify actual local LeRobot dataset path/layout: v3.0 shared file-000.parquet
- [ ] Verify v3.0 episode decoding against the user's actual 53-episode dataset
- [ ] Measure camera calibration while keeping current physical positions
- [ ] Calibrate joint mapping/URDF and gripper representation
- [ ] Validate per-frame Real/Sim reprojection errors
- [ ] Add movable objects, collision/contact, pick/lift/place success metrics
- [ ] Add actuators and System Identification of real tracking lag
- [ ] Domain Randomization and BC/RL evaluation only after validation

## Safety

No connection to leader/follower, USB serial ports or cameras. No commands to the robot. The 8° tracking-error, 250ms control-gap and communication fault guards are untouched. NEVER use these kinematic simulator poses as physical-motion approval.

## References

- [Notion plan](https://app.notion.com/p/real2sim-sim2real-40d85e16f4dc80608fcbf85d6fa1b93a)
- [MuJoCo MjSpec documentation](https://mujoco.readthedocs.io/en/stable/python.html#model-editing)
