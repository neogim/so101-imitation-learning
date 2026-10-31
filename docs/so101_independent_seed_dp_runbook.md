# SO-101 offline DP2 reproducibility — run locally after git pull

This stage requires **no Codex**. The scripts were committed to the feature branch. No experiment has been run by the assistant on the Mac's private LeRobot dataset. Execute locally and share **only a short sanitized summary**, not private calibration content.

## Purpose and safeguards
Repeat the 31-frame, ±3 mm, 4-hypothesis study with **frame-independent IK starting seeds**. No `previous_witness` seed and no greedy-derived candidate-generation history. Compare new DP2 quality gates to the previously reported finite candidate results (historical ++ and --: DP2 6/6 each). Do **not** construe this as physical sign/offset validation or permission for ACT augmentation/hardware replay.

No robot connection. No motor power, torque changes, calibration changes, URDF edits, or dataset writes. All results go to **new** gitignored output directories. **Do not add outputs/, calibration JSON, or source datasets to Git.**

## A. Pull and test

```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python -m unittest discover -s scripts/so101_capture -p 'test_generate_independent_ik_candidates_offline.py' -v
python -m py_compile scripts/so101_capture/generate_independent_ik_candidates_offline.py scripts/so101_capture/select_ik_trajectory_dp_offline.py
```

If the environment has multiple Python interpreters use the interpreter from prior SO-101 IK runs. Confirm `placo`, `numpy`, `scipy`, `pyarrow` and all existing local dataset paths are available.

## B. Generate candidates without greedy history

Ensure `FOLLOWER_CALIBRATION_JSON` points to the **existing private local** calibration cache. It is used only by the DP comparator, not by the new generator. Never print or commit the cache.

```bash
python scripts/so101_capture/generate_independent_ik_candidates_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --split-dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --output outputs/so101-ik-independent-seeds-565-595-20261028 \
  --episode 0 --frame-start 565 --frame-end 595 \
  --offset-mm 3 --feasible-residual-mm 2 --max-nfev 100
```

If the output directory already exists, do not overwrite: specify a new unique path and reuse that path in step C.

## C. Run the existing greedy/DP1/DP2 comparison on the new pool

```bash
python scripts/so101_capture/select_ik_trajectory_dp_offline.py \
  --candidates outputs/so101-ik-independent-seeds-565-595-20261028/per_seed.json \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --output outputs/so101-ik-independent-dp-565-595-20261028 \
  --episode 0
```

The selector is currently hardcoded to 565–595; **do not run 648 frames with this version**.

## D. Extract ONLY aggregate metrics to paste in ChatGPT

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path("outputs/so101-ik-independent-dp-565-595-20261028/summary.json")
d=json.loads(p.read_text())
for name, by_direction in d["results"].items():
    print("\n",name)
    for method in ("greedy","dp_first_order","dp_second_order"):
        rows=[x["methods"][method] for x in by_direction.values()]
        passing=sum(x["quality_pass"] for x in rows)
        coverage=sum(x["coverage"] for x in rows)
        print(f"  {method}: {passing}/6 pass, {coverage}/186 candidate-coverage")
        for kind in ("deviation_peak","jump_peak","velocity_peak","acceleration_peak"):
            vals=[m[kind]["value"] for r in rows for m in r["segment_metrics"] if m[kind] is not None]
            print("   ",kind, max(vals) if vals else "N/A")
PY
```

Paste the compact result and any error message into ChatGPT. No need to upload `per_seed.json`. The assistant can analyze, update Sheets/Docs, and provide **step 2 (31→648 gate and reusable script changes)** only after independently checking step 1.

## Acceptance and next stage
For the new independent seed pool require **full 186/186 coverage**, position residual ≤2 mm, deviation ≤15°, jump ≤5°, rate ≤100°/s, acceleration ≤2000°/s², tool-angle difference ≤15° in the DP2 result for the hypothesis under review. The thresholds are study diagnostics, not certified physical safety limits. A 6/6 outcome is conditional numerical evidence only. If all candidate sets do not pass, do not silently lower thresholds or expand to 648; review missing candidates / solver branches.

## Known limitations
Candidate starts are a finite set of recorded pose and shoulder/elbow perturbations only; there is no global reachability guarantee, collision check, orientation constraint, physical camera registration or physical angle calibration. Hypothesis-specific FK defines different targets for each mapping.
