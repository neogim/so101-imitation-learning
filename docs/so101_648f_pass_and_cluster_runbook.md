# 2026-10-28 648-frame DP2 PASS and optional cluster study

## Confirmed user-provided Mac output (Episode0 f0–647)
Mapping shoulder-_elbow-: DP2 6/6 PASS, all directions 648/648 coverage.
Direction / max-jump deg / max-velocity deg/s / max-acceleration deg/s²:
x+: 3.477861 / 69.557608 / 1413.419523
x-: 2.857137 / 57.143603 / 731.025701
y+: 2.397599 / 47.950892 / 720.997527
y-: 2.385834 / 47.715579 / 717.445035
z+: 2.842888 / 56.858633 / 1087.413711
z-: 2.506438 / 50.128577 / 822.483445
FULL_NUMERICAL_GATE: PASS, as a *single mapping* selection. This is strictly a simulated numerical result, NOT evidence physical motor/URDF sign mapping is correct. Placo still emits self-collision warnings in neutral pose; neither collisions nor torque nor real-world replay verified. Recorded timestamp gaps were accepted; inspect original summary's timestamp_diagnostics for actual dt outliers.

## Optional lightweight IK candidate clustering (120 frames, no new IK)
Goal: test whether original ++ failures correlate with clusters of 5-joint candidate motor angle vectors rather than interpreting broad J5 ranges as branches. Per-frame connected components in 5-dimensional Euclidean joint-angle space at exploratory radius 5deg. A local cluster label is NOT a persistent IK branch across frames. Default method uses no scipy/sklearn clustering.

```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python -m py_compile scripts/so101_capture/cluster_existing_ik_candidates_offline.py

python scripts/so101_capture/cluster_existing_ik_candidates_offline.py \
  --candidates outputs/so101-ik-independent-000-119-stage-v1/per_seed.json \
  --paths outputs/so101-ik-dp-000-119-stage-v2/selected_paths.json \
  --mapping 'shoulder+_elbow+' --start 0 --end 119 \
  --radius-deg 5 \
  --output outputs/so101-ik-dp-000-119-stage-v2/clusters_plus_plus_5deg_v1.json
```

Only share console output and summary count. Interpretation: compare frames 51–99 with others; clusters alone do not establish mode switching without frame-to-frame cluster correspondence. High numbers of clusters could arise from sparse seeds or threshold choice. The research priority is now physical mapping/collision validation after numeric PASS; no hardware commands included or authorized.
