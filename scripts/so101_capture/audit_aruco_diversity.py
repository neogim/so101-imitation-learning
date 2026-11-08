#!/usr/bin/env python3
"""Read-only pairwise diversity audit for ArUco/FK calibration captures.

Check all pairs, not just consecutive poses. Report joint-space near repeats,
per-marker rigid-link FK translational/rotational diversity, and warnings.
No camera/serial access; no source data modifications. FK assumes the stored
LeRobot angles directly correspond to URDF angles (not yet validated).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from audit_fk_aruco import JOINTS, forward_kinematics, parse_urdf, rotation_distance_deg

LINK_BY_MARKER = {0: "gripper_link", 1: "gripper_link", 2: "wrist_link"}


def audit_diversity(urdf, manifest, *, duplicate_deg=3.0,
                    near_translation_mm=5.0, near_rotation_deg=5.0):
    if any(not math.isfinite(x) or x <= 0 for x in
           (duplicate_deg, near_translation_mm, near_rotation_deg)):
        raise ValueError("Thresholds must be positive and finite")
    graph, _ = parse_urdf(urdf)
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if not samples:
        raise ValueError("No extracted poses in manifest")
    angles = []
    tags = []
    ids = []
    transforms = []
    for sample in samples:
        q = {name: float(sample["joints"][name]) for name in JOINTS}
        if not all(math.isfinite(v) for v in q.values()):
            raise ValueError("Non-finite joint angles")
        angles.append([q[name] for name in JOINTS])
        tags.append({"episode": int(sample["episode_index"]),
                     "frame": int(sample["frame_index"])})
        ids.append({int(marker["id"]) for marker in sample["markers"]})
        transforms.append({link: forward_kinematics(graph, q, link)
                           for link in set(LINK_BY_MARKER.values())})
    a = np.asarray(angles, dtype=float)
    pairs = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            d_all = float(np.max(np.abs(a[i] - a[j])))
            d_upstream = float(np.max(np.abs(a[i, :4] - a[j, :4])))
            if d_all < duplicate_deg or d_upstream < duplicate_deg:
                pairs.append({
                    "a": tags[i], "b": tags[j],
                    "max_joint_delta_deg": round(d_all, 4),
                    "max_j1_to_j4_delta_deg": round(d_upstream, 4),
                    "all_five_near_duplicate": d_all < duplicate_deg,
                    "wrist_fk_near_duplicate_candidate": d_upstream < duplicate_deg,
                })
    grouped = {}
    for marker_id, link in LINK_BY_MARKER.items():
        relevant = [i for i, visible in enumerate(ids) if marker_id in visible]
        distances = []
        rotations = []
        near_pairs = []
        for k, i in enumerate(relevant):
            for j in relevant[k+1:]:
                t1, t2 = transforms[i][link], transforms[j][link]
                dist_mm = float(np.linalg.norm(t1[:3, 3] - t2[:3, 3]) * 1000)
                rot_deg = rotation_distance_deg(t1, t2)
                distances.append(dist_mm)
                rotations.append(rot_deg)
                if dist_mm < near_translation_mm and rot_deg < near_rotation_deg:
                    near_pairs.append({
                        "a": tags[i], "b": tags[j],
                        "distance_mm": round(dist_mm, 3),
                        "rotation_deg": round(rot_deg, 3)
                    })
        pts = np.array([transforms[i][link][:3, 3] for i in relevant])
        if len(pts) > 1:
            centered = pts - pts.mean(axis=0)
            singular_values = np.linalg.svd(centered, compute_uv=False)
            position_spread_mm = np.round(np.ptp(pts, axis=0) * 1000, 3).tolist()
            centered_singular_values_mm = np.round(singular_values * 1000, 3).tolist()
        else:
            position_spread_mm = None
            centered_singular_values_mm = None
        grouped[str(marker_id)] = {
            "candidate_link": link,
            "observations": len(relevant),
            "frames": [tags[i] for i in relevant],
            "max_fk_translation_separation_mm": round(max(distances), 3) if distances else None,
            "max_fk_rotation_separation_deg": round(max(rotations), 3) if rotations else None,
            "fk_origin_position_xyz_span_mm": position_spread_mm,
            "centered_position_singular_values_mm": centered_singular_values_mm,
            "fk_pose_near_repeat_pairs": near_pairs,
        }
    return {
        "kind": "so101_aruco_fk_pairwise_diversity_v1",
        "samples": len(samples),
        "distinct_deg": duplicate_deg,
        "pairwise_joint_near_repeats": pairs,
        "per_marker": grouped,
        "all_ids_observation_counts": dict(sorted(Counter(
            mid for sample_ids in ids for mid in sample_ids).items())),
        "caveats": [
            "Pairs with near-identical J1-J4 can be duplicates for wrist_link/ID2 even if J5 differs.",
            "Independent observation count cannot be determined solely from arbitrary thresholds.",
            "The FK positions are computed under unverified LeRobot-to-URDF zero/sign mapping.",
            "This is not an observability or reprojection-error proof; camera intrinsics/extrinsics and marker mounting transforms are unknown.",
            "IDs 3/4/5 have no verified link assignment and are not used for FK grouping.",
        ],
    }


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", type=Path,
                   default=root / "data/urdf/so101/so101_new_calib.urdf")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--duplicate-deg", type=float, default=3.)
    p.add_argument("--near-translation-mm", type=float, default=5.)
    p.add_argument("--near-rotation-deg", type=float, default=5.)
    args = p.parse_args()
    report = audit_diversity(args.urdf, args.manifest,
        duplicate_deg=args.duplicate_deg,
        near_translation_mm=args.near_translation_mm,
        near_rotation_deg=args.near_rotation_deg)
    print("Pose count:", report["samples"])
    print("Joint-space repeat pairs:", len(report["pairwise_joint_near_repeats"]))
    for pair in report["pairwise_joint_near_repeats"]:
        print("  frame", pair["a"]["frame"], "vs", pair["b"]["frame"],
              "J1-J5 delta", pair["max_joint_delta_deg"],
              "J1-J4 delta", pair["max_j1_to_j4_delta_deg"])
    for mid, item in report["per_marker"].items():
        print(f"ID {mid}: {item['observations']} observations, "
              f"FK max translation {item['max_fk_translation_separation_mm']} mm, "
              f"max rotation {item['max_fk_rotation_separation_deg']} deg, "
              f"near-repeat pairs {len(item['fk_pose_near_repeat_pairs'])}")
        if item["fk_pose_near_repeat_pairs"]:
            for near in item["fk_pose_near_repeat_pairs"]:
                print("  near:", near["a"]["frame"], near["b"]["frame"],
                      "distance(mm)", near["distance_mm"],
                      "rotation(deg)", near["rotation_deg"])
    print("\n--- Full report (JSON) ---")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
