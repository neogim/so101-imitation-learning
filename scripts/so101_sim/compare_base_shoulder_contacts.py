#!/usr/bin/env python3
"""One-at-a-time SO-101 base/shoulder collision ablation in offline MuJoCo.

The normal URDF and RL environment stay unchanged. This script builds new
in-memory models for each condition and uses joint-target PD oracle only.
Never treat exclusions as safe for hardware or contact-rich RL training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from inspect_reach_collisions import (
    BASE_SHOULDER_PAIRS, geom_inventory, rollout,
)
from so101_reach_env import SO101ReachEnv


CASES = {
    "baseline": None,
    "reencoded_masks_control": (),
    "exclude_geom0_6": ((0, 6),),
    "exclude_geom1_6": ((1, 6),),
    "exclude_geom2_6": ((2, 6),),
    "exclude_all_three": BASE_SHOULDER_PAIRS,
}


def compare_pairs(seed=101, episode=0, max_steps=150, frame_skip=10):
    if episode < 0 or max_steps < 1 or frame_skip < 1:
        raise ValueError("Invalid episode or physics horizon")
    env = SO101ReachEnv(seed=seed)
    try:
        inventory = geom_inventory(env.model)
        if env.model.ngeom != 17:
            raise ValueError("SO-101 collision model changed: expected 17 geoms")
        expected = {0: "base_motor_holder_so101_v1",
                    1: "base_so101_v2",
                    2: "sts3215_03a_v1",
                    6: "rotation_pitch_so101_v1"}
        for gid, mesh in expected.items():
            if inventory[gid]["mesh_name"] != mesh:
                raise ValueError(f"Unexpected geom{gid}: expected {mesh}")
    finally:
        env.close()

    results = {}
    for case, pairs in CASES.items():
        results[case] = rollout(
            None, seed, episode, max_steps, frame_skip, False,
            excluded_geom_pairs=pairs)
    results["all_contacts_disabled_reference"] = rollout(
        None, seed, episode, max_steps, frame_skip, True)

    ref = results["baseline"]
    for case, result in results.items():
        np.testing.assert_allclose(result["goal_joint_rad"], ref["goal_joint_rad"])
        np.testing.assert_allclose(result["goal_tcp_base_m"], ref["goal_tcp_base_m"])
        np.testing.assert_allclose(result["initial_joint_rad"], ref["initial_joint_rad"])

    # Critical control: re-encoding *without* exclusions must preserve baseline
    # contacts at reset, even though the geom contact masks are changed.
    initial = lambda r: sorted((tuple(v["geom_id_pair"]), v["contact_points"])
                               for v in r["initial_contact_geom_pairs"])
    if initial(results["reencoded_masks_control"]) != initial(ref):
        raise RuntimeError("Contact-mask re-encoding modified baseline contacts")
    for case, pairs in CASES.items():
        if not pairs:
            continue
        wanted = {tuple(sorted(v)) for v in pairs}
        active = {tuple(v["geom_id_pair"])
                  for v in results[case]["initial_contact_geom_pairs"]}
        if wanted & active:
            raise RuntimeError(f"Excluded pairs still contact at reset: {case}")

    return {
        "status": "DIAGNOSTIC_ONLY_NOT_VALIDATED_COLLISION_MODEL",
        "seed": seed, "episode": episode,
        "max_steps": max_steps, "frame_skip": frame_skip,
        "excluded_pairs": {case: ([list(x) for x in pairs] if pairs is not None
                                   else []) for case, pairs in CASES.items()},
        "geom_inventory": inventory,
        "results": results,
        "warnings": [
            "This temporary contact mask modification is for causal isolation only.",
            "Excluding contact may permit physical intersections and is not a valid real-world safety model.",
            "PD still uses the same 3Nm generalized torque limit for each joint.",
            "A contact point is not necessarily a harmful collision.",
            "Initial contacts are validated, but interpretation must consider later contact pairs.",
        ],
    }


def save_report(report, output):
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    output.mkdir(parents=True)
    (output / "pair_ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "pair_ablation.csv").open("x", newline="", encoding="utf-8") as f:
        cols = ["condition", "success", "final_tcp_distance_m",
                "min_tcp_distance_m", "final_joint_error_l2_rad",
                "contact_substep_fraction", "max_contact_points",
                "mean_joint_saturation_fraction", "initial_contact_points",
                "max_abs_joint_velocity_rad_s"]
        writer = csv.DictWriter(f, cols)
        writer.writeheader()
        for name, result in report["results"].items():
            writer.writerow({
                "condition": name,
                "success": result["success"],
                "final_tcp_distance_m": result["final_tcp_distance_m"],
                "min_tcp_distance_m": result["min_tcp_distance_m"],
                "final_joint_error_l2_rad": result["final_joint_error_l2_rad"],
                "contact_substep_fraction": result["contact_substep_fraction"],
                "max_contact_points": result["max_contact_points"],
                "mean_joint_saturation_fraction": float(np.mean(
                    result["torque_saturation_fraction"])),
                "initial_contact_points": result["initial_contact_points"],
                "max_abs_joint_velocity_rad_s": max(
                    result["max_abs_joint_velocity_rad_s"]),
            })
    return output / "pair_ablation.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare_pairs(seed=args.seed, episode=args.episode,
                           max_steps=args.max_steps, frame_skip=args.frame_skip)
    path = save_report(result, args.out)
    print(f"Saved {path}")
    print("Condition                    TCP(mm)   joint-error(rad)   contact-rate   mean-saturation")
    for name, value in result["results"].items():
        print(f"{name:<28} {1000*value['final_tcp_distance_m']:>7.1f} "
              f"{value['final_joint_error_l2_rad']:>17.3f} "
              f"{value['contact_substep_fraction']:>13.1%} "
              f"{np.mean(value['torque_saturation_fraction']):>16.1%}")
    print("Do not use diagnostic exclusions for production contact physics or hardware.")


if __name__ == "__main__":
    main()
