#!/usr/bin/env python3
"""Compare temporary SO-101 collision primitives with full meshes (offline).

Primitive sizes are rough origin-centered bounds scaled by factor and
NOT hardware-validated proxies. Never run SAC or physical SO-101 with these
unvalidated settings. Only compares the same PD-oracle goal across models.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from inspect_reach_collisions import rollout


def cases(scale=.65):
    if not 0 < scale <= 1:
        raise ValueError("scale must be (0,1]")
    return {
        "original_meshes": [],
        "shoulder_box": [(6, "box", scale)],
        "shoulder_sphere": [(6, "sphere", scale)],
        "shoulder_capsule": [(6, "capsule", scale)],
        "four_base_shoulder_boxes": [(gid, "box", scale) for gid in (0, 1, 2, 6)],
    }


def compare_proxies(seed=101, episode=0, max_steps=150,
                    frame_skip=10, scale=.65):
    results = {}
    for key, proxies in cases(scale).items():
        results[key] = rollout(
            None, seed, episode, max_steps, frame_skip, False,
            primitive_proxies=proxies)
    baseline = results["original_meshes"]
    for result in results.values():
        np.testing.assert_allclose(result["goal_joint_rad"], baseline["goal_joint_rad"])
        np.testing.assert_allclose(result["goal_tcp_base_m"], baseline["goal_tcp_base_m"])
        np.testing.assert_allclose(result["initial_joint_rad"], baseline["initial_joint_rad"])
    return {
        "status": "DIAGNOSTIC_GEOM_PROXY_ONLY_NOT_VALIDATED_PHYSICS",
        "seed": seed, "episode": episode, "scale": scale,
        "max_steps": max_steps, "frame_skip": frame_skip,
        "results": results,
        "notes": [
            "Scaled boxes/spheres/capsules are coarse sensitivity probes, not validated SO-101 collision models.",
            "Never infer safe collision-free paths from these simplified contact geoms.",
            "Use same-seed per-environment evaluation only; results may differ across platforms.",
            "Do not retrain SAC until a physically sensible collision model and stable PD are established.",
        ],
    }


def export(report, out):
    if out.exists():
        raise ValueError("Output directory exists")
    out.mkdir(parents=True)
    (out / "proxy_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "proxy_comparison.csv").open("x", newline="", encoding="utf-8") as f:
        cols = ["condition", "final_tcp_distance_m", "min_tcp_distance_m",
                "joint_error_l2_rad", "success", "contact_fraction",
                "avg_saturation", "peak_velocity_rad_s", "initial_contacts"]
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for key, result in report["results"].items():
            w.writerow({
                "condition": key,
                "final_tcp_distance_m": result["final_tcp_distance_m"],
                "min_tcp_distance_m": result["min_tcp_distance_m"],
                "joint_error_l2_rad": result["final_joint_error_l2_rad"],
                "success": result["success"],
                "contact_fraction": result["contact_substep_fraction"],
                "avg_saturation": float(np.mean(result["torque_saturation_fraction"])),
                "peak_velocity_rad_s": float(max(result["max_abs_joint_velocity_rad_s"])),
                "initial_contacts": result["initial_contact_points"],
            })
    return out / "proxy_comparison.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--scale", type=float, default=.65)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--frame-skip", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = compare_proxies(seed=a.seed, episode=a.episode,
                             max_steps=a.max_steps, frame_skip=a.frame_skip,
                             scale=a.scale)
    path = export(report, a.out)
    print(f"Saved {path}")
    print("Case                      final(mm) joint_err(rad) contact% saturation%")
    for name, r in report["results"].items():
        print(f"{name:<27} {1000*r['final_tcp_distance_m']:>8.1f}"
              f" {r['final_joint_error_l2_rad']:>13.3f}"
              f" {100*r['contact_substep_fraction']:>7.1f}"
              f" {100*np.mean(r['torque_saturation_fraction']):>11.1f}")


if __name__ == "__main__":
    main()
