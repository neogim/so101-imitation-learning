#!/usr/bin/env python3
"""Evaluate deterministic SAC and Jacobian-DLS weakness over many FK goals.

This is a kinematics-only evaluation. It does not train SAC, step physics, or
communicate with robot hardware. Different goal-joint ranges reuse seed values,
but generally produce different TCP targets because the sampling bounds differ.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_sac_dls_rollout import CSV_FIELDS
from learn_reach_kinematic import KinematicReachEnv, jacobian_reach_action


POLICIES = ("sac", "jacobian_dls")
MAX_STEPS = 60
ACTION_DELTA_RAD = 0.04
SUCCESS_RADIUS_M = 0.025
DISTANCE_EDGES_MM = (50, 75, 100, 125, 150, 175, 200, math.inf)
RESULT_FIELDS = [
    "goal_joint_delta_rad", "goal_seed",
    "target_x", "target_y", "target_z",
    "initial_tcp_x", "initial_tcp_y", "initial_tcp_z",
    "initial_tcp_distance_mm", "direction_x", "direction_y", "direction_z",
    *(f"{policy}_{metric}" for policy in ("sac", "dls") for metric in (
        "success", "final_tcp_distance_mm", "steps", "min_tcp_distance_mm",
        "final_minus_min_mm", "timeout", "tcp_path_length_m")),
]


def _record(env, *, policy, goal_seed, step, action):
    tcp = env._tcp()
    target = env.target_tcp.copy()
    distance_mm = float(np.linalg.norm(target - tcp) * 1000)
    row = {
        "policy": policy, "goal_seed": goal_seed, "step": step,
        "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
        "target_x": target[0], "target_y": target[1], "target_z": target[2],
        "tcp_distance_mm": distance_mm,
        "success": distance_mm <= env.success_radius_m * 1000,
    }
    row.update({f"joint_{i + 1}": float(value) for i, value in enumerate(
                env.data.qpos[env.qids])})
    row.update({f"action_{i + 1}": float(value) for i, value in enumerate(action)})
    return row


def _run_policy(env, model, *, policy, goal_seed):
    obs, _ = env.reset(seed=goal_seed)
    rows = [_record(env, policy=policy, goal_seed=goal_seed, step=0,
                    action=np.zeros(5))]
    for step in range(1, MAX_STEPS + 1):
        if policy == "sac":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = jacobian_reach_action(env)
        obs, _, terminated, truncated, _ = env.step(action)
        rows.append(_record(env, policy=policy, goal_seed=goal_seed,
                            step=step, action=np.asarray(action, dtype=float)))
        if terminated or truncated:
            break
    return rows


def _path_length_m(rows):
    points = np.array([[row["tcp_x"], row["tcp_y"], row["tcp_z"]]
                       for row in rows])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _policy_metrics(rows):
    distances = [row["tcp_distance_mm"] for row in rows]
    success = bool(rows[-1]["success"])
    return {
        "success": success,
        "final_tcp_distance_mm": distances[-1],
        "steps": int(rows[-1]["step"]),
        "min_tcp_distance_mm": min(distances),
        "final_minus_min_mm": distances[-1] - min(distances),
        "timeout": not success and rows[-1]["step"] == MAX_STEPS,
        "tcp_path_length_m": _path_length_m(rows),
    }


def evaluate(model_path, goal_deltas, seeds):
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required") from exc

    model = SAC.load(str(model_path), device="cpu")
    results = []
    traces = {}
    for goal_delta in goal_deltas:
        envs = {
            policy: KinematicReachEnv(
                max_steps=MAX_STEPS, action_delta_rad=ACTION_DELTA_RAD,
                success_radius_m=SUCCESS_RADIUS_M,
                goal_joint_delta_rad=goal_delta)
            for policy in POLICIES
        }
        try:
            for index, seed in enumerate(seeds, start=1):
                policy_rows = {
                    policy: _run_policy(
                        envs[policy], model, policy=policy, goal_seed=seed)
                    for policy in POLICIES
                }
                first = policy_rows["sac"][0]
                other = policy_rows["jacobian_dls"][0]
                initial_fields = [
                    *(f"joint_{i}" for i in range(1, 6)),
                    "tcp_x", "tcp_y", "tcp_z",
                    "target_x", "target_y", "target_z",
                ]
                if not all(first[name] == other[name] for name in initial_fields):
                    raise RuntimeError(
                        f"Initial condition mismatch at delta={goal_delta}, seed={seed}")
                initial = np.array([first[f"tcp_{axis}"] for axis in "xyz"])
                target = np.array([first[f"target_{axis}"] for axis in "xyz"])
                direction = target - initial
                row = {
                    "goal_joint_delta_rad": goal_delta,
                    "goal_seed": seed,
                    **{f"target_{axis}": target[i] for i, axis in enumerate("xyz")},
                    **{f"initial_tcp_{axis}": initial[i]
                       for i, axis in enumerate("xyz")},
                    "initial_tcp_distance_mm": float(np.linalg.norm(direction) * 1000),
                    **{f"direction_{axis}": direction[i]
                       for i, axis in enumerate("xyz")},
                }
                for policy, prefix in (("sac", "sac"),
                                       ("jacobian_dls", "dls")):
                    metrics = _policy_metrics(policy_rows[policy])
                    row.update({f"{prefix}_{name}": value
                                for name, value in metrics.items()})
                results.append(row)
                traces[(goal_delta, seed)] = policy_rows
                if index % 25 == 0:
                    print(f"delta={goal_delta:.2f}: {index}/{len(seeds)} goals")
        finally:
            for env in envs.values():
                env.close()
    return results, traces


def _distance_label(low, high):
    return f"{low:.0f}+" if math.isinf(high) else f"{low:.0f}-{high:.0f}"


def _distance_band(distance):
    for low, high in zip(DISTANCE_EDGES_MM[:-1], DISTANCE_EDGES_MM[1:]):
        if low <= distance < high:
            return _distance_label(low, high)
    raise ValueError(f"Initial distance outside configured bands: {distance}")


def _group_summary(rows):
    output = {}
    for policy, prefix in (("sac", "sac"), ("jacobian_dls", "dls")):
        successes = [bool(row[f"{prefix}_success"]) for row in rows]
        output[policy] = {
            "success_count": sum(successes),
            "sample_count": len(rows),
            "success_rate": sum(successes) / len(rows),
            "mean_final_tcp_distance_mm": float(np.mean(
                [row[f"{prefix}_final_tcp_distance_mm"] for row in rows])),
            "mean_steps": float(np.mean([row[f"{prefix}_steps"] for row in rows])),
            "mean_tcp_path_length_m": float(np.mean(
                [row[f"{prefix}_tcp_path_length_m"] for row in rows])),
        }
    return output


def _coordinate_stats(rows, subset):
    selected = [row for row in rows if bool(row["sac_success"]) is subset]
    if not selected:
        return {"sample_count": 0}
    result = {"sample_count": len(selected)}
    for name in ("target_x", "target_y", "target_z",
                 "direction_x", "direction_y", "direction_z"):
        values = [row[name] for row in selected]
        result[f"mean_{name}"] = float(np.mean(values))
        result[f"std_{name}"] = float(np.std(values))
    return result


def build_summary(results, goal_deltas, seeds):
    by_range = {}
    for delta in goal_deltas:
        rows = [row for row in results if row["goal_joint_delta_rad"] == delta]
        by_range[f"{delta:.2f}"] = {
            "mean_initial_tcp_distance_mm": float(np.mean(
                [row["initial_tcp_distance_mm"] for row in rows])),
            **_group_summary(rows),
        }

    by_distance = {}
    for low, high in zip(DISTANCE_EDGES_MM[:-1], DISTANCE_EDGES_MM[1:]):
        label = _distance_label(low, high)
        rows = [row for row in results
                if low <= row["initial_tcp_distance_mm"] < high]
        by_distance[label] = ({"sample_count": 0} if not rows else {
            "sample_count": len(rows), **_group_summary(rows)})

    by_direction_sign = {}
    for axis in "xyz":
        for sign, predicate in (("negative", lambda value: value < 0),
                                ("positive", lambda value: value >= 0)):
            rows = [row for row in results if predicate(row[f"direction_{axis}"])]
            by_direction_sign[f"{axis}_{sign}"] = (
                {"sample_count": 0} if not rows else
                {"sample_count": len(rows), **_group_summary(rows)})

    by_range_direction_sign = {}
    for delta in goal_deltas:
        range_rows = [row for row in results
                      if row["goal_joint_delta_rad"] == delta]
        groups = {}
        for axis in "xyz":
            for sign, predicate in (("negative", lambda value: value < 0),
                                    ("positive", lambda value: value >= 0)):
                rows = [row for row in range_rows
                        if predicate(row[f"direction_{axis}"])]
                groups[f"{axis}_{sign}"] = {
                    "sample_count": len(rows),
                    "sac_success_count": sum(row["sac_success"] for row in rows),
                    "sac_success_rate": (sum(row["sac_success"] for row in rows)
                                         / len(rows) if rows else None),
                }
        by_range_direction_sign[f"{delta:.2f}"] = groups

    magnitude_edges_m = (0, .025, .050, .075, .100, math.inf)
    by_direction_magnitude = {}
    for axis in "xyz":
        groups = {}
        for low, high in zip(magnitude_edges_m[:-1], magnitude_edges_m[1:]):
            label = (f"{low * 1000:.0f}+" if math.isinf(high) else
                     f"{low * 1000:.0f}-{high * 1000:.0f}")
            rows = [row for row in results
                    if low <= abs(row[f"direction_{axis}"]) < high]
            groups[label] = {
                "sample_count": len(rows),
                "sac_success_count": sum(row["sac_success"] for row in rows),
                "sac_success_rate": (sum(row["sac_success"] for row in rows)
                                     / len(rows) if rows else None),
            }
        by_direction_magnitude[axis] = groups

    failures = [row for row in results
                if not row["sac_success"] and row["dls_success"]]
    if not failures:
        raise RuntimeError("No SAC-failure/DLS-success representative exists")
    failure = max(failures, key=lambda row: row["sac_final_minus_min_mm"])
    both = [row for row in results if row["sac_success"] and row["dls_success"]]
    if not both:
        raise RuntimeError("No both-success representative exists")
    both_success = max(both, key=lambda row: (
        row["goal_joint_delta_rad"], row["initial_tcp_distance_mm"]))

    sac_failures = [row for row in results if not row["sac_success"]]
    worsened = [row for row in sac_failures if row["sac_final_minus_min_mm"] > 1.0]
    return {
        "status": "KINEMATIC_REACH_ANALYSIS_ONLY_NO_TRAINING_NO_PHYSICS",
        "goal_joint_delta_rad_values": list(goal_deltas),
        "evaluation_seeds": [min(seeds), max(seeds)],
        "seed_note": (
            "Seed values are shared across ranges, but different sampling bounds "
            "generally produce different TCP targets."),
        "case_count": len(results),
        "success_threshold_mm": SUCCESS_RADIUS_M * 1000,
        "max_steps": MAX_STEPS,
        "action_delta_rad": ACTION_DELTA_RAD,
        "by_goal_range": by_range,
        "by_initial_distance_mm": by_distance,
        "by_direction_sign": by_direction_sign,
        "by_goal_range_and_direction_sign": by_range_direction_sign,
        "by_absolute_direction_magnitude_mm": by_direction_magnitude,
        "sac_target_coordinate_stats": {
            "success": _coordinate_stats(results, True),
            "failure": _coordinate_stats(results, False),
        },
        "sac_failure_error_behavior": {
            "failure_count": len(sac_failures),
            "final_more_than_1mm_above_min_count": len(worsened),
            "final_more_than_1mm_above_min_rate": (
                len(worsened) / len(sac_failures) if sac_failures else 0),
            "mean_final_minus_min_mm": float(np.mean(
                [row["sac_final_minus_min_mm"] for row in sac_failures]))
            if sac_failures else 0,
            "median_final_minus_min_mm": float(np.median(
                [row["sac_final_minus_min_mm"] for row in sac_failures]))
            if sac_failures else 0,
            "mean_min_tcp_distance_mm": float(np.mean(
                [row["sac_min_tcp_distance_mm"] for row in sac_failures]))
            if sac_failures else 0,
            "min_under_30mm_count": sum(
                row["sac_min_tcp_distance_mm"] < 30 for row in sac_failures),
            "min_under_35mm_count": sum(
                row["sac_min_tcp_distance_mm"] < 35 for row in sac_failures),
        },
        "representatives": {
            "sac_fail_dls_success": {
                "goal_joint_delta_rad": failure["goal_joint_delta_rad"],
                "goal_seed": failure["goal_seed"],
            },
            "both_success": {
                "goal_joint_delta_rad": both_success["goal_joint_delta_rad"],
                "goal_seed": both_success["goal_seed"],
            },
        },
    }


def _plot_success_by_range(results, goal_deltas, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(goal_deltas))
    width = 0.36
    for offset, (label, prefix, color) in enumerate((
            ("SAC", "sac", "tab:blue"), ("DLS", "dls", "tab:orange"))):
        rates = []
        for delta in goal_deltas:
            rows = [r for r in results if r["goal_joint_delta_rad"] == delta]
            rates.append(np.mean([r[f"{prefix}_success"] for r in rows]) * 100)
        ax.bar(x + (offset - .5) * width, rates, width, label=label, color=color)
    ax.set(xticks=x, xticklabels=[f"±{d:.2f}" for d in goal_deltas],
           xlabel="Goal joint range (rad)", ylabel="Success rate (%)", ylim=(0, 105))
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "success_by_goal_range.png", dpi=160)
    plt.close(fig)


def _plot_success_by_distance(results, out):
    labels, samples, sac, dls = [], [], [], []
    for low, high in zip(DISTANCE_EDGES_MM[:-1], DISTANCE_EDGES_MM[1:]):
        rows = [r for r in results if low <= r["initial_tcp_distance_mm"] < high]
        if not rows:
            continue
        labels.append(_distance_label(low, high))
        samples.append(len(rows))
        sac.append(np.mean([r["sac_success"] for r in rows]) * 100)
        dls.append(np.mean([r["dls_success"] for r in rows]) * 100)
    x = np.arange(len(labels))
    width = .36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, sac, width, label="SAC", color="tab:blue")
    ax.bar(x + width / 2, dls, width, label="DLS", color="tab:orange")
    ax.set(xticks=x, xticklabels=[f"{label}\n(n={n})" for label, n in zip(labels, samples)],
           xlabel="Initial TCP distance band (mm)", ylabel="Success rate (%)",
           ylim=(0, 105))
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "success_by_initial_distance.png", dpi=160)
    plt.close(fig)


def _plot_target_xyz(results, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    pairs = (("x", "y"), ("x", "z"), ("y", "z"))
    for ax, (first, second) in zip(axes, pairs):
        for success, label, color, marker in (
                (True, "SAC success", "tab:blue", "o"),
                (False, "SAC failure", "tab:red", "x")):
            rows = [row for row in results if row["sac_success"] is success]
            ax.scatter([row[f"target_{first}"] for row in rows],
                       [row[f"target_{second}"] for row in rows],
                       s=20, alpha=.65, color=color, marker=marker, label=label)
        ax.set(xlabel=f"Target {first.upper()} (m)",
               ylabel=f"Target {second.upper()} (m)")
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "target_xyz_success.png", dpi=160)
    plt.close(fig)


def _plot_direction_sign(results, out):
    labels, samples, rates = [], [], []
    for axis in "xyz":
        for sign, predicate in (("−", lambda value: value < 0),
                                ("+", lambda value: value >= 0)):
            rows = [row for row in results if predicate(row[f"direction_{axis}"])]
            labels.append(axis.upper() + sign)
            samples.append(len(rows))
            rates.append((np.mean([row["sac_success"] for row in rows]) * 100)
                         if rows else 0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    bars = ax.bar(labels, rates, color="tab:blue")
    for bar, sample in zip(bars, samples):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={sample}", ha="center", va="bottom", fontsize=9)
    ax.set(xlabel="Initial-to-target direction component",
           ylabel="SAC success rate (%)", ylim=(0, 110))
    ax.grid(axis="y", alpha=.3)

    ax = axes[1]
    magnitude_edges = (0, .025, .050, .075, .100, math.inf)
    x = np.arange(len(magnitude_edges) - 1)
    width = .25
    for offset, (axis, color) in enumerate(zip("xyz", ("tab:blue", "tab:green",
                                                        "tab:purple"))):
        axis_rates = []
        for low, high in zip(magnitude_edges[:-1], magnitude_edges[1:]):
            rows = [row for row in results
                    if low <= abs(row[f"direction_{axis}"]) < high]
            axis_rates.append(np.mean([row["sac_success"] for row in rows]) * 100)
        ax.bar(x + (offset - 1) * width, axis_rates, width,
               label=f"|Δ{axis.upper()}|", color=color)
    magnitude_labels = [
        (f"{low * 1000:.0f}+" if math.isinf(high) else
         f"{low * 1000:.0f}-{high * 1000:.0f}")
        for low, high in zip(magnitude_edges[:-1], magnitude_edges[1:])]
    ax.set(xticks=x, xticklabels=magnitude_labels,
           xlabel="Absolute direction component (mm)",
           ylabel="SAC success rate (%)", ylim=(0, 105))
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sac_success_by_direction_sign.png", dpi=160)
    plt.close(fig)


def _plot_final_error(results, out):
    upper_mm = 80
    bins = np.arange(0, upper_mm + 5, 5)
    sac_values = [r["sac_final_tcp_distance_mm"] for r in results]
    dls_values = [r["dls_final_tcp_distance_mm"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist([value for value in sac_values if value <= upper_mm], bins=bins,
            alpha=.6, label="SAC", color="tab:blue")
    ax.hist([value for value in dls_values if value <= upper_mm], bins=bins,
            alpha=.6, label="DLS", color="tab:orange")
    ax.axvline(25, color="black", linestyle="--", label="25 mm threshold")
    ax.set(xlabel="Final TCP distance (mm)", ylabel="Case count")
    overflow = (sum(value > upper_mm for value in sac_values),
                sum(value > upper_mm for value in dls_values))
    if any(overflow):
        ax.text(.98, .72, f">{upper_mm} mm: SAC n={overflow[0]}, DLS n={overflow[1]}",
                transform=ax.transAxes, ha="right", va="top")
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "final_tcp_error_distribution.png", dpi=160)
    plt.close(fig)


def _plot_representative_failure(summary, traces, out):
    representative = summary["representatives"]["sac_fail_dls_success"]
    key = (representative["goal_joint_delta_rad"], representative["goal_seed"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy, label, color in (("sac", "SAC", "tab:blue"),
                                 ("jacobian_dls", "DLS", "tab:orange")):
        rows = traces[key][policy]
        ax.plot([r["step"] for r in rows], [r["tcp_distance_mm"] for r in rows],
                label=label, color=color, marker="o", markersize=3)
    ax.axhline(25, color="black", linestyle="--", label="25 mm threshold")
    ax.set(xlabel="Step", ylabel="TCP distance (mm)",
           title=(f"Representative SAC failure: delta={key[0]:.2f}, "
                  f"seed={key[1]}"))
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "representative_failure_distance_curve.png", dpi=160)
    plt.close(fig)


def _write_rollout_csv(path, traces):
    path.parent.mkdir(parents=True)
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for policy in POLICIES:
            writer.writerows(traces[policy])


def save(results, traces, summary, goal_deltas, out):
    out.mkdir(parents=True)
    with (out / "results.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, representative in summary["representatives"].items():
        key = (representative["goal_joint_delta_rad"], representative["goal_seed"])
        _write_rollout_csv(out / f"representative_{name}" / "rollout.csv", traces[key])
    _plot_success_by_range(results, goal_deltas, out)
    _plot_success_by_distance(results, out)
    _plot_target_xyz(results, out)
    _plot_direction_sign(results, out)
    _plot_final_error(results, out)
    _plot_representative_failure(summary, traces, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--goal-joint-deltas", type=float, nargs="+",
                        default=[0.35, 0.50, 0.70])
    parser.add_argument("--seed-start", type=int, default=1000001)
    parser.add_argument("--num-goals", type=int, default=100)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"SAC model was not found: {args.model}")
    if args.out.exists():
        parser.error(f"--out must not already exist: {args.out}")
    if args.num_goals < 1:
        parser.error("--num-goals must be positive")
    if (not args.goal_joint_deltas or
            any(not 0 < value <= 1 for value in args.goal_joint_deltas)):
        parser.error("goal joint deltas must be in (0, 1]")
    if len(set(args.goal_joint_deltas)) != len(args.goal_joint_deltas):
        parser.error("goal joint deltas must be unique")
    seeds = list(range(args.seed_start, args.seed_start + args.num_goals))
    results, traces = evaluate(args.model, args.goal_joint_deltas, seeds)
    summary = build_summary(results, args.goal_joint_deltas, seeds)
    save(results, traces, summary, args.goal_joint_deltas, args.out)
    print(f"Saved {len(results)} cases to {args.out}")
    for delta, metrics in summary["by_goal_range"].items():
        print(f"delta={delta}: SAC {metrics['sac']['success_count']}/"
              f"{metrics['sac']['sample_count']}, DLS "
              f"{metrics['jacobian_dls']['success_count']}/"
              f"{metrics['jacobian_dls']['sample_count']}")


if __name__ == "__main__":
    main()
