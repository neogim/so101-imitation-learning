#!/usr/bin/env python3
"""Experiment C-1: evaluate saved reward-scale-10 SAC models at 10 mm.

This is a kinematic-only evaluation.  It neither trains a policy nor accesses
robot hardware.  The ordinary environment default remains 25 mm; this script
passes a 10 mm success radius only to its evaluation environments.
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

import experiment_b_reward_scale as exp_b
from compare_sac_dls_rollout import CSV_FIELDS
from learn_reach_kinematic import KinematicReachEnv, jacobian_reach_action


TRAINING_SEEDS = (101, 202, 303)
GOAL_DELTAS = exp_b.GOAL_DELTAS
EVALUATION_SEEDS = range(exp_b.SEED_START, exp_b.SEED_START + exp_b.NUM_GOALS)
MAX_STEPS = 60
ACTION_DELTA_RAD = 0.04
SUCCESS_RADIUS_M = 0.010
REWARD_SCALE = 10.0
REPRESENTATIVE_KEYS = ((0.70, 1000046), (0.70, 1000047), (0.70, 1000099))


def _record(env, *, policy, goal_seed, step, action):
    tcp = env._tcp()
    target = env.target_tcp.copy()
    distance_mm = float(np.linalg.norm(target - tcp) * 1000)
    row = {
        "policy": policy, "goal_seed": goal_seed, "step": step,
        "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
        "target_x": target[0], "target_y": target[1], "target_z": target[2],
        "tcp_distance_mm": distance_mm,
        "success": distance_mm <= SUCCESS_RADIUS_M * 1000,
    }
    row.update({f"joint_{index + 1}": float(value) for index, value in enumerate(
        env.data.qpos[env.qids])})
    row.update({f"action_{index + 1}": float(value) for index, value in enumerate(
        np.asarray(action, dtype=float))})
    return row


def _run_rollout(env, model, *, policy_name, goal_seed):
    obs, _ = env.reset(seed=goal_seed)
    rows = [_record(env, policy=policy_name, goal_seed=goal_seed, step=0,
                    action=np.zeros(5))]
    for step in range(1, MAX_STEPS + 1):
        if policy_name == "jacobian_dls":
            action = jacobian_reach_action(env)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        rows.append(_record(env, policy=policy_name, goal_seed=goal_seed,
                            step=step, action=action))
        if terminated or truncated:
            break
    return rows


def _first_step_at_or_below(rows, threshold_mm):
    return next((int(row["step"]) for row in rows
                 if row["tcp_distance_mm"] <= threshold_mm), None)


def _metrics(training_seed, goal_delta, evaluation_seed, rows):
    distances = np.array([row["tcp_distance_mm"] for row in rows])
    first25 = _first_step_at_or_below(rows, 25.0)
    first10 = _first_step_at_or_below(rows, 10.0)
    min_index = int(np.argmin(distances))
    post25_start = first25 if first25 is not None else 0
    post25_changes = np.diff(distances[post25_start:])
    nonzero_signs = np.sign(post25_changes[np.abs(post25_changes) > 1e-9])
    direction_reversals = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
    initial = rows[0]
    result = {
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "goal_joint_delta_rad": goal_delta,
        **{f"initial_tcp_{axis}": initial[f"tcp_{axis}"] for axis in "xyz"},
        **{f"target_{axis}": initial[f"target_{axis}"] for axis in "xyz"},
        "initial_tcp_distance_mm": float(distances[0]),
        "first_25mm_step": first25,
        "first_10mm_step": first10,
        "success_10mm": first10 is not None,
        "min_tcp_distance_mm": float(distances[min_index]),
        "min_tcp_distance_step": min_index,
        "final_tcp_distance_mm": float(distances[-1]),
        "end_step": int(rows[-1]["step"]),
        "timeout_60_steps": first10 is None and rows[-1]["step"] == MAX_STEPS,
        "reached_25mm_not_10mm": first25 is not None and first10 is None,
        "final_minus_min_mm": float(distances[-1] - distances[min_index]),
        "max_retreat_after_min_mm": float(
            distances[min_index:].max() - distances[min_index]),
        "post_25mm_increase_step_count": int(np.sum(post25_changes > 1e-9)),
        "post_25mm_decrease_step_count": int(np.sum(post25_changes < -1e-9)),
        "post_25mm_direction_reversal_count": direction_reversals,
    }
    result["steps_from_25mm_to_10mm"] = (
        None if first10 is None or first25 is None else first10 - first25)
    return result


def _load_b2_evaluation(path):
    output = {}
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            key = (int(raw["training_seed"]),
                   float(raw["goal_joint_delta_rad"]),
                   int(raw["evaluation_seed"]))
            output[key] = {
                "success": raw["success"] == "True",
                "steps": int(raw["steps"]),
                "final_tcp_distance_mm": float(raw["final_tcp_distance_mm"]),
            }
    if len(output) != 900:
        raise ValueError("Experiment B-2 evaluation must contain 900 cases")
    return output


def _verify_b2_prefix(result, rows, b2):
    key = (result["training_seed"], result["goal_joint_delta_rad"],
           result["evaluation_seed"])
    previous = b2[key]
    first25 = result["first_25mm_step"]
    if previous["success"]:
        if first25 != previous["steps"]:
            raise RuntimeError(f"25 mm first-arrival step differs from B-2: {key}")
        current_distance = rows[first25]["tcp_distance_mm"]
        if not math.isclose(current_distance, previous["final_tcp_distance_mm"],
                            rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"25 mm first-arrival distance differs from B-2: {key}")
    elif first25 is not None:
        raise RuntimeError(f"C-1 reached 25 mm although B-2 did not: {key}")


def evaluate(models, fixed_goals, b2):
    results, traces = [], {}
    for training_seed in TRAINING_SEEDS:
        traces[training_seed] = {}
        for goal_delta in GOAL_DELTAS:
            env = KinematicReachEnv(
                max_steps=MAX_STEPS, action_delta_rad=ACTION_DELTA_RAD,
                goal_joint_delta_rad=goal_delta,
                success_radius_m=SUCCESS_RADIUS_M, reward_scale=REWARD_SCALE)
            try:
                for evaluation_seed in EVALUATION_SEEDS:
                    key = (goal_delta, evaluation_seed)
                    rows = _run_rollout(
                        env, models[training_seed], policy_name="sac",
                        goal_seed=evaluation_seed)
                    source = fixed_goals[key]
                    first = rows[0]
                    for axis in "xyz":
                        if first[f"tcp_{axis}"] != source[f"initial_tcp_{axis}"]:
                            raise RuntimeError(f"Initial TCP mismatch: {key}")
                        if first[f"target_{axis}"] != source[f"target_{axis}"]:
                            raise RuntimeError(f"Target mismatch: {key}")
                    metrics = _metrics(training_seed, goal_delta,
                                       evaluation_seed, rows)
                    _verify_b2_prefix(metrics, rows, b2)
                    results.append(metrics)
                    traces[training_seed][key] = rows
            finally:
                env.close()
    return results, traces


def _distribution(values):
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)), "max": float(np.max(values)),
    }


def _arrival_summary(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return {
        "reached_count": len(values), "not_reached_count": len(rows) - len(values),
        "mean_reached_cases": None if not values else float(np.mean(values)),
        "median_reached_cases": None if not values else float(np.median(values)),
        "p90_reached_cases": None if not values else float(np.percentile(values, 90)),
    }


def _aggregate(rows):
    reached25 = sum(row["first_25mm_step"] is not None for row in rows)
    reached10 = sum(row["success_10mm"] for row in rows)
    return {
        "sample_count": len(rows),
        "reached_25mm_count": reached25,
        "reached_25mm_rate": reached25 / len(rows),
        "reached_10mm_count": reached10,
        "reached_10mm_rate": reached10 / len(rows),
        "reached_25mm_not_10mm_count": sum(
            row["reached_25mm_not_10mm"] for row in rows),
        "timeout_count": sum(row["timeout_60_steps"] for row in rows),
        "final_tcp_distance_mm": _distribution(
            [row["final_tcp_distance_mm"] for row in rows]),
        "min_tcp_distance_mm": _distribution(
            [row["min_tcp_distance_mm"] for row in rows]),
        "first_25mm_step": _arrival_summary(rows, "first_25mm_step"),
        "first_10mm_step": _arrival_summary(rows, "first_10mm_step"),
        "steps_from_25mm_to_10mm": _arrival_summary(
            rows, "steps_from_25mm_to_10mm"),
        "max_retreat_after_min_mm": _distribution(
            [row["max_retreat_after_min_mm"] for row in rows]),
        "failed_10mm_behavior": {
            "failed_count": sum(not row["success_10mm"] for row in rows),
            "retreat_at_most_1mm_count": sum(
                not row["success_10mm"] and
                row["max_retreat_after_min_mm"] <= 1 for row in rows),
            "retreat_over_5mm_count": sum(
                not row["success_10mm"] and
                row["max_retreat_after_min_mm"] > 5 for row in rows),
            "four_or_more_direction_reversals_count": sum(
                not row["success_10mm"] and
                row["post_25mm_direction_reversal_count"] >= 4 for row in rows),
            "definitions": {
                "limited_retreat": "maximum retreat after the minimum is at most 1 mm",
                "substantial_retreat": "maximum retreat after the minimum is over 5 mm",
                "repeated_direction_change": "at least four error-direction reversals after first reaching 25 mm",
            },
        },
    }


def build_summary(results, model_paths, representative_csvs):
    per_seed = {}
    for seed in TRAINING_SEEDS:
        selected = [row for row in results if row["training_seed"] == seed]
        per_seed[str(seed)] = {
            "all_300": _aggregate(selected),
            "by_goal_range": {
                f"{delta:.2f}": _aggregate([
                    row for row in selected
                    if row["goal_joint_delta_rad"] == delta])
                for delta in GOAL_DELTAS},
        }
    return {
        "status": "EXPERIMENT_C1_KINEMATIC_ONLY_NO_TRAINING_NO_HARDWARE",
        "success_radius_mm": 10,
        "reference_radius_mm": 25,
        "max_steps": MAX_STEPS,
        "action_delta_rad": ACTION_DELTA_RAD,
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seed_range": [min(EVALUATION_SEEDS), max(EVALUATION_SEEDS)],
        "goal_joint_delta_rad_values": list(GOAL_DELTAS),
        "same_300_goals_verified_for_all_models": True,
        "b2_25mm_first_arrival_exactly_reproduced_for_all_900_cases": True,
        "model_paths": model_paths,
        "per_training_seed": per_seed,
        "representative_csvs": representative_csvs,
    }


def _write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_viewer_csv(path, sac_rows, dls_rows):
    path.parent.mkdir(parents=True)
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sac_rows)
        writer.writerows(dls_rows)


def save_representatives(results, traces, out):
    failed = [row for row in results if not row["success_10mm"]]
    keys = list(REPRESENTATIVE_KEYS)
    if failed:
        representative_failure = max(
            failed, key=lambda row: row["max_retreat_after_min_mm"])
        failure_key = (representative_failure["goal_joint_delta_rad"],
                       representative_failure["evaluation_seed"])
        if failure_key not in keys:
            keys.append(failure_key)
    saved = []
    for goal_delta, evaluation_seed in keys:
        dls_env = KinematicReachEnv(
            max_steps=MAX_STEPS, action_delta_rad=ACTION_DELTA_RAD,
            goal_joint_delta_rad=goal_delta, success_radius_m=SUCCESS_RADIUS_M)
        try:
            dls_rows = _run_rollout(
                dls_env, None, policy_name="jacobian_dls",
                goal_seed=evaluation_seed)
        finally:
            dls_env.close()
        for training_seed in TRAINING_SEEDS:
            relative = (Path("representatives") /
                        f"train{training_seed}_eval{evaluation_seed}_delta{goal_delta:.2f}" /
                        "rollout.csv")
            _write_viewer_csv(
                out / relative, traces[training_seed][(goal_delta, evaluation_seed)],
                dls_rows)
            saved.append(str(relative))
    return saved, keys


def make_plots(results, traces, representative_keys, out):
    colors = {101: "tab:green", 202: "tab:blue", 303: "tab:purple"}
    x = np.arange(len(TRAINING_SEEDS)); width = .34
    reached25 = []; reached10 = []
    for seed in TRAINING_SEEDS:
        rows = [row for row in results if row["training_seed"] == seed]
        reached25.append(100 * np.mean([row["first_25mm_step"] is not None for row in rows]))
        reached10.append(100 * np.mean([row["success_10mm"] for row in rows]))
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.bar(x-width/2, reached25, width, label="Reached 25 mm")
    ax.bar(x+width/2, reached10, width, label="Reached 10 mm")
    ax.set_xticks(x, [str(seed) for seed in TRAINING_SEEDS])
    ax.set(xlabel="Training seed", ylabel="Arrival rate (%)", ylim=(0, 105))
    ax.grid(axis="y", alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "arrival_rate_25mm_vs_10mm.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, field, label in (
            (axes[0], "min_tcp_distance_mm", "Minimum TCP error (mm)"),
            (axes[1], "final_tcp_distance_mm", "Final TCP error (mm)")):
        ax.boxplot([[row[field] for row in results if row["training_seed"] == seed]
                    for seed in TRAINING_SEEDS],
                   tick_labels=[str(seed) for seed in TRAINING_SEEDS], showfliers=True)
        ax.axhline(10, color="black", linestyle="--", label="10 mm")
        ax.set(xlabel="Training seed", ylabel=label); ax.grid(axis="y", alpha=.3)
    axes[0].legend(); fig.tight_layout()
    fig.savefig(out / "minimum_and_final_tcp_error.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    mean25 = []; mean10 = []
    for seed in TRAINING_SEEDS:
        rows = [row for row in results if row["training_seed"] == seed]
        mean25.append(np.mean([row["first_25mm_step"] for row in rows
                               if row["first_25mm_step"] is not None]))
        mean10.append(np.mean([row["first_10mm_step"] for row in rows
                               if row["first_10mm_step"] is not None]))
    ax.bar(x-width/2, mean25, width, label="25 mm reached cases")
    ax.bar(x+width/2, mean10, width, label="10 mm reached cases")
    ax.set_xticks(x, [str(seed) for seed in TRAINING_SEEDS])
    ax.set(xlabel="Training seed", ylabel="Mean first-arrival step")
    ax.grid(axis="y", alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "arrival_steps_25mm_vs_10mm.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(len(representative_keys), 1,
                             figsize=(9, 3.7 * len(representative_keys)),
                             squeeze=False)
    for ax, key in zip(axes[:, 0], representative_keys):
        for seed in TRAINING_SEEDS:
            rows = traces[seed][key]
            ax.plot([row["step"] for row in rows],
                    [row["tcp_distance_mm"] for row in rows],
                    color=colors[seed], label=str(seed), linewidth=1.6)
        ax.axhline(25, color="gray", linestyle="--", label="25 mm")
        ax.axhline(10, color="black", linestyle=":", label="10 mm")
        ax.set(title=f"Evaluation seed {key[1]}, ±{key[0]:.2f} rad",
               xlabel="Step", ylabel="TCP error (mm)")
        ax.grid(alpha=.3)
    axes[0, 0].legend(ncol=5, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "representative_tcp_error_curves.png", dpi=160)
    plt.close(fig)


def write_report(summary, out):
    lines = ["# Experiment C-1: saved SAC precision at 10 mm", "",
             "Kinematic-only evaluation. No training or robot hardware was used.", "",
             "| Training seed | 25 mm arrival | 10 mm arrival | 25 mm but not 10 mm | Min error mean / median / P90 (mm) | Final error mean / median / P90 (mm) |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed in TRAINING_SEEDS:
        metrics = summary["per_training_seed"][str(seed)]["all_300"]
        minimum = metrics["min_tcp_distance_mm"]
        final = metrics["final_tcp_distance_mm"]
        lines.append(
            f"| {seed} | {metrics['reached_25mm_count']}/300 "
            f"({metrics['reached_25mm_rate']:.1%}) | "
            f"{metrics['reached_10mm_count']}/300 "
            f"({metrics['reached_10mm_rate']:.1%}) | "
            f"{metrics['reached_25mm_not_10mm_count']} | "
            f"{minimum['mean']:.2f} / {minimum['median']:.2f} / {minimum['p90']:.2f} | "
            f"{final['mean']:.2f} / {final['median']:.2f} / {final['p90']:.2f} |")
    lines += ["", "## Success by goal range", "",
              "| Goal range | 25 mm arrival | 10 mm arrival |",
              "| ---: | ---: | ---: |"]
    for delta in GOAL_DELTAS:
        range_metrics = [
            summary["per_training_seed"][str(seed)]["by_goal_range"][f"{delta:.2f}"]
            for seed in TRAINING_SEEDS]
        reached25 = sum(item["reached_25mm_count"] for item in range_metrics)
        reached10 = sum(item["reached_10mm_count"] for item in range_metrics)
        lines.append(f"| ±{delta:.2f} rad | {reached25}/300 ({reached25/300:.1%}) | "
                     f"{reached10}/300 ({reached10/300:.1%}) |")
    failed_count = sum(
        summary["per_training_seed"][str(seed)]["all_300"]
        ["failed_10mm_behavior"]["failed_count"] for seed in TRAINING_SEEDS)
    limited = sum(
        summary["per_training_seed"][str(seed)]["all_300"]
        ["failed_10mm_behavior"]["retreat_at_most_1mm_count"]
        for seed in TRAINING_SEEDS)
    substantial = sum(
        summary["per_training_seed"][str(seed)]["all_300"]
        ["failed_10mm_behavior"]["retreat_over_5mm_count"]
        for seed in TRAINING_SEEDS)
    reversals = sum(
        summary["per_training_seed"][str(seed)]["all_300"]
        ["failed_10mm_behavior"]["four_or_more_direction_reversals_count"]
        for seed in TRAINING_SEEDS)
    lines += ["", "The first 25 mm arrival step and distance exactly matched Experiment B-2 in all 900 cases.",
              "Arrival-step averages exclude cases that did not reach the corresponding threshold.",
              "Experiment B-2 stopped at 25 mm, so its final errors and episode lengths are not compared directly with the continued C-1 rollouts.", ""]
    lines += ["## Behavior among 10 mm failures", "",
              f"- Total failures: {failed_count}",
              f"- Maximum retreat after the minimum was at most 1 mm: {limited}",
              f"- Maximum retreat after the minimum exceeded 5 mm: {substantial}",
              f"- At least four error-direction reversals after first reaching 25 mm: {reversals}",
              "- These categories are descriptive and may overlap; they are not learned-state labels.",
              "", "## Conclusion", "",
              "- The saved policies can reach 10 mm in many cases without retraining, but do not meet a reliable 10 mm requirement over the fixed 300 goals.",
              "- A separate C-2 training run with a 10 mm termination condition is justified if 10 mm is the required operating criterion.",
              "- This C-1 result does not determine which C-2 reward or curriculum design would work best.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--experiment-a-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    required = (args.b2_dir / "summary.json", args.b2_dir / "evaluation_900.csv",
                args.experiment_a_csv)
    if not all(path.is_file() for path in required):
        parser.error("One or more Experiment A/B-2 inputs are missing")
    if args.out.exists():
        parser.error("--out must not exist")

    from stable_baselines3 import SAC
    b2_summary = json.loads((args.b2_dir / "summary.json").read_text(encoding="utf-8"))
    model_paths = b2_summary["model_paths"]
    if set(model_paths) != {str(seed) for seed in TRAINING_SEEDS}:
        parser.error("Experiment B-2 model paths do not cover seeds 101, 202, 303")
    if not all(Path(path).is_file() for path in model_paths.values()):
        parser.error("One or more saved SAC models are missing")
    models = {seed: SAC.load(model_paths[str(seed)], device="cpu")
              for seed in TRAINING_SEEDS}
    if any(model.num_timesteps != 100000 for model in models.values()):
        raise RuntimeError("Expected three unchanged 100,000-step models")

    fixed_goals = exp_b.load_experiment_a(args.experiment_a_csv)
    b2 = _load_b2_evaluation(args.b2_dir / "evaluation_900.csv")
    results, traces = evaluate(models, fixed_goals, b2)
    if len(results) != 900:
        raise RuntimeError("C-1 evaluation did not produce 900 rows")

    args.out.mkdir(parents=True)
    _write_csv(args.out / "evaluation_900.csv", results)
    representative_csvs, representative_keys = save_representatives(
        results, traces, args.out)
    summary = build_summary(results, model_paths, representative_csvs)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(results, traces, representative_keys, args.out)
    write_report(summary, args.out)
    print("Saved Experiment C-1 to", args.out)
    for seed in TRAINING_SEEDS:
        metrics = summary["per_training_seed"][str(seed)]["all_300"]
        print(f"training seed {seed}: 10 mm "
              f"{metrics['reached_10mm_count']}/300")


if __name__ == "__main__":
    main()
