#!/usr/bin/env python3
"""Experiment C-2b: reproduce 10 mm SAC training across seeds 101/202/303."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import experiment_b_reward_scale as exp_b
import experiment_c1_precision10 as c1
import experiment_c2_train_precision10 as c2
from learn_reach_kinematic import KinematicReachEnv


TRAINING_SEEDS = (101, 202, 303)
REPRESENTATIVE_KEYS = tuple((0.70, seed) for seed in
                            (1000062, 1000046, 1000047, 1000099))


def verify_model_settings(models):
    settings = {str(seed): c2._model_settings(model)
                for seed, model in models.items()}
    if any(value != settings["101"] for value in settings.values()):
        raise RuntimeError("SAC settings differ across training seeds")
    expected = {
        "learning_rate": 3e-4, "buffer_size": 100000,
        "learning_starts": 100, "batch_size": 64,
        "train_freq_frequency": 1, "train_freq_unit": "step",
        "gradient_steps": 1,
        "policy_kwargs": {"net_arch": [64, 64], "use_sde": False},
    }
    if settings["101"] != expected:
        raise RuntimeError("SAC settings differ from Experiment C-2")
    if any(model.num_timesteps != 100000 for model in models.values()):
        raise RuntimeError("Expected three 100,000-step models")
    return settings


def evaluate(models, fixed_goals):
    results, traces = [], {seed: {} for seed in TRAINING_SEEDS}
    for training_seed, model in models.items():
        for goal_delta in c1.GOAL_DELTAS:
            env = KinematicReachEnv(
                max_steps=c1.MAX_STEPS, action_delta_rad=c1.ACTION_DELTA_RAD,
                goal_joint_delta_rad=goal_delta,
                success_radius_m=c1.SUCCESS_RADIUS_M,
                reward_scale=c1.REWARD_SCALE)
            try:
                for evaluation_seed in c1.EVALUATION_SEEDS:
                    key = (goal_delta, evaluation_seed)
                    rollout = c1._run_rollout(
                        env, model, policy_name="sac", goal_seed=evaluation_seed)
                    source = fixed_goals[key]
                    first = rollout[0]
                    for axis in "xyz":
                        if first[f"tcp_{axis}"] != source[f"initial_tcp_{axis}"]:
                            raise RuntimeError(f"Initial TCP mismatch: {key}")
                        if first[f"target_{axis}"] != source[f"target_{axis}"]:
                            raise RuntimeError(f"Target mismatch: {key}")
                    results.append(c1._metrics(
                        training_seed, goal_delta, evaluation_seed, rollout))
                    traces[training_seed][key] = rollout
            finally:
                env.close()
    return results, traces


def load_c2_seed101(path):
    output = {}
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            if raw["model"] == "trained_10mm":
                output[(float(raw["goal_joint_delta_rad"]),
                        int(raw["evaluation_seed"]))] = raw
    if len(output) != 300:
        raise ValueError("C-2 evaluation must contain 300 new-model rows")
    return output


def verify_seed101_reproduction(results, c2_rows):
    fields = (
        "first_25mm_step", "first_10mm_step", "success_10mm",
        "min_tcp_distance_mm", "final_tcp_distance_mm", "end_step",
        "timeout_60_steps", "reached_25mm_not_10mm",
        "max_retreat_after_min_mm", "post_25mm_direction_reversal_count")
    checked = 0
    for row in results:
        if row["training_seed"] != 101:
            continue
        old = c2_rows[(row["goal_joint_delta_rad"], row["evaluation_seed"])]
        for field in fields:
            actual, expected = row[field], old[field]
            if isinstance(actual, bool):
                matches = actual == (expected == "True")
            elif actual is None:
                matches = expected == ""
            else:
                matches = math.isclose(float(actual), float(expected),
                                       rel_tol=0, abs_tol=1e-12)
            if not matches:
                raise RuntimeError(f"C-2 seed 101 mismatch: {field}")
        checked += 1
    if checked != 300:
        raise RuntimeError("C-2 seed 101 reproduction did not cover 300 cases")


def training_metric_stats(rows, name):
    values = [row[name] for row in rows if row[name] is not None]
    if not values:
        return None
    return {
        "sample_count": len(values), "final": values[-1],
        "min": min(values), "max": max(values),
        "last_10_mean": float(np.mean(values[-10:])),
        "nonfinite_count": sum(not math.isfinite(value) for value in values),
    }


def build_summary(results, training_rows, training_seconds, model_paths,
                  model_settings, representative_csvs):
    per_seed, failure_sets = {}, {}
    for seed in TRAINING_SEEDS:
        selected = [row for row in results if row["training_seed"] == seed]
        failures = {(row["goal_joint_delta_rad"], row["evaluation_seed"])
                    for row in selected if not row["success_10mm"]}
        failure_sets[seed] = failures
        per_seed[str(seed)] = {
            "all_300": c1._aggregate(selected),
            "by_goal_range": {
                f"{delta:.2f}": c1._aggregate([
                    row for row in selected
                    if row["goal_joint_delta_rad"] == delta])
                for delta in c1.GOAL_DELTAS},
            "training_seconds": training_seconds[str(seed)],
            "training_metrics": {
                metric: training_metric_stats(training_rows[seed], metric)
                for metric in ("rollout/success_rate", "train/ent_coef",
                               "train/actor_loss", "train/critic_loss")},
        }
    rates = [per_seed[str(seed)]["all_300"]["reached_10mm_rate"]
             for seed in TRAINING_SEEDS]
    common = set.intersection(*failure_sets.values())
    union = set.union(*failure_sets.values())
    return {
        "status": "EXPERIMENT_C2B_KINEMATIC_ONLY_NO_HARDWARE",
        "training_seeds": list(TRAINING_SEEDS),
        "training_steps": 100000,
        "success_radius_mm": 10,
        "reward_scale": 10.0,
        "training_goal_joint_delta_rad": 0.35,
        "max_steps": 60,
        "action_delta_rad": 0.04,
        "only_training_seed_changed": True,
        "same_300_goals_verified_for_all_models": True,
        "seed101_c2_result_exactly_reproduced": True,
        "model_settings": model_settings,
        "model_paths": model_paths,
        "per_training_seed": per_seed,
        "all_300_success_rate_across_training_seeds": {
            "mean": float(np.mean(rates)),
            "std_population": float(np.std(rates)),
            "min": min(rates), "max": max(rates),
            "training_seed_count": 3,
            "note": "Three training seeds are too few to establish statistical stability.",
        },
        "failures": {
            "common_to_all_seeds": [list(item) for item in sorted(common)],
            "observed_by_any_seed": [list(item) for item in sorted(union)],
            "by_training_seed": {
                str(seed): [list(item) for item in sorted(failure_sets[seed])]
                for seed in TRAINING_SEEDS},
        },
        "representative_csvs": representative_csvs,
    }


def save_representatives(traces, out):
    saved = []
    for goal_delta, evaluation_seed in REPRESENTATIVE_KEYS:
        env = KinematicReachEnv(
            max_steps=c1.MAX_STEPS, action_delta_rad=c1.ACTION_DELTA_RAD,
            goal_joint_delta_rad=goal_delta,
            success_radius_m=c1.SUCCESS_RADIUS_M)
        try:
            dls = c1._run_rollout(
                env, None, policy_name="jacobian_dls", goal_seed=evaluation_seed)
        finally:
            env.close()
        for seed in TRAINING_SEEDS:
            relative = (Path("representatives") /
                        f"train{seed}_eval{evaluation_seed}_delta070" /
                        "rollout.csv")
            c1._write_viewer_csv(
                out / relative, traces[seed][(goal_delta, evaluation_seed)], dls)
            saved.append(str(relative))
    return saved


def make_plots(results, traces, training_rows, out):
    colors = {101: "tab:green", 202: "tab:blue", 303: "tab:purple"}
    x = np.arange(len(c1.GOAL_DELTAS)); width = .24
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for index, seed in enumerate(TRAINING_SEEDS):
        values = [100 * np.mean([
            row["success_10mm"] for row in results
            if row["training_seed"] == seed and
            row["goal_joint_delta_rad"] == delta]) for delta in c1.GOAL_DELTAS]
        ax.bar(x + (index - 1) * width, values, width,
               label=f"train seed {seed}", color=colors[seed])
    ax.set_xticks(x, [f"±{delta:.2f}" for delta in c1.GOAL_DELTAS])
    ax.set(xlabel="Goal joint range (rad)", ylabel="10 mm success rate (%)",
           ylim=(0, 105)); ax.grid(axis="y", alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "success_rate_10mm_by_training_seed.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, field, label in (
            (axes[0], "min_tcp_distance_mm", "Minimum TCP error (mm)"),
            (axes[1], "first_10mm_step", "10 mm first-arrival step")):
        ax.boxplot([[row[field] for row in results
                     if row["training_seed"] == seed and row[field] is not None]
                    for seed in TRAINING_SEEDS],
                   tick_labels=[str(seed) for seed in TRAINING_SEEDS], showfliers=True)
        ax.set(xlabel="Training seed", ylabel=label); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "precision_and_steps_by_training_seed.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    specifications = (("rollout/success_rate", "Training success rate"),
                      ("train/ent_coef", "Entropy coefficient"),
                      ("train/actor_loss", "Actor loss"),
                      ("train/critic_loss", "Critic loss"))
    for ax, (metric, ylabel) in zip(axes, specifications):
        for seed in TRAINING_SEEDS:
            points = [(row["time/total_timesteps"], row[metric])
                      for row in training_rows[seed] if row[metric] is not None]
            ax.plot([point[0] for point in points], [point[1] for point in points],
                    color=colors[seed], label=str(seed))
        if metric == "train/ent_coef":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel); ax.grid(alpha=.3)
    axes[0].legend(title="Training seed"); axes[-1].set_xlabel("Training timesteps")
    fig.tight_layout(); fig.savefig(out / "training_curves_by_seed.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(REPRESENTATIVE_KEYS), 1,
                             figsize=(9, 3.7 * len(REPRESENTATIVE_KEYS)),
                             squeeze=False)
    for ax, key in zip(axes[:, 0], REPRESENTATIVE_KEYS):
        for seed in TRAINING_SEEDS:
            rows = traces[seed][key]
            ax.plot([row["step"] for row in rows],
                    [row["tcp_distance_mm"] for row in rows],
                    color=colors[seed], label=str(seed))
        ax.axhline(25, color="gray", linestyle="--", label="25 mm")
        ax.axhline(10, color="black", linestyle=":", label="10 mm")
        ax.set(title=f"Evaluation seed {key[1]}, ±0.70 rad",
               xlabel="Step", ylabel="TCP error (mm)"); ax.grid(alpha=.3)
    axes[0, 0].legend(ncol=5, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "representative_tcp_error_curves.png", dpi=160)
    plt.close(fig)


def write_report(summary, out):
    lines = ["# Experiment C-2b: 10 mm SAC training-seed reproducibility", "",
             "Kinematic-only experiment. No robot hardware was used.", "",
             "| Training seed | ±0.35 | ±0.50 | ±0.70 | All 300 | Minimum error mean / median / P90 (mm) | 10 mm step mean / median / P90 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed in TRAINING_SEEDS:
        item = summary["per_training_seed"][str(seed)]
        ranges = item["by_goal_range"]
        all_cases = item["all_300"]
        minimum = all_cases["min_tcp_distance_mm"]
        steps = all_cases["first_10mm_step"]
        lines.append(
            f"| {seed} | " + " | ".join(
                f"{ranges[f'{delta:.2f}']['reached_10mm_count']}/100"
                for delta in c1.GOAL_DELTAS) +
            f" | {all_cases['reached_10mm_count']}/300 "
            f"({all_cases['reached_10mm_rate']:.1%}) | "
            f"{minimum['mean']:.2f} / {minimum['median']:.2f} / {minimum['p90']:.2f} | "
            f"{steps['mean_reached_cases']:.2f} / {steps['median_reached_cases']:.2f} / "
            f"{steps['p90_reached_cases']:.2f} |")
    across = summary["all_300_success_rate_across_training_seeds"]
    failures = summary["failures"]
    lines += ["", "## Across training seeds", "",
              f"- Mean 10 mm success rate: {across['mean']:.2%}",
              f"- Population standard deviation: {across['std_population']:.3%}",
              f"- Minimum / maximum: {across['min']:.2%} / {across['max']:.2%}",
              f"- Unique failed goals: {len(failures['observed_by_any_seed'])}",
              "- Three seeds are insufficient to establish statistical stability.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed101-model", type=Path, required=True)
    parser.add_argument("--seed101-log", type=Path, required=True)
    parser.add_argument("--c2-evaluation", type=Path, required=True)
    parser.add_argument("--experiment-a-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    required = (args.seed101_model, args.seed101_log,
                args.c2_evaluation, args.experiment_a_csv)
    if not all(path.is_file() for path in required):
        parser.error("One or more Experiment A/C-2 inputs are missing")
    if args.out.exists():
        parser.error("--out must not exist")
    args.out.mkdir(parents=True)

    from stable_baselines3 import SAC
    models = {101: SAC.load(str(args.seed101_model), device="cpu")}
    training_seconds = {"101": c2._logged_elapsed_seconds(args.seed101_log)}
    seed_dirs = {}
    for seed in (202, 303):
        seed_dir = args.out / f"train_seed_{seed}"
        seed_dir.mkdir()
        model, seconds = c2.train_precision10(seed_dir, train_seed=seed)
        models[seed] = model
        training_seconds[str(seed)] = seconds
        seed_dirs[seed] = seed_dir
    settings = verify_model_settings(models)

    fixed_goals = exp_b.load_experiment_a(args.experiment_a_csv)
    results, traces = evaluate(models, fixed_goals)
    if len(results) != 900:
        raise RuntimeError("C-2b evaluation did not produce 900 rows")
    verify_seed101_reproduction(results, load_c2_seed101(args.c2_evaluation))
    c1._write_csv(args.out / "evaluation_900.csv", results)

    log_paths = {101: args.seed101_log}
    log_paths.update({seed: seed_dirs[seed] / "training_log" / "progress.csv"
                      for seed in (202, 303)})
    copied_logs = {}
    for seed, source in log_paths.items():
        target = args.out / f"training_seed_{seed}_progress.csv"
        shutil.copyfile(source, target)
        copied_logs[seed] = target
    training_rows = {seed: exp_b.load_training_metrics(path)
                     for seed, path in copied_logs.items()}
    representative_csvs = save_representatives(traces, args.out)
    model_paths = {
        "101": str(args.seed101_model),
        "202": str(seed_dirs[202] / "sac_reward10_success10mm.zip"),
        "303": str(seed_dirs[303] / "sac_reward10_success10mm.zip"),
    }
    summary = build_summary(
        results, training_rows, training_seconds, model_paths,
        settings, representative_csvs)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "training_settings.json").write_text(json.dumps({
        "training_seeds": list(TRAINING_SEEDS), "timesteps": 100000,
        "success_radius_mm": 10, "reward_scale": 10.0,
        "goal_joint_delta_rad": 0.35, "max_steps": 60,
        "action_delta_rad": 0.04, "model_settings": settings,
        "only_variable": "training_seed",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(results, traces, training_rows, args.out)
    write_report(summary, args.out)
    print("Saved Experiment C-2b to", args.out)
    for seed in TRAINING_SEEDS:
        metric = summary["per_training_seed"][str(seed)]["all_300"]
        print(seed, f"{metric['reached_10mm_count']}/300")


if __name__ == "__main__":
    main()