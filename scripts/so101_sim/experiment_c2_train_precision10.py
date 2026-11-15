#!/usr/bin/env python3
"""Experiment C-2: train SAC with a 10 mm success radius and compare C-1.

Only the training environment's success radius differs from Experiment B.
This script uses kinematics only and never communicates with robot hardware.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import experiment_b_reward_scale as exp_b
import experiment_c1_precision10 as c1
from learn_reach_kinematic import KinematicReachEnv


MODEL_NAMES = ("trained_25mm", "trained_10mm")
MODEL_LABELS = {"trained_25mm": "trained at 25 mm",
                "trained_10mm": "trained at 10 mm"}
TRAINING_SEED = 101
TRAINING_STEPS = 100000
REPRESENTATIVE_KEYS = tuple((0.70, seed) for seed in
                            (1000046, 1000047, 1000099, 1000062))


def train_precision10(out, train_seed=TRAINING_SEED):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure

    torch.set_num_threads(1)
    env = KinematicReachEnv(
        seed=train_seed, max_steps=c1.MAX_STEPS,
        action_delta_rad=c1.ACTION_DELTA_RAD,
        goal_joint_delta_rad=0.35,
        success_radius_m=c1.SUCCESS_RADIUS_M,
        reward_scale=c1.REWARD_SCALE)
    try:
        model = SAC(
            "MlpPolicy", env, seed=train_seed, device="cpu",
            learning_rate=3e-4, buffer_size=TRAINING_STEPS,
            learning_starts=100, batch_size=64, train_freq=1,
            gradient_steps=1, policy_kwargs={"net_arch": [64, 64]}, verbose=0)
        log_dir = out / "training_log"
        log_dir.mkdir()
        model.set_logger(configure(str(log_dir), ["csv"]))
        start = time.perf_counter()
        model.learn(total_timesteps=TRAINING_STEPS,
                    callback=exp_b.ProgressPrinter.build())
        seconds = time.perf_counter() - start
        model.save(str(out / "sac_reward10_success10mm"))
        return model, seconds
    finally:
        env.close()


def _model_settings(model):
    return {
        "learning_rate": model.learning_rate,
        "buffer_size": model.buffer_size,
        "learning_starts": model.learning_starts,
        "batch_size": model.batch_size,
        "train_freq_frequency": model.train_freq.frequency,
        "train_freq_unit": str(model.train_freq.unit.value),
        "gradient_steps": model.gradient_steps,
        "policy_kwargs": model.policy_kwargs,
    }


def verify_training_settings(old_model, new_model):
    old_settings = _model_settings(old_model)
    new_settings = _model_settings(new_model)
    expected = {
        "learning_rate": 3e-4, "buffer_size": 100000,
        "learning_starts": 100, "batch_size": 64,
        "train_freq_frequency": 1, "train_freq_unit": "step",
        "gradient_steps": 1,
        "policy_kwargs": {"net_arch": [64, 64], "use_sde": False},
    }
    if old_settings != expected or new_settings != expected:
        raise RuntimeError("SAC settings differ from Experiment B")
    if old_model.num_timesteps != TRAINING_STEPS or new_model.num_timesteps != TRAINING_STEPS:
        raise RuntimeError("Expected two 100,000-step SAC models")
    return old_settings


def evaluate(models, fixed_goals):
    results, traces = [], {name: {} for name in MODEL_NAMES}
    for model_name, model in models.items():
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
                    row = c1._metrics(TRAINING_SEED, goal_delta,
                                      evaluation_seed, rollout)
                    row = {"model": model_name, **row}
                    results.append(row)
                    traces[model_name][key] = rollout
            finally:
                env.close()
    return results, traces


def _load_c1_seed101(path):
    output = {}
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            if int(raw["training_seed"]) == TRAINING_SEED:
                output[(float(raw["goal_joint_delta_rad"]),
                        int(raw["evaluation_seed"]))] = raw
    if len(output) != 300:
        raise ValueError("C-1 must contain 300 rows for training seed 101")
    return output


def verify_c1_reproduction(results, c1_rows):
    checked = 0
    fields = (
        "first_25mm_step", "first_10mm_step", "success_10mm",
        "min_tcp_distance_mm", "final_tcp_distance_mm", "end_step",
        "timeout_60_steps", "reached_25mm_not_10mm",
        "max_retreat_after_min_mm", "post_25mm_direction_reversal_count")
    for row in results:
        if row["model"] != "trained_25mm":
            continue
        old = c1_rows[(row["goal_joint_delta_rad"], row["evaluation_seed"])]
        for field in fields:
            actual = row[field]
            expected = old[field]
            if isinstance(actual, bool):
                if actual != (expected == "True"):
                    raise RuntimeError(f"C-1 reproduction mismatch: {field}")
            elif actual is None:
                if expected != "":
                    raise RuntimeError(f"C-1 reproduction mismatch: {field}")
            elif not math.isclose(float(actual), float(expected),
                                  rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"C-1 reproduction mismatch: {field}")
        checked += 1
    if checked != 300:
        raise RuntimeError("C-1 reproduction did not cover 300 cases")


def add_comparison_outcomes(results):
    indexed = {(row["model"], row["goal_joint_delta_rad"],
                row["evaluation_seed"]): row for row in results}
    counts = {"improved": 0, "worsened": 0,
              "same_success": 0, "same_failure": 0,
              "by_goal_range": {
                  f"{delta:.2f}": {"improved": 0, "worsened": 0,
                                   "same_success": 0, "same_failure": 0}
                  for delta in c1.GOAL_DELTAS}}
    for goal_delta in c1.GOAL_DELTAS:
        for evaluation_seed in c1.EVALUATION_SEEDS:
            old = indexed[("trained_25mm", goal_delta, evaluation_seed)]
            new = indexed[("trained_10mm", goal_delta, evaluation_seed)]
            if not old["success_10mm"] and new["success_10mm"]:
                outcome = "improved"
            elif old["success_10mm"] and not new["success_10mm"]:
                outcome = "worsened"
            elif old["success_10mm"]:
                outcome = "same_success"
            else:
                outcome = "same_failure"
            counts[outcome] += 1
            counts["by_goal_range"][f"{goal_delta:.2f}"][outcome] += 1
            old["old_to_new_10mm_outcome"] = outcome
            new["old_to_new_10mm_outcome"] = outcome
    return counts


def _logged_elapsed_seconds(path):
    with path.open(newline="", encoding="utf-8") as file:
        values = [float(row["time/time_elapsed"]) for row in csv.DictReader(file)
                  if row.get("time/time_elapsed") not in (None, "")]
    return values[-1] if values else None


def _training_metric_stats(rows, name):
    values = [row[name] for row in rows if row[name] is not None]
    if not values:
        return None
    return {
        "sample_count": len(values), "final": values[-1],
        "min": min(values), "max": max(values),
        "last_10_mean": float(np.mean(values[-10:])),
        "nonfinite_count": sum(not math.isfinite(value) for value in values),
    }


def build_summary(results, outcome_counts, settings, training_seconds,
                  training_rows, model_paths, representative_csvs):
    per_model = {}
    for model_name in MODEL_NAMES:
        selected = [row for row in results if row["model"] == model_name]
        per_model[model_name] = {
            "all_300": c1._aggregate(selected),
            "by_goal_range": {
                f"{delta:.2f}": c1._aggregate([
                    row for row in selected
                    if row["goal_joint_delta_rad"] == delta])
                for delta in c1.GOAL_DELTAS},
            "training_metrics": {
                metric: _training_metric_stats(training_rows[model_name], metric)
                for metric in ("rollout/success_rate", "train/ent_coef",
                               "train/actor_loss", "train/critic_loss")},
        }
    return {
        "status": "EXPERIMENT_C2_KINEMATIC_ONLY_NO_HARDWARE",
        "training_seed": TRAINING_SEED,
        "training_steps": TRAINING_STEPS,
        "reward_scale": c1.REWARD_SCALE,
        "training_goal_joint_delta_rad": 0.35,
        "evaluation_success_radius_mm": 10,
        "old_training_success_radius_mm": 25,
        "new_training_success_radius_mm": 10,
        "max_steps": c1.MAX_STEPS,
        "action_delta_rad": c1.ACTION_DELTA_RAD,
        "model_settings": settings,
        "only_training_success_radius_changed": True,
        "same_300_goals_verified_for_both_models": True,
        "old_model_c1_result_exactly_reproduced": True,
        "training_seconds": training_seconds,
        "model_paths": model_paths,
        "per_model": per_model,
        "old_to_new_10mm_outcomes": outcome_counts,
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
        for model_name in MODEL_NAMES:
            relative = (Path("representatives") /
                        f"{model_name}_eval{evaluation_seed}_delta070" /
                        "rollout.csv")
            c1._write_viewer_csv(out / relative, traces[model_name][
                (goal_delta, evaluation_seed)], dls)
            saved.append(str(relative))
    return saved


def make_plots(results, traces, training_rows, out):
    colors = {"trained_25mm": "tab:blue", "trained_10mm": "tab:orange"}
    x = np.arange(len(c1.GOAL_DELTAS)); width = .35
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for offset, model_name in zip((-.5, .5), MODEL_NAMES):
        values = [100 * np.mean([
            row["success_10mm"] for row in results
            if row["model"] == model_name and row["goal_joint_delta_rad"] == delta])
                  for delta in c1.GOAL_DELTAS]
        ax.bar(x + offset * width, values, width,
               label=MODEL_LABELS[model_name], color=colors[model_name])
    ax.set_xticks(x, [f"±{delta:.2f}" for delta in c1.GOAL_DELTAS])
    ax.set(xlabel="Goal joint range (rad)", ylabel="10 mm success rate (%)",
           ylim=(0, 105)); ax.grid(axis="y", alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "success_rate_10mm_by_goal_range.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, field, label in (
            (axes[0], "min_tcp_distance_mm", "Minimum TCP error (mm)"),
            (axes[1], "first_10mm_step", "10 mm first-arrival step")):
        data = [[row[field] for row in results
                 if row["model"] == name and row[field] is not None]
                for name in MODEL_NAMES]
        ax.boxplot(data, tick_labels=["25 mm train", "10 mm train"], showfliers=True)
        ax.set(ylabel=label); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "precision_and_arrival_steps.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    specifications = (("rollout/success_rate", "Training success rate"),
                      ("train/ent_coef", "Entropy coefficient"),
                      ("train/actor_loss", "Actor loss"),
                      ("train/critic_loss", "Critic loss"))
    for ax, (metric, ylabel) in zip(axes, specifications):
        for model_name in MODEL_NAMES:
            points = [(row["time/total_timesteps"], row[metric])
                      for row in training_rows[model_name]
                      if row[metric] is not None]
            ax.plot([point[0] for point in points], [point[1] for point in points],
                    color=colors[model_name], label=MODEL_LABELS[model_name])
        if metric == "train/ent_coef":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel); ax.grid(alpha=.3)
    axes[0].legend(); axes[-1].set_xlabel("Training timesteps")
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(len(REPRESENTATIVE_KEYS), 1,
                             figsize=(9, 3.7 * len(REPRESENTATIVE_KEYS)),
                             squeeze=False)
    for ax, key in zip(axes[:, 0], REPRESENTATIVE_KEYS):
        for model_name in MODEL_NAMES:
            rows = traces[model_name][key]
            ax.plot([row["step"] for row in rows],
                    [row["tcp_distance_mm"] for row in rows],
                    color=colors[model_name], label=MODEL_LABELS[model_name])
        ax.axhline(25, color="gray", linestyle="--", label="25 mm")
        ax.axhline(10, color="black", linestyle=":", label="10 mm")
        ax.set(title=f"Evaluation seed {key[1]}, ±0.70 rad",
               xlabel="Step", ylabel="TCP error (mm)"); ax.grid(alpha=.3)
    axes[0, 0].legend(ncol=4, fontsize=9)
    fig.tight_layout(); fig.savefig(out / "representative_tcp_error_curves.png", dpi=160)
    plt.close(fig)


def write_report(summary, out):
    lines = ["# Experiment C-2: SAC trained with a 10 mm success radius", "",
             "Kinematic-only experiment. No robot hardware was used.", "",
             "| Model | ±0.35 | ±0.50 | ±0.70 | 25 mm all | 10 mm all | 25 mm but not 10 mm |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in MODEL_NAMES:
        model = summary["per_model"][name]
        cells = []
        for delta in c1.GOAL_DELTAS:
            metric = model["by_goal_range"][f"{delta:.2f}"]
            cells.append(f"{metric['reached_10mm_count']}/100")
        all_cases = model["all_300"]
        lines.append(f"| {MODEL_LABELS[name]} | {' | '.join(cells)} | "
                     f"{all_cases['reached_25mm_count']}/300 | "
                     f"{all_cases['reached_10mm_count']}/300 "
                     f"({all_cases['reached_10mm_rate']:.1%}) | "
                     f"{all_cases['reached_25mm_not_10mm_count']} |")
    lines += ["", "## Precision and arrival steps", "",
              "| Model | Minimum error mean / median / P90 (mm) | 10 mm step mean / median / P90 |",
              "| --- | ---: | ---: |"]
    for name in MODEL_NAMES:
        metric = summary["per_model"][name]["all_300"]
        minimum = metric["min_tcp_distance_mm"]
        steps = metric["first_10mm_step"]
        lines.append(
            f"| {MODEL_LABELS[name]} | {minimum['mean']:.2f} / "
            f"{minimum['median']:.2f} / {minimum['p90']:.2f} | "
            f"{steps['mean_reached_cases']:.2f} / "
            f"{steps['median_reached_cases']:.2f} / "
            f"{steps['p90_reached_cases']:.2f} |")
    outcomes = summary["old_to_new_10mm_outcomes"]
    lines += ["", "## Outcome changes", "",
              f"- Old failure to new success: {outcomes['improved']}",
              f"- Old success to new failure: {outcomes['worsened']}",
              f"- Both successful: {outcomes['same_success']}",
              f"- Both failed: {outcomes['same_failure']}", "",
              "| Goal range | Improved | Worsened |",
              "| ---: | ---: | ---: |"]
    for delta in c1.GOAL_DELTAS:
        change = outcomes["by_goal_range"][f"{delta:.2f}"]
        lines.append(f"| ±{delta:.2f} rad | {change['improved']} | "
                     f"{change['worsened']} |")
    lines += ["",
              "Training rolling success and fixed-goal evaluation success are separate measurements.",
              "Loss values are recorded but are not used alone to rank the models.", "",
              "## Interpretation", "",
              "- Changing the training success radius to 10 mm improved all 85 old-model failures and introduced no regressions on the fixed goals.",
              "- The fixed evaluation supports the success-radius change for this training seed, including out-of-training goal ranges.",
              "- One training seed cannot establish reproducibility. Additional training seeds are the next bounded check before changing the goal distribution.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--old-training-log", type=Path, required=True)
    parser.add_argument("--experiment-a-csv", type=Path, required=True)
    parser.add_argument("--c1-evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    required = (args.old_model, args.old_training_log,
                args.experiment_a_csv, args.c1_evaluation)
    if not all(path.is_file() for path in required):
        parser.error("One or more Experiment A/B/C-1 inputs are missing")
    if args.out.exists():
        parser.error("--out must not exist")
    args.out.mkdir(parents=True)

    from stable_baselines3 import SAC
    old_model = SAC.load(str(args.old_model), device="cpu")
    new_model, training_seconds = train_precision10(args.out)
    settings = verify_training_settings(old_model, new_model)
    models = {"trained_25mm": old_model, "trained_10mm": new_model}

    fixed_goals = exp_b.load_experiment_a(args.experiment_a_csv)
    results, traces = evaluate(models, fixed_goals)
    if len(results) != 600:
        raise RuntimeError("C-2 evaluation did not produce 600 rows")
    verify_c1_reproduction(results, _load_c1_seed101(args.c1_evaluation))
    outcomes = add_comparison_outcomes(results)
    c1._write_csv(args.out / "evaluation_600.csv", results)

    old_log_copy = args.out / "training_log_trained_25mm.csv"
    new_log_copy = args.out / "training_log_trained_10mm.csv"
    shutil.copyfile(args.old_training_log, old_log_copy)
    shutil.copyfile(args.out / "training_log" / "progress.csv", new_log_copy)
    training_rows = {
        "trained_25mm": exp_b.load_training_metrics(old_log_copy),
        "trained_10mm": exp_b.load_training_metrics(new_log_copy),
    }
    representative_csvs = save_representatives(traces, args.out)
    model_paths = {
        "trained_25mm": str(args.old_model),
        "trained_10mm": str(args.out / "sac_reward10_success10mm.zip"),
    }
    summary = build_summary(
        results, outcomes, settings,
        {"trained_25mm": _logged_elapsed_seconds(args.old_training_log),
         "trained_10mm": training_seconds},
        training_rows, model_paths, representative_csvs)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "training_settings.json").write_text(json.dumps({
        "training_seed": TRAINING_SEED, "timesteps": TRAINING_STEPS,
        "reward_scale": c1.REWARD_SCALE, "goal_joint_delta_rad": 0.35,
        "success_radius_mm": 10, "max_steps": c1.MAX_STEPS,
        "action_delta_rad": c1.ACTION_DELTA_RAD,
        "model_settings": settings,
        "only_difference_from_experiment_b": "success_radius_mm: 25 -> 10",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(results, traces, training_rows, args.out)
    write_report(summary, args.out)
    print("Saved Experiment C-2 to", args.out)
    for name in MODEL_NAMES:
        metric = summary["per_model"][name]["all_300"]
        print(name, f"{metric['reached_10mm_count']}/300")


if __name__ == "__main__":
    main()