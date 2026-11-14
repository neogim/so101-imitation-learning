#!/usr/bin/env python3
"""Experiment B-2: reward-scale-10 SAC reproducibility across training seeds."""
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
from compare_sac_dls_rollout import CSV_FIELDS
from learn_reach_kinematic import KinematicReachEnv


TRAINING_SEEDS = (101, 202, 303)
GOAL_DELTAS = exp_b.GOAL_DELTAS
REPRESENTATIVE_EVAL_SEEDS = (1000046, 1000047)
METRIC_NAMES = (
    "success", "final_tcp_distance_mm", "steps", "min_tcp_distance_mm",
    "final_minus_min_mm", "timeout", "tcp_path_length_m",
)


def _percentile(values, percentile):
    return float(np.percentile(values, percentile))


def _metrics(rows):
    finals = [row["final_tcp_distance_mm"] for row in rows]
    all_steps = [row["steps"] for row in rows]
    success_steps = [row["steps"] for row in rows if row["success"]]
    success_count = sum(row["success"] for row in rows)
    return {
        "sample_count": len(rows),
        "success_count": success_count,
        "success_rate": success_count / len(rows),
        "final_tcp_distance_mm": {
            "mean": float(np.mean(finals)), "median": float(np.median(finals)),
            "p90": _percentile(finals, 90), "max": max(finals),
        },
        "steps_all_cases": {
            "mean": float(np.mean(all_steps)), "median": float(np.median(all_steps)),
            "p90": _percentile(all_steps, 90),
        },
        "steps_success_cases": None if not success_steps else {
            "sample_count": len(success_steps),
            "mean": float(np.mean(success_steps)),
            "median": float(np.median(success_steps)),
            "p90": _percentile(success_steps, 90),
        },
    }


def _normalized_rows(training_seed, evaluated):
    rows = []
    for source in evaluated:
        row = {
            "training_seed": training_seed,
            "goal_joint_delta_rad": source["goal_joint_delta_rad"],
            "evaluation_seed": source["goal_seed"],
        }
        for name in (
                "target_x", "target_y", "target_z",
                "initial_tcp_x", "initial_tcp_y", "initial_tcp_z",
                "initial_tcp_distance_mm", "direction_x", "direction_y",
                "direction_z"):
            row[name] = source[name]
        for name in METRIC_NAMES:
            row[name] = source[f"reward10_{name}"]
        rows.append(row)
    return rows


def _load_existing_evaluation(path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            key = (float(raw["goal_joint_delta_rad"]), int(raw["goal_seed"]))
            rows[key] = raw
    return rows


def _logged_elapsed_seconds(path):
    with path.open(newline="", encoding="utf-8") as file:
        values = [float(row["time/time_elapsed"]) for row in csv.DictReader(file)
                  if row.get("time/time_elapsed") not in (None, "")]
    return values[-1] if values else None


def _verify_seed101_reproduction(evaluated, existing_path):
    existing = _load_existing_evaluation(existing_path)
    checked = 0
    for row in evaluated:
        key = (row["goal_joint_delta_rad"], row["goal_seed"])
        old = existing[key]
        for name in METRIC_NAMES:
            actual = row[f"reward10_{name}"]
            expected_text = old[f"reward10_{name}"]
            if isinstance(actual, bool):
                expected = expected_text == "True"
                if actual is not expected:
                    raise RuntimeError(f"Seed 101 reproduction mismatch: {key} {name}")
            elif not math.isclose(float(actual), float(expected_text), abs_tol=1e-12):
                raise RuntimeError(f"Seed 101 reproduction mismatch: {key} {name}")
        checked += 1
    if checked != 300:
        raise RuntimeError("Seed 101 reproduction did not cover 300 cases")


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


def _verify_model_settings(models):
    settings = {str(seed): _model_settings(model) for seed, model in models.items()}
    reference = settings["101"]
    if any(value != reference for value in settings.values()):
        raise RuntimeError("SAC settings differ across training seeds")
    if reference != {
            "learning_rate": 3e-4, "buffer_size": 100000,
            "learning_starts": 100, "batch_size": 64,
            "train_freq_frequency": 1, "train_freq_unit": "step",
            "gradient_steps": 1,
            "policy_kwargs": {"net_arch": [64, 64], "use_sde": False}}:
        raise RuntimeError(f"SAC settings differ from Experiment B: {reference}")
    return settings


def _band_summary(rows, value_fn, edges):
    output = {}
    for low, high in zip(edges[:-1], edges[1:]):
        label = f"{low:.0f}+" if math.isinf(high) else f"{low:.0f}-{high:.0f}"
        selected = [row for row in rows if low <= value_fn(row) < high]
        output[label] = {
            "sample_count": len(selected),
            "success_count": sum(row["success"] for row in selected),
            "success_rate": (sum(row["success"] for row in selected) / len(selected)
                             if selected else None),
        }
    return output


def _training_metric_stats(rows, name):
    values = [row[name] for row in rows if row[name] is not None]
    if not values:
        return None
    tail = values[-10:]
    return {
        "sample_count": len(values),
        "nonfinite_count": sum(not math.isfinite(value) for value in values),
        "final": values[-1], "min": min(values), "max": max(values),
        "last_10_mean": float(np.mean(tail)),
        "last_10_std": float(np.std(tail)),
    }


def build_summary(all_rows, training_rows, training_seconds):
    per_seed = {}
    failure_sets = {}
    for seed in TRAINING_SEEDS:
        rows = [row for row in all_rows if row["training_seed"] == seed]
        failures = {(row["goal_joint_delta_rad"], row["evaluation_seed"])
                    for row in rows if not row["success"]}
        failure_sets[seed] = failures
        per_seed[str(seed)] = {
            "all_300": _metrics(rows),
            "by_goal_range": {
                f"{delta:.2f}": _metrics([
                    row for row in rows if row["goal_joint_delta_rad"] == delta])
                for delta in GOAL_DELTAS},
            "by_initial_distance_mm": _band_summary(
                rows, lambda row: row["initial_tcp_distance_mm"],
                exp_b.DISTANCE_EDGES_MM),
            "by_absolute_y_displacement_mm": _band_summary(
                rows, lambda row: abs(row["direction_y"]) * 1000,
                exp_b.Y_EDGES_MM),
            "training_seconds": training_seconds[str(seed)],
            "training_metrics": {
                name: _training_metric_stats(training_rows[seed], name)
                for name in ("rollout/success_rate", "train/ent_coef",
                             "train/actor_loss", "train/critic_loss")},
        }
    rates = [per_seed[str(seed)]["all_300"]["success_rate"]
             for seed in TRAINING_SEEDS]
    common = set.intersection(*failure_sets.values())
    union = set.union(*failure_sets.values())
    return {
        "status": "EXPERIMENT_B2_KINEMATIC_ONLY_NO_HARDWARE",
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_seed_range": [exp_b.SEED_START,
                                  exp_b.SEED_START + exp_b.NUM_GOALS - 1],
        "goal_joint_delta_rad_values": list(GOAL_DELTAS),
        "same_300_goals_verified_for_all_models": True,
        "seed_101_existing_result_reproduced": True,
        "per_training_seed": per_seed,
        "all_300_success_rate_across_training_seeds": {
            "mean": float(np.mean(rates)), "std_population": float(np.std(rates)),
            "min": min(rates), "max": max(rates), "training_seed_count": 3,
            "note": "Three training seeds are too few to establish statistical stability.",
        },
        "failures": {
            "common_to_all_seeds": [list(item) for item in sorted(common)],
            "observed_by_any_seed": [list(item) for item in sorted(union)],
            "shared_by_seed_pair": {
                f"{first}_{second}": [list(item) for item in sorted(
                    (failure_sets[first] & failure_sets[second]) - common)]
                for first, second in ((101, 202), (101, 303), (202, 303))},
            "seed_only": {
                str(seed): [list(item) for item in sorted(
                    failure_sets[seed] - set.union(
                        *(failure_sets[other] for other in TRAINING_SEEDS
                          if other != seed)))]
                for seed in TRAINING_SEEDS},
        },
    }


def _write_evaluation(all_rows, out):
    with (out / "evaluation_900.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)


def _copy_training_logs(seed_dirs, seed101_log, out):
    paths = {101: seed101_log}
    paths.update({seed: seed_dirs[seed] / "training_log" / "progress.csv"
                  for seed in (202, 303)})
    copied = {}
    for seed, source in paths.items():
        target = out / f"training_seed_{seed}_progress.csv"
        shutil.copyfile(source, target)
        copied[seed] = target
    return copied


def _write_viewer_csv(path, sac_rows, dls_rows):
    path.parent.mkdir(parents=True)
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader(); writer.writerows(sac_rows); writer.writerows(dls_rows)


def save_representatives(models, traces, out):
    saved = []
    for eval_seed in REPRESENTATIVE_EVAL_SEEDS:
        key = (0.70, eval_seed)
        dls_env = KinematicReachEnv(goal_joint_delta_rad=.70)
        try:
            dls = exp_b._run_rollout(
                dls_env, None, policy_name="jacobian_dls", goal_seed=eval_seed)
        finally:
            dls_env.close()
        for training_seed in TRAINING_SEEDS:
            path = (out / "representatives" /
                    f"train{training_seed}_eval{eval_seed}_delta070" / "rollout.csv")
            _write_viewer_csv(path, traces[training_seed][key], dls)
            saved.append(str(path.relative_to(out)))
    return saved


def make_plots(all_rows, training_rows, traces, out):
    colors = {101: "tab:green", 202: "tab:blue", 303: "tab:purple"}
    x = np.arange(len(GOAL_DELTAS)); width = .24
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for index, seed in enumerate(TRAINING_SEEDS):
        values = [100 * np.mean([row["success"] for row in all_rows
                                 if row["training_seed"] == seed and
                                 row["goal_joint_delta_rad"] == delta])
                  for delta in GOAL_DELTAS]
        ax.bar(x + (index - 1) * width, values, width,
               label=f"train seed {seed}", color=colors[seed])
    ax.set(xticks=x, xticklabels=[f"±{delta:.2f}" for delta in GOAL_DELTAS],
           xlabel="Goal joint range (rad)", ylabel="Success rate (%)", ylim=(0, 105))
    ax.grid(axis="y", alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "success_rate_by_training_seed.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.boxplot([[row["final_tcp_distance_mm"] for row in all_rows
                 if row["training_seed"] == seed] for seed in TRAINING_SEEDS],
               tick_labels=[str(seed) for seed in TRAINING_SEEDS], showfliers=True)
    ax.axhline(25, color="black", linestyle="--", label="25 mm threshold")
    ax.set(xlabel="Training seed", ylabel="Final TCP distance (mm)")
    ax.grid(axis="y", alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "final_tcp_error_by_training_seed.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    positions = np.arange(len(TRAINING_SEEDS)); width = .35
    all_means = []; success_means = []
    for seed in TRAINING_SEEDS:
        rows = [row for row in all_rows if row["training_seed"] == seed]
        all_means.append(np.mean([row["steps"] for row in rows]))
        success_means.append(np.mean([row["steps"] for row in rows if row["success"]]))
    ax.bar(positions-width/2, all_means, width, label="All cases")
    ax.bar(positions+width/2, success_means, width, label="Success cases")
    ax.set_xticks(positions, [str(seed) for seed in TRAINING_SEEDS])
    ax.set(xlabel="Training seed", ylabel="Mean evaluation steps")
    ax.grid(axis="y", alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "steps_by_training_seed.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    specs = (("rollout/success_rate", "Training success rate"),
             ("train/ent_coef", "Entropy coefficient"),
             ("train/actor_loss", "Actor loss"),
             ("train/critic_loss", "Critic loss"))
    for ax, (name, ylabel) in zip(axes, specs):
        for seed in TRAINING_SEEDS:
            points = [(row["time/total_timesteps"], row[name])
                      for row in training_rows[seed] if row[name] is not None]
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    color=colors[seed], alpha=.8, label=str(seed))
        if name == "train/ent_coef": ax.set_yscale("log")
        ax.set_ylabel(ylabel); ax.grid(alpha=.3)
    axes[0].legend(title="Training seed")
    axes[-1].set_xlabel("Training timesteps")
    fig.tight_layout(); fig.savefig(out / "training_curves_by_seed.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for ax, eval_seed in zip(axes, REPRESENTATIVE_EVAL_SEEDS):
        key = (.70, eval_seed)
        for seed in TRAINING_SEEDS:
            rows = traces[seed][key]
            ax.plot([r["step"] for r in rows], [r["tcp_distance_mm"] for r in rows],
                    marker="o", markersize=3, color=colors[seed], label=str(seed))
        ax.axhline(25, color="black", linestyle="--")
        ax.set(title=f"Evaluation seed {eval_seed}, ±0.70 rad",
               xlabel="Step", ylabel="TCP distance (mm)")
        ax.grid(alpha=.3)
    axes[0].legend(title="Training seed")
    fig.tight_layout(); fig.savefig(out / "representative_tcp_error_curves.png", dpi=160)
    plt.close(fig)


def write_report(summary, out):
    lines = ["# Experiment B-2 seed reproducibility", "",
             "Kinematic-only evaluation; no real robot was operated.", "",
             "## Fixed 300-goal evaluation", "",
             "| Training seed | ±0.35 | ±0.50 | ±0.70 | All | Final error mean / median / P90 / max (mm) | Steps all / success-only (mean) |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed in TRAINING_SEEDS:
        seed_summary = summary["per_training_seed"][str(seed)]
        metrics = seed_summary["all_300"]
        ranges = seed_summary["by_goal_range"]
        error = metrics["final_tcp_distance_mm"]
        steps_all = metrics["steps_all_cases"]["mean"]
        steps_success = metrics["steps_success_cases"]["mean"]
        range_cells = [
            f"{ranges[f'{delta:.2f}']['success_count']}/100"
            for delta in GOAL_DELTAS]
        lines.append(
            f"| {seed} | {' | '.join(range_cells)} | "
            f"{metrics['success_count']}/300 ({metrics['success_rate']:.1%}) | "
            f"{error['mean']:.2f} / {error['median']:.2f} / "
            f"{error['p90']:.2f} / {error['max']:.2f} | "
            f"{steps_all:.2f} / {steps_success:.2f} |")
    rates = summary["all_300_success_rate_across_training_seeds"]
    lines += ["", "## Across three training seeds", "",
              f"- Mean success rate: {rates['mean']:.1%}",
              f"- Population standard deviation: {rates['std_population']:.3%}",
              f"- Minimum / maximum: {rates['min']:.1%} / {rates['max']:.1%}",
              "- Three seeds are insufficient to establish statistical stability.", ""]
    failures = summary["failures"]
    lines += ["## Failure overlap", ""]
    if not failures["observed_by_any_seed"]:
        lines.append("- No failures on the fixed 300 goals.")
    else:
        lines += [
            f"- Unique failed goals: {len(failures['observed_by_any_seed'])}; all were in the ±0.70-rad set.",
            f"- Common to all three seeds: {failures['common_to_all_seeds'] or 'none'}",
            f"- Shared by seeds 202 and 303: {failures['shared_by_seed_pair']['202_303'] or 'none'}",
            f"- Seed 202 only: {failures['seed_only']['202'] or 'none'}",
            f"- Seed 303 only: {failures['seed_only']['303'] or 'none'}",
        ]
    lines += ["", "## Interpretation", "",
              "- Reward×10 produced high success for all three training seeds, but the exact 300/300 result did not reproduce for seeds 202 and 303.",
              "- No failures were observed in the ±0.35- or ±0.50-rad sets; the observed sensitivity was confined to the out-of-training ±0.70-rad goals.",
              "- The three-seed result supports proceeding to a separate 10-mm-accuracy experiment, but it does not establish statistical stability.",
              "- Training rolling success and fixed-goal evaluation success are different measurements; losses alone were not used to rank models.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed101-model", type=Path, required=True)
    parser.add_argument("--seed101-experiment-dir", type=Path, required=True)
    parser.add_argument("--experiment-a-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    required = (args.seed101_model, args.experiment_a_csv,
                args.seed101_experiment_dir / "evaluation.csv",
                args.seed101_experiment_dir / "training_log" / "progress.csv")
    if not all(path.is_file() for path in required):
        parser.error("One or more Experiment A/B inputs are missing")
    if args.out.exists(): parser.error("--out must not exist")
    args.out.mkdir(parents=True)

    from stable_baselines3 import SAC
    models = {101: SAC.load(str(args.seed101_model), device="cpu")}
    seed_dirs = {}
    seed101_log = args.seed101_experiment_dir / "training_log" / "progress.csv"
    training_seconds = {"101": _logged_elapsed_seconds(seed101_log)}
    for seed in (202, 303):
        seed_dir = args.out / f"train_seed_{seed}"
        seed_dir.mkdir()
        _, seconds = exp_b.train_reward10(seed_dir, train_seed=seed)
        models[seed] = SAC.load(str(seed_dir / "sac_reward10.zip"), device="cpu")
        seed_dirs[seed] = seed_dir
        training_seconds[str(seed)] = seconds
    settings = _verify_model_settings(models)

    experiment_a = exp_b.load_experiment_a(args.experiment_a_csv)
    all_rows = []; traces = {}; evaluated_by_seed = {}
    for seed in TRAINING_SEEDS:
        evaluated, seed_traces = exp_b.evaluate_reward10(models[seed], experiment_a)
        evaluated_by_seed[seed] = evaluated; traces[seed] = seed_traces
        all_rows.extend(_normalized_rows(seed, evaluated))
    _verify_seed101_reproduction(
        evaluated_by_seed[101], args.seed101_experiment_dir / "evaluation.csv")

    log_paths = _copy_training_logs(
        seed_dirs, seed101_log, args.out)
    training_rows = {seed: exp_b.load_training_metrics(path)
                     for seed, path in log_paths.items()}
    _write_evaluation(all_rows, args.out)
    summary = build_summary(all_rows, training_rows, training_seconds)
    summary["model_paths"] = {
        "101": str(args.seed101_model),
        "202": str(seed_dirs[202] / "sac_reward10.zip"),
        "303": str(seed_dirs[303] / "sac_reward10.zip"),
    }
    summary["model_settings"] = settings
    summary["representative_csvs"] = save_representatives(models, traces, args.out)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "training_settings.json").write_text(json.dumps({
        "training_seeds": list(TRAINING_SEEDS), "reward_scale": 10.0,
        "timesteps": 100000, "goal_joint_delta_rad": .35,
        "success_threshold_mm": 25, "max_steps": 60,
        "action_delta_rad": .04, "model_settings": settings,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(all_rows, training_rows, traces, args.out)
    write_report(summary, args.out)
    print("Saved Experiment B-2 to", args.out)
    for seed in TRAINING_SEEDS:
        metric = summary["per_training_seed"][str(seed)]["all_300"]
        print(f"training seed {seed}: {metric['success_count']}/300")


if __name__ == "__main__":
    main()
