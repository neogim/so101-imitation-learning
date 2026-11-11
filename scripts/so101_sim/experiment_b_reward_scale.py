#!/usr/bin/env python3
"""Train reward-scale-10 SAC and compare it with the saved baseline SAC.

Experiment B changes only the kinematic reach reward multiplier. The saved
baseline model and Experiment A's fixed 300-case goal table are reused. This
script never accesses robot hardware and does not run MuJoCo dynamics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_sac_dls_rollout import CSV_FIELDS
from learn_reach_kinematic import KinematicReachEnv, jacobian_reach_action


GOAL_DELTAS = (0.35, 0.50, 0.70)
SEED_START = 1000001
NUM_GOALS = 100
MAX_STEPS = 60
ACTION_DELTA_RAD = 0.04
SUCCESS_RADIUS_M = 0.025
REWARD_SCALE = 10.0
DISTANCE_EDGES_MM = (50, 75, 100, 125, 150, 175, 200, math.inf)
Y_EDGES_MM = (0, 25, 50, 75, 100, math.inf)


class ProgressPrinter:
    """Create a small SB3 callback that prints every 10k environment steps."""

    @staticmethod
    def build():
        from stable_baselines3.common.callbacks import BaseCallback

        class Callback(BaseCallback):
            def _on_step(self):
                if self.num_timesteps % 10000 == 0:
                    print(f"training step {self.num_timesteps}/100000", flush=True)
                return True

        return Callback()


def train_reward10(out, timesteps=100000, train_seed=101):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure

    torch.set_num_threads(1)
    env = KinematicReachEnv(
        seed=train_seed, max_steps=MAX_STEPS,
        action_delta_rad=ACTION_DELTA_RAD,
        goal_joint_delta_rad=0.35,
        success_radius_m=SUCCESS_RADIUS_M,
        reward_scale=REWARD_SCALE)
    try:
        model = SAC(
            "MlpPolicy", env, seed=train_seed, device="cpu",
            learning_rate=3e-4, buffer_size=max(10000, timesteps),
            learning_starts=min(100, max(1, timesteps // 4)),
            batch_size=64, train_freq=1, gradient_steps=1,
            policy_kwargs={"net_arch": [64, 64]}, verbose=0)
        log_dir = out / "training_log"
        log_dir.mkdir()
        model.set_logger(configure(str(log_dir), ["csv"]))
        start = time.perf_counter()
        model.learn(total_timesteps=timesteps, callback=ProgressPrinter.build())
        seconds = time.perf_counter() - start
        model.save(str(out / "sac_reward10"))
        return model, seconds
    finally:
        env.close()


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
    row.update({f"joint_{i + 1}": float(value) for i, value in enumerate(
                env.data.qpos[env.qids])})
    row.update({f"action_{i + 1}": float(value) for i, value in enumerate(action)})
    return row


def _run_rollout(env, policy, *, policy_name, goal_seed):
    obs, _ = env.reset(seed=goal_seed)
    rows = [_record(env, policy=policy_name, goal_seed=goal_seed, step=0,
                    action=np.zeros(5))]
    for step in range(1, MAX_STEPS + 1):
        if policy_name == "jacobian_dls":
            action = jacobian_reach_action(env)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        rows.append(_record(env, policy=policy_name, goal_seed=goal_seed,
                            step=step, action=np.asarray(action, dtype=float)))
        if terminated or truncated:
            break
    return rows


def _rollout_metrics(rows):
    distances = [row["tcp_distance_mm"] for row in rows]
    success = bool(rows[-1]["success"])
    points = np.array([[row[f"tcp_{axis}"] for axis in "xyz"] for row in rows])
    return {
        "success": success,
        "final_tcp_distance_mm": distances[-1],
        "steps": int(rows[-1]["step"]),
        "min_tcp_distance_mm": min(distances),
        "final_minus_min_mm": distances[-1] - min(distances),
        "timeout": not success and rows[-1]["step"] == MAX_STEPS,
        "tcp_path_length_m": float(
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
    }


def load_experiment_a(path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            key = (float(raw["goal_joint_delta_rad"]), int(raw["goal_seed"]))
            row = {}
            for name, value in raw.items():
                if name.endswith(("success", "timeout")):
                    row[name] = value == "True"
                elif name in ("goal_seed", "sac_steps", "dls_steps"):
                    row[name] = int(float(value))
                else:
                    row[name] = float(value)
            rows[key] = row
    expected = len(GOAL_DELTAS) * NUM_GOALS
    if len(rows) != expected:
        raise ValueError(f"Experiment A must contain {expected} cases")
    return rows


def evaluate_reward10(model, experiment_a):
    results = []
    traces = {}
    for goal_delta in GOAL_DELTAS:
        env = KinematicReachEnv(
            max_steps=MAX_STEPS, action_delta_rad=ACTION_DELTA_RAD,
            goal_joint_delta_rad=goal_delta,
            success_radius_m=SUCCESS_RADIUS_M)
        try:
            for index, seed in enumerate(
                    range(SEED_START, SEED_START + NUM_GOALS), start=1):
                key = (goal_delta, seed)
                source = experiment_a[key]
                rollout = _run_rollout(
                    env, model, policy_name="sac", goal_seed=seed)
                first = rollout[0]
                for axis in "xyz":
                    if first[f"tcp_{axis}"] != source[f"initial_tcp_{axis}"]:
                        raise RuntimeError(f"Initial TCP mismatch for {key}")
                    if first[f"target_{axis}"] != source[f"target_{axis}"]:
                        raise RuntimeError(f"Target mismatch for {key}")
                metrics = _rollout_metrics(rollout)
                baseline_success = source["sac_success"]
                reward10_success = metrics["success"]
                if not baseline_success and reward10_success:
                    outcome = "improved"
                elif baseline_success and not reward10_success:
                    outcome = "worsened"
                elif baseline_success:
                    outcome = "same_success"
                else:
                    outcome = "same_failure"
                row = dict(source)
                row.update({f"reward10_{name}": value
                            for name, value in metrics.items()})
                row["baseline_to_reward10"] = outcome
                results.append(row)
                traces[key] = rollout
                if index % 25 == 0:
                    print(f"evaluation delta={goal_delta:.2f}: "
                          f"{index}/{NUM_GOALS}", flush=True)
        finally:
            env.close()
    return results, traces


def _policy_summary(rows, prefix):
    successes = [row[f"{prefix}_success"] for row in rows]
    return {
        "success_count": sum(successes),
        "sample_count": len(rows),
        "success_rate": sum(successes) / len(rows),
        "mean_final_tcp_distance_mm": float(np.mean(
            [row[f"{prefix}_final_tcp_distance_mm"] for row in rows])),
        "median_final_tcp_distance_mm": float(np.median(
            [row[f"{prefix}_final_tcp_distance_mm"] for row in rows])),
        "mean_steps": float(np.mean([row[f"{prefix}_steps"] for row in rows])),
    }


def load_training_metrics(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row = {}
            for name in ("time/total_timesteps", "train/ent_coef",
                         "train/actor_loss", "train/critic_loss",
                         "train/n_updates", "rollout/success_rate"):
                value = raw.get(name, "")
                row[name] = float(value) if value not in (None, "") else None
            if row["time/total_timesteps"] is not None:
                rows.append(row)
    if not rows:
        raise ValueError("SB3 training CSV has no metric rows")
    return rows


def _metric_stats(rows, name):
    values = [row[name] for row in rows if row[name] is not None]
    tail = values[-10:]
    return {
        "sample_count": len(values),
        "nonfinite_count": sum(not math.isfinite(value) for value in values),
        "final": values[-1],
        "min": min(values), "max": max(values),
        "last_10_mean": float(np.mean(tail)),
        "last_10_std": float(np.std(tail)),
    }


def build_summary(results, training_rows, baseline_ent_coef, train_seconds):
    by_range = {}
    for delta in GOAL_DELTAS:
        rows = [row for row in results if row["goal_joint_delta_rad"] == delta]
        by_range[f"{delta:.2f}"] = {
            "baseline": _policy_summary(rows, "sac"),
            "reward10": _policy_summary(rows, "reward10"),
            "dls": _policy_summary(rows, "dls"),
        }
    by_distance = {}
    for low, high in zip(DISTANCE_EDGES_MM[:-1], DISTANCE_EDGES_MM[1:]):
        label = f"{low:.0f}+" if math.isinf(high) else f"{low:.0f}-{high:.0f}"
        rows = [row for row in results
                if low <= row["initial_tcp_distance_mm"] < high]
        by_distance[label] = {
            "sample_count": len(rows),
            "baseline": _policy_summary(rows, "sac"),
            "reward10": _policy_summary(rows, "reward10"),
        }
    by_y = {}
    for low, high in zip(Y_EDGES_MM[:-1], Y_EDGES_MM[1:]):
        label = f"{low:.0f}+" if math.isinf(high) else f"{low:.0f}-{high:.0f}"
        rows = [row for row in results
                if low <= abs(row["direction_y"]) * 1000 < high]
        by_y[label] = {
            "sample_count": len(rows),
            "baseline": _policy_summary(rows, "sac"),
            "reward10": _policy_summary(rows, "reward10"),
        }
    outcomes = {name: sum(row["baseline_to_reward10"] == name for row in results)
                for name in ("improved", "worsened", "same_success", "same_failure")}
    seed_cases = {}
    for seed in (1000046, 1000047):
        rows = [row for row in results if row["goal_seed"] == seed]
        seed_cases[str(seed)] = {
            f"{row['goal_joint_delta_rad']:.2f}": {
                "initial_tcp_distance_mm": row["initial_tcp_distance_mm"],
                "baseline_success": row["sac_success"],
                "baseline_final_tcp_distance_mm": row["sac_final_tcp_distance_mm"],
                "reward10_success": row["reward10_success"],
                "reward10_final_tcp_distance_mm": row["reward10_final_tcp_distance_mm"],
                "reward10_steps": row["reward10_steps"],
                "dls_success": row["dls_success"],
            } for row in rows}
    return {
        "status": "EXPERIMENT_B_KINEMATIC_ONLY_NO_HARDWARE",
        "changed_parameter": {"reward_scale": REWARD_SCALE},
        "unchanged_training_parameters": {
            "timesteps": 100000, "train_seed": 101,
            "goal_joint_delta_rad": 0.35, "learning_rate": 3e-4,
            "buffer_size": 100000, "learning_starts": 100,
            "batch_size": 64, "train_freq": 1, "gradient_steps": 1,
            "network": [64, 64], "action_delta_rad": ACTION_DELTA_RAD,
        },
        "evaluation": {
            "case_count": len(results), "seed_start": SEED_START,
            "seed_end": SEED_START + NUM_GOALS - 1,
            "goal_joint_delta_rad_values": list(GOAL_DELTAS),
            "same_goals_as_experiment_a_verified": True,
        },
        "training_seconds": train_seconds,
        "baseline_final_ent_coef": baseline_ent_coef,
        "baseline_training_history_available": False,
        "reward10_training_metrics": {
            name: _metric_stats(training_rows, name)
            for name in ("rollout/success_rate", "train/ent_coef",
                         "train/actor_loss", "train/critic_loss")
        },
        "by_goal_range": by_range,
        "by_initial_distance_mm": by_distance,
        "by_absolute_y_displacement_mm": by_y,
        "baseline_to_reward10_outcomes": outcomes,
        "selected_seed_results": seed_cases,
    }


def _plot_grouped(labels, baseline, reward10, ylabel, xlabel, path,
                  samples=None, dls=None):
    x = np.arange(len(labels))
    series = [("Baseline", baseline, "tab:blue"),
              ("Reward ×10", reward10, "tab:green")]
    if dls is not None:
        series.append(("DLS", dls, "tab:orange"))
    width = .8 / len(series)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.25), 4.7))
    for index, (name, values, color) in enumerate(series):
        ax.bar(x + (index - (len(series) - 1) / 2) * width,
               values, width, label=name, color=color)
    tick_labels = labels if samples is None else [
        f"{label}\n(n={sample})" for label, sample in zip(labels, samples)]
    ax.set(xticks=x, xticklabels=tick_labels, xlabel=xlabel,
           ylabel=ylabel, ylim=(0, 105))
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_plots(results, training_rows, baseline_ent_coef, out):
    labels = [f"±{delta:.2f}" for delta in GOAL_DELTAS]
    grouped = [[row for row in results if row["goal_joint_delta_rad"] == delta]
               for delta in GOAL_DELTAS]
    rate = lambda rows, prefix: np.mean([row[f"{prefix}_success"] for row in rows]) * 100
    _plot_grouped(labels, [rate(rows, "sac") for rows in grouped],
                  [rate(rows, "reward10") for rows in grouped], "Success rate (%)",
                  "Goal joint range (rad)", out / "success_by_goal_range.png",
                  samples=[len(rows) for rows in grouped],
                  dls=[rate(rows, "dls") for rows in grouped])

    groups, labels, samples = [], [], []
    for low, high in zip(DISTANCE_EDGES_MM[:-1], DISTANCE_EDGES_MM[1:]):
        rows = [row for row in results
                if low <= row["initial_tcp_distance_mm"] < high]
        groups.append(rows); samples.append(len(rows))
        labels.append(f"{low:.0f}+" if math.isinf(high) else f"{low:.0f}-{high:.0f}")
    _plot_grouped(labels, [rate(rows, "sac") for rows in groups],
                  [rate(rows, "reward10") for rows in groups], "Success rate (%)",
                  "Initial TCP distance (mm)", out / "success_by_initial_distance.png",
                  samples=samples)

    groups, labels, samples = [], [], []
    for low, high in zip(Y_EDGES_MM[:-1], Y_EDGES_MM[1:]):
        rows = [row for row in results
                if low <= abs(row["direction_y"]) * 1000 < high]
        groups.append(rows); samples.append(len(rows))
        labels.append(f"{low:.0f}+" if math.isinf(high) else f"{low:.0f}-{high:.0f}")
    _plot_grouped(labels, [rate(rows, "sac") for rows in groups],
                  [rate(rows, "reward10") for rows in groups], "Success rate (%)",
                  "Absolute Y displacement (mm)", out / "success_by_y_displacement.png",
                  samples=samples)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    upper = 100
    bins = np.arange(0, upper + 5, 5)
    for prefix, label, color in (("sac", "Baseline", "tab:blue"),
                                 ("reward10", "Reward ×10", "tab:green")):
        values = [row[f"{prefix}_final_tcp_distance_mm"] for row in results]
        ax.hist([value for value in values if value <= upper], bins=bins,
                alpha=.55, label=label, color=color)
        overflow = sum(value > upper for value in values)
        if overflow:
            ax.text(.98, .82 if prefix == "sac" else .73,
                    f">{upper} mm {label}: n={overflow}", transform=ax.transAxes,
                    ha="right")
    ax.axvline(25, color="black", linestyle="--", label="25 mm threshold")
    ax.set(xlabel="Final TCP distance (mm)", ylabel="Case count")
    ax.grid(axis="y", alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "final_tcp_error_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
    metric_specs = (("rollout/success_rate", "Training success rate"),
                    ("train/ent_coef", "Entropy coefficient"),
                    ("train/actor_loss", "Actor loss"),
                    ("train/critic_loss", "Critic loss"))
    for ax, (name, ylabel) in zip(axes, metric_specs):
        points = [(row["time/total_timesteps"], row[name]) for row in training_rows
                  if row[name] is not None]
        ax.plot([point[0] for point in points], [point[1] for point in points],
                color="tab:green", label="Reward ×10")
        if name == "train/ent_coef":
            ax.axhline(baseline_ent_coef, color="tab:blue", linestyle="--",
                       label="Baseline final value")
            ax.set_yscale("log")
            ax.legend()
        ax.set_ylabel(ylabel); ax.grid(alpha=.3)
    axes[-1].set_xlabel("Training timesteps")
    fig.tight_layout(); fig.savefig(out / "training_metrics.png", dpi=160)
    plt.close(fig)

    outcomes = ("improved", "worsened", "same_success", "same_failure")
    counts = [sum(row["baseline_to_reward10"] == name for row in results)
              for name in outcomes]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(outcomes, counts,
                  color=("tab:green", "tab:red", "tab:blue", "tab:gray"))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, count + 1, str(count), ha="center")
    ax.set(xlabel="Baseline to reward ×10 outcome", ylabel="Case count")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out / "outcome_transitions.png", dpi=160)
    plt.close(fig)


def _write_viewer_csv(path, reward10_rows, dls_rows):
    path.parent.mkdir(parents=True)
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(reward10_rows)
        writer.writerows(dls_rows)


def save_representatives(results, traces, out, baseline_model):
    improved = [row for row in results if row["baseline_to_reward10"] == "improved"]
    failures = [row for row in results if not row["reward10_success"]]
    success_row = max(improved or [row for row in results if row["reward10_success"]],
                      key=lambda row: row["initial_tcp_distance_mm"])
    selected = {
        "reward10_success": (success_row, "reward10"),
    }
    if failures:
        failure_row = max(
            failures, key=lambda row: row["reward10_final_tcp_distance_mm"])
        selected["reward10_failure"] = (failure_row, "reward10")
    else:
        worst_success = max(
            results, key=lambda row: row["reward10_final_tcp_distance_mm"])
        selected["reward10_worst_success"] = (worst_success, "reward10")
        baseline_failure = max(
            (row for row in results if not row["sac_success"]),
            key=lambda row: row["sac_final_tcp_distance_mm"])
        selected["baseline_failure"] = (baseline_failure, "baseline")
    for seed in (1000046, 1000047):
        row = next(
            row for row in results
            if row["goal_seed"] == seed and row["goal_joint_delta_rad"] == .70)
        selected[f"seed_{seed}_delta070"] = (row, "reward10")
    saved = {}
    for name, (row, trace_kind) in selected.items():
        key = (row["goal_joint_delta_rad"], row["goal_seed"])
        dls_env = KinematicReachEnv(
            max_steps=MAX_STEPS, action_delta_rad=ACTION_DELTA_RAD,
            goal_joint_delta_rad=key[0], success_radius_m=SUCCESS_RADIUS_M)
        try:
            dls = _run_rollout(
                dls_env, None, policy_name="jacobian_dls", goal_seed=key[1])
        finally:
            dls_env.close()
        sac_rows = traces[key]
        if trace_kind == "baseline":
            baseline_env = KinematicReachEnv(
                max_steps=MAX_STEPS, action_delta_rad=ACTION_DELTA_RAD,
                goal_joint_delta_rad=key[0], success_radius_m=SUCCESS_RADIUS_M)
            try:
                sac_rows = _run_rollout(
                    baseline_env, baseline_model, policy_name="sac",
                    goal_seed=key[1])
            finally:
                baseline_env.close()
        _write_viewer_csv(out / f"representative_{name}" / "rollout.csv",
                          sac_rows, dls)
        saved[name] = {
            "goal_joint_delta_rad": row["goal_joint_delta_rad"],
            "goal_seed": row["goal_seed"], "sac_trace": trace_kind,
        }
    return saved


def write_results(results, out):
    fields = list(results[0])
    with (out / "evaluation.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--experiment-a-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.baseline_model.is_file():
        parser.error(f"Baseline model not found: {args.baseline_model}")
    if not args.experiment_a_csv.is_file():
        parser.error(f"Experiment A CSV not found: {args.experiment_a_csv}")
    if args.out.exists():
        parser.error(f"--out must not exist: {args.out}")
    args.out.mkdir(parents=True)

    from stable_baselines3 import SAC
    baseline = SAC.load(str(args.baseline_model), device="cpu")
    baseline_ent_coef = float(baseline.log_ent_coef.detach().cpu().exp().item())
    experiment_a = load_experiment_a(args.experiment_a_csv)
    reward10_model, seconds = train_reward10(args.out)
    results, traces = evaluate_reward10(reward10_model, experiment_a)
    write_results(results, args.out)
    training_rows = load_training_metrics(args.out / "training_log" / "progress.csv")
    summary = build_summary(results, training_rows, baseline_ent_coef, seconds)
    summary["representatives"] = save_representatives(
        results, traces, args.out, baseline)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(results, training_rows, baseline_ent_coef, args.out)
    print(f"Saved Experiment B to {args.out}")
    for delta, values in summary["by_goal_range"].items():
        print(f"delta={delta}: baseline={values['baseline']['success_count']}/100, "
              f"reward10={values['reward10']['success_count']}/100, "
              f"DLS={values['dls']['success_count']}/100")
    print("outcomes", summary["baseline_to_reward10_outcomes"])


if __name__ == "__main__":
    main()
