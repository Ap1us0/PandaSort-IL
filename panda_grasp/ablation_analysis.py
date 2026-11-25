"""Summarize early retention and the two matched-budget test comparisons."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ablation_cli import BUDGET, CHECKPOINTS
from .experiment import TEST_SEEDS, VALIDATION_SEEDS, write_csv
from .sac_ablation import ABLATIONS

METHODS = ("bc", "sac", "bc_sac", *ABLATIONS)
LABELS = {
    "bc": "Pure BC", "sac": "Pure SAC", "bc_sac": "BC + SAC",
    "critic_warmup": "Critic warm-up", "low_exploration": "Lower exploration",
    "bc_regularized": "BC regularization",
}
COLORS = {
    "bc": "#777777", "sac": "#373c55", "bc_sac": "#a66d22",
    "critic_warmup": "#7c306d", "low_exploration": "#306c9d", "bc_regularized": "#157658",
}
METRICS = {
    "closed_near_rate": "Close gripper near cube",
    "bilateral_rate": "Establish bilateral grip",
    "lift_rate": "Reach 5 cm in a grip",
    "success_rate": "Hold steadily for 1 second",
}


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_result(directory):
    test = load_csv(directory / "test_episodes.csv")
    if [int(row["scenario_seed"]) for row in test] != list(TEST_SEEDS):
        raise ValueError(f"Test scenario mismatch in {directory}")
    summary = load_json(directory / "test_summary.json")
    if sum(row["is_success"] == "True" for row in test) != summary["successes"]:
        raise ValueError(f"Test summary does not agree with the replay in {directory}")
    return test, summary


def collect(root):
    results = []
    curves = {}
    for parent in sorted(root.glob("seed_*")):
        seed = int(parent.name.removeprefix("seed_"))
        for method in METHODS:
            directory = parent / method
            baseline = method in ("bc", "sac", "bc_sac")
            points = directory / "learning_curve.csv"
            short = parent / "bc_sac_total"
            if baseline and (directory / "reference.json").is_file():
                reference = load_json(directory / "reference.json")
                points = Path(reference["archived_learning_curve"])
                cost = reference["archived_cost"]["environment_steps"]
                demo_count = (load_json(parent / "bc" / "reference.json")["archived_cost"]["environment_steps"]
                              if method in ("bc", "bc_sac") else 0)
                online_steps = 0 if method == "bc" else cost
                if method == "bc_sac" and (short / "learning_curve.csv").exists():
                    stock = {int(row["environment_steps"]): row for row in load_csv(points)}
                    for row in load_csv(short / "learning_curve.csv"):
                        step = int(row["environment_steps"])
                        if step in stock:
                            for metric in ("bilateral_rate", "lift_rate", "success_rate"):
                                if not np.isclose(float(row[metric]), float(stock[step][metric])):
                                    raise ValueError(f"Stock SAC did not reproduce checkpoint {step}, seed {seed}")
                        stock[step] = row
                    curves[seed, method] = [stock[step] for step in sorted(stock)]
            elif not baseline and (directory / "config.json").is_file():
                config = load_json(directory / "config.json")
                demo_count = config["demo_steps"]
                online_steps = config["environment_steps"]
                for csv_path in (directory / "validation").glob("step_*.csv"):
                    if [int(row["scenario_seed"]) for row in load_csv(csv_path)] != list(VALIDATION_SEEDS):
                        raise ValueError(f"Validation scenario mismatch in {csv_path}")
            else:
                continue
            if (seed, method) not in curves and points.is_file():
                curves[seed, method] = load_csv(points)
            entries = [("online", directory, online_steps)]
            if method in ("bc", "sac"):
                entries.append(("total", directory, online_steps))
            elif method == "bc_sac" and (short / "config.json").exists():
                entries.append(("total", short, load_json(short / "config.json")["environment_steps"]))
            elif not baseline:
                entries.append(("total", directory / "total_budget", config["cost_match_online_steps"]))
            for budget, path, steps in entries:
                if not (path / "test_summary.json").is_file():
                    continue
                if budget == "total" and steps + demo_count > BUDGET:
                    raise ValueError(f"Total interaction budget exceeded in {path}")
                _, summary = test_result(path)
                results.append({
                    "budget": budget, "seed": seed, "method": method, "online_steps": steps,
                    "demo_steps": demo_count, "total_steps": steps + demo_count,
                    **summary,
                })
    return results, curves


def chart(root, curves):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), layout="constrained")
    for axis, (metric, label) in zip(axes.flat, METRICS.items()):
        for method in METHODS:
            runs = []
            for (seed, name), rows in sorted(curves.items()):
                if name != method:
                    continue
                if method == "bc":
                    baseline_actor = next((row for row in curves.get((seed, "bc_sac"), [])
                                           if int(row["environment_steps"]) == 0 and row.get(metric)), None)
                    if baseline_actor is not None:
                        value = float(baseline_actor[metric])
                        runs.append({0: value, BUDGET: value})
                else:
                    runs.append({int(row["environment_steps"]): float(row[metric]) for row in rows
                                 if int(row["environment_steps"]) in CHECKPOINTS and row.get(metric)})
            if not runs:
                continue
            common = sorted(set.intersection(*(set(run) for run in runs)))
            if not common:
                continue
            values = np.asarray([[run[step] for step in common] for run in runs])
            average = values.mean(axis=0)
            if method != "bc":
                axis.fill_between(common, values.min(axis=0), values.max(axis=0),
                                  color=COLORS[method], alpha=0.09, linewidth=0)
            axis.plot(common, average, label=LABELS[method], color=COLORS[method],
                      ls="--" if method == "bc" else "-", marker=None if method == "bc" else "o",
                      markersize=3)
        axis.set_title(label)
        axis.set(xlim=(-400, 24_400), ylim=(-0.055, 1.065), xlabel="Online environment steps")
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.savefig(root / "retention_curves.png", dpi=180)
    plt.close(fig)


def early_rows(curves):
    rows = []
    for (seed, method), history in sorted(curves.items()):
        if method == "bc":
            continue
        for row in history:
            if int(row["environment_steps"]) in (0, 512, 1000, 2000, 6000):
                rows.append({"seed": seed, "method": method, "environment_steps": row["environment_steps"],
                             **{key: row.get(key) for key in METRICS},
                             "mean_max_hold_s": row["mean_max_hold_s"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/sac_ablation"))
    args = parser.parse_args()
    results, curves = collect(args.root)
    if results:
        write_csv(args.root / "summary.csv", results)
    early = early_rows(curves)
    if early:
        write_csv(args.root / "early_validation.csv", early)
    if curves:
        chart(args.root, curves)


if __name__ == "__main__":
    main()
