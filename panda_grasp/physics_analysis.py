"""Validate paired scenario sets and summarize physics-shift experiments."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import write_csv, write_json
from .physics_shift import (
    ADAPT_TEST_SCENES, CONDITIONS, NOMINAL_TEST_SCENES, RUNTIME, SCREEN_SCENES, checkpoint_path,
)

COLORS = {"expert": "#6b6470", "bc": "#306ca0", "bc_sac": "#27814d"}
LABELS = {"expert": "Expert", "bc": "Pure BC", "bc_sac": "BC-regularized SAC"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_rows(file, scenes, expected_successes):
    with file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["scenario_seed"]) for row in rows] != list(scenes):
        raise ValueError(f"Scenario mismatch in {file}")
    if sum(row["is_success"] == "True" for row in rows) != expected_successes:
        raise ValueError(f"Invalid success count in {file}")
    return rows


def summarize_survey(root):
    output = []
    for mode, scenes in (("screen", SCREEN_SCENES), ("test", NOMINAL_TEST_SCENES)):
        for file in sorted((root / mode).glob("*/*/*/summary.json")):
            value = read_json(file)
            verify_rows(file.with_name("episodes.csv"), scenes, value["successes"])
            output.append({"mode": mode, "seed": file.parents[2].name, "method": file.parents[1].name, **value})
    if output:
        write_csv(root / "frozen_summary.csv", output)
    return output


def adaptation_summary(root):
    summary = []
    paired = []
    for parent in sorted((root / "adapt").glob("seed_*")) + [root / "adapt" / "expert"]:
        history = {}
        for directory in (parent.iterdir() if parent.is_dir() and parent.name != "expert" else [parent]):
            path = directory / "test_summary.json"
            if not path.is_file():
                continue
            row = read_json(path)
            matched = verify_rows(path.with_name("test_episodes.csv"), ADAPT_TEST_SCENES, row["successes"])
            cost = read_json(directory / "cost.json") if (directory / "cost.json").is_file() else {}
            method = directory.name
            summary.append({"seed": parent.name, "method": method, "new_demo_steps": cost.get("new_demo_steps", 0),
                            "online_steps": cost.get("online_steps", 0), "total_new_steps": cost.get("total_new_steps", 0),
                            **row})
            history[method] = matched
        for adapted, frozen in (("bc", "frozen_bc"), ("bc_prefix", "frozen_bc"),
                                ("bc_prefix_selected", "frozen_bc"), ("bc_sac", "frozen_bc_sac")):
            if adapted not in history or frozen not in history:
                continue
            middle = [(base["is_success"] == "True", changed["is_success"] == "True")
                      for base, changed in zip(history[frozen], history[adapted])]
            paired.append({
                "seed": parent.name, "adapted": adapted, "frozen": frozen,
                "failure_to_success": sum(not a and b for a, b in middle),
                "success_to_failure": sum(a and not b for a, b in middle),
            })
    if summary:
        write_csv(root / "adaptation_summary.csv", summary)
    if paired:
        write_csv(root / "paired_changes.csv", paired)
    return summary


def stress_chart(rows, root):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for axis, target in zip(axes, ("sliding_scale", "mass_scale")):
        fixed = "mass_scale" if target == "sliding_scale" else "sliding_scale"
        x_scale = 100 if target == "sliding_scale" else 75
        for method in COLORS:
            for mode in ("screen", "test"):
                groups = {}
                for row in rows:
                    if row["mode"] == mode and row["method"] == method and row[fixed] == 1.0:
                        groups.setdefault(row[target] * x_scale, []).append(row["success_rate"])
                xs = sorted(groups)
                values = [float(np.mean(groups[x])) for x in xs]
                if xs:
                    axis.plot(xs, values, color=COLORS[method], marker="o" if mode == "test" else None,
                              ls="-" if mode == "test" else ":", alpha=1 if mode == "test" else 0.7,
                              label=LABELS[method] if mode == "test" else None)
                    if mode == "test" and method != "expert":
                        axis.fill_between(xs, [min(groups[x]) for x in xs], [max(groups[x]) for x in xs],
                                          color=COLORS[method], alpha=0.10, linewidth=0)
        axis.set(xlabel="Sliding friction (% of nominal)" if target == "sliding_scale" else "Cube mass (g)",
                 ylabel="Successful lift and hold", ylim=(-0.04, 1.05))
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Frozen policies | dotted: 12 screening scenes; markers: 100 test scenes")
    fig.savefig(root / "friction_and_mass.png", dpi=175)
    plt.close(fig)


def adaptation_chart(rows, root):
    groups = ("frozen_bc", "frozen_bc_sac", "bc_prefix", "bc_prefix_selected", "bc", "bc_sac")
    labels = ("BC frozen", "SAC frozen", "BC 12 demos", "BC 12 demos\nselected", "BC full demos", "SAC + 12 demos")
    values = [[row["success_rate"] for row in rows if row["method"] == name] for name in groups]
    if not all(values):
        return
    fig, axis = plt.subplots(figsize=(10.5, 4.6), layout="constrained")
    colors = [COLORS["bc_sac"] if "sac" in name else COLORS["bc"] for name in groups]
    means = [np.mean(group) for group in values]
    axis.bar(labels, means, color=colors, alpha=0.67)
    for index, group in enumerate(values):
        axis.scatter(index + np.linspace(-0.12, 0.12, len(group)), group, c="#25252a", s=22, zorder=3)
    for index, method in enumerate(groups):
        costs = [row["total_new_steps"] for row in rows if row["method"] == method]
        cost = str(min(costs)) if min(costs) == max(costs) else f"{min(costs)}–{max(costs)}"
        axis.text(index, 1.035, f"{cost} steps", ha="center", fontsize=9)
    axis.set(title="Adaptation at 50% sliding friction | dots: training seeds",
             ylabel="Successful lift and hold on 100 held-out scenes", ylim=(0, 1.12))
    axis.grid(axis="y", alpha=0.23)
    fig.savefig(root / "adaptation_cost.png", dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = args.root
    frozen = summarize_survey(root)
    adapted = adaptation_summary(root)
    if any(row["mode"] == "test" for row in frozen):
        stress_chart(frozen, root)
    if adapted:
        adaptation_chart(adapted, root)
    fingerprints = []
    for seed in range(3):
        for method in ("bc", "bc_sac"):
            model = checkpoint_path(args.results, method, seed)
            fingerprints.append({"seed": seed, "method": method, "file": model.as_posix(),
                                 "sha256": hashlib.sha256(model.read_bytes()).hexdigest()})
    write_json(root / "source_checkpoints.json", fingerprints)


if __name__ == "__main__":
    main()
