"""Summarize the placement experiment and the paired lift diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .experiment import TEST_SEEDS, write_csv
from .place import GOAL_RADIUS
from .place_cli import TEST_SCENES


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def placement_summary(root):
    rows = []
    curves = []
    endpoint_errors = []
    for directory in sorted(root.glob("seed_*")) + [root / "expert"]:
        path = directory / "test_summary.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        raw = csv_rows(directory / "test_episodes.csv")
        if [int(row["scenario_seed"]) for row in raw] != list(TEST_SCENES):
            raise ValueError(f"Incomplete or unexpected test scenarios in {directory}")
        for row in raw:
            if row["is_success"] == "True" and not (
                row["supported_at_target"] == row["released_at_target"] == "True"
                and float(row["settled_at_target_s"]) >= 1.0
            ):
                raise ValueError(f"Invalid successful placement in {directory}")
        endpoint_errors.append([float(row["goal_distance_m"]) * 1000 for row in raw])
        metadata = directory / "dataset.json"
        if metadata.is_file():
            data = json.loads(metadata.read_text(encoding="utf-8"))
            curve = csv_rows(directory / "learning_curve.csv")
            curves.append((directory.name, curve))
            result.update(demo_steps=data["transitions"], demo_successes=data["successful"],
                          collection_s=data["collection_s"], training_s=float(curve[-1]["training_s"]))
        else:
            result.update(demo_steps=None, demo_successes=None, collection_s=None, training_s=None)
        rows.append({"run": directory.name, **result})
    if not rows:
        return
    write_csv(root / "summary.csv", rows)
    if not curves:
        return
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for seed, curve in curves:
        left.plot([int(row["updates"]) for row in curve], [float(row["success_rate"]) for row in curve],
                  label=seed.replace("_", " "), marker="o")
    left.set(xlabel="BC optimizer updates", ylabel="Success rate on 12 validation scenes",
             title="From lift weights to placement", ylim=(-0.05, 1.06))
    left.legend(frameon=False, loc="lower right")
    right.boxplot(endpoint_errors, tick_labels=[row["run"].replace("_", " ") for row in rows],
                  showmeans=True, meanprops={"marker": "D", "markerfacecolor": "#37846f"})
    right.set(ylabel="Final XY error (mm)", title="Placement on 100 test scenes", ylim=(0, None))
    right.text(0.97, 0.95, f"Goal tolerance: {GOAL_RADIUS * 1000:g} mm",
               ha="right", va="top", transform=right.transAxes)
    for axis in (left, right):
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(root / "learning_and_precision.png", dpi=170)
    plt.close(figure)


def lift_summary(root):
    rows = []
    for path in sorted(root.glob("*/seed_*/*/summary.json")):
        raw = csv_rows(path.with_name("episodes.csv"))
        baseline = [row for row in raw if row["mode"] == "policy"]
        if [int(row["scenario_seed"]) for row in baseline] != list(TEST_SEEDS):
            raise ValueError(f"Incomplete lift replay in {path.parent}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for mode, result in data.items():
            rows.append({
                "profile": path.parents[2].name, "seed": path.parents[1].name,
                "method": path.parent.name, "mode": mode,
                **{k: v for k, v in result.items() if k not in ("hold_breaks", "post_lift_violation_fraction")},
                **result["hold_breaks"], **result["post_lift_violation_fraction"],
            })
    if rows:
        write_csv(root / "summary.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    placement_summary(args.results / "pick_place")
    lift_summary(args.results / "lift_diagnostics")


if __name__ == "__main__":
    main()
