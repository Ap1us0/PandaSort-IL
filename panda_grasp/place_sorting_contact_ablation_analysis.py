"""Summarize tactile-source and phase ablations on the same sorting scenes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .experiment import write_csv
from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_sorting_analysis import audit, record_cases
from .place_sorting_bc import OUTPUT as SORTING_RESULTS
from .place_sorting_contact import OUTPUT as CONTACT_RESULTS
from .place_sorting_contact_ablation import (
    METHODS, OUTPUT, TEST, VARIANTS, load_evaluation, scene_type, training_sources,
)

LABELS = ["Binary", "Force only", "Slip only", "Force + slip\nin transport", "Force + slip\nthroughout"]
COLORS = ("#657584", "#4089AA", "#886BB1", "#218474", "#BA783B")


def audit_training(output, sorting, contact):
    records = []
    for seed in range(3):
        _, previous = training_sources(seed, contact, sorting)
        updates = previous["optimizer_updates_each"]
        datasets = [r["augmented_sha256"] for r in previous["datasets"]]
        for method in VARIANTS:
            directory = output / "training" / f"seed_{seed}" / method
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            if (config["optimizer_updates"] != updates or config["learning_rate"] != previous["learning_rate"]
                    or config["starting_sha256"] != previous["starting_sha256"]
                    or config["demonstration_sha256"] != datasets or config["test_scenarios"] != list(TEST)
                    or config["new_demonstration_scenes"] != 0 or config["new_online_training_steps"] != 0):
                raise ValueError(f"Training inputs or budget differ in {directory}")
        records.append({"seed": seed, "optimizer_updates_per_method": updates,
                        "demonstration_sources": len(datasets),
                        "demonstration_transitions": sum(row["transitions"] for row in previous["datasets"]),
                        "starting_sha256": previous["starting_sha256"],
                        "binary_sha256": previous["binary_checkpoint_sha256"],
                        "continuous_sha256": previous["continuous_checkpoint_sha256"]})
    write_csv(output / "budget.csv", records)


def focused(output, records, objects, *, methods=METHODS):
    detailed = []
    for method in methods:
        for seed in range(3):
            scene = next(row for row in records if row["method"] == method and row["seed"] == seed)
            cube = next(row for row in objects if row["method"] == method and row["seed"] == seed and row["shape"] == "cube")
            detailed.append({"method": method, "seed": seed, "scenes": scene["scenes"],
                             "all_sorted": scene["all_sorted"], "sorted_items": scene["sorted_items"],
                             "cube_sorted": cube["sorted"], "cube_lifted": cube["lifted"],
                             "cube_transport_drops": cube["transport_drops"]})
    write_csv(output / "by_seed.csv", detailed)
    aggregate = []
    for method in methods:
        rows = [r for r in detailed if r["method"] == method]
        aggregate.append({"method": method, **{key: sum(row[key] for row in rows) for key in
                          ("scenes", "all_sorted", "sorted_items", "cube_sorted", "cube_lifted", "cube_transport_drops")}})
    write_csv(output / "overall.csv", aggregate)
    return detailed, aggregate


def paired(output, cases):
    results, samples = [], []
    pairs = [("binary", next_method) for next_method in METHODS if next_method != "binary"]
    pairs.append(("continuous", "transport"))
    for reference, method in pairs:
        for seed in range(3):
            before, old_items = cases[reference, seed]
            after, new_items = cases[method, seed]
            cubes_before, cubes_after = ([r for r in items if r["shape"] == "cube"] for items in (old_items, new_items))
            if [r["scenario_seed"] for r in cubes_before] != [r["scenario_seed"] for r in cubes_after]:
                raise ValueError(f"Unmatched cube results for seed {seed}, {method}")
            scene_pairs = list(zip(before, after, strict=True))
            cube_pairs = list(zip(cubes_before, cubes_after, strict=True))
            results.append({
                "reference": reference, "method": method, "seed": seed,
                "whole_scene_gains": sum(a["all_sorted"] == "False" and b["all_sorted"] == "True" for a, b in scene_pairs),
                "whole_scene_losses": sum(a["all_sorted"] == "True" and b["all_sorted"] == "False" for a, b in scene_pairs),
                "cube_drops_avoided": sum(a["transport_drop"] == "True" and b["transport_drop"] == "False" for a, b in cube_pairs),
                "cube_drops_added": sum(a["transport_drop"] == "False" and b["transport_drop"] == "True" for a, b in cube_pairs),
            })
            if seed == 0 and reference == "binary" and method in VARIANTS:
                for (old, new), (old_cube, new_cube) in zip(scene_pairs, cube_pairs, strict=True):
                    recovery = old_cube["transport_drop"] == "True" and new_cube["sorted"] == "True"
                    regression = old_cube["sorted"] == "True" and new_cube["transport_drop"] == "True"
                    if recovery or regression:
                        samples.append((method, old, new, recovery, new_cube["final_color"] == "outside"))
    write_csv(output / "paired.csv", results)
    return samples


def plot(output, detailed, aggregate):
    x = np.arange(len(METHODS))
    fig, panels = plt.subplots(1, 2, figsize=(11.5, 3.9), layout="constrained")
    seed0 = [r["cube_transport_drops"] for r in detailed if r["seed"] == 0]
    scenes = [100 * r["all_sorted"] / r["scenes"] for r in aggregate]
    for ax, values, title, limits, labels, y_label in (
        (panels[0], seed0, "Cube transport drops, seed 0", (0, max(10, 1.23 * max(seed0))), "%.0f", "Drops per 30 scenes"),
        (panels[1], scenes, "All three objects sorted", (0, max(50, 1.23 * max(scenes))), "%.1f%%", "Percent of 90 runs"),
    ):
        bars = ax.bar(x, values, width=0.6, color=COLORS)
        ax.bar_label(bars, fmt=labels, padding=3, fontsize=9)
        ax.set(xticks=x, xticklabels=LABELS, ylim=limits, title=title, ylabel=y_label)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=OBJECT_RESULTS)
    parser.add_argument("--sorting", type=Path, default=SORTING_RESULTS)
    parser.add_argument("--contact", type=Path, default=CONTACT_RESULTS)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit_training(args.out, args.sorting, args.contact)
    def load(method, seed):
        return load_evaluation(method, seed, args.source, args.sorting, args.contact, args.out)

    runs = tuple((method, seed) for method in METHODS for seed in range(3))
    records, objects, cases = audit(args.out, args.source, methods=runs, scenarios=TEST,
                                   checkpoint_identity=lambda method, seed: load(method, seed)[1])
    detailed, aggregate = focused(args.out, records, objects)
    examples = paired(args.out, cases)
    plot(args.out, detailed, aggregate)
    if args.videos and examples:
        pooled_success = {row["method"]: row["all_sorted"] for row in aggregate}
        method, old, new, _, _ = max(examples, key=lambda item: (
            item[3] and item[2]["all_sorted"] == "True", pooled_success[item[0]],
            abs(int(item[2]["sorted_items"]) - int(item[1]["sorted_items"])), item[4],
        ))
        record_cases(args.out, cases, (("binary", 0, old), (method, 0, new)),
                     lambda method, seed: load(method, seed)[0], scene_type=scene_type)


if __name__ == "__main__":
    main()
