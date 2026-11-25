"""Verify the tactile comparison and report per-seed cube transport changes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import write_csv
from .place_contact_signals import ContactSortingScene, TACTILE_SIZE
from .place_objects import OBS_SIZE, SHAPES
from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_objects_train import sha
from .place_sorting import SortingScene
from .place_sorting_analysis import audit, draw, record_cases
from .place_sorting_bc import OUTPUT as SORTING_RESULTS
from .place_sorting_contact import METHODS, OUTPUT, TEST, load_evaluation

STYLES = (("binary", "Binary contact BC", "#667886"), ("continuous", "Force + slip BC", "#217C71"))


def audit_training(output, sorting):
    rows = []
    for seed in range(3):
        root = output / "training" / f"seed_{seed}"
        old = json.loads((sorting / "training" / f"seed_{seed}" / "config.json").read_text(encoding="utf-8"))
        receipt = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if (receipt["optimizer_updates_each"] != old["selected_updates"]
                or receipt["binary_checkpoint_sha256"] != old["checkpoint_sha256"]
                or receipt["starting_sha256"] != old["initial_sha256"]
                or not receipt["binary_checkpoint_reproduced_exactly"]):
            raise ValueError(f"Mismatched training budget or checkpoint for seed {seed}")
        for original, metadata in zip(old["datasets"], receipt["datasets"], strict=True):
            path = root / "datasets" / metadata["domain"] / metadata["shape"] / "expert.npz"
            if sha(Path(original)) != metadata["original_sha256"] or sha(path) != metadata["augmented_sha256"]:
                raise ValueError(f"A demonstration archive changed: {path}")
            with np.load(original) as base, np.load(path) as observed:
                if any(not np.array_equal(base[key], observed[key][:, :OBS_SIZE] if key in
                           ("observations", "next_observations") else observed[key]) for key in base.files):
                    raise ValueError(f"Data changed outside the new sensor columns: {path}")
                if observed["observations"].shape[1] != TACTILE_SIZE:
                    raise ValueError(f"Invalid observation size: {path}")
            rows.append({"seed": seed, "matched_optimizer_updates": receipt["optimizer_updates_each"], **metadata})
    write_csv(output / "matched_demonstrations.csv", rows)


def compare(output, cases):
    records, recoveries, regressions = [], [], []
    for seed in range(3):
        before, items_before = cases["binary", seed]
        after, items_after = cases["continuous", seed]
        counts = {
            "seed": seed,
            "all_sorted_before": sum(row["all_sorted"] == "True" for row in before),
            "all_sorted_after": sum(row["all_sorted"] == "True" for row in after),
            "scene_gains": sum(a["all_sorted"] == "False" and b["all_sorted"] == "True" for a, b in zip(before, after)),
            "scene_losses": sum(a["all_sorted"] == "True" and b["all_sorted"] == "False" for a, b in zip(before, after)),
        }
        for shape in SHAPES:
            a = [item for item in items_before if item["shape"] == shape]
            b = [item for item in items_after if item["shape"] == shape]
            if [row["scenario_seed"] for row in a] != [row["scenario_seed"] for row in b]:
                raise ValueError(f"Unmatched items for {shape}, seed {seed}")
            counts.update({
                f"{shape}_sorted_before": sum(item["sorted"] == "True" for item in a),
                f"{shape}_sorted_after": sum(item["sorted"] == "True" for item in b),
                f"{shape}_drops_before": sum(item["transport_drop"] == "True" for item in a),
                f"{shape}_drops_after": sum(item["transport_drop"] == "True" for item in b),
                f"{shape}_drops_avoided": sum(x["transport_drop"] == "True" and y["transport_drop"] == "False"
                                              for x, y in zip(a, b)),
                f"{shape}_drops_added": sum(x["transport_drop"] == "False" and y["transport_drop"] == "True"
                                            for x, y in zip(a, b)),
            })
            if shape == "cube":
                for old_scene, new_scene, old_cube, new_cube in zip(before, after, a, b):
                    if old_cube["transport_drop"] == "True" and new_cube["sorted"] == "True":
                        recoveries.append((seed, old_scene, new_scene))
                    if old_cube["sorted"] == "True" and new_cube["transport_drop"] == "True":
                        regressions.append((seed, old_scene, new_scene))
        records.append(counts)
    write_csv(output / "paired.csv", records)
    return recoveries, regressions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=OBJECT_RESULTS)
    parser.add_argument("--sorting", type=Path, default=SORTING_RESULTS)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit_training(args.out, args.sorting)
    load = lambda method, seed: load_evaluation(method, seed, args.source, args.sorting, args.out)
    runs = tuple((method, seed) for method in METHODS for seed in range(3))
    records, objects, cases = audit(args.out, args.source, methods=runs, scenarios=TEST,
                                   checkpoint_identity=lambda method, seed: load(method, seed)[1])
    draw(args.out, records, objects, STYLES)
    recoveries, regressions = compare(args.out, cases)
    if args.videos and (regressions or recoveries):
        candidates = regressions or recoveries
        seed, old, new = max(candidates, key=lambda pair: (
            pair[0] == 0, abs(int(pair[2]["sorted_items"]) - int(pair[1]["sorted_items"])),
            pair[1]["all_sorted"] == "True",
        ))
        record_cases(args.out, cases, (("binary", seed, old), ("continuous", seed, new)),
                     lambda method, seed: load(method, seed)[0],
                     scene_type=lambda method: ContactSortingScene if method == "continuous" else SortingScene)


if __name__ == "__main__":
    main()
