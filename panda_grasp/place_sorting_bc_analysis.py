"""Audit paired three-object sorting tests and the additional demonstration data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import write_csv
from .place_objects import ACTION_NAMES, OBS_SIZE, SHAPES
from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_objects_train import sha
from .place_sorting_analysis import audit, draw, record_cases
from .place_sorting_bc import OUTPUT, TEST, TRAIN_SCENES, VALIDATION, load_policy

METHODS = (("expert", None), *((method, seed) for method in ("frozen", "adapted") for seed in range(3)))
STYLES = (("frozen", "Frozen BC", "#6D7888"), ("adapted", "BC with sorting demos", "#217C71"),
          ("expert", "Expert", "#A6844C"))


def audit_demonstrations(output):
    rows = []
    for seed in range(3):
        root = output / "training" / f"seed_{seed}"
        receipt = json.loads((root / "demonstrations" / "dataset.json").read_text(encoding="utf-8"))
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if receipt["scenes"] != TRAIN_SCENES or receipt["expert_action_steps"] != config["new_expert_steps"]:
            raise ValueError(f"Invalid budget in {root}")
        sampled = set()
        for item in receipt["shape_data"]:
            name = item["shape"]
            data = root / "demonstrations" / name / "expert.npz"
            if sha(data) != item["sha256"]:
                raise ValueError(f"Dataset hash does not match: {data}")
            with np.load(data) as archive:
                steps = item["expert_action_steps"]
                scenarios = set(archive["scenario_seeds"])
                successful = archive["successful"]
                if (archive["observations"].shape != (steps, OBS_SIZE)
                        or archive["next_observations"].shape != (steps, OBS_SIZE)
                        or archive["actions"].shape != (steps, len(ACTION_NAMES))
                        or not np.array_equal(archive["actions"], archive["expert_actions"])
                        or int(archive["offsets"][-1]) != steps
                        or int(successful.sum()) != item["successful"]
                        or len(scenarios) != TRAIN_SCENES
                        or scenarios.intersection(TEST) or scenarios.intersection(VALIDATION)):
                    raise ValueError(f"Invalid dataset: {data}")
                selected_steps = int(np.diff(archive["offsets"])[successful].sum())
            if sampled and sampled != scenarios:
                raise ValueError("Each shape must have been attempted in all training scenes")
            sampled = scenarios
            rows.append({"seed": seed, **item, "used_expert_steps": selected_steps,
                         "all_homing_steps": receipt["homing_action_steps"], "selected_updates": config["selected_updates"]})
    write_csv(output / "demo_audit.csv", rows)


def pairs(output, cases):
    counts, examples = [], []
    for seed in range(3):
        before, previous = cases["frozen", seed]
        after, updated = cases["adapted", seed]
        for first, second in zip(before, after):
            gain = int(second["sorted_items"]) - int(first["sorted_items"])
            if gain > 0:
                examples.append((gain, seed, first, second))
        metrics = {}
        for name, first, second, field in (
            ("scene", before, after, "all_sorted"), ("item", previous, updated, "sorted"),
            *((shape, [row for row in previous if row["shape"] == shape],
               [row for row in updated if row["shape"] == shape], "sorted") for shape in SHAPES),
        ):
            metrics[f"{name}_gains"] = sum(x[field] == "False" and y[field] == "True" for x, y in zip(first, second))
            metrics[f"{name}_losses"] = sum(x[field] == "True" and y[field] == "False" for x, y in zip(first, second))
        counts.append({"seed": seed, **metrics,
                       "transport_drops_before": sum(row["transport_drop"] == "True" for row in previous),
                       "transport_drops_after": sum(row["transport_drop"] == "True" for row in updated)})
    write_csv(output / "paired.csv", counts)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=OBJECT_RESULTS)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit_demonstrations(args.out)
    load = lambda method, seed: load_policy(method, seed, args.source, args.out)
    records, objects, cases = audit(args.out, args.source, methods=METHODS, scenarios=TEST,
                                   checkpoint_identity=lambda method, seed: load(method, seed)[1])
    draw(args.out, records, objects, STYLES)
    examples = pairs(args.out, cases)
    if args.videos and examples:
        _, seed, first, second = max(examples, key=lambda item: (
            item[3]["all_sorted"] == "True", item[0], item[1] == 1,
        ))
        record_cases(args.out, cases, (("frozen", seed, first), ("adapted", seed, second)),
                     lambda method, training_seed: load(method, training_seed)[0])


if __name__ == "__main__":
    main()
