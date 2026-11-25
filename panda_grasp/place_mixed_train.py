"""Train one state-only placement actor from three existing physical conditions."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .experiment import evaluate, write_csv, write_json
from .learning import train_clone
from .physics_shift import CONDITIONS
from .place_cli import clone_policy
from .place_mixed_benchmark import OUTPUT, VALIDATION_SCENES, dataset_path, evaluate_model, model_path
from .place_robustness import STRESS_CONDITIONS, TEST_SCENES, ShiftedPlacementEnv, load_clone, placement_metrics

UPDATES = (0, 250, 750, 1500, 3000, 6000)


def dataset_metadata(condition, path):
    with np.load(path) as data:
        eligible = np.repeat(data["successful"], np.diff(data["offsets"]))
        return {
            "condition": condition, "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "episodes": len(data["successful"]), "successful_episodes": int(data["successful"].sum()),
            "collected_steps": len(eligible), "eligible_steps": int(eligible.sum()),
            "eligible_phases": sorted(set(data["phase"][eligible])),
        }


def train(seed, results, out):
    directory = out / "mixed" / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    source = model_path("nominal", seed, results, out)
    datasets = [dataset_path(condition, seed, results) for condition in STRESS_CONDITIONS]
    model = load_clone(source)
    chronological = []
    details = []
    best = None
    started = perf_counter()
    validation_s = 0.0
    with ExitStack() as stack:
        envs = {name: stack.enter_context(ShiftedPlacementEnv(CONDITIONS[name])) for name in STRESS_CONDITIONS}

        def report(step, current, xyz, grip):
            nonlocal best, validation_s
            now = perf_counter()
            policy = clone_policy(current)
            outcomes = {name: placement_metrics(evaluate(env, policy, VALIDATION_SCENES))
                        for name, env in envs.items()}
            successes = [outcomes[name]["successes"] for name in STRESS_CONDITIONS]
            drop_rate = sum(row["transport_drop_rate"] for row in outcomes.values())
            rank = (min(successes), sum(successes), -drop_rate, -step)
            if best is None or rank > best:
                best = rank
                torch.save(current.state_dict(), directory / "model.pt")
            chronological.append({
                "optimizer_updates": step, "training_s": now - started - validation_s,
                "xyz_mse": xyz, "gripper_bce": grip, "worst_successes_of_12": rank[0],
                "total_successes_of_36": rank[1],
                **{f"{name}_successes": row["successes"] for name, row in outcomes.items()},
            })
            details.extend({"optimizer_updates": step, "condition": name, **row} for name, row in outcomes.items())
            write_csv(directory / "learning_curve.csv", chronological)
            write_csv(directory / "validation_conditions.csv", details)
            validation_s += perf_counter() - now
            print(f"{directory}: {step} updates, validation by condition {successes}/12", flush=True)

        report(0, model, None, None)
        train_clone(datasets[0], seed + 19000, UPDATES[-1], set(UPDATES[1:]), report,
                    model=model, additional_datasets=datasets[1:])

    write_json(directory / "config.json", {
        "seed": seed, "starting_checkpoint": source.as_posix(),
        "starting_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "datasets": [dataset_metadata(name, path) for name, path in zip(STRESS_CONDITIONS, datasets)],
        "condition_metadata_input": False, "new_demo_steps": 0, "new_online_training_steps": 0,
        "trained_optimizer_updates": UPDATES[-1], "selected_updates": -best[-1],
        "selected_worst_successes_of_12": best[0], "selected_total_successes_of_36": best[1],
        "validation_scenes": list(VALIDATION_SCENES), "test_scenes": list(TEST_SCENES),
        "conditions": {name: asdict(CONDITIONS[name]) for name in STRESS_CONDITIONS},
    })
    evaluate_model("mixed", seed, {name: CONDITIONS[name] for name in STRESS_CONDITIONS},
                   TEST_SCENES, results, out, "cross")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args.seed, args.results, args.out)


if __name__ == "__main__":
    main()
