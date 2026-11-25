"""Cross-evaluate existing placement policies and check unseen combined shifts."""

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from .physics_shift import CONDITIONS, Condition
from .place_cli import clone_policy
from .place_robustness import (
    STRESS_CONDITIONS, TEST_SCENES, ShiftedPlacementEnv, load_clone, run_evaluation, teacher_policy,
)

OUTPUT = Path("results/place_mixed")
VALIDATION_SCENES = range(1_450_000, 1_450_012)
COMBINED_SCENES = range(1_460_000, 1_460_100)
COMBINED = {
    "combined_moderate": Condition(mass_scale=2.0, sliding_scale=0.75),
    "combined_full": Condition(mass_scale=2.5, sliding_scale=0.50),
}


def model_path(source, seed, results, output):
    seed_dir = f"seed_{seed}"
    if source == "nominal":
        return results / "pick_place" / seed_dir / "model.pt"
    if source == "mixed":
        return output / "mixed" / seed_dir / "model.pt"
    return results / "place_robustness" / "adapt" / seed_dir / source / "model.pt"


def dataset_path(source, seed, results):
    seed_dir = f"seed_{seed}"
    if source == "nominal":
        return results / "pick_place" / seed_dir / "expert.npz"
    return results / "place_robustness" / "adapt" / seed_dir / source / "demonstrations" / "expert.npz"


def prior_cell(source, condition, seed, results):
    seed_dir = f"seed_{seed}"
    base = results / "place_robustness"
    if source == "nominal":
        return base / "test" / seed_dir / condition
    if source == condition:
        return base / "adapt" / seed_dir / condition
    return None


def evaluate_model(source, seed, conditions, scenes, results, out, folder):
    model = model_path(source, seed, results, out)
    policy = clone_policy(load_clone(model))
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    for name, condition in conditions.items():
        if folder == "cross" and prior_cell(source, name, seed, results) is not None:
            print(f"Reusing {source} seed {seed}, {name} from the same held-out scenes", flush=True)
            continue
        with ShiftedPlacementEnv(condition) as env:
            run_evaluation(env, policy, scenes, out / folder / source / f"seed_{seed}" / name, {
                "source": source, "condition": name, "seed": seed,
                "checkpoint": model.as_posix(), "checkpoint_sha256": sha,
                **asdict(condition), "mass_kg": env.mass_kg, "sliding_coefficient": env.cube_sliding,
            })


def evaluate_expert(out):
    for name, condition in COMBINED.items():
        with ShiftedPlacementEnv(condition) as env:
            run_evaluation(env, teacher_policy(1.0), COMBINED_SCENES, out / "combined" / "expert" / name, {
                "source": "expert", "condition": name, "seed": None,
                **asdict(condition), "mass_kg": env.mass_kg, "sliding_coefficient": env.cube_sliding,
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cross", "expert", "combined"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--source", choices=("mixed", "nominal"), default="mixed")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "expert":
        evaluate_expert(args.out)
    elif args.seed is None:
        parser.error("This command requires --seed")
    elif args.command == "cross":
        conditions = {name: CONDITIONS[name] for name in STRESS_CONDITIONS}
        for source in ("friction_50", "mass_250"):
            evaluate_model(source, args.seed, conditions, TEST_SCENES, args.results, args.out, "cross")
    else:
        evaluate_model(args.source, args.seed, COMBINED, COMBINED_SCENES, args.results, args.out, "combined")


if __name__ == "__main__":
    main()
