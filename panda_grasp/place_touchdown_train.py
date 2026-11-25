"""Add twelve verified scene demonstrations and preserve the four-domain actor."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from .experiment import collect_demonstrations, evaluate, write_csv, write_json
from .learning import train_clone
from .place_cli import clone_policy
from .place_mixed_benchmark import dataset_path
from .place_mixed_train import dataset_metadata
from .place_robustness import ShiftedPlacementEnv, load_clone, placement_metrics
from .place_touchdown import (
    DOMAINS, MODERATE, OUTPUT, TEST_SCENES, VALIDATION_SCENES, build_teacher, origin_path, test_checkpoint,
)

DEMONSTRATIONS = 12
UPDATES = (0, 250, 750, 1500)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def train(seed, results, output):
    teacher_spec = read_json(output / "teacher_selection.json")
    for training_seed in range(3):
        if not (output / "probe" / f"seed_{training_seed}" / "summary.json").is_file():
            raise ValueError("Finish the paired diagnostics before collecting adaptation data")
    teacher = teacher_spec["teacher"]
    root = output / "training" / f"seed_{seed}"
    demo = root / "demonstrations"
    dataset = demo / "expert.npz"
    if not dataset.exists():
        start = 81_000_000 + 10_000 * seed
        with ShiftedPlacementEnv(DOMAINS[MODERATE]) as env:
            collect_demonstrations(
                demo, env, seed, range(start, start + DEMONSTRATIONS),
                lambda rng, perturb: build_teacher(env, teacher), {"condition": MODERATE, "teacher": teacher},
                enable_perturbation=False,
            )
    data = read_json(demo / "dataset.json")
    if (data["teacher"], data["seed"], data["episodes"]) != (teacher, seed, DEMONSTRATIONS):
        raise ValueError(f"Incompatible existing demonstration set in {demo}")
    paths = [dataset if name == MODERATE else dataset_path(name, seed, results) for name in DOMAINS]
    source = origin_path(results / "place_mixed", seed)
    model = load_clone(source)
    curve = []
    history = []
    best = None
    start = perf_counter()
    eval_s = 0.0
    with ExitStack() as stack:
        envs = {name: stack.enter_context(ShiftedPlacementEnv(condition)) for name, condition in DOMAINS.items()}

        def report(step, actor, xyz, grip):
            nonlocal best, eval_s
            now = perf_counter()
            scores = {name: placement_metrics(evaluate(env, clone_policy(actor), VALIDATION_SCENES))
                      for name, env in envs.items()}
            successes = [score["successes"] for score in scores.values()]
            drops = sum(score["transport_drop_rate"] for score in scores.values())
            rank = (min(successes), sum(successes), -drops, -step)
            if best is None or rank > best:
                best = rank
                torch.save(actor.state_dict(), root / "model.pt")
            curve.append({
                "optimizer_updates": step, "training_s": now - start - eval_s,
                "xyz_mse": xyz, "gripper_bce": grip, "worst_successes_of_12": rank[0],
                "total_successes_of_48": rank[1],
                **{f"{name}_successes": score["successes"] for name, score in scores.items()},
            })
            history.extend({"optimizer_updates": step, "condition": name, **score} for name, score in scores.items())
            write_csv(root / "learning_curve.csv", curve)
            write_csv(root / "validation_conditions.csv", history)
            eval_s += perf_counter() - now
            print(f"{root}: {step} updates, validation by condition {successes}", flush=True)

        report(0, model, None, None)
        train_clone(paths[0], seed + 27000, UPDATES[-1], set(UPDATES[1:]), report,
                    model=model, additional_datasets=paths[1:])
    write_json(root / "config.json", {
        "seed": seed, "teacher": teacher, "new_demo_episodes": data["episodes"],
        "successful_new_demos": data["successful"], "new_demo_steps": data["transitions"],
        "new_online_training_steps": 0, "trained_updates": UPDATES[-1], "selected_updates": -best[-1],
        "selected_worst_successes_of_12": best[0], "selected_total_successes_of_48": best[1],
        "dataset_sources": [dataset_metadata(name, path) for name, path in zip(DOMAINS, paths)],
        "initial_checkpoint": source.as_posix(), "initial_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "validation_seeds": list(VALIDATION_SCENES), "test_seeds": list(TEST_SCENES),
    })
    test_checkpoint(root / "model.pt", output, seed, "adapted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args.seed, args.results, args.out)


if __name__ == "__main__":
    main()
