"""Select feasible changed conditions, then adapt BC with 12 full demonstrations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import collect_demonstrations, evaluate, write_csv, write_json
from .learning import train_clone
from .physics_shift import CONDITIONS
from .place_cli import clone_policy
from .place_robustness import (
    OUTPUT, SCREEN_SCENES, TEST_SCENES, ShiftedPlacementEnv, TransportTeacher,
    load_clone, placement_metrics, run_evaluation,
)

DEMO_EPISODES = 12
UPDATES = (0, 250, 750, 1500, 3000)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_conditions(root):
    screen = root / "screen"
    cases = []
    for condition in ("friction_50", "mass_250"):
        runs = [read_json(screen / f"seed_{seed}" / condition / "summary.json") for seed in range(3)]
        expert = read_json(screen / "expert" / condition / "summary.json")
        slower = screen / "expert_speed_0.75" / condition / "summary.json"
        if expert["successes"] < 10 and slower.exists():
            alternative = read_json(slower)
            if alternative["successes"] > expert["successes"]:
                expert = alternative
        failures = Counter()
        for result in runs:
            failures.update(result["failure_stages"])
        frozen_successes = sum(result["successes"] for result in runs)
        cases.append({
            "condition": condition, "frozen_successes_of_36": frozen_successes,
            "expert_successes_of_12": expert["successes"], "expert_speed": expert["teacher_speed"],
            "frozen_failures_by_stage": dict(failures),
            "adapt": expert["successes"] >= 10 and frozen_successes <= 30,
        })
    write_json(root / "selection.json", {
        "screen_seeds": list(SCREEN_SCENES), "expert_minimum": 10,
        "frozen_maximum": 30, "cases": cases,
    })
    print(cases, flush=True)


def adapt(seed, condition, root, checkpoints):
    config = read_json(root / "selection.json")
    case = next(case for case in config["cases"] if case["condition"] == condition)
    if not case["adapt"]:
        raise ValueError(f"{condition} did not pass the adaptation screen")
    directory = root / "adapt" / f"seed_{seed}" / condition
    demos = directory / "demonstrations"
    origin = checkpoints / f"seed_{seed}" / "model.pt"
    speed = case["expert_speed"]
    with ShiftedPlacementEnv(CONDITIONS[condition]) as env:
        dataset = demos / "expert.npz"
        if not dataset.exists():
            seed_base = 71_000_000 if condition == "friction_50" else 72_000_000
            first = seed_base + 10_000 * seed
            collect_demonstrations(
                demos, env, seed, range(first, first + DEMO_EPISODES),
                lambda rng, perturb: TransportTeacher(env, speed),
                {"condition": condition, "teacher_speed": speed},
            )
        data = read_json(demos / "dataset.json")
        if (data["seed"], data["episodes"], data["condition"], data["teacher_speed"]) != (
            seed, DEMO_EPISODES, condition, speed,
        ):
            raise ValueError("Existing demonstrations do not match the selected condition")
        if not data["successful"]:
            raise ValueError(f"No successful demonstrations in {demos}")
        curve = []
        best = None

        def report(step, model, xyz, grip):
            nonlocal best
            result = placement_metrics(evaluate(env, clone_policy(model), SCREEN_SCENES))
            curve.append({"optimizer_updates": step, "xyz_mse": xyz, "gripper_bce": grip, **result})
            write_csv(directory / "learning_curve.csv", curve)
            candidate = (result["successes"], -result["transport_drop_rate"], -step)
            if best is None or candidate > best:
                best = candidate
                torch.save(model.state_dict(), directory / "model.pt")
            print(f"{directory}: BC {step}, validation {result['successes']}/12", flush=True)

        model = load_clone(origin)
        report(0, model, None, None)
        train_clone(dataset, seed + 8000, UPDATES[-1], set(UPDATES[1:]), report, model=model)
        config = {
            "condition": condition, "seed": seed, "new_demonstration_steps": data["transitions"],
            "new_demonstration_episodes": data["episodes"], "successful_demonstrations": data["successful"],
            "training_updates": UPDATES[-1], "selected_updates": -best[-1],
            "selected_screen_successes": best[0], "original_checkpoint": origin.as_posix(),
            "original_checkpoint_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        }
        write_json(directory / "config.json", config)
        run_evaluation(env, clone_policy(load_clone(directory / "model.pt")), TEST_SCENES, directory, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "train"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--condition", choices=("friction_50", "mass_250"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoints", type=Path, default=Path("results/pick_place"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "select":
        select_conditions(args.out)
    elif args.seed is None or args.condition is None:
        parser.error("train requires --seed and --condition")
    else:
        adapt(args.seed, args.condition, args.out, args.checkpoints)


if __name__ == "__main__":
    main()
