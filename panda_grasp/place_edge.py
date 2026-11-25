"""Compare equal-budget full expert and learner-conditioned landing demonstrations."""

import argparse
from enum import Enum, auto
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ._vendor.pickup import State
from .experiment import collect_demonstrations, evaluate, write_csv, write_json
from .learning import train_clone
from .place_boundary import BoundaryEvaluation, CANDIDATES, run_case
from .place_cli import clone_policy
from .place_edge_diagnosis import OUTPUT
from .place_low_friction import BUDGET, digest
from .place_robustness import load_clone
from .place_touchdown import build_teacher
from .place_transport_diagnosis import LOW

TEST = range(1_530_000, 1_530_100)
VALIDATION = range(1_531_000, 1_531_024)
VIEWS = ("frozen_previous", "generic_landing", "edge_landing")
CHECKPOINTS = (250, 750, 1500)
LEARNING_RATE = 0.00015


class CollectionPhase(Enum):
    BEFORE_ARRIVAL = auto()


class LandingDemonstrator:
    """Hand over to the same center-seeking expert from either prefix policy."""

    def __init__(self, env, prefix_actor=None):
        self.env = env
        self.prefix_actor = prefix_actor
        self.teacher = build_teacher(env, "verified_touch")
        self.state = CollectionPhase.BEFORE_ARRIVAL
        self.handed_over = False

    def __call__(self, obs):
        if not self.handed_over:
            if not self.env.transported:
                if self.prefix_actor is None:
                    return self.teacher(obs)
                action = self.prefix_actor.action(obs)
                return action, action
            self.teacher.state = State.TRANSPORT
            self.handed_over = True
        data = self.teacher(obs)
        self.state = self.teacher.state
        return data


def previous_path(results, seed):
    cost_path = results / "place_low_friction" / f"seed_{seed}" / "extra_demo_bc" / "cost.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    source = Path(cost["selected_checkpoint"])
    identity = digest(source)
    if identity != cost["checkpoint_sha256"]:
        raise ValueError(f"Previous BC checkpoint has changed: {source}")
    return source, identity


def landing_suffix(directory):
    """Keep only expert actions after arrival in complete, successful demonstrations."""
    dataset = directory / "expert.npz"
    with np.load(dataset) as data:
        phases = data["phase"]
        counts = np.diff(data["offsets"])
        successful = data["successful"]
        suffix = (phases != CollectionPhase.BEFORE_ARRIVAL.name) & np.repeat(successful, counts)
        lengths = np.add.reduceat(suffix.astype(np.int64), data["offsets"][:-1])
        episodes = lengths > 0
        if not episodes.any():
            raise ValueError(f"No successful landing suffixes in {directory}")
        destination = directory / "landing.npz"
        np.savez_compressed(
            destination,
            **{name: data[name][suffix] for name in
               ("observations", "actions", "expert_actions", "next_observations", "rewards", "terminated", "truncated", "phase")},
            offsets=np.r_[0, np.cumsum(lengths[episodes])],
            scenario_seeds=data["scenario_seeds"][episodes], successful=successful[episodes],
        )
    metadata = {"successful_suffixes": int(episodes.sum()), "supervised_suffix_steps": int(suffix.sum())}
    write_json(directory / "landing.json", metadata)
    return destination, metadata


def collect_data(seed, variant, actor, directory):
    # Shared reset order for the two training methods, with separate physical interactions.
    first = 83_300_000 + 100_000 * seed
    with BoundaryEvaluation(CANDIDATES[LOW]) as env:
        collect_demonstrations(
            directory, env, seed, range(first, first + BUDGET),
            lambda rng, perturb: LandingDemonstrator(env, actor if variant == "edge_landing" else None),
            {"condition": LOW, "prefix_policy": variant, "interaction_budget": BUDGET,
             "landing_teacher": "verified_touch from carried arrival"},
            interaction_budget=BUDGET, enable_perturbation=False,
        )
    data = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    if data["transitions"] != BUDGET:
        raise ValueError(f"Incorrect simulation budget in {directory}")
    path, info = landing_suffix(directory)
    print(f"{directory}: {info['successful_suffixes']} successful suffixes, "
          f"{info['supervised_suffix_steps']} labeled steps from {BUDGET} interactions", flush=True)
    return path, {**data, **info}


def score(env, actor):
    rows = evaluate(env, clone_policy(actor), VALIDATION)
    return {
        "stable": sum(row["stable_placement_success"] for row in rows),
        "drops": sum(row["transport_drop"] for row in rows),
        "strict": sum(row["is_success"] for row in rows),
        "median_error_m": float(np.median([row["goal_distance_m"] for row in rows])),
    }


def train(seed, variant, results, output):
    root = output / f"seed_{seed}" / variant
    source, digest_source = previous_path(results, seed)
    actor = load_clone(source)
    dataset, collection = collect_data(seed, variant, actor, root / "demos")
    anchors = [results / "place_low_friction" / f"seed_{seed}" / "demos" / name / "expert.npz"
               for name in ("shared", "bc_only")]
    records = []
    best = None
    elapsed_evaluation = 0.0
    started = perf_counter()
    with BoundaryEvaluation(CANDIDATES[LOW]) as env:
        baseline = score(env, actor)
        elapsed_evaluation = perf_counter() - started

        def report(step, model, xyz, grip):
            nonlocal best, elapsed_evaluation
            begin = perf_counter()
            training_s = max(0.0, begin - started - elapsed_evaluation)
            row = baseline if step == 0 else score(env, model)
            elapsed_evaluation += perf_counter() - begin
            # Within the initial observed drop count, maximize stable placements first.
            rank = (row["drops"] <= baseline["drops"], row["stable"], -row["drops"], -step)
            if best is None or rank > best:
                best = rank
                torch.save(model.state_dict(), root / "model.pt")
            records.append({"optimizer_updates": step, "training_s": training_s, **row,
                            "xyz_mse": xyz, "gripper_bce": grip})
            write_csv(root / "validation.csv", records)
            print(f"{root}: update {step}, stable {row['stable']}/{len(VALIDATION)}, drops {row['drops']}", flush=True)

        report(0, actor, None, None)
        train_clone(anchors[0], seed + 28100, CHECKPOINTS[-1], set(CHECKPOINTS), report,
                    model=actor, additional_datasets=[anchors[1], dataset], learning_rate=LEARNING_RATE)
    write_json(root / "config.json", {
        "method": variant, "seed": seed, "prior_checkpoint": source.as_posix(), "prior_sha256": digest_source,
        "selected_sha256": digest(root / "model.pt"), "selected_updates": -best[-1],
        "attempted_updates": CHECKPOINTS[-1], "learning_rate": LEARNING_RATE,
        "new_demo_steps": BUDGET, "new_online_steps": 0, "training_s": records[-1]["training_s"],
        "collection_s": collection["collection_s"], "supervised_suffix_steps": collection["supervised_suffix_steps"],
        "successful_suffixes": collection["successful_suffixes"],
        "anchor_datasets": [path.as_posix() for path in anchors], "validation": list(VALIDATION), "test": list(TEST),
    })


def test(seed, variant, results, output):
    root = output / f"seed_{seed}" / variant
    prior, identity = previous_path(results, seed)
    path = prior
    if variant != VIEWS[0]:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        path = root / "model.pt"
        if config["selected_sha256"] != digest(path) or config["prior_sha256"] != identity:
            raise ValueError(f"Checkpoint verification failed in {root}")
    run_case(root / "test", LOW, clone_policy(load_clone(path)), TEST,
             {"method": variant, "seed": seed, "checkpoint": path.as_posix(),
              "checkpoint_sha256": digest(path), "initial_sha256": identity})


def fixed_checkpoint(seed, variant, results, output):
    """Rebuild the first planned update checkpoint using the already charged demonstrations."""
    root = output / f"seed_{seed}" / variant
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    fixed = CHECKPOINTS[0]
    if config["selected_updates"] == fixed:
        print(f"{root}: reuse selected evaluation at {fixed} updates", flush=True)
        return
    source, identity = previous_path(results, seed)
    if config["prior_sha256"] != identity:
        raise ValueError(f"Initial checkpoint changed in {root}")
    anchors = [Path(path) for path in config["anchor_datasets"]]
    actor = train_clone(anchors[0], seed + 28100, fixed, set(), None, model=load_clone(source),
                        additional_datasets=[anchors[1], root / "demos" / "landing.npz"],
                        learning_rate=LEARNING_RATE)
    path = root / "model_fixed_250.pt"
    torch.save(actor.state_dict(), path)
    run_case(root / "fixed_250_test", LOW, clone_policy(actor), TEST,
             {"method": variant, "seed": seed, "checkpoint": path.as_posix(), "selected_updates": fixed,
              "checkpoint_sha256": digest(path), "initial_sha256": identity})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test", "fixed"))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--method", choices=VIEWS, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        if args.method == VIEWS[0]:
            parser.error("Frozen weights only support test")
        train(args.seed, args.method, args.results, args.out)
    elif args.command == "fixed":
        if args.method == VIEWS[0]:
            parser.error("Frozen weights have no fixed-update checkpoint")
        fixed_checkpoint(args.seed, args.method, args.results, args.out)
    else:
        test(args.seed, args.method, args.results, args.out)


if __name__ == "__main__":
    main()
