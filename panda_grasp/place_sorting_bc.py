"""Continue mixed BC on expert trajectories collected in the live sorting scene."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .experiment import save_demonstrations, write_csv, write_json
from .learning import Clone, train_clone
from .place_objects import ACTION_NAMES, OBS_SIZE, SHAPES
from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_objects_train import sha
from .place_sorting import SortingScene, evaluate_actor, load_mixed, run_scene

OUTPUT = Path("results/place_sorting_bc")
TRAIN_SCENES = 24
VALIDATION = range(1_562_000, 1_562_010)
TEST = range(1_563_000, 1_563_030)
CHECKPOINTS = (250, 700, 1300)
LEARNING_RATE = 0.00025


def collect(seed, root):
    directory = root / "demonstrations"
    start = 86_300_000 + 100_000 * seed
    scenes, items = [], []
    groups = {name: [] for name in SHAPES}
    started = perf_counter()
    with SortingScene() as env:
        for scenario in range(start, start + TRAIN_SCENES):
            traces = []
            scene, rows = run_scene(env, scenario, demonstrations=traces)
            scenes.append(scene)
            items.extend(rows)
            for row, trajectory in zip(rows, traces, strict=True):
                if len(trajectory) != row["attempt_steps"] or not any(trajectory[-1][5:7]):
                    raise ValueError(f"Incomplete expert trace: {scenario}, {row['shape']}")
                groups[row["shape"]].append((trajectory, {**row, "is_success": row["sorted"]}))
    paths, records = [], []
    for name, pairs in groups.items():
        trajectories, outcomes = zip(*pairs)
        if not any(row["is_success"] for row in outcomes):
            raise ValueError(f"No complete expert placement for {name}, training seed {seed}")
        path = directory / name / "expert.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_demonstrations(path, trajectories, outcomes)
        paths.append(path)
        records.append({"shape": name, "attempts": len(pairs),
                        "successful": sum(row["is_success"] for row in outcomes),
                        "expert_action_steps": sum(len(t) for t in trajectories),
                        "sha256": sha(path)})
    write_csv(directory / "scenes.csv", scenes)
    write_csv(directory / "items.csv", items)
    metadata = {
        "seed": seed, "scenes": TRAIN_SCENES, "expert_all_sorted": sum(r["all_sorted"] for r in scenes),
        "expert_action_steps": sum(r["expert_action_steps"] for r in records),
        "homing_action_steps": sum(row["homing_steps"] for row in scenes),
        "collection_s": perf_counter() - started, "action_names": ACTION_NAMES, "obs_size": OBS_SIZE,
        "shape_data": records,
    }
    write_json(directory / "dataset.json", metadata)
    print(f"Seed {seed}: collected {TRAIN_SCENES} scenes, "
          f"{sum(r['successful'] for r in records)}/{len(items)} complete placements", flush=True)
    return paths, metadata


def validate(env, actor):
    scenes = items = drops = lifts = 0
    sorted_by_shape = dict.fromkeys(SHAPES, 0)
    for seed in VALIDATION:
        result, rows = run_scene(env, seed, actor)
        scenes += result["all_sorted"]
        items += result["sorted_items"]
        drops += sum(row["transport_drop"] for row in rows)
        lifts += sum(row["lifted"] for row in rows)
        for row in rows:
            sorted_by_shape[row["shape"]] += row["sorted"]
    return {"all_sorted": scenes, "items_sorted": items, "transport_drops": drops, "lifted": lifts,
            **{f"{name}_sorted": value for name, value in sorted_by_shape.items()}}


def train(seed, source, output):
    root = output / "training" / f"seed_{seed}"
    fresh, cost = collect(seed, root)
    actor, identity = load_mixed(seed, source)
    historical = [source / "training" / f"seed_{seed}" / "demonstrations" / name / "expert.npz" for name in SHAPES]
    paths = fresh + historical
    evaluations, best = [], None
    spent_validation = 0.0
    start = perf_counter()
    with SortingScene() as env:
        baseline = validate(env, actor)
        spent_validation = perf_counter() - start

        def report(step, model, xyz, grip):
            nonlocal best, spent_validation
            begin = perf_counter()
            result = baseline if step == 0 else validate(env, model)
            rank = (result["cylinder_sorted"] >= baseline["cylinder_sorted"],
                    result["all_sorted"], result["items_sorted"], -result["transport_drops"], -step)
            if best is None or rank > best:
                best = rank
                torch.save(actor.state_dict(), root / "model.pt")
            evaluations.append({"updates": step, "training_s": max(0.0, begin - start - spent_validation),
                                "xyz_mse": xyz, "gripper_bce": grip, **result})
            spent_validation += perf_counter() - begin
            write_csv(root / "validation.csv", evaluations)
            print(f"Seed {seed}, updates {step}: {result['all_sorted']}/{len(VALIDATION)} full, "
                  f"{result['items_sorted']}/{3*len(VALIDATION)} items, drops {result['transport_drops']}", flush=True)

        report(0, actor, None, None)
        train_clone(paths[0], seed + 168270, CHECKPOINTS[-1], CHECKPOINTS, report, model=actor,
                    additional_datasets=paths[1:], learning_rate=LEARNING_RATE)
    write_json(root / "config.json", {
        "seed": seed, "initial_sha256": identity, "checkpoint_sha256": sha(root / "model.pt"),
        "attempted_updates": CHECKPOINTS[-1], "selected_updates": -best[-1],
        "learning_rate": LEARNING_RATE, "training_s": evaluations[-1]["training_s"],
        "new_expert_steps": cost["expert_action_steps"], "new_homing_steps": cost["homing_action_steps"],
        "new_online_learning_steps": 0, "validation_scenarios": list(VALIDATION), "test_scenarios": list(TEST),
        "datasets": [path.as_posix() for path in paths], "expert_scenes": TRAIN_SCENES,
    })


def load_policy(method, seed, source, output):
    if method == "expert":
        return None, None
    if method == "frozen":
        return load_mixed(seed, source)
    directory = output / "training" / f"seed_{seed}"
    source_weights = directory / "model.pt"
    receipt = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    identity = sha(source_weights)
    if identity != receipt["checkpoint_sha256"]:
        raise ValueError(f"Trained checkpoint differs from its saved hash: {source_weights}")
    actor = Clone(OBS_SIZE, len(ACTION_NAMES))
    actor.load_state_dict(torch.load(source_weights, weights_only=True, map_location="cpu"))
    actor.eval()
    return actor, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--source", type=Path, default=OBJECT_RESULTS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--method", choices=("expert", "frozen", "adapted"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        if args.seed is None:
            parser.error("train requires a training seed")
        train(args.seed, args.source, args.out)
    else:
        if args.method is None or (args.method != "expert" and args.seed is None):
            parser.error("test requires a method, and learned policies require a training seed")
        actor, identity = load_policy(args.method, args.seed, args.source, args.out)
        evaluate_actor(args.method, args.seed, actor, identity, args.out / "test", TEST)


if __name__ == "__main__":
    main()
