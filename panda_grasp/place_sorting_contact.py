"""Replay old demonstrations with contact signals and compare budget-matched BC."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .experiment import write_json
from .learning import train_clone
from .place_contact_signals import CONTACTS, TACTILE_SIZE, ContactClone, ContactObjectScene, ContactSortingScene
from .place_objects import OBS_SIZE, SHAPES
from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_objects_train import sha
from .place_sorting import SortingScene, evaluate_actor, load_mixed, run_scene
from .place_sorting_bc import LEARNING_RATE, OUTPUT as SORTING_RESULTS, load_policy as load_sorting_policy

OUTPUT = Path("results/place_sorting_contact")
TEST = range(1_564_000, 1_564_030)
METHODS = ("binary", "continuous")


def read_dataset(path):
    with np.load(path) as data:
        return dict(data)


def traces(data):
    return {int(seed): slice(start, stop)
            for seed, start, stop in zip(data["scenario_seeds"], data["offsets"][:-1], data["offsets"][1:], strict=True)}


def verify_equal(observed, expected, note):
    if not np.array_equal(observed, expected):
        raise ValueError(f"Expert replay differs in {note}")


def with_signals(data, observations, following, path):
    verify_equal(observations[:, :OBS_SIZE], data["observations"], f"{path}: observations")
    verify_equal(following[:, :OBS_SIZE], data["next_observations"], f"{path}: next observations")
    if not np.isfinite(observations).all() or not np.isfinite(following).all():
        raise ValueError(f"Nonfinite sensor readings: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{**data, "observations": observations, "next_observations": following})
    extra = observations[:, OBS_SIZE:]
    return {
        "transitions": len(extra), "episodes": len(data["successful"]),
        "successful": int(data["successful"].sum()), "augmented_sha256": sha(path),
        "contact_samples": int(np.any(extra[:, :2] > 0, axis=1).sum()),
        "slip_samples": int(np.any(extra[:, 2:] != 0, axis=1).sum()),
    }


def augment_sorting(paths, root):
    archives = {name: read_dataset(path) for name, path in zip(SHAPES, paths, strict=True)}
    indexed = {name: traces(data) for name, data in archives.items()}
    seeds = list(indexed["cube"])
    if any(list(indexed[name]) != seeds for name in SHAPES):
        raise ValueError("Sorting demonstration objects must share the same scenes")
    before = {name: np.empty((len(data["actions"]), TACTILE_SIZE), dtype=np.float32) for name, data in archives.items()}
    after = {name: np.empty_like(values) for name, values in before.items()}
    homing_steps = 0
    with ContactSortingScene() as env:
        for index, seed in enumerate(seeds):
            recorded = []
            scene, outcomes = run_scene(env, seed, demonstrations=recorded)
            homing_steps += scene["homing_steps"]
            for row, episode in zip(outcomes, recorded, strict=True):
                name = row["shape"]
                data = archives[name]
                segment = indexed[name][seed]
                obs, actions, labels, following, rewards, terminated, truncated, phase = map(np.asarray, zip(*episode))
                for key, array in (("actions", actions), ("expert_actions", labels), ("rewards", rewards.astype(np.float32)),
                                   ("terminated", terminated), ("truncated", truncated), ("phase", phase)):
                    verify_equal(array, data[key][segment], f"sorting {seed}, {name}, {key}")
                verify_equal(row["sorted"], data["successful"][index], f"sorting {seed}, {name}, success")
                before[name][segment] = obs
                after[name][segment] = following
    augmented, metadata = [], []
    for name, source in zip(SHAPES, paths, strict=True):
        path = root / "sorting" / name / "expert.npz"
        record = with_signals(archives[name], before[name], after[name], path)
        augmented.append(path)
        metadata.append({"domain": "sorting", "shape": name, "original_sha256": sha(source), **record})
    return augmented, metadata, homing_steps


def augment_single(paths, root):
    augmented, metadata = [], []
    for name, source in zip(SHAPES, paths, strict=True):
        data = read_dataset(source)
        before = np.empty((len(data["actions"]), TACTILE_SIZE), dtype=np.float32)
        after = np.empty_like(before)
        with ContactObjectScene(name, True, True) as env:
            for seed, segment in traces(data).items():
                obs, _ = env.reset(seed=seed)
                for i in range(segment.start, segment.stop):
                    before[i] = obs
                    obs, reward, terminated, truncated, _ = env.step(data["actions"][i])
                    after[i] = obs
                    verify_equal(data["rewards"][i], np.float32(reward), f"single {name}, {i}, reward")
                    verify_equal(terminated, data["terminated"][i], f"single {name}, {i}, termination")
                    budget_cutoff = i == len(before) - 1 and data["truncated"][i] and not terminated
                    if not budget_cutoff:
                        verify_equal(truncated, data["truncated"][i], f"single {name}, {i}, truncation")
        path = root / "single" / name / "expert.npz"
        record = with_signals(data, before, after, path)
        augmented.append(path)
        metadata.append({"domain": "single", "shape": name, "original_sha256": sha(source), **record})
    return augmented, metadata


def train(seed, source, sorting, output):
    previous = sorting / "training" / f"seed_{seed}"
    config = json.loads((previous / "config.json").read_text(encoding="utf-8"))
    original = [Path(path) for path in config["datasets"]]
    updates = config["selected_updates"]
    if len(original) != 2 * len(SHAPES) or config["learning_rate"] != LEARNING_RATE or not updates:
        raise ValueError("Expected the six archived BC sources with the matched update budget")
    root = output / "training" / f"seed_{seed}"
    initial, identity = load_mixed(seed, source)
    control, control_identity = load_sorting_policy("adapted", seed, source, sorting)
    if config["initial_sha256"] != identity:
        raise ValueError(f"The starting weights differ for seed {seed}")
    fit_seed = seed + 168270
    rebuilt = train_clone(original[0], fit_seed, updates, set(), None, model=load_mixed(seed, source)[0],
                          additional_datasets=original[1:], learning_rate=LEARNING_RATE)
    for key, parameter in rebuilt.state_dict().items():
        verify_equal(parameter.numpy(), control.state_dict()[key].numpy(), f"binary checkpoint {seed}, {key}")
    print(f"Seed {seed}: binary checkpoint exactly reproduced at {updates} updates", flush=True)
    start = perf_counter()
    sorting_paths, sorting_meta, homing = augment_sorting(original[:len(SHAPES)], root / "datasets")
    print(f"Seed {seed}: verified all archived sorting actions and base observations", flush=True)
    single_paths, single_meta = augment_single(original[len(SHAPES):], root / "datasets")
    replay_s = perf_counter() - start
    all_paths = sorting_paths + single_paths
    model = ContactClone().initialize(initial)
    start = perf_counter()
    train_clone(all_paths[0], fit_seed, updates, set(), None, model=model,
                additional_datasets=all_paths[1:], learning_rate=LEARNING_RATE)
    training_s = perf_counter() - start
    path = root / "model.pt"
    torch.save(model.state_dict(), path)
    write_json(root / "config.json", {
        "seed": seed, "starting_sha256": identity, "binary_checkpoint_sha256": control_identity,
        "continuous_checkpoint_sha256": sha(path), "optimizer_updates_each": updates, "learning_rate": LEARNING_RATE,
        "same_actions_and_base_observations": True, "binary_checkpoint_reproduced_exactly": True,
        "sensor_features": CONTACTS, "replayed_homing_steps": homing, "replay_s": replay_s, "training_s": training_s,
        "new_demonstration_scenes": 0, "new_online_training_steps": 0, "datasets": sorting_meta + single_meta,
        "test_scenarios": list(TEST),
    })
    print(f"Seed {seed}: trained the continuous-contact model for {updates} updates", flush=True)


def load_evaluation(method, seed, source, sorting, output):
    if method == "binary":
        return load_sorting_policy("adapted", seed, source, sorting)
    root = output / "training" / f"seed_{seed}"
    path = root / "model.pt"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    identity = sha(path)
    if identity != config["continuous_checkpoint_sha256"]:
        raise ValueError(f"Continuous-contact checkpoint hash mismatch: {path}")
    model = ContactClone()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    return model, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--source", type=Path, default=OBJECT_RESULTS)
    parser.add_argument("--sorting", type=Path, default=SORTING_RESULTS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        train(args.seed, args.source, args.sorting, args.out)
    elif args.method is None:
        parser.error("test requires --method")
    else:
        model, identity = load_evaluation(args.method, args.seed, args.source, args.sorting, args.out)
        scene_type = ContactSortingScene if args.method == "continuous" else SortingScene
        evaluate_actor(args.method, args.seed, model, identity, args.out / "test", TEST, scene_type)


if __name__ == "__main__":
    main()
