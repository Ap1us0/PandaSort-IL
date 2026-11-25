"""Warm-start mixed-object cloning with wrist-yaw actions."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from .experiment import collect_demonstrations, evaluate, write_csv, write_json
from .learning import Clone, train_clone
from .place import PLACE_OBS_SIZE
from .place_cli import clone_policy
from .place_objects import ACTION_NAMES, OBS_SIZE, ObjectPlacementEnv, YawTeacher
from .place_objects_benchmark import OUTPUT, TEST, brick_long_teacher, evaluate_cell, source_policy, yaw_teacher
from .place_robustness import load_clone

BUDGET = 9000
UPDATES = (250, 700, 1300, 2000)
VALIDATION = range(1_551_000, 1_551_012)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(path):
    prior = load_clone(path)
    model = Clone(OBS_SIZE, len(ACTION_NAMES))
    weights = prior.state_dict()
    weights["features.0.weight"] = torch.nn.functional.pad(weights["features.0.weight"], (0, OBS_SIZE - PLACE_OBS_SIZE))
    weights["mean.weight"] = torch.nn.functional.pad(weights["mean.weight"], (0, 0, 0, 1))
    weights["mean.bias"] = torch.nn.functional.pad(weights["mean.bias"], (0, 1))
    model.load_state_dict(weights)
    with torch.no_grad():
        observations = torch.linspace(-1, 1, 11 * OBS_SIZE).reshape(11, OBS_SIZE)
        actions = model(observations)
        torch.testing.assert_close(actions[:, :4], prior(observations[:, :PLACE_OBS_SIZE]), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actions[:, 4], torch.zeros_like(actions[:, 4]), rtol=0, atol=0)
    return model


def protocol(output):
    return json.loads((output / "selection.json").read_text(encoding="utf-8"))


def demonstrations(seed, output, shapes, teacher_modes):
    if not shapes or BUDGET % len(shapes):
        raise ValueError("The validated shapes must divide the demonstration budget")
    steps = BUDGET // len(shapes)
    paths = []
    costs = []
    for i, shape in enumerate(shapes):
        directory = output / "demonstrations" / shape
        start = 85_200_000 + seed * 100_000 + i * 10_000
        with ObjectPlacementEnv(shape, True, True) as env:
            collect_demonstrations(
                directory, env, seed, range(start, start + steps),
                lambda rng, perturb: YawTeacher(env, brick_long_axis=teacher_modes[shape] == "long_axis"),
                {"shape": shape, "random_yaw": True, "expert": "aligned_verified_touch",
                 "teacher_mode": teacher_modes[shape]},
                interaction_budget=steps, enable_perturbation=False, action_names=ACTION_NAMES,
            )
        info = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
        if info["transitions"] != steps or info["successful"] == 0:
            raise ValueError(f"Incorrect budget or no usable demonstrations for {shape}")
        paths.append(directory / "expert.npz")
        costs.append(info)
        print(f"Seed {seed}, {shape}: {steps} steps, {info['successful']}/{info['episodes']} demonstrations usable", flush=True)
    return paths, costs


def train(output, results, seed):
    design = protocol(output)
    shapes = design["selected_shapes"]
    directory = output / "training" / f"seed_{seed}"
    paths, costs = demonstrations(seed, directory, shapes, design["teacher_modes"])
    source, identity = source_policy(results, seed)
    actor = expand(source)
    started = perf_counter()
    evaluation_s = 0.0
    history, best = [], None
    with ExitStack() as stack:
        envs = {shape: stack.enter_context(ObjectPlacementEnv(shape, True, True)) for shape in shapes}

        def score(step, model, xyz, grip):
            nonlocal best, evaluation_s
            begin = perf_counter()
            trained_s = begin - started - evaluation_s
            results = {shape: evaluate(env, clone_policy(model), VALIDATION) for shape, env in envs.items()}
            stable = {name: sum(row["stable_placement_success"] for row in runs) for name, runs in results.items()}
            drops = sum(row["transport_drop"] for runs in results.values() for row in runs)
            ordering = (min(stable.values()), sum(stable.values()), -drops, -step)
            if best is None or ordering > best:
                best = ordering
                torch.save(model.state_dict(), directory / "model.pt")
            evaluation_s += perf_counter() - begin
            history.append({"updates": step, "training_s": trained_s, "xyz_mse": xyz, "gripper_bce": grip,
                            "transport_drops": drops, **{f"{shape}_stable": value for shape, value in stable.items()}})
            write_csv(directory / "validation.csv", history)
            print(f"Seed {seed}, updates {step}: {stable}, drops {drops}", flush=True)

        score(0, actor, None, None)
        train_clone(paths[0], seed + 55123, UPDATES[-1], set(UPDATES), score, model=actor,
                    additional_datasets=paths[1:], learning_rate=0.00035)
    write_json(directory / "config.json", {
        "seed": seed, "source_checkpoint_sha256": identity, "checkpoint_sha256": sha(directory / "model.pt"),
        "trained_updates": UPDATES[-1], "selected_updates": -best[-1], "training_seconds": history[-1]["training_s"],
        "new_expert_steps": sum(d["transitions"] for d in costs), "new_online_steps": 0,
        "demonstrations": costs, "action_names": ACTION_NAMES, "input_size": OBS_SIZE,
    })


def no_yaw_policy(actor):
    def factory(env, scene):
        def action(obs):
            values = actor.action(obs)
            values[4] = 0.0
            return values
        return action
    return factory


def test(output, results, which, seed):
    if which == "expert":
        design = protocol(output)
        for shape in TEST:
            teacher = brick_long_teacher if design["teacher_modes"][shape] == "long_axis" else yaw_teacher
            evaluate_cell(output / "test" / which / shape, shape, True, True, teacher, TEST[shape],
                          method=which, seed=None)
        return
    if which in ("mixed", "mixed_no_yaw"):
        directory = output / "training" / f"seed_{seed}"
        path = directory / "model.pt"
        metadata = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if sha(path) != metadata["checkpoint_sha256"]:
            raise ValueError(f"Saved model changed: {path}")
        actor = Clone(OBS_SIZE, len(ACTION_NAMES))
        actor.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        actor.eval()
    else:
        path, _ = source_policy(results, seed)
        actor = load_clone(path)
    disabled = which == "mixed_no_yaw"
    shapes = ("cube", "brick") if disabled else TEST
    policy = no_yaw_policy(actor) if disabled else clone_policy(actor)
    for shape in shapes:
        directory = output / "test" / which / f"seed_{seed}" / shape
        scenes = TEST[shape][:12] if disabled else TEST[shape]
        evaluate_cell(directory, shape, True, which != "frozen", policy, scenes,
                      method=which, seed=seed, checkpoint=path.as_posix(), checkpoint_sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--method", choices=("frozen", "mixed", "expert", "mixed_no_yaw"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        if args.seed is None:
            parser.error("train requires --seed")
        train(args.out, args.results, args.seed)
    else:
        if args.method is None or (args.method != "expert" and args.seed is None):
            parser.error("test requires --method, and learned methods require --seed")
        test(args.out, args.results, args.method, args.seed)


if __name__ == "__main__":
    main()
