"""Collect, train, evaluate and render target-conditioned placement policies."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .env import OBS_SIZE
from .experiment import (
    collect_demonstrations, evaluate, record_video, summarize, write_csv, write_json,
)
from .learning import Clone, train_clone
from .place import PLACE_OBS_SIZE, PandaPickPlaceEnv
from .place_expert import PlacementExpert

VALIDATION_SCENES = range(1_100_000, 1_100_012)
TEST_SCENES = range(1_200_000, 1_200_100)
RESULTS = Path("results/pick_place")


def expert_policy(env, seed):
    expert = PlacementExpert(env)
    return lambda obs: expert(obs)[1]


def clone_policy(model):
    return lambda env, seed: model.action


@torch.no_grad()
def expand_lift_clone(path):
    model = Clone(PLACE_OBS_SIZE)
    weights = torch.load(path, map_location="cpu", weights_only=True)
    weights["features.0.weight"] = torch.nn.functional.pad(
        weights["features.0.weight"], (0, PLACE_OBS_SIZE - OBS_SIZE),
    )
    model.load_state_dict(weights, strict=True)
    return model


def metrics(rows):
    output = summarize(rows)
    output.update(
        transport_rate=float(np.mean([row["transported"] for row in rows])),
        release_rate=float(np.mean([row["released_at_target"] for row in rows])),
        mean_goal_error_m=float(np.mean([row["goal_distance_m"] for row in rows])),
        mean_stable_placement_s=float(np.mean([row["settled_at_target_s"] for row in rows])),
    )
    return output


def test_policy(env, policy, directory):
    rows = evaluate(env, policy, TEST_SCENES)
    write_csv(directory / "test_episodes.csv", rows)
    result = metrics(rows)
    write_json(directory / "test_summary.json", result)
    print(f"{directory}: {result['successes']}/{len(rows)} successful placements", flush=True)


def train(args, env):
    directory = args.out / f"seed_{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    dataset_file = directory / "expert.npz"
    metadata_path = directory / "dataset.json"
    lift_path = args.lift_dir / f"seed_{args.seed}" / "bc" / "model.pt"
    metadata = {"task": "PandaPickPlace-v0", "obs_size": PLACE_OBS_SIZE, "contact_trigger_expert": True}
    if dataset_file.is_file():
        data_config = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(data_config.get(key) != value for key, value in
               {**metadata, "seed": args.seed, "episodes": args.demos}.items()):
            raise ValueError("Existing demonstration data does not match this run")
    else:
        start_scene = 30_000_000 + 10_000 * args.seed
        collect_demonstrations(
            directory, env, args.seed, range(start_scene, start_scene + args.demos),
            lambda rng, perturb: PlacementExpert(env, rng, perturb=perturb), metadata,
        )

    model = expand_lift_clone(lift_path)
    write_json(directory / "config.json", {
        "seed": args.seed, "demos": args.demos, "updates": args.updates, "max_steps": env.max_steps,
        "lift_checkpoint": lift_path.as_posix(),
        "lift_checkpoint_sha256": hashlib.sha256(lift_path.read_bytes()).hexdigest(),
        "validation_seeds": list(VALIDATION_SCENES), "test_seeds": list(TEST_SCENES),
    })
    curve = []
    eval_time = 0
    started = perf_counter()

    def report(step, current, xyz_loss, grip_loss):
        nonlocal eval_time
        now = perf_counter()
        result = metrics(evaluate(env, clone_policy(current), VALIDATION_SCENES))
        curve.append({"updates": step, "training_s": now - started - eval_time,
                      "xyz_mse": xyz_loss, "gripper_bce": grip_loss, **result})
        write_csv(directory / "learning_curve.csv", curve)
        eval_time += perf_counter() - now
        print(f"{directory}: update {step}, validation {result['successes']}/{len(VALIDATION_SCENES)}", flush=True)

    report(0, model, None, None)
    model = train_clone(dataset_file, args.seed, args.updates, {1000, 3000, 6000, args.updates}, report, model=model)
    torch.save(model.state_dict(), directory / "model.pt")
    test_policy(env, clone_policy(model), directory)


def video(args, env):
    directory = args.out / f"seed_{args.seed}"
    if args.use_expert:
        policy = expert_policy
        directory = args.out / "expert"
    else:
        model = Clone(PLACE_OBS_SIZE)
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
        policy = clone_policy(model)
    with (directory / "test_episodes.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for outcome, success in (("success", "True"), ("failure", "False")):
        row = next((row for row in rows if row["is_success"] == success), None)
        if row:
            destination = directory / f"{outcome}_{row['scenario_seed']}.mp4"
            record_video(env, policy, row, destination)
            print(destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "expert", "videos"))
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demos", type=int, default=160)
    parser.add_argument("--updates", type=int, default=9000)
    parser.add_argument("--lift-dir", type=Path, default=Path("results/contact_trigger"))
    parser.add_argument("--use-expert", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or not 1 <= args.demos <= 10000 or args.updates < 1:
        parser.error("Expected a nonnegative seed, 1–10,000 demonstrations and a positive update budget")
    torch.set_num_threads(1)
    with PandaPickPlaceEnv(render_mode="rgb_array" if args.command == "videos" else None) as env:
        if args.command == "train":
            train(args, env)
        elif args.command == "expert":
            test_policy(env, expert_policy, args.out / "expert")
        else:
            video(args, env)


if __name__ == "__main__":
    main()
