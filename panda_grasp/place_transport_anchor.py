"""Compare a fixed 250-update BC fit with a transport-action preservation loss."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .env import OBS_SIZE
from .experiment import write_csv, write_json
from .learning import train_clone
from .place_anchor_diagnosis import OUTPUT
from .place_boundary import run_case
from .place_cli import clone_policy
from .place_edge import BUDGET, CHECKPOINTS, LEARNING_RATE, previous_path
from .place_low_friction import digest
from .place_robustness import load_clone
from .place_transport_diagnosis import LOW

TEST = range(1_540_000, 1_540_100)
METHODS = ("frozen", "ordinary", "transport_anchored")
STEPS = CHECKPOINTS[0]
ANCHOR_WEIGHT = 20.0
ACTION_WEIGHTS = (1.0, 1.0, 3.0, 2.0)
NEAR_M = 0.085
CONTACT_APPROACH_M = 0.05
BATCH_PER_REGION = 48


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_data(results, seed):
    root = results / "place_edge" / f"seed_{seed}" / "generic_landing"
    previous = read_json(root / "config.json")
    if previous["new_demo_steps"] != BUDGET or previous["new_online_steps"] != 0:
        raise ValueError("The control must use the same demonstration budget")
    comparison_dir = root / ("test" if previous["selected_updates"] == STEPS else "fixed_250_test")
    archived = read_json(comparison_dir / "summary.json")
    ordinary = Path(archived["checkpoint"])
    if digest(ordinary) != archived["checkpoint_sha256"]:
        raise ValueError(f"Ordinary policy differs from archive: {ordinary}")
    if previous["learning_rate"] != LEARNING_RATE:
        raise ValueError("The control must use the same cloning learning rate")
    frozen, identity = previous_path(results, seed)
    if previous["prior_sha256"] != identity or archived["initial_sha256"] != identity:
        raise ValueError("The ordinary policy has a different parent checkpoint")
    anchors = [Path(path) for path in previous["anchor_datasets"]]
    raw = root / "demos" / "expert.npz"
    suffix = root / "demos" / "landing.npz"
    if read_json(root / "demos" / "dataset.json")["transitions"] != BUDGET:
        raise ValueError("The supplementary demonstration data must use exactly the same budget")
    return frozen, identity, ordinary, anchors, raw, suffix


def transport_observations(paths):
    regions = [[], [], []]
    for path in paths:
        with np.load(path) as data:
            successful = np.repeat(data["successful"], np.diff(data["offsets"]))
            obs = data["observations"][successful]
        not_arrived = obs[:, OBS_SIZE + 5] < 0.5
        lifted = obs[:, OBS_SIZE + 4] > 0.5
        distance_to_cube = np.linalg.norm(obs[:, OBS_SIZE - 6:OBS_SIZE - 3], axis=1) * 0.12
        near_grasp = not_arrived & ~lifted & (distance_to_cube <= CONTACT_APPROACH_M)
        regions[0].append(obs[near_grasp])
        carrying = obs[lifted & not_arrived]
        near = np.linalg.norm(carrying[:, OBS_SIZE:OBS_SIZE + 2], axis=1) * 0.25 <= NEAR_M
        regions[1].append(carrying[~near])
        regions[2].append(carrying[near])
    tensors = [torch.from_numpy(np.concatenate(parts)) for parts in regions]
    if any(len(region) == 0 for region in tensors):
        raise ValueError("Transport anchoring requires grasp, transport and near-goal observations")
    return tensors


class TransportActionAnchor:
    def __init__(self, reference, regions, seed):
        self.regions = regions
        with torch.no_grad():
            self.targets = [reference(obs) for obs in regions]
        self.generator = torch.Generator().manual_seed(seed + 73900)
        self.action_weights = torch.tensor(ACTION_WEIGHTS)
        self.losses = []

    def __call__(self, model):
        observations, labels = [], []
        for xs, ys in zip(self.regions, self.targets):
            batch = torch.randint(len(xs), (BATCH_PER_REGION,), generator=self.generator)
            observations.append(xs[batch])
            labels.append(ys[batch])
        delta = model(torch.cat(observations)) - torch.cat(labels)
        loss = (delta.square() * self.action_weights).mean()
        self.losses.append(float(loss.detach()))
        return ANCHOR_WEIGHT * loss

    @torch.no_grad()
    def action_drift(self, model):
        rows = []
        for name, xs, targets in zip(("final_approach_and_lift", "transport_far", "transport_near"), self.regions, self.targets):
            difference = (model(xs) - targets).abs().numpy()
            for index, action in enumerate(("x", "y", "z", "grip")):
                rows.append({"region": name, "action": action, "samples": len(xs),
                             "mean_absolute_delta": float(np.mean(difference[:, index])),
                             "p95_absolute_delta": float(np.quantile(difference[:, index], 0.95))})
        return rows


def train(seed, results, output):
    frozen, identity, ordinary, anchors, raw, suffix = source_data(results, seed)
    root = output / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    anchor = TransportActionAnchor(load_clone(frozen), transport_observations([*anchors, raw]), seed)
    start = perf_counter()
    adapted = train_clone(
        anchors[0], seed + 28100, STEPS, set(), None, model=load_clone(frozen),
        additional_datasets=[anchors[1], suffix], learning_rate=LEARNING_RATE, regularizer=anchor,
    )
    training_s = perf_counter() - start
    destination = root / "model.pt"
    torch.save(adapted.state_dict(), destination)
    write_csv(root / "loss.csv", [
        {"update": step, "unweighted_transport_loss": loss} for step, loss in enumerate(anchor.losses, 1)
    ])
    write_csv(root / "transport_action_drift.csv", [
        {"method": name, **r} for name, model in (("ordinary", load_clone(ordinary)), ("transport_anchored", adapted))
        for r in anchor.action_drift(model)
    ])
    write_json(root / "config.json", {
        "training_seed": seed, "parent_sha256": identity, "model_sha256": digest(destination),
        "updates": STEPS, "learning_rate": LEARNING_RATE, "training_s": training_s,
        "shared_demo_steps": BUDGET, "new_demo_steps": 0, "new_online_steps": 0,
        "raw_demonstrations": raw.as_posix(), "ordinary_demonstration_sha256": digest(raw),
        "landing_suffix_sha256": digest(suffix), "historical_demonstration_paths": [p.as_posix() for p in anchors],
        "anchor_weight": ANCHOR_WEIGHT, "action_weights_xy_z_grip": ACTION_WEIGHTS,
        "near_goal_m": NEAR_M, "near_grasp_m": CONTACT_APPROACH_M, "sample_count_per_region": BATCH_PER_REGION,
        "pool_sizes": [len(r) for r in anchor.regions], "test_scenarios": list(TEST),
    })
    print(f"Seed {seed}: {STEPS} updates from {BUDGET} shared demonstration steps; loss recorded", flush=True)


def test(seed, method, results, output):
    frozen, identity, ordinary, _, raw, _ = source_data(results, seed)
    root = output / f"seed_{seed}"
    if method == "transport_anchored":
        config = read_json(root / "config.json")
        path = root / "model.pt"
        if (config["model_sha256"] != digest(path) or config["parent_sha256"] != identity
                or config["ordinary_demonstration_sha256"] != digest(raw)):
            raise ValueError(f"Candidate weights or training data changed in {root}")
    else:
        path = frozen if method == "frozen" else ordinary
    run_case(root / method, LOW, clone_policy(load_clone(path)), TEST, {
        "seed": seed, "method": method, "checkpoint": path.as_posix(), "checkpoint_sha256": digest(path),
        "initial_sha256": identity, "ordinary_demo_sha256": digest(raw),
        "shared_demo_steps": 0 if method == "frozen" else BUDGET, "updates": 0 if method == "frozen" else STEPS,
    })


def verify_ordinary(results, output):
    records = []
    for seed in range(3):
        frozen, _, saved, anchors, _, suffix = source_data(results, seed)
        rebuilt = train_clone(
            anchors[0], seed + 28100, STEPS, set(), None, model=load_clone(frozen),
            additional_datasets=[anchors[1], suffix], learning_rate=LEARNING_RATE,
        )
        target = load_clone(saved).state_dict()
        if any(not torch.equal(weights, target[name]) for name, weights in rebuilt.state_dict().items()):
            raise ValueError(f"Seed {seed}: the ordinary checkpoint did not reproduce")
        records.append({"seed": seed, "source": saved.as_posix(), "steps": STEPS, "all_tensors_exact": True})
    write_json(output / "ordinary_checkpoint_rebuild.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test", "verify-ordinary"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "verify-ordinary":
        verify_ordinary(args.results, args.out)
        return
    if args.seed is None:
        parser.error("train and test require --seed")
    if args.command == "train":
        train(args.seed, args.results, args.out)
    elif args.method is None:
        parser.error("test requires --method")
    else:
        test(args.seed, args.method, args.results, args.out)


if __name__ == "__main__":
    main()
