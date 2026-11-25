"""Match BC training while separating tactile sources and their use during transport."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from .env import OBS_SIZE as GRASP_OBS_SIZE
from .experiment import write_json
from .learning import train_clone
from .place_contact_signals import CONTACTS, ContactClone, ContactSortingScene
from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_objects_train import sha
from .place_sorting import SortingScene, evaluate_actor, load_mixed
from .place_sorting_bc import LEARNING_RATE, OUTPUT as SORTING_RESULTS
from .place_sorting_contact import OUTPUT as CONTACT_RESULTS, load_evaluation as load_control

OUTPUT = Path("results/place_sorting_contact_ablation")
TEST = range(1_565_000, 1_565_030)
VARIANTS = ("force", "slip", "transport")
METHODS = ("binary", "force", "slip", "transport", "continuous")
SWITCHED = "force_in_transport"
THREE_PHASE = "binary_then_force_then_slip"
LIFTED_INDEX = GRASP_OBS_SIZE + 4
ARRIVED_INDEX = GRASP_OBS_SIZE + 5


class MaskedContactClone(ContactClone):
    """Mask observations only. Every Cartesian and gripper action is learned."""

    def __init__(self, method):
        super().__init__()
        self.method = method
        coefficients = {
            "force": [1.0] * 2 + [0.0] * (len(CONTACTS) - 2),
            "slip": [0.0] * 2 + [1.0] * (len(CONTACTS) - 2),
            "transport": [1.0] * len(CONTACTS),
            SWITCHED: [1.0] * len(CONTACTS),
            THREE_PHASE: [1.0] * len(CONTACTS),
        }
        if method not in coefficients:
            raise ValueError(f"Unknown feature selection: {method}")
        self.register_buffer("contact_mask", torch.tensor(coefficients[method]), persistent=False)

    def contact_inputs(self, observation):
        sensors = super().contact_inputs(observation) * self.contact_mask
        if self.method in ("transport", SWITCHED, THREE_PHASE):
            lifted = observation[..., LIFTED_INDEX:LIFTED_INDEX + 1] > 0.5
            arrived = observation[..., ARRIVED_INDEX:ARRIVED_INDEX + 1] > 0.5
            transport = lifted & ~arrived
            if self.method == THREE_PHASE:
                return torch.cat((sensors[..., :2] * transport, sensors[..., 2:] * (lifted & arrived)), dim=-1)
            if self.method == SWITCHED:
                return torch.cat((sensors[..., :2] * transport, sensors[..., 2:] * ~transport), dim=-1)
            sensors = sensors * transport
        return sensors


def training_sources(seed, contact, sorting):
    sensory = contact / "training" / f"seed_{seed}"
    original = sorting / "training" / f"seed_{seed}"
    sensory_receipt = json.loads((sensory / "config.json").read_text(encoding="utf-8"))
    original_receipt = json.loads((original / "config.json").read_text(encoding="utf-8"))
    updates = sensory_receipt["optimizer_updates_each"]
    if (updates != original_receipt["selected_updates"] or not updates
            or sensory_receipt["learning_rate"] != LEARNING_RATE
            or sensory_receipt["starting_sha256"] != original_receipt["initial_sha256"]
            or sensory_receipt["binary_checkpoint_sha256"] != original_receipt["checkpoint_sha256"]
            or not sensory_receipt["same_actions_and_base_observations"]
            or not sensory_receipt["binary_checkpoint_reproduced_exactly"]):
        raise ValueError(f"Unmatched control experiment for training seed {seed}")
    paths = []
    for metadata, base in zip(sensory_receipt["datasets"], original_receipt["datasets"], strict=True):
        archived = sensory / "datasets" / metadata["domain"] / metadata["shape"] / "expert.npz"
        if sha(archived) != metadata["augmented_sha256"] or sha(Path(base)) != metadata["original_sha256"]:
            raise ValueError(f"Demonstration archive changed: {archived}")
        paths.append(archived)
    return paths, sensory_receipt


def train(seed, source, sorting, contact, output, *, methods=VARIANTS, scenarios=TEST):
    paths, reference = training_sources(seed, contact, sorting)
    initial, identity = load_mixed(seed, source)
    if identity != reference["starting_sha256"]:
        raise ValueError(f"Unexpected starting weights for seed {seed}")
    updates = reference["optimizer_updates_each"]
    for method in methods:
        root = output / "training" / f"seed_{seed}" / method
        root.mkdir(parents=True, exist_ok=True)
        model = MaskedContactClone(method).initialize(initial)
        started = perf_counter()
        train_clone(paths[0], seed + 168270, updates, set(), None, model=model,
                    additional_datasets=paths[1:], learning_rate=LEARNING_RATE)
        training_s = perf_counter() - started
        checkpoint = root / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        write_json(root / "config.json", {
            "seed": seed, "method": method, "starting_sha256": identity,
            "checkpoint_sha256": sha(checkpoint), "optimizer_updates": updates,
            "learning_rate": LEARNING_RATE, "training_s": training_s,
            "new_demonstration_scenes": 0, "new_online_training_steps": 0,
            "demonstration_sha256": [entry["augmented_sha256"] for entry in reference["datasets"]],
            "test_scenarios": list(scenarios),
        })
        print(f"Seed {seed}: {method} trained for {updates} matched updates", flush=True)


def load_evaluation(method, seed, source, sorting, contact, output):
    if method in ("binary", "continuous"):
        return load_control(method, seed, source, sorting, contact)
    root = output / "training" / f"seed_{seed}" / method
    path = root / "model.pt"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    identity = sha(path)
    if identity != config["checkpoint_sha256"] or method != config["method"]:
        raise ValueError(f"Checkpoint or mask differs from archive: {path}")
    model = MaskedContactClone(method)
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    return model, identity


def scene_type(method):
    return SortingScene if method == "binary" else ContactSortingScene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--source", type=Path, default=OBJECT_RESULTS)
    parser.add_argument("--sorting", type=Path, default=SORTING_RESULTS)
    parser.add_argument("--contact", type=Path, default=CONTACT_RESULTS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        train(args.seed, args.source, args.sorting, args.contact, args.out)
    elif args.method is None:
        parser.error("test requires --method")
    else:
        model, identity = load_evaluation(args.method, args.seed, args.source, args.sorting, args.contact, args.out)
        evaluate_actor(args.method, args.seed, model, identity, args.out / "test", TEST, scene_type(args.method))


if __name__ == "__main__":
    main()
