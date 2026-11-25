"""Compare phase-dependent tactile BC with matched archived controls."""

import argparse
from pathlib import Path

import torch

from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_sorting import evaluate_actor
from .place_sorting_bc import OUTPUT as SORTING_RESULTS
from .place_sorting_contact import OUTPUT as CONTACT_RESULTS
from .place_sorting_contact_ablation import (
    OUTPUT as ABLATION_RESULTS, SWITCHED, load_evaluation, scene_type, train,
)

OUTPUT = Path("results/place_sorting_contact_phase")
TEST = range(1_566_000, 1_566_030)
METHODS = ("binary", "force", "slip", SWITCHED)


def load(method, seed, output, prior=ABLATION_RESULTS):
    archive = output if method == SWITCHED else prior
    return load_evaluation(method, seed, OBJECT_RESULTS, SORTING_RESULTS, CONTACT_RESULTS, archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        train(args.seed, OBJECT_RESULTS, SORTING_RESULTS, CONTACT_RESULTS, args.out,
              methods=(SWITCHED,), scenarios=TEST)
    elif args.method is None:
        parser.error("test requires --method")
    else:
        model, identity = load(args.method, args.seed, args.out)
        evaluate_actor(args.method, args.seed, model, identity, args.out / "test", TEST, scene_type(args.method))


if __name__ == "__main__":
    main()
