"""Evaluate contact switches before lift, continuous force in transport, and slip at placement."""

import argparse
from pathlib import Path

import torch

from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_sorting import evaluate_actor
from .place_sorting_bc import OUTPUT as SORTING_RESULTS
from .place_sorting_contact import OUTPUT as CONTACT_RESULTS
from .place_sorting_contact_ablation import THREE_PHASE, load_evaluation, scene_type, train
from .place_sorting_contact_phase import METHODS as CONTROLS, OUTPUT as PRIOR, load as load_prior

OUTPUT = Path("results/place_sorting_contact_schedule")
TEST = range(1_567_000, 1_567_030)
METHODS = (*CONTROLS, THREE_PHASE)


def load(method, seed, output):
    if method == THREE_PHASE:
        return load_evaluation(method, seed, OBJECT_RESULTS, SORTING_RESULTS, CONTACT_RESULTS, output)
    return load_prior(method, seed, PRIOR)


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
              methods=(THREE_PHASE,), scenarios=TEST)
    elif args.method is None:
        parser.error("test requires --method")
    else:
        model, identity = load(args.method, args.seed, args.out)
        evaluate_actor(args.method, args.seed, model, identity, args.out / "test", TEST, scene_type(args.method))


if __name__ == "__main__":
    main()
