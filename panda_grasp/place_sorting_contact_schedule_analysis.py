"""Audit the three-stage tactile experiment and render verified examples."""

import argparse
from pathlib import Path

import torch

from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_sorting_analysis import audit
from .place_sorting_contact_ablation import (
    OUTPUT as ABLATION, TEST as ABLATION_TEST, SWITCHED, THREE_PHASE,
)
from .place_sorting_contact_ablation_analysis import focused
from .place_sorting_contact_phase import TEST as PRIOR_TEST
from .place_sorting_contact_phase_analysis import COLORS, audit_budget, paired_comparison, plot, record_new
from .place_sorting_contact_schedule import METHODS, OUTPUT, PRIOR, TEST, load

LABELS = ("Contact\nswitches", "Force only", "Slip only", "Slip\nbefore lift", "Switches\nbefore lift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    archives = {"force": (ABLATION, ABLATION_TEST), "slip": (ABLATION, ABLATION_TEST),
                SWITCHED: (PRIOR, PRIOR_TEST), THREE_PHASE: (args.out, TEST)}
    audit_budget(args.out, methods=METHODS, archives=archives, load_actor=load)
    methods = tuple((method, seed) for method in METHODS for seed in range(3))
    recorded, objects, cases = audit(args.out, OBJECT_RESULTS, methods=methods, scenarios=TEST,
                                    checkpoint_identity=lambda method, seed: load(method, seed, args.out)[1])
    _, overall = focused(args.out, recorded, objects, methods=METHODS)
    paired_comparison(args.out, cases, candidate=THREE_PHASE, controls=METHODS[:-1])
    plot(args.out, overall, labels=LABELS, colors=(*COLORS, "#AE6B2F"))
    if args.videos:
        record_new(args.out, cases, candidate=THREE_PHASE, load_actor=load)


if __name__ == "__main__":
    main()
