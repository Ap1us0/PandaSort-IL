"""Evaluate the frozen four-condition placement policy at nearby physics limits."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .experiment import evaluate, write_csv, write_json
from .physics_shift import Condition
from .place_cli import clone_policy
from .place_robustness import load_clone, placement_metrics
from .place_touchdown import DOMAINS, MODERATE, teacher_policy
from .place_touchdown_rescore import OutcomeEvaluation

OUTPUT = Path("results/place_boundary")
SCREEN_SCENES = range(1_500_000, 1_500_024)
TEST_SCENES = range(1_510_000, 1_510_100)
EXPERT_GATE = 22
REFERENCE = "friction_75_mass_150"
SWEEPS = {
    "friction": tuple(f"friction_{scale}_mass_150" for scale in (70, 65, 60, 55)),
    "mass": tuple(f"friction_75_mass_{grams}" for grams in (165, 175, 185, 200)),
}
CANDIDATES = {
    REFERENCE: DOMAINS[MODERATE],
    **{name: Condition(mass_scale=2.0, sliding_scale=scale / 100)
       for name, scale in zip(SWEEPS["friction"], (70, 65, 60, 55))},
    **{name: Condition(mass_scale=grams / 75, sliding_scale=0.75)
       for name, grams in zip(SWEEPS["mass"], (165, 175, 185, 200))},
}


class BoundaryEvaluation(OutcomeEvaluation):
    """Expose both task outcome and process criterion in each archived episode."""

    def _info(self, contacts, status):
        return {
            **super()._info(contacts, status),
            "stable_placement_success": self.max_outcome_ticks >= self.hold_ticks,
            "outcome_stable_s": self.max_outcome_ticks * self.dt,
        }


def run_case(directory, name, policy, scenes, extra):
    condition = CANDIDATES[name]
    with BoundaryEvaluation(condition) as env:
        rows = evaluate(env, policy, scenes)
        if any(row["is_success"] and not row["stable_placement_success"] for row in rows):
            raise ValueError(f"Strict success without stable placement in {name}")
        stable = sum(row["stable_placement_success"] for row in rows)
        strict = sum(row["is_success"] for row in rows)
        metrics = {
            **extra, "condition": name, **asdict(condition), "mass_kg": env.mass_kg,
            "sliding_coefficient": env.cube_sliding, **placement_metrics(rows),
            "strict_successes": strict, "stable_placement_successes": stable,
            "stable_without_gripped_touchdown": stable - strict,
        }
    write_csv(directory / "episodes.csv", rows)
    write_json(directory / "summary.json", metrics)
    print(f"{directory}: stable {stable}/{len(rows)}, strict {strict}/{len(rows)}", flush=True)


def screen(output, axis):
    cases = [REFERENCE] if axis == "reference" else SWEEPS[axis]
    for name in cases:
        run_case(output / "screen" / name, name, teacher_policy("verified_touch"),
                 SCREEN_SCENES, {"source": "expert", "seed": None})


def select(output):
    chosen = [REFERENCE]
    experts_only = []
    decisions = []
    for axis, cases in SWEEPS.items():
        records = [json.loads((output / "screen" / name / "summary.json").read_text(encoding="utf-8"))
                   for name in cases]
        qualifying = [index for index, row in enumerate(records) if row["stable_placement_successes"] >= EXPERT_GATE]
        frontier = qualifying[-1] if qualifying else None
        harder = (frontier + 1 if frontier is not None else 0)
        if frontier is not None:
            chosen.append(cases[frontier])
        if harder < len(cases):
            experts_only.append(cases[harder])
        decisions.append({
            "axis": axis, "chosen": cases[frontier] if frontier is not None else None,
            "next_harder": cases[harder] if harder < len(cases) else None,
            "screen_outcomes": {row["condition"]: row["stable_placement_successes"] for row in records},
        })
    write_json(output / "selection.json", {
        "expert_gate_of_24": EXPERT_GATE, "screen_seeds": list(SCREEN_SCENES),
        "test_seeds": list(TEST_SCENES), "policy_cases": chosen, "expert_only_cases": experts_only,
        "decisions": decisions,
    })
    print(f"Policy cases: {chosen}. Additional expert cases: {experts_only}.", flush=True)


def test(output, results, seed):
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    cases = selection["policy_cases"]
    if seed is None:
        policy = teacher_policy("verified_touch")
        extra = {"source": "expert", "seed": None}
        name = "expert"
        cases = cases + selection["expert_only_cases"]
    else:
        source = results / "place_touchdown" / "training" / f"seed_{seed}" / "model.pt"
        reference = results / "place_touchdown" / "artifact_verification.json"
        identity = next(row["selected_sha256"] for row in json.loads(reference.read_text(encoding="utf-8"))
                        if row["seed"] == seed)
        if hashlib.sha256(source.read_bytes()).hexdigest() != identity:
            raise ValueError(f"Frozen checkpoint has changed: {source}")
        policy = clone_policy(load_clone(source))
        extra = {"source": "frozen", "seed": seed, "checkpoint": source.as_posix(), "checkpoint_sha256": identity}
        name = f"seed_{seed}"
    for condition in cases:
        run_case(output / "test" / name / condition, condition, policy, TEST_SCENES, extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "select", "test"))
    parser.add_argument("--axis", choices=("reference", *SWEEPS))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "screen":
        if args.axis is None:
            parser.error("screen requires --axis")
        screen(args.out, args.axis)
    elif args.command == "select":
        select(args.out)
    else:
        test(args.out, args.results, args.seed)


if __name__ == "__main__":
    main()
