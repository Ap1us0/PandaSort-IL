"""Paired touchdown interventions and four-condition held-out evaluation."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from ._vendor.pickup import State
from .experiment import rollout, write_csv, write_json
from .physics_shift import CONDITIONS
from .place_cli import clone_policy
from .place_expert import PlacementExpert
from .place_mixed_benchmark import COMBINED, COMBINED_SCENES
from .place_robustness import (
    STRESS_CONDITIONS, ShiftedPlacementEnv, grip_probe, load_clone, placement_metrics, run_evaluation,
)

OUTPUT = Path("results/place_touchdown")
VALIDATION_SCENES = range(1_470_000, 1_470_012)
TEST_SCENES = range(1_480_000, 1_480_100)
MODERATE = "combined_moderate"
DOMAINS = {**{name: CONDITIONS[name] for name in STRESS_CONDITIONS}, MODERATE: COMBINED[MODERATE]}
PROBE_SCALE = 0.5
TEACHERS = {"standard": (False, 1.0), "verified_touch": (True, 1.0),
            "slow_verified_touch": (True, PROBE_SCALE)}


class TouchdownTeacher(PlacementExpert):
    """Construct demonstrations with a physically verified release transition."""

    def __init__(self, env, *, verify, descent_scale):
        super().__init__(env)
        self.verify = verify
        self.descent_scale = descent_scale

    def _ready_to_retreat(self):
        return bool(self.env.supported_at_target) if self.verify else super()._ready_to_retreat()

    def __call__(self, observation):
        label, action = super().__call__(observation)
        if self.state == State.PLACE and action[2] < 0:
            label[2] *= self.descent_scale
            action[2] *= self.descent_scale
        return label, action


def build_teacher(env, name):
    verify, scale = TEACHERS[name]
    return TouchdownTeacher(env, verify=verify, descent_scale=scale)


def teacher_policy(name):
    def factory(env, seed):
        teacher = build_teacher(env, name)
        return lambda obs: teacher(obs)[1]
    return factory


def slow_descent(policy):
    """Change only commanded downward motion between arrival and touchdown."""
    def factory(env, seed):
        predict = policy(env, seed)

        def choose(observation):
            action = predict(observation)
            if env.transported and not env.supported_at_target and action[2] < 0:
                action = action.copy()
                action[2] *= PROBE_SCALE
            return action

        return choose

    return factory


def origin_path(mixed, seed):
    return mixed / "mixed" / f"seed_{seed}" / "model.pt"


def diagnostics(mixed, output, seed):
    reference = mixed / "combined" / "mixed" / f"seed_{seed}" / MODERATE
    with (reference / "episodes.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if [int(row["scenario_seed"]) for row in rows] != list(COMBINED_SCENES):
        raise ValueError(f"Expected the previous round's ordered scenes in {reference}")
    path = origin_path(mixed, seed)
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    if json.loads((reference / "summary.json").read_text(encoding="utf-8"))["checkpoint_sha256"] != identity:
        raise ValueError(f"Checkpoint differs from the previous round: {path}")
    failed = [row for row in rows if row["is_success"] == "False"]
    successes = [row for row in rows if row["is_success"] == "True"][:12]
    sample = sorted([*failed, *successes], key=lambda row: int(row["scenario_seed"]))
    original = clone_policy(load_clone(path))
    paired = []
    with ShiftedPlacementEnv(DOMAINS[MODERATE]) as env:
        for mode, policy in (("hold_grip", grip_probe(original)), ("slow_descent", slow_descent(original))):
            for stored in sample:
                trial = rollout(env, policy, int(stored["scenario_seed"]))
                paired.append({
                    "intervention": mode, "baseline_success": stored["is_success"] == "True",
                    "baseline_stage": stored["failure_stage"],
                    "baseline_early_open": stored["open_without_gripped_touchdown"] == "True",
                    **trial,
                })
    output_dir = output / "probe" / f"seed_{seed}"
    write_csv(output_dir / "paired_episodes.csv", paired)
    summary = []
    for mode in ("hold_grip", "slow_descent"):
        group = [row for row in paired if row["intervention"] == mode]
        summary.append({
            "intervention": mode, "initial_failures": len(failed), "initial_success_controls": len(successes),
            "rescued_failures": sum(not row["baseline_success"] and row["is_success"] for row in group),
            "regressed_successes": sum(row["baseline_success"] and not row["is_success"] for row in group),
            "rescued_touchdowns": sum(row["baseline_stage"] == "touchdown" and row["is_success"] for row in group),
            "failed_after_probe": placement_metrics(group)["failure_stages"],
        })
    write_json(output_dir / "summary.json", {
        "seed": seed, "checkpoint_sha256": identity, "descent_scale": PROBE_SCALE, "outcomes": summary,
    })
    print(f"{output_dir}: {summary}", flush=True)


def evaluate_teachers(output):
    with ShiftedPlacementEnv(DOMAINS[MODERATE]) as env:
        candidates = {}
        for name in TEACHERS:
            candidates[name] = run_evaluation(env, teacher_policy(name), VALIDATION_SCENES,
                                              output / "teacher_validation" / name)
    chosen = max(candidates, key=lambda name: (
        candidates[name]["successes"], -candidates[name]["early_open_rate"],
        -candidates[name]["transport_drop_rate"],
    ))
    write_json(output / "teacher_selection.json", {
        "teacher": chosen, "validation_seeds": list(VALIDATION_SCENES),
        "candidates": {name: {k: stats[k] for k in ("successes", "early_open_rate", "transport_drop_rate")}
                       for name, stats in candidates.items()},
    })


def test_checkpoint(path, output, seed, profile):
    weights = load_clone(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, condition in DOMAINS.items():
        with ShiftedPlacementEnv(condition) as env:
            run_evaluation(env, clone_policy(weights), TEST_SCENES, output / profile / f"seed_{seed}" / name, {
                "condition": name, "seed": seed, "source": profile, "checkpoint": path.as_posix(),
                "checkpoint_sha256": sha, **asdict(condition),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "frozen", "teacher", "expert-test"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--mixed", type=Path, default=Path("results/place_mixed"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "teacher":
        evaluate_teachers(args.out)
    elif args.command == "expert-test":
        selection = json.loads((args.out / "teacher_selection.json").read_text(encoding="utf-8"))
        with ShiftedPlacementEnv(DOMAINS[MODERATE]) as env:
            run_evaluation(env, teacher_policy(selection["teacher"]), TEST_SCENES, args.out / "expert" / MODERATE,
                           {"condition": MODERATE, "source": "expert", "teacher": selection["teacher"]})
    elif args.seed is None:
        parser.error("probe and frozen require --seed")
    elif args.command == "probe":
        diagnostics(args.mixed, args.out, args.seed)
    else:
        test_checkpoint(origin_path(args.mixed, args.seed), args.out, args.seed, "frozen")


if __name__ == "__main__":
    main()
