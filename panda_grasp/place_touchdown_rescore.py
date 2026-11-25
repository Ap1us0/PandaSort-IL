"""Re-score saved placement policies with and without the gripped-touchdown requirement."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .env import ANGULAR_LIMIT, LINEAR_LIMIT
from .experiment import write_csv, write_json
from .place import GOAL_RADIUS, GRIP_OPEN_M, REST_TOLERANCE_M, RETRACT_M
from .place_robustness import ShiftedPlacementEnv, load_clone
from .place_touchdown import DOMAINS, MODERATE, OUTPUT, TEST_SCENES


class OutcomeEvaluation(ShiftedPlacementEnv):
    """Record stable placement independently of the benchmark's release ordering."""

    def _reset_task(self, options):
        super()._reset_task(options)
        self.outcome_ticks = 0
        self.max_outcome_ticks = 0

    def _sample_hold(self, height, contacts):
        super()._sample_hold(height, contacts)
        if not self.transported:
            return
        on_target = (self.goal_distance <= GOAL_RADIUS and contacts.supported
                     and abs(height) <= REST_TOLERANCE_M)
        released = (np.sum(self.data.qpos[list(self.gripper.fingers)]) >= GRIP_OPEN_M
                    and not (contacts.left_count or contacts.right_count))
        clear = self.tcp[2] >= self.cube_xyz[2] + RETRACT_M
        stable = (np.linalg.norm(self.cube_velocity[:3]) <= LINEAR_LIMIT
                  and np.linalg.norm(self.cube_velocity[3:]) <= ANGULAR_LIMIT)
        self.outcome_ticks = self.outcome_ticks + 1 if on_target and released and clear and stable else 0
        self.max_outcome_ticks = max(self.max_outcome_ticks, self.outcome_ticks)


def replay(output, profile, seed):
    path = output / profile / f"seed_{seed}" / MODERATE
    record = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    weights = Path(record["checkpoint"])
    if hashlib.sha256(weights.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
        raise ValueError(f"The checkpoint has changed: {weights}")
    policy = load_clone(weights)
    with (path / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        archived = list(csv.DictReader(stream))
    if [int(row["scenario_seed"]) for row in archived] != list(TEST_SCENES):
        raise ValueError(f"Unexpected scenario set in {path}")
    results = []
    with OutcomeEvaluation(DOMAINS[MODERATE]) as env:
        for old in archived:
            scene = int(old["scenario_seed"])
            obs, _ = env.reset(seed=scene)
            while True:
                obs, _, terminated, truncated, info = env.step(policy.action(obs))
                if terminated or truncated:
                    break
            strict = info["is_success"]
            if (str(strict) != old["is_success"] or env.episode.steps != int(old["steps"])
                    or info["failure_stage"] != old["failure_stage"]
                    or not np.isclose(info["goal_distance_m"], float(old["goal_distance_m"]), rtol=0, atol=1e-12)):
                raise ValueError(f"Archived trial did not reproduce in {path}: {scene}")
            stable = env.max_outcome_ticks >= env.hold_ticks
            if strict and not stable:
                raise ValueError(f"Strict success did not satisfy outcome criterion: {path}, {scene}")
            results.append({
                "scenario_seed": scene, "strict_success": strict, "stable_placement_success": stable,
                "stable_placement_s": env.max_outcome_ticks * env.dt,
                "goal_error_m": info["goal_distance_m"], "strict_failure_stage": info["failure_stage"],
            })
    destination = output / "outcome_rescore" / profile / f"seed_{seed}"
    write_csv(destination / "episodes.csv", results)
    strict_count = sum(row["strict_success"] for row in results)
    stable_count = sum(row["stable_placement_success"] for row in results)
    write_json(destination / "summary.json", {
        "profile": profile, "seed": seed, "episodes": len(results),
        "strict_successes": strict_count, "stable_placement_successes": stable_count,
        "stable_without_gripped_touchdown": stable_count - strict_count,
    })
    print(f"{profile} seed {seed}: strict {strict_count}/100, stable placement {stable_count}/100", flush=True)


def summarize(output):
    directory = output / "outcome_rescore"
    summaries = []
    pairs = []
    for seed in range(3):
        outcomes = {}
        for profile in ("frozen", "adapted"):
            case = directory / profile / f"seed_{seed}"
            with (case / "episodes.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if [int(row["scenario_seed"]) for row in rows] != list(TEST_SCENES):
                raise ValueError(f"Mismatched scenes in {case}")
            outcomes[profile] = [row["stable_placement_success"] == "True" for row in rows]
            summaries.append(json.loads((case / "summary.json").read_text(encoding="utf-8")))
        differences = list(zip(outcomes["frozen"], outcomes["adapted"]))
        pairs.append({
            "seed": seed,
            "fail_to_success": sum(not before and after for before, after in differences),
            "success_to_fail": sum(before and not after for before, after in differences),
        })
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "paired_changes.csv", pairs)
    for profile in ("frozen", "adapted"):
        rows = [row for row in summaries if row["profile"] == profile]
        print(profile, sum(row["strict_successes"] for row in rows),
              sum(row["stable_placement_successes"] for row in rows), flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--profile", choices=("frozen", "adapted"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.profile is None and args.seed is None:
        summarize(args.out)
    elif args.profile is None or args.seed is None:
        parser.error("Pass both --profile and --seed to replay a checkpoint")
    else:
        replay(args.out, args.profile, args.seed)


if __name__ == "__main__":
    main()
