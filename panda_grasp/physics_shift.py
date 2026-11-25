"""Contact-physics stress tests for the frozen PandaGrasp policies."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from stable_baselines3 import SAC

from .ablation_cli import RetentionEvaluation, measure
from .cli import load_policy, make_policy
from .experiment import write_csv, write_json
from .expert import PickupExpert


@dataclass(frozen=True)
class Condition:
    mass_scale: float = 1.0
    sliding_scale: float = 1.0


CONDITIONS = {
    "nominal": Condition(),
    **{f"friction_{value}": Condition(sliding_scale=int(value) / 100)
       for value in ("75", "50", "35", "27", "23", "20", "16")},
    **{f"mass_{value}": Condition(mass_scale=int(value) / 100)
       for value in ("150", "200", "250", "300", "400")},
}
SCREEN_SCENES = range(1_310_000, 1_310_012)
NOMINAL_TEST_SCENES = range(1_320_000, 1_320_100)
ADAPT_TEST_SCENES = range(1_330_000, 1_330_100)
RUNTIME = Path("results/physics_shift")


def apply_physics_condition(env, condition):
    model = env.model
    body = model.geom_bodyid[env.cube]
    if condition.sliding_scale != 1.0:
        finger_geoms = np.flatnonzero(np.isin(model.geom_bodyid, env.gripper.bodies))
        model.geom_friction[np.r_[env.cube, finger_geoms], 0] *= condition.sliding_scale
    if condition.mass_scale != 1.0:
        model.body_mass[body] *= condition.mass_scale
        model.body_inertia[body] *= condition.mass_scale
        mujoco.mj_setConst(model, env.data)
    return float(model.body_mass[body]), float(model.geom_friction[env.cube, 0])


class ShiftedGraspEnv(RetentionEvaluation):
    def __init__(self, condition, **kwargs):
        super().__init__(**kwargs)
        self.condition = condition
        self.mass_kg, self.cube_sliding = apply_physics_condition(self, condition)


def teacher_policy(env, seed):
    expert = PickupExpert(env, contact_trigger=True)
    return lambda obs: expert(obs)[1]


def checkpoint_path(results, method, seed):
    if method == "bc":
        return results / "contact_trigger" / f"seed_{seed}" / "bc" / "model.pt"
    return results / "sac_ablation" / f"seed_{seed}" / "bc_regularized" / "model.zip"


def source_policy(results, method, seed):
    if method == "expert":
        return teacher_policy
    path = checkpoint_path(results, method, seed)
    if method == "bc":
        return load_policy(method, path.parent)
    return make_policy("sac", SAC.load(path, device="cpu"))


def evaluate_conditions(results, output, mode, seed, conditions):
    methods = ("expert",) if seed is None else ("bc", "bc_sac")
    scenes = SCREEN_SCENES if mode == "screen" else NOMINAL_TEST_SCENES
    summaries = []
    for method in methods:
        policy = source_policy(results, method, seed)
        for label in conditions:
            with ShiftedGraspEnv(CONDITIONS[label]) as env:
                rows, summary = measure(env, policy, scenes)
                config = {"condition": label, **asdict(CONDITIONS[label]),
                          "cube_mass_kg": env.mass_kg, "cube_sliding_coefficient": env.cube_sliding}
            directory = output / mode / ("expert" if seed is None else f"seed_{seed}") / method / label
            write_csv(directory / "episodes.csv", rows)
            write_json(directory / "summary.json", {**config, **summary})
            summaries.append({"seed": seed, "method": method, **config, **summary})
            print(f"{mode} {seed} {method} {label}: {summary['successes']}/{len(rows)}", flush=True)
    write_csv(output / mode / ("expert.csv" if seed is None else f"seed_{seed}.csv"), summaries)


def select_adaptation(output):
    """Choose the mildest recoverable shift using only the screen scenes."""
    root = output / "screen"
    selected = None
    evidence = []
    candidates = sorted(
        (label for label in CONDITIONS if label.startswith("friction_")),
        key=lambda label: -CONDITIONS[label].sliding_scale,
    ) + sorted(
        (label for label in CONDITIONS if label.startswith("mass_")),
        key=lambda label: CONDITIONS[label].mass_scale,
    )
    for label in candidates:
        metrics = {}
        for policy in ("expert", "bc", "bc_sac"):
            seeds = ["expert"] if policy == "expert" else [f"seed_{seed}" for seed in range(3)]
            summaries = [root / seed / policy / label / "summary.json" for seed in seeds]
            metrics[policy] = sum(json.loads(path.read_text(encoding="utf-8"))["successes"] for path in summaries)
        qualifies = metrics["expert"] >= 10 and all(metrics[method] <= 29 for method in ("bc", "bc_sac"))
        evidence.append({"condition": label, **metrics, "qualifies": qualifies})
        if qualifies:
            selected = label
            break
    write_json(output / "selection.json", {
        "screen_seeds": list(SCREEN_SCENES), "adaptation_condition": selected,
        "expert_minimum_successes_of_12": 10, "each_learned_policy_max_successes_of_36": 29,
        "evaluated_candidates_in_priority_order": evidence,
    })
    print(f"Adaptation condition: {selected}", flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("screen", "test", "select"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=RUNTIME)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=tuple(CONDITIONS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "select":
        select_adaptation(args.out)
    else:
        evaluate_conditions(args.results, args.out, args.mode, args.seed, args.conditions)


if __name__ == "__main__":
    main()
