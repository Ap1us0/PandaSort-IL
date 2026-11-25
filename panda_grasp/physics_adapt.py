"""Matched-cost adaptation of existing BC and BC-regularized SAC checkpoints."""

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch

from .ablation_cli import measure
from .cli import make_policy
from .experiment import collect_demonstrations, write_csv, write_json
from .expert import PickupExpert
from .learning import Clone, train_clone
from .physics_shift import (
    ADAPT_TEST_SCENES, CONDITIONS, RUNTIME, SCREEN_SCENES, ShiftedGraspEnv,
    checkpoint_path, source_policy, teacher_policy,
)
from .sac_ablation import AblationSAC, DemonstrationPenalty

BUDGET = 8000
PREFIX_EPISODES = 12
CLONE_UPDATES = 6000


def collect(env, seed, path, condition):
    start_seed = 60_000_000 + 10_000 * seed
    collect_demonstrations(
        path, env, seed, range(start_seed, start_seed + BUDGET),
        lambda rng, perturb: PickupExpert(env, contact_trigger=True),
        {"condition": condition, "interaction_budget": BUDGET},
        interaction_budget=BUDGET,
    )
    with np.load(path / "expert.npz") as dataset:
        offsets = dataset["offsets"]
        if offsets[-1] != BUDGET or len(offsets) <= PREFIX_EPISODES:
            raise ValueError("Invalid demonstration budget or insufficient episodes")
        prefix_steps = int(offsets[PREFIX_EPISODES])
        state_fields = ("observations", "actions", "expert_actions", "next_observations",
                        "rewards", "terminated", "truncated", "phase")
        np.savez_compressed(
            path / "expert_prefix.npz",
            **{key: dataset[key][:prefix_steps] for key in state_fields},
            offsets=offsets[:PREFIX_EPISODES + 1],
            scenario_seeds=dataset["scenario_seeds"][:PREFIX_EPISODES],
            successful=dataset["successful"][:PREFIX_EPISODES],
        )
    return prefix_steps


def evaluate_profile(env, policy, path):
    rows, result = measure(env, policy, ADAPT_TEST_SCENES)
    write_csv(path / "test_episodes.csv", rows)
    write_json(path / "test_summary.json", result)
    print(f"{path}: {result['successes']}/{len(rows)}", flush=True)


def train_bc(env, results, directory, dataset, demo_steps, seed, updates=CLONE_UPDATES):
    directory.mkdir(parents=True, exist_ok=True)
    origin = checkpoint_path(results, "bc", seed)
    model = Clone()
    model.load_state_dict(torch.load(origin, weights_only=True, map_location="cpu"))
    curve = []

    def report(step, current, xyz, gripper):
        _, stats = measure(env, make_policy("bc", current), SCREEN_SCENES)
        curve.append({"optimizer_updates": step, "online_steps": 0, "new_demo_steps": demo_steps,
                      "xyz_mse": xyz, "gripper_bce": gripper, **stats})
        write_csv(directory / "learning_curve.csv", curve)
        print(f"{directory}: {step} BC updates, screen success {stats['successes']}/{len(SCREEN_SCENES)}",
              flush=True)

    report(0, model, None, None)
    if updates:
        model = train_clone(dataset, seed + 7000, updates, {1000, 3000, updates}, report, model=model)
    torch.save(model.state_dict(), directory / "model.pt")
    write_json(directory / "cost.json", {
        "initial_checkpoint": origin.as_posix(), "new_demo_steps": demo_steps,
        "online_steps": 0, "optimizer_updates": updates, "total_new_steps": demo_steps,
    })
    return make_policy("bc", model)


class AdaptationCurve(BaseCallback):
    def __init__(self, env, directory, prefix_steps, checkpoints):
        super().__init__()
        self.env = env
        self.directory = directory
        self.prefix_steps = prefix_steps
        self.checkpoints = checkpoints
        self.rows = []

    def record(self, model, steps):
        _, stats = measure(self.env, make_policy("sac", model), SCREEN_SCENES)
        self.rows.append({"online_steps": steps, "new_demo_steps": self.prefix_steps,
                          "total_new_steps": self.prefix_steps + steps,
                          "actor_updates": model.actor_gradient_updates, **stats})
        write_csv(self.directory / "learning_curve.csv", self.rows)
        print(f"{self.directory}: {steps} online steps, screen success {stats['successes']}/{len(SCREEN_SCENES)}",
              flush=True)

    def _on_rollout_start(self):
        if self.num_timesteps in self.checkpoints:
            self.record(self.model, self.num_timesteps)

    def _on_step(self):
        return True


def train_sac(env, results, directory, demonstrations, seed, prefix_steps):
    directory.mkdir(parents=True, exist_ok=True)
    origin = checkpoint_path(results, "bc_sac", seed)
    with Monitor(ShiftedGraspEnv(env.condition), str(directory / "train")) as train_env:
        model = AblationSAC.load(origin, env=train_env, device="cpu", seed=seed + 7000)
        model.actor_gradient_updates = 0
        model._n_updates = 0
        frequency = model.train_freq.frequency
        online = (BUDGET - prefix_steps) // frequency * frequency
        if online <= model.learning_starts:
            raise ValueError("No room for SAC updates in the chosen budget")
        model.set_logger(configure(str(directory), ["csv"]))
        penalty = DemonstrationPenalty(model.actor, demonstrations / "expert_prefix.npz", model.ablation.bc_weight, seed)
        callback = AdaptationCurve(env, directory, prefix_steps, {0, 1000, 3000, 5000})
        with model.actor.optimizer.register_step_pre_hook(penalty):
            model.learn(total_timesteps=online, callback=callback, log_interval=10)
        callback.record(model, model.num_timesteps)
        model.save(directory / "model")
        write_json(directory / "cost.json", {
            "initial_checkpoint": origin.as_posix(), "new_demo_steps": prefix_steps,
            "online_steps": model.num_timesteps, "optimizer_updates": model.actor_gradient_updates,
            "total_new_steps": prefix_steps + model.num_timesteps, "fresh_online_buffer": True,
        })
    return make_policy("sac", model)


def run(seed, results, root):
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    condition = selection["adaptation_condition"]
    if condition is None:
        raise ValueError("No feasible shift passed the predeclared screen")
    output = root / "adapt" / f"seed_{seed}"
    with ShiftedGraspEnv(CONDITIONS[condition]) as env:
        demos = output / "demonstrations"
        if (demos / "expert_prefix.npz").is_file():
            meta = json.loads((demos / "dataset.json").read_text(encoding="utf-8"))
            if meta["condition"] != condition or meta["interaction_budget"] != BUDGET:
                raise ValueError("Existing demonstrations do not match the selected setting")
            with np.load(demos / "expert_prefix.npz") as prefix:
                prefix_steps = len(prefix["actions"])
        else:
            prefix_steps = collect(env, seed, demos, condition)
        bc = train_bc(env, results, output / "bc", demos / "expert.npz", BUDGET, seed)
        evaluate_profile(env, bc, output / "bc")
        bc_prefix = train_bc(env, results, output / "bc_prefix", demos / "expert_prefix.npz", prefix_steps, seed)
        evaluate_profile(env, bc_prefix, output / "bc_prefix")
        enhanced_sac = train_sac(env, results, output / "bc_sac", demos, seed, prefix_steps)
        evaluate_profile(env, enhanced_sac, output / "bc_sac")
        for method in ("bc", "bc_sac"):
            evaluate_profile(env, source_policy(results, method, seed), output / f"frozen_{method}")



def replay_selected_prefix(seed, results, root):
    """Replay the earliest best screening checkpoint using the same cached 12 demonstrations."""
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    home = root / "adapt" / f"seed_{seed}"
    with (home / "bc_prefix" / "learning_curve.csv").open(encoding="utf-8", newline="") as stream:
        chosen = max(csv.DictReader(stream), key=lambda row: (int(row["successes"]), -int(row["optimizer_updates"])))
    steps = int(chosen["optimizer_updates"])
    target = home / "bc_prefix_selected"
    path = home / "demonstrations" / "expert_prefix.npz"
    with ShiftedGraspEnv(CONDITIONS[selection["adaptation_condition"]]) as env:
        policy = train_bc(env, results, target, path, int(chosen["new_demo_steps"]), seed, updates=steps)
        write_json(target / "selection.json", {
            "optimizer_updates": steps, "screen_successes": int(chosen["successes"]),
            "rule": "maximize screening successes, choose earliest on ties",
        })
        evaluate_profile(env, policy, target)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=RUNTIME)
    parser.add_argument("--prefix-early-stop", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.seed is None:
        if args.prefix_early_stop:
            parser.error("--prefix-early-stop requires a training seed")
        selection = json.loads((args.out / "selection.json").read_text(encoding="utf-8"))
        with ShiftedGraspEnv(CONDITIONS[selection["adaptation_condition"]]) as env:
            evaluate_profile(env, teacher_policy, args.out / "adapt" / "expert")
    else:
        execute = replay_selected_prefix if args.prefix_early_stop else run
        execute(args.seed, args.results, args.out)


if __name__ == "__main__":
    main()
