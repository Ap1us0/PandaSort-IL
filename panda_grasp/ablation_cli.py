"""Run matched grasp-lift ablations and replay the archived reference policies."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import torch

from .cli import load_policy, make_policy
from .env import PandaGraspEnv
from .experiment import TEST_SEEDS, VALIDATION_SEEDS, evaluate, summarize, write_csv, write_json
from .learning import ACTIVATION, WIDTHS, Clone, initialize_actor
from .sac_ablation import ABLATIONS, Ablation, AblationSAC, DemonstrationPenalty

BUDGET = 24_000
CHECKPOINTS = (0, 512, 1000, 2000, 6000, 12000, 18000, BUDGET)


class RetentionEvaluation(PandaGraspEnv):
    """Observe policy closure beside the cube without changing actions or physics."""

    def reset(self, **kwargs):
        self.closed_near_cube = False
        return super().reset(**kwargs)

    def step(self, action):
        near = np.linalg.norm(self.tcp - self.cube_xyz) <= 0.02
        result = super().step(action)
        self.closed_near_cube |= bool(near and action[3] <= -0.5)
        result[-1]["closed_near_cube"] = self.closed_near_cube
        return result


def measure(env, policy, seeds):
    rows = evaluate(env, policy, seeds)
    summary = summarize(rows)
    summary["closed_near_rate"] = sum(row["closed_near_cube"] for row in rows) / len(rows)
    return rows, summary


def test_policy(env, policy, directory):
    rows, summary = measure(env, policy, TEST_SEEDS)
    write_csv(directory / "test_episodes.csv", rows)
    write_json(directory / "test_summary.json", summary)
    print(f"{directory}: test {summary['successes']}/{len(TEST_SEEDS)}", flush=True)
    return rows


class RetentionCurve(BaseCallback):
    def __init__(self, env, output, limit, cost_match, penalty=None):
        super().__init__()
        self.env = env
        self.output = output
        self.penalty = penalty
        self.limit = limit
        self.cost_match = cost_match
        self.rows = []
        self.started = perf_counter()
        self.evaluation_s = 0.0

    def record(self, model, steps):
        now = perf_counter()
        rows, summary = measure(self.env, make_policy("sac", model), VALIDATION_SEEDS)
        self.rows.append({
            "environment_steps": steps,
            "training_s": now - self.started - self.evaluation_s,
            "critic_updates": model._n_updates,
            "actor_updates": getattr(model, "actor_gradient_updates", 0),
            "demo_loss_since_last_eval": self.penalty.drain() if self.penalty else None,
            **summary,
        })
        write_csv(self.output / "learning_curve.csv", self.rows)
        write_csv(self.output / "validation" / f"step_{steps:05}.csv", rows)
        if steps == self.limit:
            model.save(self.output / "model")
        elif steps == self.cost_match:
            model.save(self.output / "model_total_budget")
        elif steps == 2000:
            model.save(self.output / "model_02000")
        self.evaluation_s += perf_counter() - now
        print(
            f"{self.output}: {steps} steps, close {summary['closed_near_rate']:.0%}, "
            f"grip {summary['bilateral_rate']:.0%}, lift {summary['lift_rate']:.0%}, "
            f"hold {summary['successes']}/{len(rows)}",
            flush=True,
        )

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        if self.num_timesteps in CHECKPOINTS or self.num_timesteps == self.cost_match:
            self.record(self.model, self.num_timesteps)


def train(method, seed, results, output):
    directory = output / f"seed_{seed}" / method
    directory.mkdir(parents=True, exist_ok=True)
    clone_dir = results / "contact_trigger" / f"seed_{seed}" / "bc"
    source_config = results / "contact_trigger" / f"seed_{seed}" / "bc_sac" / "config.json"
    baseline = json.loads(source_config.read_text(encoding="utf-8"))
    parameters = {name: baseline[name] for name in (
        "learning_rate", "buffer_size", "batch_size", "learning_starts", "train_freq",
        "gradient_steps", "gamma", "ent_coef", "tau",
    )}
    if baseline["budget"] != BUDGET or baseline["net_arch"] != WIDTHS:
        raise ValueError("Archived baseline configuration does not match the ablation")
    demo_steps = json.loads((clone_dir / "dataset.json").read_text(encoding="utf-8"))["transitions"]
    if not 0 < demo_steps < BUDGET:
        raise ValueError("The total interaction comparison requires room for online training")
    cost_match = (BUDGET - demo_steps) // parameters["train_freq"] * parameters["train_freq"]
    limit = cost_match if method == "bc_sac_total" else BUDGET
    settings = ABLATIONS.get(method, Ablation())
    clone = Clone()
    clone.load_state_dict(torch.load(clone_dir / "model.pt", weights_only=True, map_location="cpu"))
    model_path = clone_dir / "model.pt"
    write_json(directory / "config.json", {
        "seed": seed, "method": method, "environment_steps": limit,
        "demo_steps": demo_steps, "cost_match_online_steps": cost_match,
        "baseline_config": source_config.as_posix(), "bc_checkpoint": model_path.as_posix(),
        "bc_checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "parameters": parameters, "ablation": asdict(settings),
        "validation_seeds": list(VALIDATION_SEEDS), "test_seeds": list(TEST_SEEDS),
    })
    with Monitor(PandaGraspEnv(), str(directory / "train")) as train_env, RetentionEvaluation() as evaluation_env:
        model = AblationSAC(
            "MlpPolicy", train_env, seed=seed, device="cpu",
            policy_kwargs={"net_arch": WIDTHS, "activation_fn": ACTIVATION}, **parameters,
        )
        model.ablation = settings
        initialize_actor(model.actor, clone)
        model.set_logger(configure(str(directory), ["csv"]))
        penalty = None
        hook = None
        if settings.bc_weight:
            penalty = DemonstrationPenalty(model.actor, clone_dir / "expert.npz", settings.bc_weight, seed)
            hook = model.actor.optimizer.register_step_pre_hook(penalty)
        curve = RetentionCurve(evaluation_env, directory, limit, cost_match, penalty)
        try:
            model.learn(total_timesteps=limit, callback=curve, log_interval=10)
            curve.record(model, model.num_timesteps)
        finally:
            if hook:
                hook.remove()
        write_json(directory / "cost.json", {
            "environment_steps": model.num_timesteps, "critic_updates": model._n_updates,
            "actor_updates": model.actor_gradient_updates, "training_s": curve.rows[-1]["training_s"],
        })
        test_policy(evaluation_env, make_policy("sac", model), directory)
        if cost_match < limit:
            matched = AblationSAC.load(directory / "model_total_budget.zip", device="cpu")
            test_policy(evaluation_env, make_policy("sac", matched), directory / "total_budget")


def reference(seed, results, output):
    known = {"bc": ("contact_trigger", "bc"), "sac": ("pilot", "sac"),
             "bc_sac": ("contact_trigger", "bc_sac")}
    with RetentionEvaluation() as env:
        for label, (profile, method) in known.items():
            source = results / profile / f"seed_{seed}" / method
            destination = output / f"seed_{seed}" / label
            actual = test_policy(env, load_policy(method, source), destination)
            with (source / "test_episodes.csv").open(encoding="utf-8", newline="") as stream:
                archived = list(csv.DictReader(stream))
            if len(archived) != len(actual):
                raise ValueError(f"Incomplete archived scenarios in {source}")
            for previous, replay in zip(archived, actual):
                if not all(str(replay[field]) == previous[field] for field in (
                    "scenario_seed", "is_success", "failure_reason", "steps", "max_hold_s",
                )):
                    raise ValueError(f"Archived policy failed to replay in {source}: {replay}")
            write_json(destination / "reference.json", {
                "source": source.as_posix(), "archived_cost": json.loads((source / "cost.json").read_text()),
                "archived_learning_curve": (source / "learning_curve.csv").as_posix(),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("reference", "bc_sac_total", *ABLATIONS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.out if args.out is not None else args.results / "sac_ablation"
    if args.method == "reference":
        reference(args.seed, args.results, output)
    else:
        train(args.method, args.seed, args.results, output)


if __name__ == "__main__":
    main()
