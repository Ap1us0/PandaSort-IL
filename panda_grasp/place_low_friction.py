"""Match additional simulation interactions for low-friction placement adaptation."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import torch

from .cli import make_policy
from .experiment import collect_demonstrations, evaluate, write_csv, write_json
from .learning import ACTIVATION, WIDTHS, initialize_actor, train_clone
from .place_boundary import BoundaryEvaluation, CANDIDATES, run_case
from .place_cli import clone_policy
from .place_mixed_benchmark import dataset_path
from .place_robustness import load_clone
from .place_touchdown import DOMAINS, MODERATE, build_teacher
from .place_transport_diagnosis import LOW, OUTPUT, checkpoint
from .sac_ablation import ABLATIONS, DemonstrationPenalty

TEST = range(1_520_000, 1_520_100)
VALIDATION = range(1_521_000, 1_521_012)
BUDGET = 6000
SHARED_DEMOS = 2000
UPDATES = (500, 1000, 1500)
METHODS = ("frozen_bc", "extra_demo_bc", "bc_regularized_sac")
SAC_SETTINGS = dict(
    learning_rate=3e-4, buffer_size=10000, batch_size=128, learning_starts=0,
    train_freq=4, gradient_steps=2, gamma=0.99, ent_coef="auto_0.08", tau=0.005,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_checkpoint(results, seed):
    source = checkpoint(results, seed)
    manifest = results / "place_touchdown" / "artifact_verification.json"
    expected = next(row["selected_sha256"] for row in json.loads(manifest.read_text(encoding="utf-8"))
                    if row["seed"] == seed)
    if digest(source) != expected:
        raise ValueError(f"Initial checkpoint has changed: {source}")
    return source, expected


def demonstration(directory, seed, start, steps):
    with BoundaryEvaluation(CANDIDATES[LOW]) as env:
        collect_demonstrations(
            directory, env, seed, range(start, start + steps),
            lambda rng, perturb: build_teacher(env, "verified_touch"),
            {"condition": LOW, "teacher": "verified_touch", "budget": steps},
            interaction_budget=steps, enable_perturbation=False,
        )
    metadata = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    if metadata["transitions"] != steps or metadata["successful"] == 0:
        raise ValueError(f"Invalid demonstration budget or data at {directory}")
    print(f"{directory}: {metadata['transitions']} steps, {metadata['successful']} successful demonstrations", flush=True)
    return directory / "expert.npz", metadata


def validation(policy, env):
    rows = evaluate(env, policy, VALIDATION)
    return {"validation_stable": sum(row["stable_placement_success"] for row in rows),
            "validation_transport_drops": sum(row["transport_drop"] for row in rows),
            "validation_strict": sum(row["is_success"] for row in rows)}


def extra_bc(seed, results, directory, source, demonstrations):
    directory.mkdir(parents=True, exist_ok=True)
    base = results / "place_touchdown" / "training" / f"seed_{seed}" / "demonstrations" / "expert.npz"
    old = [base if name == MODERATE else dataset_path(name, seed, results) for name in DOMAINS]
    model = load_clone(source)
    history = []
    started = perf_counter()
    eval_s = 0.0
    with BoundaryEvaluation(CANDIDATES[LOW]) as env:
        def report(step, actor, xyz, grip):
            nonlocal eval_s
            now = perf_counter()
            row = {"optimizer_updates": step, "training_s": now - started - eval_s,
                   "xyz_mse": xyz, "gripper_bce": grip, **validation(clone_policy(actor), env)}
            eval_s += perf_counter() - now
            history.append(row)
            write_csv(directory / "validation.csv", history)
            print(f"{directory}: update {step}: {row['validation_stable']}/12 stable", flush=True)

        report(0, model, None, None)
        train_clone(old[0], seed + 16200, UPDATES[-1], set(UPDATES), report,
                    model=model, additional_datasets=[*old[1:], *demonstrations])
    torch.save(model.state_dict(), directory / "model.pt")
    return {"optimizer_updates": UPDATES[-1], "training_s": history[-1]["training_s"],
            "validation_s": eval_s, "datasets": [path.as_posix() for path in [*old, *demonstrations]],
            "selection": "fixed final update"}


def seed_replay(buffer, path):
    with np.load(path) as data:
        for i, terminal in enumerate(data["terminated"]):
            truncated = bool(data["truncated"][i])
            done = bool(terminal or truncated)
            buffer.add(data["observations"][i], data["next_observations"][i], data["actions"][i],
                       data["rewards"][i:i + 1], np.array([done]),
                       [{"TimeLimit.truncated": truncated and not bool(terminal)}])
    if buffer.size() != SHARED_DEMOS:
        raise ValueError("SAC replay buffer did not receive the shared demonstration budget")


class OnlineCurve(BaseCallback):
    def __init__(self, env, directory, penalty):
        super().__init__()
        self.env = env
        self.directory = directory
        self.penalty = penalty
        self.rows = []
        self.started = perf_counter()
        self.evaluation_s = 0.0

    def record(self):
        now = perf_counter()
        row = {"online_steps": self.num_timesteps, "training_s": now - self.started - self.evaluation_s,
               "critic_updates": self.model._n_updates, "bc_loss": self.penalty.drain(),
               **validation(make_policy("sac", self.model), self.env)}
        self.evaluation_s += perf_counter() - now
        self.rows.append(row)
        write_csv(self.directory / "validation.csv", self.rows)
        print(f"{self.directory}: {self.num_timesteps} online: {row['validation_stable']}/12 stable", flush=True)

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        if self.num_timesteps in (0, 1000, 2000, 3000):
            self.record()


def regularized_sac(seed, directory, source, shared):
    directory.mkdir(parents=True, exist_ok=True)
    clone = load_clone(source)
    with Monitor(BoundaryEvaluation(CANDIDATES[LOW]), str(directory / "train")) as env, \
            BoundaryEvaluation(CANDIDATES[LOW]) as evaluation_env:
        model = SAC("MlpPolicy", env, seed=seed + 208200, device="cpu", **SAC_SETTINGS,
                    policy_kwargs={"net_arch": WIDTHS, "activation_fn": ACTIVATION})
        initialize_actor(model.actor, clone)
        model.set_logger(configure(str(directory), ["csv"]))
        seed_replay(model.replay_buffer, shared)
        penalty = DemonstrationPenalty(model.actor, shared, ABLATIONS["bc_regularized"].bc_weight, seed)
        curve = OnlineCurve(evaluation_env, directory, penalty)
        with model.actor.optimizer.register_step_pre_hook(penalty):
            model.learn(total_timesteps=BUDGET - SHARED_DEMOS, callback=curve)
            curve.num_timesteps = model.num_timesteps
            curve.record()
        if model.num_timesteps != BUDGET - SHARED_DEMOS:
            raise ValueError("SAC exceeded its allocated environment interactions")
        model.save(directory / "model")
    return {"optimizer_updates": model._n_updates, "training_s": curve.rows[-1]["training_s"],
            "validation_s": curve.evaluation_s, "seed": seed + 208200, "settings": SAC_SETTINGS,
            "bc_weight": penalty.weight, "replay_prefill": SHARED_DEMOS, "selection": "fixed final step"}


def run_training(seed, results, output):
    source, identity = locked_checkpoint(results, seed)
    root = output / f"seed_{seed}"
    start = 82_000_000 + 100_000 * seed
    shared, common_meta = demonstration(root / "demos" / "shared", seed, start, SHARED_DEMOS)
    extra, extra_meta = demonstration(root / "demos" / "bc_only", seed, start + 10_000, BUDGET - SHARED_DEMOS)
    bc = extra_bc(seed, results, root / "extra_demo_bc", source, (shared, extra))
    sac = regularized_sac(seed, root / "bc_regularized_sac", source, shared)
    records = {
        "frozen_bc": {"new_demo_steps": 0, "new_online_steps": 0, "training_s": 0.0,
                      "new_demo_collection_s": 0.0, "selected_checkpoint": source.as_posix()},
        "extra_demo_bc": {"new_demo_steps": BUDGET, "new_online_steps": 0,
                          "new_demo_collection_s": common_meta["collection_s"] + extra_meta["collection_s"],
                          "selected_checkpoint": (root / "extra_demo_bc" / "model.pt").as_posix(), **bc},
        "bc_regularized_sac": {"new_demo_steps": SHARED_DEMOS, "new_online_steps": BUDGET - SHARED_DEMOS,
                               "new_demo_collection_s": common_meta["collection_s"],
                               "selected_checkpoint": (root / "bc_regularized_sac" / "model.zip").as_posix(), **sac},
    }
    for method, record in records.items():
        write_json(root / method / "cost.json", {
            **record, "method": method, "training_seed": seed, "budget_ceiling": BUDGET,
            "initial_sha256": identity, "checkpoint_sha256": digest(Path(record["selected_checkpoint"])),
        })


def placement_sac_policy(model):
    """Evaluate the actor directly in the environment's identical [-1, 1] action space."""
    model.policy.set_training_mode(False)

    @torch.no_grad()
    def action(observation):
        obs = torch.as_tensor(observation, device=model.device).unsqueeze(0)
        return model.actor(obs, deterministic=True)[0].cpu().numpy()

    return lambda env, scene: action


def test(method, seed, output):
    directory = output / f"seed_{seed}" / method
    cost = json.loads((directory / "cost.json").read_text(encoding="utf-8"))
    source = Path(cost["selected_checkpoint"])
    if digest(source) != cost["checkpoint_sha256"]:
        raise ValueError(f"Test checkpoint has changed: {source}")
    if method == "bc_regularized_sac":
        policy = placement_sac_policy(SAC.load(source, device="cpu"))
    else:
        policy = clone_policy(load_clone(source))
    run_case(directory / "test", LOW, policy, TEST, {"seed": seed, "method": method, "action_path": "direct_tanh"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        run_training(args.seed, args.results, args.out)
    elif args.method is None:
        parser.error("test requires --method")
    else:
        test(args.method, args.seed, args.out)


if __name__ == "__main__":
    main()
