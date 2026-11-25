"""Run matched PandaGrasp pilots and replay deterministic evaluation videos."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import torch

from .env import GAMMA, PandaGraspEnv
from .experiment import (
    TEST_SEEDS, VALIDATION_SEEDS, demo_dataset, evaluate, record_video, summarize,
    write_csv, write_json,
)
from .expert import PickupExpert
from .learning import ACTIVATION, LOG_STD_INIT, WIDTHS, Clone, initialize_actor, train_clone


def make_policy(method, model=None):
    if method == "random":
        def uniform(env, seed):
            rng = np.random.default_rng(seed + 20_000_000)
            return lambda obs: rng.uniform(-1, 1, size=4).astype(np.float32)
        return uniform
    if method == "expert":
        def expert(env, seed):
            controller = PickupExpert(env)
            return lambda obs: controller(obs)[1]
        return expert
    if method == "bc":
        return lambda env, seed: model.action
    return lambda env, seed: lambda obs: model.predict(obs, deterministic=True)[0]


def evaluation(env, method, model, directory):
    rows = evaluate(env, make_policy(method, model), TEST_SEEDS)
    write_csv(directory / "test_episodes.csv", rows)
    metrics = summarize(rows)
    write_json(directory / "test_summary.json", metrics)
    print(f"{directory}: {metrics['successes']}/{len(rows)} successes", flush=True)


class TrainingCurve(BaseCallback):
    def __init__(self, env, method, directory, checkpoints):
        super().__init__()
        self.env = env
        self.method = method
        self.directory = directory
        self.checkpoints = checkpoints
        self.rows = []
        self.started = perf_counter()
        self.evaluation_s = 0.0

    def record(self, model, steps):
        now = perf_counter()
        rows = evaluate(self.env, make_policy(self.method, model), VALIDATION_SEEDS)
        elapsed = now - self.started - self.evaluation_s
        metrics = summarize(rows)
        self.rows.append({"environment_steps": steps, "training_s": elapsed, **metrics})
        write_csv(self.directory / "learning_curve.csv", self.rows)
        self.evaluation_s += perf_counter() - now
        print(f"{self.directory}: step {steps}, validation {metrics['successes']}/{len(rows)}", flush=True)

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # On rollout start, SB3 has already finished the preceding gradient update.
        if self.num_timesteps in self.checkpoints:
            self.record(self.model, self.num_timesteps)


def run_sac(method, seed, budget, directory, env, clone):
    directory.mkdir(parents=True, exist_ok=True)
    train_env = Monitor(PandaGraspEnv(), str(directory / "train"))
    try:
        options = dict(
            learning_rate=3e-4, buffer_size=100_000, batch_size=128,
            learning_starts=512, train_freq=4, gradient_steps=2, gamma=GAMMA,
            ent_coef="auto_0.08", tau=0.005,
        )
        model = SAC(
            "MlpPolicy", train_env, seed=seed, device="cpu",
            policy_kwargs={"net_arch": WIDTHS, "activation_fn": ACTIVATION}, **options,
        )
        model.set_logger(configure(str(directory), ["csv"]))
        initialize_actor(model.actor, clone)
        write_json(directory / "config.json", {
            "method": method, "seed": seed, "budget": budget, "device": "cpu", **options,
            "net_arch": WIDTHS, "activation": ACTIVATION.__name__, "initial_log_std": LOG_STD_INIT,
            "bc_actor_initialized": clone is not None,
        })
        checkpoints = {0, 2000, 6000, 12000, 18000, budget}
        curve = TrainingCurve(env, method, directory, checkpoints)
        model.learn(total_timesteps=budget, callback=curve, log_interval=10)
        curve.record(model, model.num_timesteps)
        model.save(directory / "model")
        write_json(directory / "cost.json", {
            "environment_steps": model.num_timesteps, "gradient_updates": model._n_updates,
            "training_s": curve.rows[-1]["training_s"],
        })
        evaluation(env, method, model, directory)
    finally:
        train_env.close()


def run_clone(seed, count, updates, directory, env, contact_trigger=False):
    directory.mkdir(parents=True, exist_ok=True)
    dataset_file = directory / "expert.npz"
    if not dataset_file.exists():
        demo_dataset(directory, seed, count, contact_trigger=contact_trigger)
    else:
        metadata = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
        if (metadata["episodes"] != count or metadata["seed"] != seed
                or metadata.get("contact_trigger_expert", False) != contact_trigger):
            raise ValueError("Dataset does not match the run arguments. Choose a fresh --out directory.")
    curve = []
    eval_time = 0.0
    started = perf_counter()

    def report(step, model, xyz_mse, grip_loss):
        nonlocal eval_time
        now = perf_counter()
        rows = evaluate(env, make_policy("bc", model), VALIDATION_SEEDS)
        curve.append({
            "optimizer_updates": step, "training_s": now - started - eval_time,
            "xyz_mse": xyz_mse, "gripper_bce": grip_loss, **summarize(rows),
        })
        write_csv(directory / "learning_curve.csv", curve)
        eval_time += perf_counter() - now
        print(f"{directory}: update {step}, validation {curve[-1]['successes']}/{len(rows)}", flush=True)

    model = train_clone(dataset_file, seed, updates, {1000, 2500, 5000, updates}, report)
    torch.save(model.state_dict(), directory / "model.pt")
    with np.load(dataset_file) as data:
        steps = len(data["actions"])
        successful = int(data["successful"].sum())
    write_json(directory / "cost.json", {
        "environment_steps": steps, "successful_demos": successful, "optimizer_updates": updates,
        "training_s": curve[-1]["training_s"],
    })
    evaluation(env, "bc", model, directory)
    return model


def load_policy(method, directory):
    if method in ("random", "expert"):
        return make_policy(method)
    if method == "bc":
        model = Clone()
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
    else:
        model = SAC.load(directory / "model.zip", device="cpu")
    return make_policy(method, model)


def run(args):
    root = args.out / f"seed_{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "runtime.txt").write_text(
        f"muJoCo {mujoco.__version__}\ngymnasium {gymnasium.__version__}\n"
        f"torch {torch.__version__}\nSB3 {stable_baselines3.__version__}\n"
        f"torch_threads {torch.get_num_threads()}\n", encoding="utf-8",
    )
    env = PandaGraspEnv()
    try:
        scenarios = []
        for split, seeds in (("validation", VALIDATION_SEEDS), ("test", TEST_SEEDS)):
            for seed in seeds:
                _, info = env.reset(seed=seed)
                scenarios.append({"split": split, "scenario_seed": seed,
                                  "cube_x": info["cube_x"], "cube_y": info["cube_y"]})
        write_csv(root / "scenarios.csv", scenarios)
        clone = None
        for method in args.methods:
            directory = root / method
            if method in ("random", "expert"):
                evaluation(env, method, None, directory)
            elif method == "bc":
                clone = run_clone(args.seed, args.demos, args.bc_updates, directory, env, args.contact_trigger_expert)
            else:
                pretrained = None
                if method == "bc_sac":
                    pretrained = clone if clone is not None else Clone()
                    if clone is None:
                        pretrained.load_state_dict(torch.load(
                            root / "bc" / "model.pt", map_location="cpu", weights_only=True,
                        ))
                run_sac(method, args.seed, args.steps, directory, env, pretrained)
    finally:
        env.close()


def videos(args):
    import csv

    root = args.out / f"seed_{args.seed}"
    env = PandaGraspEnv(render_mode="rgb_array")
    try:
        for method in args.methods:
            directory = root / method
            with (directory / "test_episodes.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            policy = load_policy(method, directory)
            for outcome, success in (("success", "True"), ("failure", "False")):
                row = next((row for row in rows if row["is_success"] == success), None)
                if row is None:
                    continue
                seed = int(row["scenario_seed"])
                path = directory / f"{outcome}_{seed}.mp4"
                record_video(env, policy, row, path)
                print(path, flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "videos"))
    parser.add_argument("--out", type=Path, default=Path("results/pilot"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=24_000)
    parser.add_argument("--demos", type=int, default=128)
    parser.add_argument("--bc-updates", type=int, default=8000)
    parser.add_argument("--contact-trigger-expert", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=("random", "expert", "bc", "sac", "bc_sac"),
                        default=["random", "expert", "bc", "sac", "bc_sac"])
    args = parser.parse_args()
    if args.seed < 0 or args.steps < 4 or args.steps % 4 or args.demos < 1 or args.bc_updates < 1:
        parser.error("Use a nonnegative seed, positive demo and BC counts and a step budget divisible by four")
    torch.set_num_threads(1)
    if args.command == "run":
        run(args)
    else:
        videos(args)


if __name__ == "__main__":
    main()
