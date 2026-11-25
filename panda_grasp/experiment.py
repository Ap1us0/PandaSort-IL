"""Shared scenario manifests, evaluation and demonstration collection."""

from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .env import OBS_FIELDS, PandaGraspEnv
from .expert import PickupExpert

VALIDATION_SEEDS = tuple(range(800_000, 800_012))
TEST_SEEDS = tuple(range(900_000, 900_100))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rollout(env, policy, seed, frames=None):
    obs, _ = env.reset(seed=seed)
    action = policy(env, seed)
    if frames is not None:
        frames.send(env.render())
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(action(obs))
        total += reward
        if frames is not None:
            frames.send(env.render())
        if terminated or truncated:
            row = {key: val for key, val in info.items() if key != "reward_parts"}
            return {"scenario_seed": seed, **row, "return": total,
                    "steps": env.episode.steps, "terminated": terminated, "truncated": truncated}


def evaluate(env, policy, seeds):
    return [rollout(env, policy, seed) for seed in seeds]


def summarize(rows):
    lifted = [row for row in rows if row["ever_carried"]]
    dropped = sum(row["dropped"] for row in rows)
    return {
        "episodes": len(rows),
        "successes": sum(row["is_success"] for row in rows),
        "success_rate": float(np.mean([row["is_success"] for row in rows])),
        "bilateral_rate": float(np.mean([row["ever_bilateral"] for row in rows])),
        "lift_rate": len(lifted) / len(rows),
        "drop_rate": dropped / len(rows),
        "drops_per_lift": dropped / len(lifted) if lifted else None,
        "mean_max_lift_m": float(np.mean([row["max_lift_m"] for row in rows])),
        "mean_max_hold_s": float(np.mean([row["max_hold_s"] for row in rows])),
        "mean_max_hold_among_lifted_s": float(np.mean([row["max_hold_s"] for row in lifted])) if lifted else None,
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "failure_reasons": dict(Counter(row["failure_reason"] for row in rows if not row["is_success"])),
    }


def demo_dataset(directory, seed, count, contact_trigger=False):
    with PandaGraspEnv() as env:
        collect_demonstrations(
            directory, env, seed, range(10_000_000 + 10_000 * seed, 10_000_000 + 10_000 * seed + count),
            lambda rng, perturb: PickupExpert(env, rng, perturb=perturb, contact_trigger=contact_trigger),
            {"contact_trigger_expert": contact_trigger, "obs_fields": OBS_FIELDS},
        )


def collect_demonstrations(directory, env, seed, scene_seeds, make_expert, metadata, *,
                           interaction_budget=None, enable_perturbation=True,
                           action_names=("dx", "dy", "dz", "gripper")):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    episodes = []
    records = []
    interactions = 0
    started = perf_counter()
    for index, scene_seed in enumerate(scene_seeds):
        obs, _ = env.reset(seed=scene_seed)
        perturbed = enable_perturbation and index >= len(scene_seeds) // 4
        expert = make_expert(rng, perturbed)
        transitions = []
        while True:
            label, action = expert(obs)
            state = expert.state.name
            following, reward, terminated, truncated, info = env.step(action)
            interactions += 1
            if interactions == interaction_budget and not (terminated or truncated):
                truncated = True
                info["failure_reason"] = "collection_budget_cutoff"
            transitions.append((obs, action, label, following, reward, terminated, truncated, state))
            obs = following
            if terminated or truncated:
                break
        row = {key: val for key, val in info.items() if key != "reward_parts"}
        row.update(episode=index, scenario_seed=scene_seed, steps=len(transitions), perturbed=perturbed)
        episodes.append(row)
        records.append(transitions)
        if interactions == interaction_budget:
            break

    save_demonstrations(directory / "expert.npz", records, episodes)
    write_csv(directory / "expert_episodes.csv", episodes)
    write_json(directory / "dataset.json", {
        "seed": seed, "episodes": len(episodes), "successful": sum(row["is_success"] for row in episodes),
        "transitions": interactions, "collection_s": perf_counter() - started,
        **metadata, "actions": list(action_names),
    })



def save_demonstrations(path, records, episodes):
    flat = [item for episode in records for item in episode]
    observations, actions, labels, next_obs, rewards, terminated, truncated, phases = map(np.asarray, zip(*flat))
    offsets = np.cumsum([0, *(len(episode) for episode in records)])
    np.savez_compressed(
        path, observations=observations, actions=actions, expert_actions=labels,
        next_observations=next_obs, rewards=rewards.astype(np.float32), terminated=terminated,
        truncated=truncated, phase=phases, offsets=offsets,
        scenario_seeds=np.asarray([row["scenario_seed"] for row in episodes]),
        successful=np.asarray([row["is_success"] for row in episodes]),
    )


def record_video(env, policy, row, path):
    import imageio_ffmpeg

    seed = int(row["scenario_seed"])
    encoder = imageio_ffmpeg.write_frames(
        str(path), (800, 480), fps=env.metadata["render_fps"], codec="libx264", macro_block_size=1,
    )
    try:
        encoder.send(None)
        replay = rollout(env, policy, seed, frames=encoder)
        if str(replay["is_success"]) != str(row["is_success"]) or replay["steps"] != int(row["steps"]):
            raise ValueError(f"Video replay did not reproduce scenario {seed}")
    finally:
        encoder.close()
