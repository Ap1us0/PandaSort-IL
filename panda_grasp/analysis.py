"""Aggregate completed runs and reproduce a failure trace from saved policies."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .cli import load_policy
from .env import LIFT_M, PandaGraspEnv
from .experiment import write_csv


PROFILES = {"pilot": "fixed wait", "contact_trigger": "contact trigger"}
METHODS = {"random": "Random", "expert": "Expert", "bc": "BC", "sac": "SAC", "bc_sac": "BC + SAC"}
COLORS = {"sac": "#4355a5", "bc_sac": "#b54b32", "bc": "#376d50"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root):
    rows, curves = [], []
    for name, profile in PROFILES.items():
        for directory in sorted((root / name).glob("seed_*/*")):
            method = directory.name
            if method not in METHODS:
                continue
            seed = int(directory.parent.name.removeprefix("seed_"))
            summary = directory / "test_summary.json"
            cost = directory / "cost.json"
            if summary.is_file():
                meta = read_json(cost) if cost.exists() else {}
                bc_dir = directory.parent / "bc"
                demos = read_json(bc_dir / "dataset.json") if method in ("bc", "bc_sac") else {}
                demo_steps = demos.get("transitions", 0)
                online = meta.get("environment_steps", 0) if method in ("sac", "bc_sac") else 0
                rows.append({
                    "profile": name, "seed": seed, "method": method, **read_json(summary),
                    "demo_steps": demo_steps, "online_steps": online,
                    "total_train_data_steps": online + demo_steps,
                    "demo_collection_s": demos.get("collection_s", 0),
                    "bc_optimizer_s": read_json(bc_dir / "cost.json")["training_s"] if demos else 0,
                    "sac_training_s": meta.get("training_s", 0) if online else 0,
                })
            curve = directory / "learning_curve.csv"
            if curve.is_file():
                curves.append((name, profile, method, seed, read_csv(curve)))
    return rows, curves


def chart(curves, destination):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7), layout="constrained")
    for panel, name in enumerate(PROFILES):
        for axis in axes[panel]:
            axis.set_ylim(-0.04, 1.04)
            axis.grid(axis="y", alpha=0.25)
        axes[panel, 0].set_title(f"BC | {PROFILES[name]}")
        axes[panel, 1].set_title(f"Online training | {PROFILES[name]}")
        axes[panel, 0].set_ylabel("Validation success rate")
        axes[panel, 0].set_xlabel("BC optimizer updates")
        axes[panel, 1].set_xlabel("Online environment interactions")
        axes[panel, 0].set_xlim(0, 8300)
        axes[panel, 1].set_xlim(0, 24800)
        groups = defaultdict(list)
        for profile_id, _, method, seed, series in curves:
            if profile_id != name and not (name == "contact_trigger" and method == "sac"):
                continue
            axis_id = int(method != "bc")
            key = "environment_steps" if axis_id else "optimizer_updates"
            xs = [int(row[key]) for row in series]
            ys = [float(row["success_rate"]) for row in series]
            axes[panel, axis_id].plot(xs, ys, color=COLORS[method], alpha=0.26, linewidth=1)
            for x, y in zip(xs, ys):
                groups[axis_id, method].append((x, y))
        for (axis_id, method), values in groups.items():
            by_step = defaultdict(list)
            for x, y in values:
                by_step[x].append(y)
            steps = sorted(by_step)
            means = [float(np.mean(by_step[step])) for step in steps]
            axes[panel, axis_id].plot(steps, means, marker="o", label=METHODS[method], color=COLORS[method])
        for axis in axes[panel]:
            if axis.get_legend_handles_labels()[0]:
                axis.legend(loc="best", frameon=False)
    fig.suptitle("PandaGrasp validation on 12 fixed scenes | faint lines: training seeds")
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def trace(directory, method, seed, scenario):
    env = PandaGraspEnv()
    try:
        obs, _ = env.reset(seed=scenario)
        choose = load_policy(method, directory)(env, scenario)
        rows = []
        for step in range(env.max_steps):
            action = choose(obs)
            obs, reward, term, trunc, info = env.step(action)
            rows.append({
                "method": method, "training_seed": seed, "scenario_seed": scenario, "step": step + 1,
                "tcp_z_m": float(env.tcp[2]), "cube_lift_m": float(env.cube_xyz[2] - env.rest_z),
                "z_action": float(action[2]), "gripper_action": float(action[3]),
                "cube_linear_speed_m_s": float(np.linalg.norm(env.cube_velocity[:3])),
                "cube_angular_speed_rad_s": float(np.linalg.norm(env.cube_velocity[3:])),
                "hold_s": env.episode.hold_ticks * env.dt,
                "left_force_n": info["left_force_n"], "right_force_n": info["right_force_n"],
                "reward": reward, "status": info["failure_reason"],
            })
            if term or trunc:
                break
        return rows
    finally:
        env.close()


def diagnostic(root, output):
    fixed = root / "pilot" / "seed_1" / "bc"
    improved = root / "contact_trigger" / "seed_1" / "bc"
    if not (fixed / "model.pt").exists():
        return
    inputs = [("Expert", fixed, "expert"), ("BC fixed wait", fixed, "bc")]
    if (improved / "model.pt").exists():
        inputs.append(("BC contact trigger", improved, "bc"))
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7), sharex=True, layout="constrained")
    combined = []
    for label, path, method in inputs:
        rows = trace(path, method, 1, 900000)
        for row in rows:
            row["policy"] = label
        combined.extend(rows)
        steps = [r["step"] for r in rows]
        for axis, key in zip(axes, ("cube_lift_m", "z_action", "hold_s")):
            axis.plot(steps, [r[key] for r in rows], label=label)
    write_csv(output / "scenario_900000_trace.csv", combined)
    axes[0].axhline(LIFT_M, linestyle="--", color="gray", linewidth=1)
    axes[0].set_ylabel("Cube lift (m)")
    axes[1].set_ylabel("Policy Z action")
    axes[2].set_ylabel("Current stable hold (s)")
    axes[2].set_xlabel("Control step")
    axes[0].legend(frameon=False)
    axes[0].set_title("Scenario 900000 | matched physical rollouts")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "scenario_900000.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, curves = collect(args.root)
    output = args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        write_csv(output / "summary.csv", rows)
    if curves:
        chart(curves, output / "learning_curves.png")
    diagnostic(args.root, output)


if __name__ == "__main__":
    main()
