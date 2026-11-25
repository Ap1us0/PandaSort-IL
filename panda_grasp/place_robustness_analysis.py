"""Audit paired placement trials and plot where condition shifts interrupt execution."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._vendor.pickup import measure_contacts
from .experiment import record_video, write_csv, write_json
from .physics_shift import CONDITIONS
from .place_cli import clone_policy
from .place_robustness import (
    OUTPUT, SCREEN_SCENES, STRESS_CONDITIONS, TEST_SCENES, ShiftedPlacementEnv, load_clone,
)

LABELS = {"nominal": "Nominal", "friction_50": "50% sliding friction", "mass_250": "187.5 g cube"}
PHASES = ("grip", "lift", "transport", "touchdown", "release", "settle")
PHASE_METRICS = ("bilateral_rate", "lift_rate", "transport_rate", "touchdown_rate", "release_rate", "success_rate")


def load(directory, scenes, manifests):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "episodes.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if [int(row["scenario_seed"]) for row in rows] != list(scenes):
        raise ValueError(f"Scenario mismatch in {directory}")
    good = [row for row in rows if row["is_success"] == "True"]
    if len(good) != summary["successes"]:
        raise ValueError(f"Success count mismatch in {directory}")
    if any(row["supported_at_target"] != "True" or row["released_at_target"] != "True"
           or float(row["settled_at_target_s"]) < 1 for row in good):
        raise ValueError(f"Invalid successful trial in {directory}")
    for row in rows:
        scene = int(row["scenario_seed"])
        layout = tuple(row[key] for key in ("cube_x", "cube_y", "goal_x", "goal_y"))
        if manifests.setdefault(scene, layout) != layout:
            raise ValueError(f"Different layout for the same scenario in {directory}")
    detail = {
        "successful_median_goal_error_m": float(np.median([float(row["goal_distance_m"]) for row in good])) if good else None,
        **{f"failed_{stage}": summary["failure_stages"].get(stage, 0) for stage in PHASES},
    }
    return rows, {**summary, **detail}


def collect(root):
    data = []
    by_key = {}
    manifests = {}
    for category, scenes in (("screen", SCREEN_SCENES), ("test", TEST_SCENES), ("adapt", TEST_SCENES)):
        for path in sorted((root / category).glob("*/*/summary.json")):
            trial, summary = load(path.parent, scenes, manifests)
            run = path.parents[1].name
            key = (category, run, summary["condition"])
            by_key[key] = trial
            result = {
                "split": category, "run": run, "condition": summary["condition"],
                "new_demonstration_steps": summary.get("new_demonstration_steps", 0),
                "selected_updates": summary.get("selected_updates"),
            }
            result.update({k: summary[k] for k in (
                "episodes", "successes", "success_rate", *PHASE_METRICS[:-1],
                "transport_drop_rate", "transport_drops_per_lift", "transport_grip_loss_rate",
                "early_open_rate", "table_at_goal_rate", "median_goal_error_m",
                "successful_median_goal_error_m",
            )})
            result.update({f"failed_{stage}": summary[f"failed_{stage}"] for stage in PHASES})
            data.append(result)
    pairs = []
    for (category, run, condition), adapted in by_key.items():
        if category != "adapt" or ("test", run, condition) not in by_key:
            continue
        frozen = by_key["test", run, condition]
        frozen_to_adapted = [(a["is_success"] == "True", b["is_success"] == "True")
                             for a, b in zip(frozen, adapted)]
        pairs.append({
            "run": run, "condition": condition,
            "failure_to_success": sum(not a and b for a, b in frozen_to_adapted),
            "success_to_failure": sum(a and not b for a, b in frozen_to_adapted),
        })
    probes = []
    for path in sorted((root / "grip_probe").glob("seed_*/friction_50/summary.json")):
        run = path.parents[1].name
        if ("test", run, "friction_50") not in by_key:
            continue
        altered, stats = load(path.parent, TEST_SCENES[::4], manifests)
        frozen = by_key["test", run, "friction_50"][::4]
        transitions = [(a["is_success"] == "True", b["is_success"] == "True")
                       for a, b in zip(frozen, altered)]
        probes.append({
            "run": run, "episodes": len(frozen),
            "frozen_successes": sum(a for a, _ in transitions), "probe_successes": stats["successes"],
            "failure_to_success": sum(not a and b for a, b in transitions),
            "success_to_failure": sum(a and not b for a, b in transitions),
            "frozen_transport_drops": sum(row["transport_drop"] == "True" for row in frozen),
            "probe_transport_drops": sum(row["transport_drop"] == "True" for row in altered),
        })
    if probes:
        write_csv(root / "paired_grip_probe.csv", probes)
    if data:
        write_csv(root / "summary.csv", data)
    if pairs:
        write_csv(root / "paired_changes.csv", pairs)
    return data


def chart(root, rows):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), layout="constrained", sharey=True)
    styles = (("test", "expert", "Expert", "#656970"), ("test", "seed_", "Frozen BC", "#315e9b"),
              ("adapt", "seed_", "Adapted BC", "#217951"))
    for ax, condition in zip(axes, STRESS_CONDITIONS):
        for split, prefix, label, color in styles:
            chosen = [row for row in rows if row["split"] == split and row["run"].startswith(prefix)
                      and row["condition"] == condition]
            if not chosen:
                continue
            points = np.array([[row[key] for key in PHASE_METRICS] for row in chosen])
            ax.plot(range(len(PHASES)), points.mean(axis=0), marker="o", color=color, label=label)
            if len(points) > 1:
                ax.fill_between(range(len(PHASES)), points.min(axis=0), points.max(axis=0),
                                color=color, alpha=0.12, linewidth=0)
        ax.set_title(LABELS[condition])
        ax.set_xticks(range(len(PHASES)), ("Grip", "Lift", "Carry", "Touch", "Release", "Settle"))
        ax.set_ylim(-0.04, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Fraction of 100 held-out scenes")
    handles, labels = axes[1].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, ncols=3, loc="outside upper center", frameon=False)
    fig.savefig(root / "placement_milestones.png", dpi=175)
    plt.close(fig)


def paired_video(root, checkpoints):
    import imageio_ffmpeg
    import torch

    torch.set_num_threads(1)
    condition = "friction_50"
    run = "seed_0"
    manifest = {}
    frozen, _ = load(root / "test" / run / condition, TEST_SCENES, manifest)
    adapted, _ = load(root / "adapt" / run / condition, TEST_SCENES, manifest)
    first, second = next(
        (a, b) for a, b in zip(frozen, adapted)
        if a["failure_stage"] == "touchdown" and a["open_without_gripped_touchdown"] == "True"
        and b["is_success"] == "True"
    )
    output = root / "videos"
    output.mkdir(exist_ok=True)
    cases = (("frozen", first, checkpoints / run / "model.pt"),
             ("adapted", second, root / "adapt" / run / condition / "model.pt"))
    verified = []
    frames_by_step = []
    with ShiftedPlacementEnv(CONDITIONS[condition], render_mode="rgb_array") as env:
        for label, row, path in cases:
            policy = clone_policy(load_clone(path))
            scene = int(row["scenario_seed"])
            obs, _ = env.reset(seed=scene)
            predict = policy(env, scene)
            while True:
                action = predict(obs)
                obs, _, terminated, truncated, info = env.step(action)
                contacts = measure_contacts(env, env.data)
                frames_by_step.append({
                    "policy": label, "scene": scene, "step": env.episode.steps,
                    "cube_lift_m": float(env.cube_xyz[2] - env.rest_z),
                    "goal_error_m": env.goal_distance, "gripper_command": float(action[3]),
                    "left_force_n": float(contacts.left_normal_n), "right_force_n": float(contacts.right_normal_n),
                    "table_force_n": float(contacts.table_normal_n), "transported": info["transported"],
                    "gripped_touchdown": info["supported_at_target"], "released": info["released_at_target"],
                    "early_open": info["open_without_gripped_touchdown"],
                })
                if terminated or truncated:
                    break
            video = output / f"{label}_friction_50_{scene}.mp4"
            record_video(env, policy, row, video)
            frames, secs = imageio_ffmpeg.count_frames_and_secs(str(video))
            if frames != int(row["steps"]) + 1:
                raise ValueError(f"Unexpected video duration for {video}")
            verified.append({"policy": label, "checkpoint": path.as_posix(), "video": video.as_posix(),
                             "scenario_seed": scene, "frames": frames, "seconds": secs,
                             "success": row["is_success"] == "True"})
    write_csv(output / "placement_trace.csv", frames_by_step)
    write_json(output / "verification.json", verified)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoints", type=Path, default=Path("results/pick_place"))
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    rows = collect(args.root)
    if any(row["split"] == "test" for row in rows):
        chart(args.root, rows)
    if args.videos:
        paired_video(args.root, args.checkpoints)


if __name__ == "__main__":
    main()
