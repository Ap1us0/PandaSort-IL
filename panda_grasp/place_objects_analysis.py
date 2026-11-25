"""Audit the object-shape comparison, wrist-yaw ablation and videos."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .experiment import record_video, write_csv, write_json
from .learning import Clone
from .place_cli import clone_policy
from .place_objects import ACTION_NAMES, OBS_SIZE, ObjectPlacementEnv, SHAPES
from .place_objects_benchmark import OUTPUT, SCREEN, SCREEN_CASES, TEST
from .place_objects_train import sha
from .place_robustness import load_clone

METHODS = ("frozen", "mixed", "expert")
LABELS = {"frozen": "Frozen cube BC", "mixed": "Yaw-enabled mixed BC", "expert": "Aligned expert"}


def audit(directory, scenes, layouts, yaw):
    log = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["scenario_seed"]) for row in rows] != list(scenes):
        raise ValueError(f"Unexpected scenes at {directory}")
    for row in rows:
        scene = int(row["scenario_seed"])
        coords = tuple(row[key] for key in ("cube_x", "cube_y", "goal_x", "goal_y"))
        if layouts.setdefault(scene, coords) != coords:
            raise ValueError(f"Pickup or goal changed between methods for {scene}")
        if log["random_initial_yaw"] and yaw.setdefault(scene, row["initial_yaw_deg"]) != row["initial_yaw_deg"]:
            raise ValueError(f"Mismatched yaw for scene {scene}")
        if row["stable_placement_success"] == "True" and float(row["outcome_stable_s"]) < 1.0:
            raise ValueError(f"Insufficient stable hold at {directory}, scene {scene}")
        if row["is_success"] == "True" and row["stable_placement_success"] != "True":
            raise ValueError(f"Strict success without stable placement at {directory}, scene {scene}")
    for key, field in (("stable", "stable_placement_success"), ("strict", "is_success"),
                       ("lifted", "ever_carried"), ("transport_drops", "transport_drop")):
        if log[key] != sum(row[field] == "True" for row in rows):
            raise ValueError(f"Unexpected {key} count in {directory}")
    if "checkpoint" in log and sha(Path(log["checkpoint"])) != log["checkpoint_sha256"]:
        raise ValueError(f"Evaluated weights changed at {directory}")
    return rows, log


def audit_training(output):
    records = []
    for seed in range(3):
        folder = output / "training" / f"seed_{seed}"
        receipt = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if receipt["new_expert_steps"] != 9000 or sha(folder / "model.pt") != receipt["checkpoint_sha256"]:
            raise ValueError(f"Budget or weights differ at {folder}")
        for info in receipt["demonstrations"]:
            shape = info["shape"]
            path = folder / "demonstrations" / shape / "expert.npz"
            with np.load(path) as data:
                count = len(data["actions"])
                if not (data["actions"].shape == (3000, len(ACTION_NAMES))
                        and data["observations"].shape == data["next_observations"].shape == (3000, OBS_SIZE)
                        and np.array_equal(data["actions"], data["expert_actions"])
                        and np.abs(data["actions"]).max() <= 1.0
                        and data["offsets"][-1] == count == info["transitions"]
                        and int(data["successful"].sum()) == info["successful"]
                        and all(scene not in TEST[shape] for scene in data["scenario_seeds"])):
                    raise ValueError(f"Inconsistent demonstration archive: {path}")
                records.append({"seed": seed, "shape": shape, "steps": count,
                                "attempted_episodes": len(data["successful"]),
                                "successful_episodes": info["successful"], "sha256": sha(path)})
    write_csv(output / "demo_audit.csv", records)


def collect(output):
    audit_training(output)
    layouts, yaw, cases = {}, {}, {}
    screen = [audit(output / "screen" / shape / method, SCREEN, layouts, yaw)[1]
              for shape in SHAPES for method in SCREEN_CASES]
    write_csv(output / "screen.csv", [{
        key: row[key] for key in ("shape", "method", "episodes", "stable", "strict", "gripped", "lifted", "transport_drops")
    } for row in screen])
    all_records = []
    for shape, scenes in TEST.items():
        for method in METHODS:
            seeds = range(3) if method != "expert" else (None,)
            for seed in seeds:
                folder = output / "test" / method
                folder = folder / shape if seed is None else folder / f"seed_{seed}" / shape
                rows, log = audit(folder, scenes, layouts, yaw)
                cases[shape, method, seed] = rows, log
                all_records.append({key: log[key] for key in ("shape", "method", "seed", "episodes",
                                               "stable", "strict", "gripped", "lifted", "transported",
                                               "transport_drops", "failure_stages")})
    write_csv(output / "comparison.csv", all_records)
    ablation = []
    for shape in ("cube", "brick"):
        for seed in range(3):
            directory = output / "test" / "mixed_no_yaw" / f"seed_{seed}" / shape
            disabled, log = audit(directory, TEST[shape][:12], layouts, yaw)
            enabled, original = cases[shape, "mixed", seed]
            if log["checkpoint_sha256"] != original["checkpoint_sha256"]:
                raise ValueError(f"Yaw ablation changed weights in {directory}")
            pairs = list(zip(enabled[:len(disabled)], disabled))
            errors = [[float(row["close_alignment_deg"]) for row in side
                       if row.get("close_alignment_deg")] for side in (enabled[:len(disabled)], disabled)]
            ablation.append({"shape": shape, "seed": seed, "episodes": len(pairs),
                             "yaw_on_stable": sum(a["stable_placement_success"] == "True" for a, b in pairs),
                             "yaw_off_stable": log["stable"],
                             "yaw_on_lifted": sum(a["ever_carried"] == "True" for a, b in pairs),
                             "yaw_off_lifted": log["lifted"],
                             "yaw_on_median_close_error_deg": float(np.median(errors[0])) if errors[0] else None,
                             "yaw_off_median_close_error_deg": float(np.median(errors[1])) if errors[1] else None,
                             "stability_lost_without_yaw": sum(a["stable_placement_success"] == "True"
                                                              and b["stable_placement_success"] == "False"
                                                              for a, b in pairs)})
    write_csv(output / "wrist_yaw_ablation.csv", ablation)
    return all_records, cases


def plot(output, rows):
    fig, panels = plt.subplots(1, 2, figsize=(9.7, 3.8), layout="constrained")
    groups = list(SHAPES)
    x = np.arange(len(groups))
    for ax, metric, title in zip(panels, ("stable", "lifted"), ("Stable placement", "Qualified lift")):
        for index, (method, color) in enumerate(zip(METHODS, ("#617686", "#208271", "#917240"))):
            values = [[100 * row[metric] / row["episodes"] for row in rows
                       if row["shape"] == shape and row["method"] == method] for shape in groups]
            dx = x + (index - 1) * 0.23
            ax.bar(dx, [np.mean(y) for y in values], width=0.22, color=color, label=LABELS[method])
            if method != "expert":
                for center, y in zip(dx, values):
                    ax.scatter(center + np.array([-0.052, 0, 0.052]), y, color="#243535", s=14, zorder=3)
        ax.set(xticks=x, xticklabels=[name.capitalize() for name in groups], ylim=(0, 107), title=title)
        ax.grid(alpha=0.22, axis="y")
        ax.set_axisbelow(True)
    panels[0].set_ylabel("Percent of independent scenarios")
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.savefig(output / "comparison.png", dpi=175)
    plt.close(fig)


def videos(output, cases):
    import imageio_ffmpeg

    recovered = []
    for shape in ("brick", "cube", "cylinder"):
        for seed in range(3):
            before, reference = cases[shape, "frozen", seed]
            after, learned = cases[shape, "mixed", seed]
            recovered.extend((shape, seed, a, b, reference, learned) for a, b in zip(before, after)
                             if a["stable_placement_success"] == "False" and b["stable_placement_success"] == "True")
        if recovered:
            break
    if not recovered:
        print("No paired frozen failure and mixed-BC success in the tested conditions", flush=True)
        return
    shape, seed, before, after, reference, learned = max(
        recovered, key=lambda item: (item[3]["is_success"] == "True", float(item[2]["goal_distance_m"])),
    )
    destination = output / "videos"
    destination.mkdir(exist_ok=True)
    clips = []
    for name, row, source in (("frozen", before, reference), ("mixed", after, learned)):
        enabled = name == "mixed"
        path = Path(source["checkpoint"])
        if enabled:
            model = Clone(OBS_SIZE, len(ACTION_NAMES))
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
        else:
            model = load_clone(path)
        mp4 = destination / f"{shape}_{name}_seed_{seed}_{row['scenario_seed']}.mp4"
        with ObjectPlacementEnv(shape, True, enabled, render_mode="rgb_array") as env:
            record_video(env, clone_policy(model), row, mp4)
            if (str(env.max_outcome_ticks >= env.hold_ticks) != row["stable_placement_success"]
                    or str(env.transport_drop) != row["transport_drop"]):
                raise ValueError(f"Video rollout no longer reproduces outcome: {mp4}")
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(mp4))
        if frames != int(row["steps"]) + 1:
            raise ValueError(f"Unexpected number of frames in {mp4}")
        clips.append({"path": mp4.as_posix(), "method": name, "seed": seed, "frames": frames, "seconds": seconds, **row})
    write_json(destination / "criteria.json", clips)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, cases = collect(args.out)
    plot(args.out, rows)
    if args.videos:
        videos(args.out, cases)


if __name__ == "__main__":
    main()
