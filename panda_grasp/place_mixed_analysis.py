"""Audit and summarize the condition-blind placement comparison."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import record_video, write_csv, write_json
from .place_cli import clone_policy
from .place_mixed_benchmark import COMBINED, COMBINED_SCENES, OUTPUT, model_path, prior_cell
from .place_robustness import STRESS_CONDITIONS, TEST_SCENES, ShiftedPlacementEnv, load_clone

SOURCES = ("nominal", "friction_50", "mass_250", "mixed")
LABELS = {"nominal": "Nominal", "friction_50": "50% friction", "mass_250": "187.5 g",
          "mixed": "Mixed", "expert": "Expert", "combined_moderate": "75% friction, 150 g",
          "combined_full": "50% friction, 187.5 g"}
METRICS = ("bilateral_rate", "lift_rate", "transport_rate", "touchdown_rate", "release_rate", "success_rate")


def read_case(directory, scenes, layouts):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["scenario_seed"]) for row in rows] != list(scenes):
        raise ValueError(f"Ordered scenario mismatch in {directory}")
    good = [r for r in rows if r["is_success"] == "True"]
    if len(good) != summary["successes"] or any(
        r["supported_at_target"] != "True" or r["released_at_target"] != "True"
        or float(r["settled_at_target_s"]) < 1 for r in good
    ):
        raise ValueError(f"Invalid placement results in {directory}")
    for row in rows:
        scene = int(row["scenario_seed"])
        layout = tuple(row[key] for key in ("cube_x", "cube_y", "goal_x", "goal_y"))
        if layouts.setdefault(scene, layout) != layout:
            raise ValueError(f"Unmatched pickup or target in {directory}, scene {scene}")
    return rows, {
        "episodes": len(rows), "successes": len(good),
        **{key: summary[key] for key in METRICS},
        "transport_drop_rate": summary["transport_drop_rate"],
        "early_open_rate": summary["early_open_rate"],
        "failure_stages": summary["failure_stages"],
        "median_goal_error_m": summary["median_goal_error_m"],
    }


def compare(root, results):
    matrix = []
    paired = []
    layouts = {}
    for seed in range(3):
        for condition in STRESS_CONDITIONS:
            cases = {}
            for source in SOURCES:
                reused = prior_cell(source, condition, seed, results)
                path = reused or root / "cross" / source / f"seed_{seed}" / condition
                if not (path / "summary.json").is_file():
                    continue
                rows, summary = read_case(path, TEST_SCENES, layouts)
                matrix.append({"source": source, "condition": condition, "seed": seed,
                               "reused": reused is not None, "path": path.as_posix(), **summary})
                cases[source] = rows
            for other in {"nominal", condition}:
                if other not in cases or "mixed" not in cases:
                    continue
                common = [(first["is_success"] == "True", second["is_success"] == "True")
                          for first, second in zip(cases[other], cases["mixed"])]
                paired.append({"seed": seed, "condition": condition, "reference": other,
                               "failure_to_success": sum(not a and b for a, b in common),
                               "success_to_failure": sum(a and not b for a, b in common)})
    return matrix, paired


def combined(root):
    rows = []
    layouts = {}
    for path in sorted((root / "combined").rglob("summary.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        _, summary = read_case(path.parent, COMBINED_SCENES, layouts)
        rows.append({"source": result["source"], "condition": result["condition"], "seed": result["seed"],
                     "path": path.parent.as_posix(), **summary})
    return rows


def cross_chart(root, rows):
    fig, ax = plt.subplots(figsize=(8.9, 4.7), layout="constrained")
    cells = [[tuple(r["successes"] for r in rows if r["source"] == source and r["condition"] == condition)
              for condition in STRESS_CONDITIONS] for source in SOURCES]
    rates = [[sum(cell) / (100 * len(cell)) if cell else float("nan") for cell in line] for line in cells]
    image = ax.imshow(rates, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for i, line in enumerate(cells):
        for j, cell in enumerate(line):
            if cell:
                ax.text(j, i, f"{sum(cell)}/{len(cell) * 100}\n" + " / ".join(map(str, cell)),
                        ha="center", va="center", color="white" if rates[i][j] > 0.57 else "#1f2937")
    ax.set_xticks(range(len(STRESS_CONDITIONS)), [LABELS[k] for k in STRESS_CONDITIONS])
    ax.set_yticks(range(len(SOURCES)), [LABELS[k] + " BC" for k in SOURCES])
    ax.set_xlabel("Test physics")
    ax.set_title("Placement successes | lower line shows the three seeds")
    fig.colorbar(image, ax=ax, label="Success rate")
    fig.savefig(root / "cross_condition_matrix.png", dpi=175)
    plt.close(fig)


def combo_chart(root, rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1), layout="constrained", sharey=True)
    for ax, condition in zip(axes, COMBINED):
        for source, color in (("nominal", "#546fa1"), ("mixed", "#26835f"), ("expert", "#666a6d")):
            chosen = [r for r in rows if (r["source"], r["condition"]) == (source, condition)]
            if not chosen:
                continue
            points = np.asarray([[row[key] for key in METRICS] for row in chosen])
            ax.plot(range(len(METRICS)), points.mean(axis=0), color=color,
                    marker="o", label=LABELS[source])
            if len(points) > 1:
                ax.fill_between(range(len(METRICS)), points.min(axis=0), points.max(axis=0),
                                color=color, alpha=0.11, linewidth=0)
        ax.set_title(LABELS[condition])
        ax.set_xticks(range(len(METRICS)), ("Grip", "Lift", "Carry", "Touch", "Release", "Settle"))
        ax.set_ylim(-0.04, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Fraction of 100 held-out scenes")
    axes[0].legend(frameon=False, loc="upper right")
    fig.savefig(root / "unseen_combination.png", dpi=175)
    plt.close(fig)


def paired_video(root, results):
    import imageio_ffmpeg
    import torch

    torch.set_num_threads(1)
    domain = "combined_moderate"
    layouts = {}
    pair = None
    for seed in range(3):
        first, _ = read_case(root / "combined" / "nominal" / f"seed_{seed}" / domain, COMBINED_SCENES, layouts)
        second, _ = read_case(root / "combined" / "mixed" / f"seed_{seed}" / domain, COMBINED_SCENES, layouts)
        pair = next(((a, b) for a, b in zip(first, second)
                     if a["is_success"] == "False" and b["is_success"] == "True"), None)
        if pair is not None:
            break
    if pair is None:
        return
    destination = root / "videos"
    destination.mkdir(exist_ok=True)
    records = []
    with ShiftedPlacementEnv(COMBINED[domain], render_mode="rgb_array") as env:
        for source, row in zip(("nominal", "mixed"), pair):
            model = model_path(source, seed, results, root)
            scene = int(row["scenario_seed"])
            path = destination / f"{source}_{domain}_{scene}.mp4"
            record_video(env, clone_policy(load_clone(model)), row, path)
            frames, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
            if frames != int(row["steps"]) + 1:
                raise ValueError(f"Unexpected video frame count in {path}")
            records.append({"policy": source, "seed": seed, "scenario_seed": scene, "file": path.as_posix(),
                            "checkpoint": model.as_posix(), "frames": frames, "seconds": duration,
                            "success": row["is_success"] == "True"})
    write_json(destination / "verification.json", records)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    matrix, paired = compare(args.root, args.results)
    if matrix:
        write_csv(args.root / "cross_summary.csv", matrix)
        cross_chart(args.root, matrix)
    if paired:
        write_csv(args.root / "paired_changes.csv", paired)
    combos = combined(args.root)
    if combos:
        write_csv(args.root / "combined_summary.csv", combos)
        combo_chart(args.root, combos)
    if args.videos:
        paired_video(args.root, args.results)


if __name__ == "__main__":
    main()
