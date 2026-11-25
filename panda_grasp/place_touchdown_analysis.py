"""Aggregate paired touchdown probes, fresh held-out outcomes and verified replay clips."""

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
from .place_mixed_analysis import METRICS, read_case
from .place_robustness import ShiftedPlacementEnv, load_clone
from .place_touchdown import DOMAINS, MODERATE, OUTPUT, TEST_SCENES, origin_path

LABELS = {"nominal": "Nominal", "friction_50": "50% friction", "mass_250": "187.5 g",
          MODERATE: "75% friction, 150 g"}
COLORS = {"frozen": "#3d669a", "adapted": "#218056", "expert": "#666875"}


def probe_summary(root):
    rows = []
    for path in sorted((root / "probe").glob("seed_*/summary.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        with path.with_name("paired_episodes.csv").open(encoding="utf-8", newline="") as stream:
            raw = list(csv.DictReader(stream))
        groups = [[r for r in raw if r["intervention"] == name] for name in ("hold_grip", "slow_descent")]
        keys = ("scenario_seed", "cube_x", "cube_y", "goal_x", "goal_y", "baseline_stage")
        assert [tuple(row[k] for k in keys) for row in groups[0]] == [
            tuple(row[k] for k in keys) for row in groups[1]
        ], f"Unmatched probe scenes in {path}"
        for summary in document["outcomes"]:
            rows.append({"seed": document["seed"], **summary})
    return rows


def evaluate_summary(root):
    groups = []
    paired = []
    layouts = {}
    cases = {}
    for profile in ("frozen", "adapted", "expert"):
        pattern = "*/*/summary.json" if profile != "expert" else "*/summary.json"
        for path in sorted((root / profile).glob(pattern)):
            condition = path.parent.name
            seed = int(path.parents[1].name.removeprefix("seed_")) if profile != "expert" else None
            rows, result = read_case(path.parent, TEST_SCENES, layouts)
            cases[profile, seed, condition] = rows
            error = [float(r["goal_distance_m"]) for r in rows if r["is_success"] == "True"]
            groups.append({
                "profile": profile, "condition": condition, "seed": seed, **result,
                "success_only_median_error_m": float(np.median(error)) if error else None,
            })
    for seed in range(3):
        for name in DOMAINS:
            frozen = cases.get(("frozen", seed, name))
            adapted = cases.get(("adapted", seed, name))
            if frozen is None or adapted is None:
                continue
            pairs = [(a["is_success"] == "True", b["is_success"] == "True") for a, b in zip(frozen, adapted)]
            paired.append({
                "seed": seed, "condition": name, "episodes": len(pairs),
                "fail_to_success": sum(not a and b for a, b in pairs),
                "success_to_fail": sum(a and not b for a, b in pairs),
            })
    return groups, paired, cases


def chart(root, rows):
    fig, (domains, stages) = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True, layout="constrained")
    names = list(DOMAINS)
    for i, profile in enumerate(("frozen", "adapted")):
        samples = [[r["success_rate"] for r in rows if r["condition"] == name and r["profile"] == profile]
                   for name in names]
        centers = np.arange(len(names)) + (i - 0.5) * 0.34
        domains.bar(centers, [np.mean(s) if s else 0 for s in samples], width=0.32,
                    color=COLORS[profile], label=profile.capitalize())
        for x, values in zip(centers, samples):
            if values:
                domains.scatter(x + np.linspace(-0.075, 0.075, len(values)), values,
                                color="#25292a", s=17, zorder=3)
    domains.set_xticks(range(len(names)), [LABELS[name] for name in names])
    domains.set(ylabel="Fraction of 100 held-out scenes", title="Four-condition strict placement success")
    for profile in COLORS:
        chosen = [row for row in rows if row["profile"] == profile and row["condition"] == MODERATE]
        if chosen:
            values = np.asarray([[row[key] for key in METRICS] for row in chosen]).mean(axis=0)
            stages.plot(range(len(METRICS)), values, marker="o", color=COLORS[profile], label=profile.capitalize())
    stages.set_xticks(range(len(METRICS)), ("Grip", "Lift", "Carry", "Touch", "Release", "Settle"))
    stages.set_title("Strict milestones at 150 g, 75% friction")
    for axis in (domains, stages):
        axis.set_ylim(-0.02, 1.05)
        axis.grid(axis="y", alpha=0.25)
    handles, labels = stages.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.savefig(root / "heldout_placement.png", dpi=175)
    plt.close(fig)


def replay_pair(root, mixed, cases):
    import imageio_ffmpeg
    import torch

    torch.set_num_threads(1)
    trial = next(
        ((seed, old, new) for seed in range(3)
         for old, new in zip(cases["frozen", seed, MODERATE], cases["adapted", seed, MODERATE])
         if old["failure_stage"] == "touchdown" and new["is_success"] == "True"),
        None,
    )
    if trial is None:
        return
    seed, old, new = trial
    directory = root / "videos"
    directory.mkdir(exist_ok=True)
    model_path = root / "training" / f"seed_{seed}" / "model.pt"
    checks = []
    with ShiftedPlacementEnv(DOMAINS[MODERATE], render_mode="rgb_array") as env:
        for profile, row, path in (("frozen", old, origin_path(mixed, seed)), ("adapted", new, model_path)):
            scene = int(row["scenario_seed"])
            output = directory / f"{profile}_moderate_{scene}.mp4"
            record_video(env, clone_policy(load_clone(path)), row, output)
            frames, duration = imageio_ffmpeg.count_frames_and_secs(str(output))
            if frames != int(row["steps"]) + 1:
                raise ValueError(f"Unexpected frame count in {output}")
            checks.append({"profile": profile, "training_seed": seed, "scenario_seed": scene,
                           "frames": frames, "seconds": duration, "path": output.as_posix(),
                           "success": row["is_success"] == "True"})
    write_json(directory / "verification.json", checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--mixed", type=Path, default=Path("results/place_mixed"))
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    root = args.out
    probes = probe_summary(root)
    if probes:
        write_csv(root / "probe_summary.csv", probes)
    scores, paired, cases = evaluate_summary(root)
    if scores:
        write_csv(root / "test_summary.csv", scores)
        chart(root, scores)
    if paired:
        write_csv(root / "paired_changes.csv", paired)
    if args.videos:
        replay_pair(root, args.mixed, cases)


if __name__ == "__main__":
    main()
