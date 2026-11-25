"""Verify, summarize and render the final held-out placement evaluation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .experiment import record_video, write_csv, write_json
from .place_boundary import BoundaryEvaluation, CANDIDATES, OUTPUT, SCREEN_SCENES, TEST_SCENES
from .place_cli import clone_policy
from .place_mixed_analysis import read_case
from .place_robustness import load_clone


def audit(directory, scenes, layouts):
    rows, _ = read_case(directory, scenes, layouts)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    failures = Counter()
    for row in rows:
        stable = row["stable_placement_success"] == "True"
        strict = row["is_success"] == "True"
        if stable != (float(row["outcome_stable_s"]) >= 1) or (strict and not stable):
            raise ValueError(f"Inconsistent placement criteria: {directory}, {row['scenario_seed']}")
        if not stable:
            failures[row["failure_stage"]] += 1
    if len(rows) - failures.total() != report["stable_placement_successes"]:
        raise ValueError(f"Inconsistent stable counts: {directory}")
    if sum(row["is_success"] == "True" for row in rows) != report["strict_successes"]:
        raise ValueError(f"Inconsistent strict counts: {directory}")
    return rows, {
        **report,
        "transport_drops": sum(row["transport_drop"] == "True" for row in rows),
        "strict_stages_among_stable_failures": dict(failures),
    }


def collect(output):
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if selection["screen_seeds"] != list(SCREEN_SCENES) or selection["test_seeds"] != list(TEST_SCENES):
        raise ValueError("Unexpected screening or held-out scenario seeds")
    layouts = {}
    screening = [audit(output / "screen" / name, SCREEN_SCENES, layouts)[1] for name in CANDIDATES]
    write_csv(output / "screen.csv", screening)
    cases, summaries = {}, []
    for name in selection["policy_cases"] + selection["expert_only_cases"]:
        sources = ["expert"]
        if name in selection["policy_cases"]:
            sources += [f"seed_{seed}" for seed in range(3)]
        for source in sources:
            rows, summary = audit(output / "test" / source / name, TEST_SCENES, layouts)
            if source != "expert":
                path = Path(summary["checkpoint"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != summary["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint has changed: {path}")
            cases[name, source] = rows, summary
            summaries.append(summary)
    # Expert and policy use different metadata keys.
    fields = dict.fromkeys(key for summary in summaries for key in summary)
    write_csv(output / "summary.csv", [{key: row.get(key) for key in fields} for row in summaries])
    return selection, cases


def plot(output, selection, cases):
    names = selection["policy_cases"] + selection["expert_only_cases"]
    labels = [f"{name.split('_')[1]}% friction\n{name.split('_')[3]} g" for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, ("stable_placement_successes", "strict_successes"),
                                 ("Stable placement", "Strict gripped touchdown")):
        x = np.arange(len(names))
        expert = [cases[name, "expert"][1][metric] for name in names]
        ax.bar(x - 0.19, expert, 0.34, color="#627583", label="Expert")
        for i, name in enumerate(selection["policy_cases"]):
            seeds = [cases[name, f"seed_{seed}"][1][metric] for seed in range(3)]
            ax.bar(i + 0.19, np.mean(seeds), 0.34, color="#167D71", label="Frozen BC" if i == 0 else None)
            ax.scatter(i + np.array([0.11, 0.19, 0.27]), seeds, color="#132F2B", s=15, zorder=4)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 105), title=title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Successes per 100 scenes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    fig.savefig(output / "heldout_placement.png", dpi=170)
    plt.close(fig)


def videos(output, selection, cases):
    import imageio_ffmpeg

    name = selection["policy_cases"][-1]
    archive = [(rows, summary) for (condition, _), (rows, summary) in cases.items()
               if condition == name and summary["source"] == "frozen"]
    entries = [(row, summary) for rows, summary in archive for row in rows]
    folder = output / "videos"
    folder.mkdir(exist_ok=True)
    clips = []
    successful = (entry for entry in entries if entry[0]["is_success"] == "True")
    failed = (entry for entry in entries if entry[0]["stable_placement_success"] == "False")
    examples = {
        "strict_success": next(successful, None),
        "stable_placement_failure": max(failed, key=lambda entry: float(entry[0]["goal_distance_m"]), default=None),
    }
    for label, example in examples.items():
        if example is None:
            continue
        row, summary = example
        path = folder / f"{label}_seed_{summary['seed']}_{row['scenario_seed']}.mp4"
        policy = clone_policy(load_clone(summary["checkpoint"]))
        with BoundaryEvaluation(CANDIDATES[name], render_mode="rgb_array") as env:
            record_video(env, policy, row, path)
            stable = env.max_outcome_ticks >= env.hold_ticks
            if str(stable) != row["stable_placement_success"]:
                raise ValueError(f"Stable result did not reproduce in {path}")
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        if frames != int(row["steps"]) + 1:
            raise ValueError(f"Unexpected frame count in {path}")
        clips.append({"path": path.as_posix(), "condition": name, "seed": summary["seed"],
                      "frames": frames, "seconds": seconds, **row})
    write_json(folder / "criteria.json", clips)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    selection, cases = collect(args.out)
    plot(args.out, selection, cases)
    if args.videos:
        videos(args.out, selection, cases)


if __name__ == "__main__":
    main()
