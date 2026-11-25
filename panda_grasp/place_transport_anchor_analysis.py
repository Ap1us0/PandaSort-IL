"""Audit and plot the matched transport-anchor study."""

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
from .place_boundary import BoundaryEvaluation, CANDIDATES
from .place_boundary_analysis import audit
from .place_cli import clone_policy
from .place_low_friction import digest
from .place_robustness import load_clone
from .place_transport_anchor import BUDGET, METHODS, OUTPUT, STEPS, TEST
from .place_transport_diagnosis import LOW

LABELS = dict(zip(METHODS, ("Frozen BC", "Ordinary demos", "Transport-anchored BC")))


def diagnose_plot(output):
    directory = output / "diagnosis"
    with (directory / "physics_trace.csv").open(encoding="utf-8", newline="") as stream:
        curves = {}
        for row in csv.DictReader(stream):
            if row["mode"] in ("reference", "ordinary") and row["group"] == "ordinary_drops":
                curves.setdefault((row["mode"], row["scenario_seed"]), []).append(row)
    records = []
    for (mode, scene), rows in curves.items():
        qualifying = [r for r in rows if r["gripping"] == "True" and float(r["height_m"]) >= 0.05]
        records.append({"mode": mode, "scenario_seed": scene,
                        "peak_after_initial_lift_mm": 1000 * max(float(r["height_m"]) for r in rows),
                        "last_qualified_goal_distance_mm": 1000 * float(qualifying[-1]["goal_distance_m"]),
                        "qualified_window_s": float(qualifying[-1]["t_s"]) - float(qualifying[0]["t_s"])})
    write_csv(directory / "clearance.csv", records)

    scene = max((r for r in records if r["mode"] == "ordinary"),
                key=lambda row: row["last_qualified_goal_distance_mm"])["scenario_seed"]
    fig, panels = plt.subplots(1, 3, figsize=(10.2, 3.45), layout="constrained", sharex=True)
    for mode, color, label in (("reference", "#617786", "Frozen BC"), ("ordinary", "#AD5539", "Ordinary demos, 250 updates")):
        rows = curves[mode, scene]
        t = np.array([float(r["t_s"]) for r in rows])
        for ax, key in zip(panels, ("height_m", "goal_distance_m", "slip_z_m")):
            ax.plot(t - t[0], [1000 * float(r[key]) for r in rows], color=color, label=label, lw=1.5)
    panels[0].axhline(50, ls="--", lw=1, color="#555")
    panels[1].axhline(22, ls="--", lw=1, color="#555")
    for ax, title in zip(panels, ("Cube lift (mm)", "Goal distance (mm)", "Vertical in-gripper slip (mm)")):
        ax.set(title=title, xlabel="Time after 5 cm lift (s)")
        ax.grid(alpha=0.2)
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False,
               title=f"Seed 1, same scene {scene}")
    fig.savefig(directory / "clearance.png", dpi=175)
    plt.close(fig)


def evaluate_outputs(output):
    all_records, cases, layouts = [], {}, {}
    for seed in range(3):
        origins, datasets = set(), set()
        root = output / f"seed_{seed}"
        for method in METHODS:
            directory = root / method
            rows, report = audit(directory, TEST, layouts)
            path = Path(report["checkpoint"])
            if digest(path) != report["checkpoint_sha256"]:
                raise ValueError(f"Evaluated checkpoint differs from {directory}: {path}")
            if (report["shared_demo_steps"], report["updates"]) != (
                (0, 0) if method == "frozen" else (BUDGET, STEPS)
            ):
                raise ValueError(f"Unmatched experiment budget: {directory}")
            origins.add(report["initial_sha256"])
            datasets.add(report["ordinary_demo_sha256"])
            cases[seed, method] = rows, path
            final_errors = [float(row["goal_distance_m"]) for row in rows]
            all_records.append({
                "seed": seed, "method": method, "episodes": len(rows), "stable": report["stable_placement_successes"],
                "strict": report["strict_successes"], "drops": report["transport_drops"],
                "lifted": sum(row["ever_carried"] == "True" for row in rows),
                "transported": sum(row["transported"] == "True" for row in rows),
                "median_goal_error_mm": float(np.median(final_errors) * 1000),
                "p90_goal_error_mm": float(np.quantile(final_errors, 0.9) * 1000),
                "shared_demo_steps": report["shared_demo_steps"], "updates": report["updates"],
            })
        if len(origins) != 1 or len(datasets) != 1:
            raise ValueError(f"Methods did not share the parent and ordinary demonstrations: seed {seed}")
    paired = []
    for seed in range(3):
        for reference, candidate in (("frozen", "ordinary"), ("frozen", "transport_anchored"),
                                      ("ordinary", "transport_anchored")):
            pairs = list(zip(cases[seed, reference][0], cases[seed, candidate][0]))
            paired.append({
                "seed": seed, "reference": reference, "candidate": candidate,
                "stable_gained": sum(a["stable_placement_success"] == "False"
                                     and b["stable_placement_success"] == "True" for a, b in pairs),
                "stable_lost": sum(a["stable_placement_success"] == "True"
                                   and b["stable_placement_success"] == "False" for a, b in pairs),
                "drops_avoided": sum(a["transport_drop"] == "True" and b["transport_drop"] == "False" for a, b in pairs),
                "drops_added": sum(a["transport_drop"] == "False" and b["transport_drop"] == "True" for a, b in pairs),
            })
    write_csv(output / "comparison.csv", all_records)
    write_csv(output / "paired.csv", paired)
    return all_records, cases


def draw(output, records):
    fig, panels = plt.subplots(1, 2, figsize=(9.6, 3.85), layout="constrained")
    x = np.arange(3)
    for ax, key, title in zip(panels, ("stable", "drops"), ("Stable placement", "Transport drops")):
        for i, (name, color) in enumerate(zip(METHODS, ("#617786", "#AD5539", "#208271"))):
            y = [row[key] for row in records if row["method"] == name]
            bars = ax.bar(x + (i - 1) * 0.23, y, 0.22, label=LABELS[name], color=color)
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set(xticks=x, xticklabels=[f"Seed {s}" for s in x], title=title)
        ax.set_ylim(0, 108 if key == "stable" else max(5, 1.24 * max(row[key] for row in records)))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    panels[0].set_ylabel("Episodes per 100 scenes")
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.savefig(output / "comparison.png", dpi=175)
    plt.close(fig)


def videos(output, cases):
    import imageio_ffmpeg

    for seed in (1, 0, 2):
        before, control = cases[seed, "ordinary"]
        after, candidate = cases[seed, "transport_anchored"]
        pairs = [(a, b) for a, b in zip(before, after)
                 if a["transport_drop"] == "True" and b["stable_placement_success"] == "True"]
        if pairs:
            break
    else:
        print("No paired ordinary transport drop recovered to stable placement", flush=True)
        return
    missed, placed = max(pairs, key=lambda item: (
        item[1]["is_success"] == "True", float(item[0]["goal_distance_m"]),
    ))
    directory = output / "videos"
    directory.mkdir(exist_ok=True)
    records = []
    for name, row, source in (("ordinary", missed, control), ("transport_anchored", placed, candidate)):
        path = directory / f"{name}_seed_{seed}_{row['scenario_seed']}.mp4"
        with BoundaryEvaluation(CANDIDATES[LOW], render_mode="rgb_array") as env:
            record_video(env, clone_policy(load_clone(source)), row, path)
            if (str(env.max_outcome_ticks >= env.hold_ticks) != row["stable_placement_success"]
                    or str(env.transport_drop) != row["transport_drop"]):
                raise ValueError(f"Video changes recorded outcome: {path}")
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        if frames != int(row["steps"]) + 1:
            raise ValueError(f"Video timing differs: {path}")
        records.append({"path": path.as_posix(), "method": name, "seed": seed, "frames": frames, "seconds": seconds, **row})
    write_json(directory / "criteria.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--diagnosis-only", action="store_true")
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    diagnose_plot(args.out)
    if not args.diagnosis_only:
        records, cases = evaluate_outputs(args.out)
        draw(args.out, records)
        if args.videos:
            videos(args.out, cases)


if __name__ == "__main__":
    main()
