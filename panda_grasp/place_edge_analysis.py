"""Audit and plot landing diagnostics and matched-budget BC evaluation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .env import OBS_FIELDS, OBS_SIZE
from .experiment import record_video, write_csv, write_json
from .place import GOAL_RADIUS
from .place_boundary import BoundaryEvaluation, CANDIDATES
from .place_boundary_analysis import audit
from .place_cli import clone_policy
from .place_edge import BUDGET, CHECKPOINTS, TEST, VIEWS
from .place_edge_diagnosis import OUTPUT
from .place_low_friction import digest
from .place_robustness import load_clone
from .place_transport_diagnosis import LOW

NAMES = dict(zip(VIEWS, ("Prior BC", "Generic landing demos", "Learner-conditioned landing demos")))


def diagnosis(output):
    folder = output / "diagnosis"
    with (folder / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    groups = {}
    for row in episodes:
        key = row["method"], row["mode"], row["group"]
        groups.setdefault(key, []).append(row)
    write_csv(folder / "summary.csv", [
        {"method": method, "mode": mode, "group": group, "episodes": len(rows),
         "stable": sum(r["stable_placement_success"] == "True" for r in rows),
         "drops": sum(r["transport_drop"] == "True" for r in rows)}
        for (method, mode, group), rows in groups.items()
    ])
    failures = [r for r in episodes if r["method"] == "extra_demo_bc" and r["mode"] == "policy"
                and r["stable_placement_success"] == "False" and r["first_support_time_s"]]
    events = []
    for row in failures:
        phase_keys = [("first_support", "touch"), ("first_exit", "exit"), ("first_open", "open")]
        event = {"scenario_seed": row["scenario_seed"], "group": row["group"],
                 "final_mm": float(row["final_error_m"]) * 1000}
        for key, label in phase_keys:
            event[f"{label}_mm"] = float(row[f"{key}_error_m"]) * 1000 if row[f"{key}_error_m"] else None
            event[f"{label}_s_after_touch"] = (float(row[f"{key}_time_s"]) - float(row["first_support_time_s"])
                                               if row[f"{key}_time_s"] else None)
        events.append(event)
    write_csv(folder / "landing_events.csv", events)

    example = max((r for r in failures if r["group"] == "new_failure"), key=lambda row: float(row["final_error_m"]))
    scene = example["scenario_seed"]
    series = {"frozen_bc": [], "extra_demo_bc": []}
    with (folder / "physics_trace.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["scenario_seed"] == scene and row["mode"] == "policy":
                series[row["method"]].append(row)
    fig, panels = plt.subplots(1, 3, figsize=(10.3, 3.5), sharex=True, layout="constrained")
    for (name, trace), color, label in zip(series.items(), ("#617682", "#A55237"), ("Prior to low-friction BC", "Low-friction BC")):
        touchdown = next(float(row["t_s"]) for row in trace if row["supported"] == "True")
        t = [float(row["t_s"]) - touchdown for row in trace]
        for ax, key, scale in zip(panels, ("error_m", "cube_vxy_m_s", "finger_gap_m"), (1000, 100, 1000)):
            ax.plot(t, [float(row[key]) * scale for row in trace], label=label, color=color, lw=1.6)
    for ax, title in zip(panels, ("Goal error (mm)", "Cube XY speed (cm/s)", "Finger opening (mm)")):
        ax.axvline(0, color="#999", lw=0.8)
        ax.set(xlim=(-0.3, 0.95), title=title, xlabel="Time from first support (s)")
        ax.grid(alpha=0.18)
    panels[0].axhline(GOAL_RADIUS * 1000, color="#555", ls="--", lw=1.0)
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False,
               title=f"Seed 2, same scene {scene}")
    fig.savefig(folder / "landing_trace.png", dpi=175)
    plt.close(fig)


def audit_demonstrations(output):
    velocity_index = 0
    velocity_scale = None
    for name, (width, scale) in OBS_FIELDS.items():
        if name == "cube_velocity_m_s":
            velocity_scale = scale
            break
        velocity_index += width
    rows = []
    for seed in range(3):
        for method in VIEWS[1:]:
            folder = output / f"seed_{seed}" / method / "demos"
            metadata = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
            with np.load(folder / "expert.npz") as raw, np.load(folder / "landing.npz") as suffix:
                wanted = np.repeat(raw["successful"], np.diff(raw["offsets"])) & (raw["phase"] != "BEFORE_ARRIVAL")
                if not (metadata["transitions"] == len(raw["actions"]) == BUDGET
                        and int(suffix["offsets"][-1]) == int(wanted.sum())
                        and np.array_equal(suffix["observations"], raw["observations"][wanted])
                        and np.array_equal(suffix["expert_actions"], raw["expert_actions"][wanted])):
                    raise ValueError(f"Demonstration budget or suffix inconsistency in {folder}")
                entries = suffix["observations"][suffix["offsets"][:-1]]
                if not np.all(entries[:, OBS_SIZE + 5] == 1):
                    raise ValueError(f"An expert suffix begins before arrival in {folder}")
                offsets = entries[:, OBS_SIZE:OBS_SIZE + 2] * 0.25
                speed = entries[:, velocity_index:velocity_index + 2] * velocity_scale
                rows.append({"method": method, "seed": seed, "experiences": len(raw["actions"]),
                             "attempted_episodes": len(raw["successful"]), "successful_suffixes": len(entries),
                             "suffix_labels": int(wanted.sum()),
                             "median_arrival_mm": float(np.median(np.linalg.norm(offsets, axis=1)) * 1000),
                             "median_arrival_speed_cm_s": float(np.median(np.linalg.norm(speed, axis=1)) * 100)})
    write_csv(output / "demo_audit.csv", rows)


def comparison(output, fixed=False):
    layouts, cases, records = {}, {}, []
    for seed in range(3):
        initial = set()
        for method in VIEWS:
            directory = output / f"seed_{seed}" / method
            config = {} if method == VIEWS[0] else json.loads((directory / "config.json").read_text(encoding="utf-8"))
            test_name = "test"
            if fixed and config and config["selected_updates"] != CHECKPOINTS[0]:
                test_name = "fixed_250_test"
            rows, report = audit(directory / test_name, TEST, layouts)
            model = Path(report["checkpoint"])
            if digest(model) != report["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint has changed: {model}")
            initial.add(report["initial_sha256"])
            if method != VIEWS[0] and (config["new_demo_steps"] != BUDGET or config["new_online_steps"] != 0):
                raise ValueError(f"Unmatched environment budget: {directory}")
            cases[seed, method] = rows, model
            errors = np.asarray([float(row["goal_distance_m"]) for row in rows])
            selected = CHECKPOINTS[0] if fixed and config else config.get("selected_updates", 0)
            records.append({
                "seed": seed, "method": method, "episodes": len(rows),
                "stable": report["stable_placement_successes"], "drops": report["transport_drops"],
                "strict": report["strict_successes"], "lifted": sum(row["ever_carried"] == "True" for row in rows),
                "median_error_mm": float(np.median(errors) * 1000), "p90_error_mm": float(np.quantile(errors, 0.9) * 1000),
                "selected_updates": selected,
                **{key: config.get(key, 0) for key in ("new_demo_steps", "new_online_steps",
                   "attempted_updates", "supervised_suffix_steps", "successful_suffixes", "training_s", "collection_s")},
            })
        if len(initial) != 1:
            raise ValueError(f"Unmatched initial checkpoint for seed {seed}")
    pairings = []
    for seed in range(3):
        for before, after in ((VIEWS[0], VIEWS[1]), (VIEWS[0], VIEWS[2]), (VIEWS[1], VIEWS[2])):
            pairs = list(zip(cases[seed, before][0], cases[seed, after][0]))
            pairings.append({
                "seed": seed, "before": before, "after": after,
                "stable_gained": sum(a["stable_placement_success"] == "False"
                                     and b["stable_placement_success"] == "True" for a, b in pairs),
                "stable_lost": sum(a["stable_placement_success"] == "True"
                                   and b["stable_placement_success"] == "False" for a, b in pairs),
                "drops_avoided": sum(a["transport_drop"] == "True" and b["transport_drop"] == "False" for a, b in pairs),
                "drops_added": sum(a["transport_drop"] == "False" and b["transport_drop"] == "True" for a, b in pairs),
            })
    prefix = "fixed_250_" if fixed else ""
    write_csv(output / f"{prefix}comparison.csv", records)
    write_csv(output / f"{prefix}paired.csv", pairings)
    return cases, records


def draw_comparison(output, records, fixed=False):
    fig, panels = plt.subplots(1, 2, figsize=(9.8, 4), layout="constrained")
    x = np.arange(3)
    for ax, key, title in zip(panels, ("stable", "drops"), ("Stable placement", "Transport drops")):
        for index, (method, color) in enumerate(zip(VIEWS, ("#617682", "#316D9A", "#208374"))):
            y = [r[key] for r in records if r["method"] == method]
            bars = ax.bar(x + (index - 1) * 0.23, y, 0.22, label=NAMES[method], color=color)
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set(xticks=x, xticklabels=[f"Seed {i}" for i in x], title=title)
        ax.set_ylim(0, 107 if key == "stable" else max(5, 1.25 * max(r[key] for r in records)))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    panels[0].set_ylabel("Episodes per 100 scenes")
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    name = "fixed_250_comparison.png" if fixed else "comparison.png"
    fig.savefig(output / name, dpi=175)
    plt.close(fig)


def videos(output, cases):
    import imageio_ffmpeg

    for seed in (2, 1, 0):
        before, before_model = cases[seed, VIEWS[0]]
        after, after_model = cases[seed, VIEWS[2]]
        candidates = [(a, b) for a, b in zip(before, after)
                      if a["transport_drop"] == "False" and a["transported"] == "True"
                      and a["stable_placement_success"] == "False" and float(a["goal_distance_m"]) > GOAL_RADIUS
                      and b["stable_placement_success"] == "True"]
        if candidates:
            break
    else:
        print("No paired edge placement recovery on the held-out scenes", flush=True)
        return
    failure, success = max(candidates, key=lambda pair: float(pair[0]["goal_distance_m"]))
    folder = output / "videos"
    folder.mkdir(exist_ok=True)
    records = []
    for method, row, model in ((VIEWS[0], failure, before_model), (VIEWS[2], success, after_model)):
        name = folder / f"{method}_seed_{seed}_{row['scenario_seed']}.mp4"
        with BoundaryEvaluation(CANDIDATES[LOW], render_mode="rgb_array") as env:
            record_video(env, clone_policy(load_clone(model)), row, name)
            if (str(env.max_outcome_ticks >= env.hold_ticks) != row["stable_placement_success"]
                    or str(env.transport_drop) != row["transport_drop"]):
                raise ValueError(f"Video changes the recorded outcome: {name}")
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(name))
        if frames != int(row["steps"]) + 1:
            raise ValueError(f"Video frame count differs: {name}")
        records.append({"path": name.as_posix(), "method": method, "seed": seed,
                        "frames": frames, "seconds": seconds, **row})
    write_json(folder / "criteria.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--diagnosis-only", action="store_true")
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    diagnosis(args.out)
    if not args.diagnosis_only:
        audit_demonstrations(args.out)
        cases, records = comparison(args.out, fixed=args.fixed)
        draw_comparison(args.out, records, fixed=args.fixed)
        if args.videos:
            videos(args.out, cases)


if __name__ == "__main__":
    main()
