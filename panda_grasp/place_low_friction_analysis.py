"""Audit matched-budget placement tests and retain a verified paired rollout."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
import torch

from .experiment import record_video, rollout, write_csv, write_json
from .place_boundary import BoundaryEvaluation, CANDIDATES
from .place_boundary_analysis import audit
from .place_cli import clone_policy
from .place_low_friction import (
    ACTIVATION, BUDGET, LOW, METHODS, OUTPUT, SAC_SETTINGS, TEST, VALIDATION, WIDTHS,
    initialize_actor, locked_checkpoint, placement_sac_policy,
)
from .place_robustness import load_clone

LABELS = dict(zip(METHODS, ("Frozen BC", "Extra-demo BC", "BC-regularized SAC")))


def collect(output):
    comparison, pairs = [], []
    cases = {}
    layouts = {}
    for seed in range(3):
        initial_hashes = set()
        for method in METHODS:
            root = output / f"seed_{seed}" / method
            rows, summary = audit(root / "test", TEST, layouts)
            if method == "bc_regularized_sac" and summary.get("action_path") != "direct_tanh":
                raise ValueError(f"Expected direct deterministic actor actions in {root}")
            cost = json.loads((root / "cost.json").read_text(encoding="utf-8"))
            steps = cost["new_demo_steps"] + cost["new_online_steps"]
            if steps != (0 if method == "frozen_bc" else BUDGET):
                raise ValueError(f"Unmatched interaction budget in {root}")
            path = Path(cost["selected_checkpoint"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != cost["checkpoint_sha256"]:
                raise ValueError(f"Evaluated checkpoint has changed: {path}")
            initial_hashes.add(cost["initial_sha256"])
            comparison.append({"method": method, "seed": seed, "episodes": summary["episodes"],
                               "stable": summary["stable_placement_successes"],
                               "strict": summary["strict_successes"],
                               "lifted": sum(row["ever_carried"] == "True" for row in rows),
                               "drops": summary["transport_drops"],
                               "new_demo_steps": cost["new_demo_steps"],
                               "new_online_steps": cost["new_online_steps"],
                               "training_s": cost["training_s"],
                               "new_demo_collection_s": cost["new_demo_collection_s"]})
            cases[seed, method] = rows, path
        if len(initial_hashes) != 1:
            raise ValueError(f"Methods do not share the same initial checkpoint for seed {seed}")
        for reference, other in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])):
            paired = list(zip(cases[seed, reference][0], cases[seed, other][0]))
            pairs.append({
                "seed": seed, "reference": reference, "other": other,
                "stable_gained": sum(a["stable_placement_success"] == "False"
                                     and b["stable_placement_success"] == "True" for a, b in paired),
                "stable_lost": sum(a["stable_placement_success"] == "True"
                                   and b["stable_placement_success"] == "False" for a, b in paired),
                "drops_avoided": sum(a["transport_drop"] == "True"
                                    and b["transport_drop"] == "False" for a, b in paired),
                "drops_added": sum(a["transport_drop"] == "False"
                                   and b["transport_drop"] == "True" for a, b in paired),
            })
    write_csv(output / "comparison.csv", comparison)
    write_csv(output / "paired.csv", pairs)
    return comparison, cases


def plot(output, comparison):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), layout="constrained")
    x = np.arange(3)
    for ax, key, title in zip(axes, ("stable", "drops"), ("Stable placement", "Transport drops")):
        for index, (method, color) in enumerate(zip(METHODS, ("#627583", "#167D71", "#8C6A38"))):
            y = [row[key] for row in comparison if row["method"] == method]
            bars = ax.bar(x + (index - 1) * 0.23, y, 0.22, label=LABELS[method], color=color)
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set(xticks=x, xticklabels=[f"Seed {seed}" for seed in x], title=title)
        ax.set_ylim(0, 108 if key == "stable" else max(10, max(row[key] for row in comparison) * 1.2))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Episodes per 100 scenes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def videos(output, comparison, cases):
    import imageio_ffmpeg

    folder = output / "videos"
    folder.mkdir(exist_ok=True)
    ranked = sorted(METHODS[1:], key=lambda method: (
        -sum(r["stable"] for r in comparison if r["method"] == method),
        sum(r["drops"] for r in comparison if r["method"] == method),
    ))
    matches = []
    for method in ranked:
        for seed in (1, 0, 2):
            frozen, model = cases[seed, METHODS[0]]
            adapted, trained = cases[seed, method]
            for before, after in zip(frozen, adapted):
                if before["transport_drop"] == "True" and after["stable_placement_success"] == "True":
                    matches.append((seed, method, before, after, model, trained))
    if not matches:
        print("No paired baseline transport drop and adapted stable placement", flush=True)
        return
    preferred = matches[0][:2]
    seed, method, before, after, model, trained = max(
        (item for item in matches if item[:2] == preferred), key=lambda item: float(item[2]["goal_distance_m"]),
    )
    clips = []
    for label, method, row, path in (("transport_drop", METHODS[0], before, model),
                                      ("stable_recovery", method, after, trained)):
        if method == "bc_regularized_sac":
            policy = placement_sac_policy(SAC.load(path, device="cpu"))
        else:
            policy = clone_policy(load_clone(path))
        video = folder / f"{label}_seed_{seed}_{row['scenario_seed']}.mp4"
        with BoundaryEvaluation(CANDIDATES[LOW], render_mode="rgb_array") as env:
            record_video(env, policy, row, video)
            if (str(env.max_outcome_ticks >= env.hold_ticks) != row["stable_placement_success"]
                    or str(env.transport_drop) != row["transport_drop"]):
                raise ValueError(f"Video does not reproduce the result: {video}")
        frames, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
        if frames != int(row["steps"]) + 1:
            raise ValueError(f"Unexpected video length: {video}")
        clips.append({"method": method, "seed": seed, "path": video.as_posix(),
                      "frames": frames, "duration_s": duration, **row})
    write_json(folder / "criteria.json", clips)


def diagnostic_events(output):
    diagnosis = output / "diagnosis"
    with (diagnosis / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        episodes = [row for row in csv.DictReader(stream)
                    if row["condition"] == LOW and row["seed"] == "1"]
    with (diagnosis / "physics_trace.csv").open(encoding="utf-8", newline="") as stream:
        traces = {}
        for row in csv.DictReader(stream):
            if row["condition"] == LOW and row["seed"] == "1":
                traces.setdefault((row["scenario_seed"], row["mode"]), []).append(row)
    dropped = [r for r in episodes if r["mode"] == "policy" and r["group"] == "original_drop"]
    recovered = {r["scenario_seed"] for r in episodes if r["mode"] == "full_close"
                 and r["stable_placement_success"] == "True"}
    rows = []
    for episode in dropped:
        scene = episode["scenario_seed"]
        trace = traces[scene, "policy"]
        indices = [i for i, point in enumerate(trace) if float(point["raw_grip"]) > -0.5]
        start = indices[0] if indices else None
        prior = trace[max(0, start - 1)] if start is not None else None
        relaxed = trace[start] if start is not None else None
        last_grip = next(row for row in reversed(trace) if row["gripping"] == "True" and row["drop"] == "False")
        rows.append({
            "scenario_seed": scene, "lift_step": trace[0]["step"],
            "last_gripped_step_before_drop": last_grip["step"],
            "drop_step": trace[-1]["step"],
            "first_relax_step": relaxed["step"] if relaxed else None,
            "first_relax_action": relaxed["raw_grip"] if relaxed else None,
            "slip_before_relax_mm": 1000 * float(prior["slip_z_m"]) if prior else None,
            "last_gripped_slip_mm": 1000 * float(last_grip["slip_z_m"]),
            "last_gripped_min_normal_n": min(float(last_grip["left_n"]), float(last_grip["right_n"])),
            "full_close_recovered_stable": scene in recovered,
        })
    write_csv(diagnosis / "slip_events.csv", rows)

    example = max((r for r in dropped if r["scenario_seed"] in recovered),
                  key=lambda row: float(row["goal_distance_m"]))
    scene = example["scenario_seed"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.3), sharex=True, layout="constrained")
    for mode, color, label in (("policy", "#AA512A", "Policy"), ("full_close", "#167D71", "Full-close probe")):
        samples = traces[scene, mode]
        t = np.array([float(row["t_s"]) for row in samples])
        y = [
            [1000 * float(row["height_m"]) for row in samples],
            [1000 * float(row["slip_z_m"]) for row in samples],
            [min(float(row["left_n"]), float(row["right_n"])) for row in samples],
            [float(row["applied_grip"]) for row in samples],
        ]
        for ax, series in zip(axes.flat, y):
            ax.plot(t - t[0], series, color=color, label=label, lw=1.5)
    for ax, label in zip(axes.flat, ("Cube lift (mm)", "Vertical slip (mm)", "Weaker finger load (N)", "Gripper action")):
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Time from 5 cm lift (s)")
    axes[1, 1].set_xlabel("Time from 5 cm lift (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False,
               title=f"Seed 1, scene {scene}")
    fig.savefig(diagnosis / "transport_trace.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def initial_actor_precision(output, results):
    """Measure rollout sensitivity to the SAC deterministic prediction path before updates."""
    rows = []
    for seed in range(3):
        source, _ = locked_checkpoint(results, seed)
        clone = load_clone(source)
        with BoundaryEvaluation(CANDIDATES[LOW]) as env:
            model = SAC("MlpPolicy", env, seed=seed + 208200, device="cpu", **SAC_SETTINGS,
                        policy_kwargs={"net_arch": WIDTHS, "activation_fn": ACTIVATION})
            initialize_actor(model.actor, clone)

            raw = placement_sac_policy(model)(env, None)

            def predict(obs):
                return model.predict(obs, deterministic=True)[0]

            for scene in VALIDATION:
                delta = [0.0, 0.0]

                def audit_action(obs):
                    reference = clone.action(obs)
                    for index, other in enumerate((raw(obs), predict(obs))):
                        delta[index] = max(delta[index], float(np.max(np.abs(reference - other))))
                    return reference

                baseline = rollout(env, lambda e, s: audit_action, scene)
                actor = rollout(env, lambda e, s: raw, scene)
                wrapped = rollout(env, lambda e, s: predict, scene)
                rows.append({"seed": seed, "scenario": scene, "max_actor_raw_diff_on_bc_states": delta[0],
                             "max_predict_diff_on_bc_states": delta[1],
                             "bc_stable": baseline["stable_placement_success"],
                             "actor_raw_stable": actor["stable_placement_success"],
                             "predict_stable": wrapped["stable_placement_success"]})
    write_csv(output / "initial_actor_precision.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--actor-precision", action="store_true")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.actor_precision:
        initial_actor_precision(args.out, args.results)
        return
    diagnostic_events(args.out)
    if args.trace_only:
        return
    comparison, cases = collect(args.out)
    plot(args.out, comparison)
    if args.videos:
        videos(args.out, comparison, cases)


if __name__ == "__main__":
    main()
