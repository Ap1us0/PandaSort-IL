"""Audit matched sorting results and retain paired evidence for the phase experiment."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .experiment import write_csv
from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_sorting_analysis import audit, read_rows, record_cases
from .place_sorting_bc import OUTPUT as SORTING_RESULTS
from .place_sorting_contact import OUTPUT as CONTACT_RESULTS
from .place_sorting_contact_ablation import OUTPUT as PRIOR, TEST as PRIOR_TEST, scene_type, training_sources
from .place_sorting_contact_ablation_analysis import focused
from .place_sorting_contact_phase import METHODS, OUTPUT, SWITCHED, TEST, load

LABELS = ("Contact\nswitches", "Force only", "Slip only", "Phase-dependent")
COLORS = ("#647586", "#4089AA", "#886BB1", "#218474")


def audit_budget(output, *, methods=METHODS, archives=None, load_actor=load):
    if archives is None:
        archives = {"force": (PRIOR, PRIOR_TEST), "slip": (PRIOR, PRIOR_TEST), SWITCHED: (output, TEST)}
    rows = []
    for seed in range(3):
        _, reference = training_sources(seed, CONTACT_RESULTS, SORTING_RESULTS)
        datasets = reference["datasets"]
        hashes = [row["augmented_sha256"] for row in datasets]
        for method in methods:
            if method == "binary":
                identity = reference["binary_checkpoint_sha256"]
            else:
                root, expected_scenarios = archives[method]
                path = root / "training" / f"seed_{seed}" / method / "config.json"
                receipt = json.loads(path.read_text(encoding="utf-8"))
                expected = {
                    "seed": seed, "method": method, "optimizer_updates": reference["optimizer_updates_each"],
                    "learning_rate": reference["learning_rate"], "starting_sha256": reference["starting_sha256"],
                    "demonstration_sha256": hashes, "new_online_training_steps": 0, "new_demonstration_scenes": 0,
                    "test_scenarios": list(expected_scenarios),
                }
                if any(receipt[key] != value for key, value in expected.items()):
                    raise ValueError(f"Different data, initialization, or budget in {path}")
                identity = receipt["checkpoint_sha256"]
            if identity != load_actor(method, seed, output)[1]:
                raise ValueError(f"Checkpoint identity mismatch for {method}, seed {seed}")
            rows.append({
                "seed": seed, "method": method, "updates": reference["optimizer_updates_each"],
                "learning_rate": reference["learning_rate"], "demonstration_sources": len(datasets),
                "demonstration_transitions": sum(row["transitions"] for row in datasets),
                "starting_sha256": reference["starting_sha256"], "checkpoint_sha256": identity,
            })
    write_csv(output / "budget.csv", rows)


def paired_comparison(output, cases, *, candidate=SWITCHED, controls=METHODS[:-1]):
    rows = []
    for seed in range(3):
        new_scenes, new_items = cases[candidate, seed]
        new_cubes = [r for r in new_items if r["shape"] == "cube"]
        for method in controls:
            old_scenes, old_items = cases[method, seed]
            old_cubes = [r for r in old_items if r["shape"] == "cube"]
            scene = list(zip(old_scenes, new_scenes, strict=True))
            cube = list(zip(old_cubes, new_cubes, strict=True))
            if any(a["scenario_seed"] != b["scenario_seed"] for a, b in scene + cube):
                raise ValueError(f"Mismatched test scenes for {method}, seed {seed}")
            avoided = [(a, b) for a, b in cube if a["transport_drop"] == "True" and b["transport_drop"] == "False"]
            added = [(a, b) for a, b in cube if a["transport_drop"] == "False" and b["transport_drop"] == "True"]
            rows.append({
                "reference": method, "seed": seed,
                "cube_drop_avoided": len(avoided), "cube_drop_added": len(added),
                "avoided_and_cube_sorted": sum(b["sorted"] == "True" for _, b in avoided),
                "avoided_but_no_lift": sum(b["lifted"] == "False" for _, b in avoided),
                "added_from_previously_unlifted": sum(a["lifted"] == "False" for a, _ in added),
                "all_three_gains": sum(a["all_sorted"] == "False" and b["all_sorted"] == "True" for a, b in scene),
                "all_three_losses": sum(a["all_sorted"] == "True" and b["all_sorted"] == "False" for a, b in scene),
            })
    write_csv(output / "paired.csv", rows)


def plot(output, rows, *, labels=LABELS, colors=COLORS):
    fig, axes = plt.subplots(1, 2, figsize=(9.2 if len(labels) == 4 else 10.6, 4), layout="constrained")
    x = np.arange(len(labels))
    for ax, title, values in (
        (axes[0], "Cube drops per qualified lift", [100 * r["cube_transport_drops"] / r["cube_lifted"] for r in rows]),
        (axes[1], "All three objects sorted", [100 * r["all_sorted"] / r["scenes"] for r in rows]),
    ):
        bars = ax.bar(x, values, width=0.6, color=colors)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, max(values) * 1.25), title=title, ylabel="Percent")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def record_diagnosis(output):
    methods = tuple((method, seed) for method in ("binary", "slip") for seed in (1, 2))
    folder = output / "diagnosis"
    archived = {}
    examples = []
    for method, seed in methods:
        path = PRIOR / "test" / method / f"seed_{seed}"
        receipt = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if receipt["checkpoint_sha256"] != load(method, seed, output)[1]:
            raise ValueError(f"Unexpected diagnosis checkpoint at {path}")
        archived[method, seed] = read_rows(path / "scenes.csv"), read_rows(path / "items.csv")
    for seed in (1, 2):
        base_scenes, base_items = archived["binary", seed]
        slip_scenes, slip_items = archived["slip", seed]
        base_cubes, slip_cubes = ({r["scenario_seed"]: r for r in items if r["shape"] == "cube"}
                                  for items in (base_items, slip_items))
        for first, second in zip(base_scenes, slip_scenes, strict=True):
            scene = first["scenario_seed"]
            if base_cubes[scene]["sorted"] == "True" and slip_cubes[scene]["transport_drop"] == "True":
                examples.append((seed, first, second))
    if examples:
        seed, first, second = max(examples, key=lambda case: (
            int(case[1]["sorted_items"]) - int(case[2]["sorted_items"]),
            case[0],
        ))
        folder.mkdir(parents=True, exist_ok=True)
        record_cases(folder, archived, (("binary", seed, first), ("slip", seed, second)),
                     lambda method, seed: load(method, seed, output)[0], scene_type=scene_type)


def record_new(output, cases, *, candidate=SWITCHED, load_actor=load):
    successes, failures = [], []
    for seed in range(3):
        scenes, items = cases[candidate, seed]
        cubes = {r["scenario_seed"]: r for r in items if r["shape"] == "cube"}
        successes.extend((candidate, seed, row) for row in scenes if row["all_sorted"] == "True")
        failures.extend((candidate, seed, row) for row in scenes if
                        cubes[row["scenario_seed"]]["transport_drop"] == "True" and row["all_sorted"] == "False")
    picks = (
        min(successes, key=lambda case: int(case[2]["homing_steps"]), default=None),
        max(failures, key=lambda case: int(case[2]["sorted_items"]), default=None),
    )
    if any(picks):
        record_cases(output, cases, picks, lambda method, seed: load_actor(method, seed, output)[0], scene_type=scene_type)


def summarize_diagnosis(output):
    root = output / "diagnosis"
    rows = []
    for seed in (1, 2):
        folder = root / f"seed_{seed}"
        events = read_rows(folder / "events.csv") + read_rows(folder / "prelift" / "events.csv")
        outcomes = {(row["scenario_seed"], row["mode"]): row for row in events}
        physics, actions = defaultdict(list), defaultdict(list)
        for filename, target in (("physics_trace.csv", physics), ("policy_trace.csv", actions)):
            for row in read_rows(folder / filename):
                if row["mode"] == "slip" and row["added_slip_drop"] == "True":
                    target[row["scenario_seed"]].append(row)
        for scene, trace in physics.items():
            slip = outcomes[scene, "slip"]
            last_contact = max(i for i, row in enumerate(trace) if row["any_finger_contact"] == "True")
            detach = trace[last_contact + 1]
            lift_t, loss_t = (float(slip[key]) for key in ("lift_t_s", "first_loss_t_s"))
            cmd_t = float(actions[scene][0]["t_s"])
            rows.append({
                "seed": seed, "scenario_seed": scene, "binary_lifted": slip["binary_lifted"],
                "first_loss_after_lift_ms": (loss_t - lift_t) * 1000,
                "first_loss_before_transport_command": loss_t <= cmd_t,
                "final_detach_after_lift_ms": (float(detach["t_s"]) - lift_t) * 1000,
                "gripper_action_at_final_detach": detach["applied_grip"],
                "goal_distance_at_final_detach_m": detach["goal_distance_m"],
                "cube_height_at_final_detach_m": detach["height_m"],
                "tool_relative_shift_along_z_m": detach["slip_z_m"],
                "other_object_contacts_before_final_detach": "/".join(sorted({
                    obstacle for row in trace[:last_contact + 1] for obstacle in row["obstacle"].split("/")
                    if obstacle in ("sort_brick", "sort_cylinder")
                })),
                **{f"{mode}_{outcome}": outcomes[scene, mode][outcome]
                   for mode in ("full_close", "half_xy", "prelift_binary", "prelift_no_slip")
                   for outcome in ("lifted", "transported", "transport_drop", "sorted")},
            })
    write_csv(root / "loss_mechanisms.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--diagnosis-video", action="store_true")
    parser.add_argument("--diagnosis-summary", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.diagnosis_video:
        record_diagnosis(args.out)
        return
    if args.diagnosis_summary:
        summarize_diagnosis(args.out)
        return
    audit_budget(args.out)
    methods = tuple((method, seed) for method in METHODS for seed in range(3))
    records, objects, cases = audit(args.out, OBJECT_RESULTS, methods=methods, scenarios=TEST,
                                   checkpoint_identity=lambda method, seed: load(method, seed, args.out)[1])
    _, results = focused(args.out, records, objects, methods=METHODS)
    paired_comparison(args.out, cases)
    plot(args.out, results)
    if args.videos:
        record_new(args.out, cases)


if __name__ == "__main__":
    main()
