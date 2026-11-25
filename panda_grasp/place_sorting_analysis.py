"""Audit matched sorting rollouts, summarize success, and render verified scenes."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .experiment import write_csv, write_json
from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_sorting import COLOR_BY_SHAPE, OUTPUT, TEST, SortingScene, load_mixed, route, run_scene

METHODS = (("expert", None), ("mixed", 0), ("mixed", 1), ("mixed", 2))
LAYOUT_FIELDS = ("shape", "order", "expected_color", "target_slot", "pickup_x", "pickup_y",
                 "target_x", "target_y", "initial_yaw_deg")


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(output, source, *, methods=METHODS, scenarios=TEST, checkpoint_identity=None):
    cases, records, objects, failures = {}, [], [], []
    layouts = {}
    for method, seed in methods:
        path = output / "test" / method
        if seed is not None:
            path /= f"seed_{seed}"
        receipt = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if seed is not None:
            identity = load_mixed(seed, source)[1] if checkpoint_identity is None else checkpoint_identity(method, seed)
            if identity != receipt["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint mismatch in {path}")
        scenes = read_rows(path / "scenes.csv")
        items = read_rows(path / "items.csv")
        if ([int(row["scenario_seed"]) for row in scenes] != list(scenarios)
                or [int(row["scenario_seed"]) for row in items] != [n for n in scenarios for _ in range(3)]):
            raise ValueError(f"Missing or reordered scenarios in {path}")
        counts = Counter()
        for scene, start in zip(scenes, range(0, len(items), 3)):
            rows = items[start:start + 3]
            scene_id = int(scene["scenario_seed"])
            plan, names = route(scene_id)
            if [row["shape"] for row in rows] != names or set(row["expected_color"] for row in rows) != set(COLOR_BY_SHAPE.values()):
                raise ValueError(f"Incorrect assignment in scene {scene_id}")
            colors = scene["tray_colors"].split("/")
            for row in rows:
                shape = row["shape"]
                if (colors[plan[shape].slot] != COLOR_BY_SHAPE[shape]
                        or int(row["target_slot"]) != plan[shape].slot
                        or row["expected_color"] != COLOR_BY_SHAPE[shape]
                        or (row["sorted"] == "True") != (row["stable_in_target"] == "True"
                            and row["final_color"] == row["expected_color"])
                        or (row["stable_in_target"] == "True" and float(row["hold_s"]) < 1)):
                    raise ValueError(f"Sorting criteria changed for {shape} in {scene_id}")
            layout = tuple(tuple(row[key] for key in LAYOUT_FIELDS) for row in rows)
            if layouts.setdefault(scene_id, layout) != layout:
                raise ValueError(f"Different physical layout in scene {scene_id}")
            sorted_count = sum(row["sorted"] == "True" for row in rows)
            if int(scene["sorted_items"]) != sorted_count or (scene["all_sorted"] == "True") != (sorted_count == 3):
                raise ValueError(f"Scene outcome does not match item outcomes in {scene_id}")
            counts[sorted_count] += 1
        if (sum(count * n for count, n in counts.items()) != receipt["sorted_items"]
                or counts[3] != receipt["all_sorted"]
                or sum(row["wrong_color"] == "True" for row in items) != receipt["wrong_color_items"]):
            raise ValueError(f"Archived summary differs from trial results in {path}")
        cases[method, seed] = scenes, items
        records.append({key: receipt[key] for key in ("method", "seed", "scenes", "all_sorted", "items",
                        "sorted_items", "wrong_color_items", "homing_failures")} | {
            f"scenes_{i}_sorted": counts[i] for i in range(4)
        })
        for shape in COLOR_BY_SHAPE:
            rows = [row for row in items if row["shape"] == shape]
            objects.append({"method": method, "seed": seed, "shape": shape,
                            "expected_color": COLOR_BY_SHAPE[shape], "attempts": len(rows),
                            "sorted": sum(row["sorted"] == "True" for row in rows),
                            "lifted": sum(row["lifted"] == "True" for row in rows),
                            "transport_drops": sum(row["transport_drop"] == "True" for row in rows),
                            "moved_before_attempt_over_10mm": sum(float(row["pickup_shift_before_attempt_m"]) > 0.01 for row in rows)})
        stages = Counter("displaced_after_placement" if row["stable_in_target"] == "True" else row["failure_stage"]
                         for row in items if row["sorted"] == "False")
        failures.extend({"method": method, "seed": seed, "failure_stage": stage, "items": count}
                        for stage, count in stages.items())
    write_csv(output / "comparison.csv", records)
    write_csv(output / "by_shape.csv", objects)
    if failures:
        write_csv(output / "failure_stages.csv", failures)
    return records, objects, cases


def draw(output, records, objects, styles=(("mixed", "Mixed BC", "#217C71"), ("expert", "Expert", "#6D7888"))):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), layout="constrained")
    names = list(COLOR_BY_SHAPE)
    x = np.arange(len(names) + 1)
    for i, (method, label, color) in enumerate(styles):
        shape_values = [100 * sum(r["sorted"] for r in objects if r["method"] == method and r["shape"] == shape)
                        / sum(r["attempts"] for r in objects if r["method"] == method and r["shape"] == shape)
                        for shape in names]
        scene_rows = [r for r in records if r["method"] == method]
        scene_value = 100 * sum(r["all_sorted"] for r in scene_rows) / sum(r["scenes"] for r in scene_rows)
        offset = (i - (len(styles) - 1) / 2) * 0.24
        bars = axes[0].bar(x + offset, shape_values + [scene_value], 0.23, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)
        outcome = [100 * sum(r[f"scenes_{n}_sorted"] for r in scene_rows) / sum(r["scenes"] for r in scene_rows)
                   for n in range(4)]
        axes[1].bar(np.arange(4) + offset, outcome, 0.23, color=color)
    axes[0].set(xticks=x, xticklabels=["Cube", "Brick", "Cylinder", "All three"], title="Sorting success", ylim=(0, 110))
    axes[1].set(xticks=range(4), xlabel="Objects sorted in a scene", title="Distribution of scene outcomes", ylim=(0, 110))
    axes[0].set_ylabel("Percent")
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(styles), frameon=False)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def record(output, source, cases):
    choices = [(method, seed, row) for (method, seed), (scenes, _) in cases.items() for row in scenes]
    successful = next(((m, s, r) for m, s, r in choices if m == "mixed" and r["all_sorted"] == "True"), None)
    if successful is None:
        successful = next(((m, s, r) for m, s, r in choices if r["all_sorted"] == "True"), None)
    partial = next(((m, s, r) for m, s, r in choices if m == "mixed" and r["sorted_items"] == "2"), None)
    if partial is None:
        partial = next(((m, s, r) for m, s, r in choices if m == "mixed" and r["all_sorted"] == "False"), None)
    record_cases(output, cases, (successful, partial),
                 lambda method, seed: load_mixed(seed, source)[0] if method == "mixed" else None)


def record_cases(output, cases, picks, load, scene_type=lambda method: SortingScene):
    import imageio_ffmpeg

    clips = []
    folder = output / "videos"
    folder.mkdir(exist_ok=True)
    for example in picks:
        if example is None:
            continue
        method, training_seed, row = example
        scene_id = int(row["scenario_seed"])
        actor = load(method, training_seed)
        path = folder / f"{method}_seed_{training_seed}_scene_{scene_id}_{row['sorted_items']}of3.mp4"
        encoder = imageio_ffmpeg.write_frames(str(path), (800, 480), fps=20, codec="libx264", macro_block_size=1)
        try:
            encoder.send(None)
            with scene_type(method)(render_mode="rgb_array") as env:
                replay, items = run_scene(env, scene_id, actor, frame=encoder.send)
            if any(str(value) != row[key] for key, value in replay.items()):
                raise ValueError(f"Rendered scene did not reproduce the recorded result: {path}")
            archived = [r for r in cases[method, training_seed][1] if int(r["scenario_seed"]) == scene_id]
            if any(str(result[key]) != previous[key] for result, previous in zip(items, archived)
                   for key in ("shape", "stable_in_target", "final_color", "sorted", "attempt_steps")):
                raise ValueError(f"Rendered scene differs on an individual item: {path}")
        finally:
            encoder.close()
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        if frames != 1 + replay["homing_steps"] + sum(item["attempt_steps"] for item in items):
            raise ValueError(f"Unexpected video frame count: {path}")
        clips.append({"path": path.as_posix(), "method": method, "seed": training_seed,
                      "frames": frames, "seconds": seconds, **replay, "items": items})
    write_json(folder / "criteria.json", clips)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=OBJECT_RESULTS)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    records, objects, cases = audit(args.out, args.source)
    draw(args.out, records, objects)
    if args.videos:
        record(args.out, args.source, cases)


if __name__ == "__main__":
    main()
