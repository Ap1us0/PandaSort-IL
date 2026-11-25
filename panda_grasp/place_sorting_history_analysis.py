"""Summarize 42.2% -> 82.2% sorting progression, plot comparison, run 1-retry eval, and render verified videos."""

import argparse
from collections import Counter, deque
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .experiment import write_csv, write_json
from .place_objects import SHAPES
from .place_objects_train import sha
from .place_sorting import COLOR_BY_SHAPE, TRAYS
from .place_sorting_aligned import AlignedContactSortingScene, OUTPUT as ALIGNED_OUTPUT
from .place_sorting_contact_schedule import OUTPUT as SCHEDULE_OUTPUT, TEST
from .place_sorting_dagger import OUTPUT as DAGGER_OUTPUT
from .place_sorting_history import HISTORY, HistoryAlignedPolicy, OUTPUT as HISTORY_OUTPUT

SOURCES = (
    ("force_in_transport", "Baseline\n(force_in_trans)", SCHEDULE_OUTPUT / "test" / "force_in_transport", True),
    ("aligned_r1", "Aligned Expert\n+ 256x256 BC", ALIGNED_OUTPUT / "test" / "round1", True),
    ("dagger", "+ Prefix\nDAgger", DAGGER_OUTPUT / "test", True),
    ("history3_ema", "+ 3-Frame\nHistory + EMA", HISTORY_OUTPUT / "test", True),
    ("aligned_expert", "Aligned\nExpert", ALIGNED_OUTPUT / "test" / "expert", False),
)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(output):
    by_seed, by_shape, overall, failures = [], [], [], []
    for method, _, root, seeded in SOURCES:
        seeds = (0, 1, 2) if seeded else (None,)
        all_scenes, all_items = [], []
        for seed in seeds:
            folder = root / f"seed_{seed}" if seed is not None else root
            scenes = read_rows(folder / "scenes.csv")
            items = read_rows(folder / "items.csv")
            all_scenes.extend(scenes)
            all_items.extend(items)
            cubes = [r for r in items if r["shape"] == "cube"]
            by_seed.append({
                "method": method, "seed": "" if seed is None else seed,
                "scenes": len(scenes),
                "all_sorted": sum(r["all_sorted"] == "True" for r in scenes),
                "sorted_items": sum(r["sorted"] == "True" for r in items),
                "cube_sorted": sum(r["sorted"] == "True" for r in cubes),
                "cube_lifted": sum(r["lifted"] == "True" for r in cubes),
                "cube_transport_drops": sum(r["transport_drop"] == "True" for r in cubes),
                "brick_sorted": sum(r["sorted"] == "True" for r in items if r["shape"] == "brick"),
                "cylinder_sorted": sum(r["sorted"] == "True" for r in items if r["shape"] == "cylinder"),
            })
            for shape in SHAPES:
                sub = [r for r in items if r["shape"] == shape]
                by_shape.append({
                    "method": method, "seed": "" if seed is None else seed, "shape": shape,
                    "attempts": len(sub),
                    "sorted": sum(r["sorted"] == "True" for r in sub),
                    "lifted": sum(r["lifted"] == "True" for r in sub),
                    "transport_drops": sum(r["transport_drop"] == "True" for r in sub),
                })
            stages = Counter(r["failure_stage"] for r in items if r["sorted"] != "True")
            for stage, cnt in sorted(stages.items()):
                failures.append({"method": method, "seed": "" if seed is None else seed, "failure_stage": stage, "items": cnt})

        cubes_all = [r for r in all_items if r["shape"] == "cube"]
        overall.append({
            "method": method,
            "scenes": len(all_scenes),
            "all_sorted": sum(r["all_sorted"] == "True" for r in all_scenes),
            "all_sorted_pct": round(100.0 * sum(r["all_sorted"] == "True" for r in all_scenes) / len(all_scenes), 1),
            "items": len(all_items),
            "sorted_items": sum(r["sorted"] == "True" for r in all_items),
            "sorted_items_pct": round(100.0 * sum(r["sorted"] == "True" for r in all_items) / len(all_items), 1),
            "cube_sorted": sum(r["sorted"] == "True" for r in cubes_all),
            "cube_lifted": sum(r["lifted"] == "True" for r in cubes_all),
            "cube_transport_drops": sum(r["transport_drop"] == "True" for r in cubes_all),
            "brick_sorted": sum(r["sorted"] == "True" for r in all_items if r["shape"] == "brick"),
            "cylinder_sorted": sum(r["sorted"] == "True" for r in all_items if r["shape"] == "cylinder"),
        })

    write_csv(output / "overall.csv", overall)
    write_csv(output / "by_seed.csv", by_seed)
    write_csv(output / "by_shape.csv", by_shape)
    write_csv(output / "failure_stages.csv", failures)
    return overall, by_seed, by_shape


def plot_comparison(output, overall, by_seed):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), layout="constrained")
    colors = ("#6D7888", "#2B7A78", "#3B6FB6", "#1D9A6C", "#B07D3B")
    x = np.arange(len(SOURCES))
    labels = [label for _, label, _, _ in SOURCES]

    # Left: Full-table 3/3 success rate (%) with per-seed dots
    ax = axes[0]
    rates = [row["all_sorted_pct"] for row in overall]
    bars = ax.bar(x, rates, color=colors, width=0.58, alpha=0.9)
    for idx, (method, _, _, seeded) in enumerate(SOURCES):
        if seeded:
            pts = [100.0 * r["all_sorted"] / r["scenes"] for r in by_seed if r["method"] == method]
            offsets = np.linspace(-0.12, 0.12, len(pts))
            ax.scatter(np.full(len(pts), idx) + offsets, pts, color="#111111", s=26, zorder=5)
        ax.text(idx, rates[idx] + 2.2, f"{rates[idx]:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 102)
    ax.set_ylabel("All-3 Sorted Rate (%)")
    ax.set_title("Full-Table Sorting Accuracy (30 Layouts x 3 Seeds)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Right: Per-shape accuracy & cube transport drop rate across learned stages
    ax2 = axes[1]
    learned = overall[:4]
    lx = np.arange(len(learned))
    l_labels = [label for _, label, _, _ in SOURCES[:4]]
    w = 0.22
    cube_r = [100.0 * r["cube_sorted"] / r["scenes"] for r in learned]
    brick_r = [100.0 * r["brick_sorted"] / r["scenes"] for r in learned]
    cyl_r = [100.0 * r["cylinder_sorted"] / r["scenes"] for r in learned]
    drop_r = [100.0 * r["cube_transport_drops"] / max(1, r["cube_lifted"]) for r in learned]
    ax2.bar(lx - w, cube_r, width=w, color="#D94848", label="Cube sorted (%)")
    ax2.bar(lx, brick_r, width=w, color="#3B6FB6", label="Brick sorted (%)")
    ax2.bar(lx + w, cyl_r, width=w, color="#2E9E5B", label="Cylinder sorted (%)")
    ax2.plot(lx, drop_r, color="#111111", marker="o", linewidth=2, linestyle="--", label="Cube drop / lift (%)")
    ax2.set_xticks(lx, l_labels, fontsize=8.5)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Per-Object Accuracy & Cube Transport Drop Rate")
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def run_history_scene_with_frame(env, seed, actor, frame=None, max_retries=0):
    obs, _ = env.reset(seed=seed)
    if frame is not None:
        frame(env.render())
    items = []
    homing_steps = 0
    retries_used = 0
    for index, name in enumerate(env.order):
        returned_home = True
        if index:
            steps, returned_home = env.return_home(frame)
            homing_steps += steps
            obs = env.select(name)
        total_steps = 0
        attempts = 0
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            if attempt > 0:
                retries_used += 1
                steps, returned_home = env.return_home(frame)
                homing_steps += steps
                obs = env.select(name)
            buf = deque([obs.copy() for _ in range(HISTORY)], maxlen=HISTORY)
            while True:
                action = actor.action(np.concatenate(list(buf), axis=0))
                obs, _, terminated, truncated, info = env.step(action)
                total_steps += 1
                buf.append(obs.copy())
                if frame is not None:
                    frame(env.render())
                if terminated or truncated:
                    break
            ok_now = bool(info["stable_placement_success"]) and env.final_tray(name) == env.plan[name].slot
            if ok_now or float(env.cube_xyz[2] - env.rest_z) < -0.045:
                break
        assignment = env.plan[name]
        items.append({
            "scenario_seed": seed, "order": index + 1, "shape": name, "expected_color": COLOR_BY_SHAPE[name],
            "target_slot": assignment.slot, "attempts": attempts,
            "attempt_steps": total_steps, "stable_in_target": info["stable_placement_success"],
            "hold_s": info["outcome_stable_s"], "failure_stage": info["failure_stage"],
            "lifted": info["ever_carried"], "transported": info["transported"], "transport_drop": info["transport_drop"],
            "terminal_error_m": info["goal_distance_m"],
        })
    slot_colors = {assignment.slot: COLOR_BY_SHAPE[name] for name, assignment in env.plan.items()}
    for row in items:
        slot = env.final_tray(row["shape"])
        row["final_color"] = slot_colors.get(slot, "outside")
        row["wrong_color"] = slot is not None and row["final_color"] != row["expected_color"]
        row["sorted"] = row["stable_in_target"] and row["final_color"] == row["expected_color"]
    scene = {
        "scenario_seed": seed, "order": "/".join(env.order),
        "tray_colors": "/".join(slot_colors[i] for i in range(len(TRAYS))),
        "sorted_items": sum(row["sorted"] for row in items), "all_sorted": all(row["sorted"] for row in items),
        "wrong_color_items": sum(row["wrong_color"] for row in items), "homing_steps": homing_steps,
        "retries_used": retries_used,
    }
    return scene, items


def evaluate_retry(output):
    summary_rows = []
    for seed in (0, 1, 2):
        weights = output / "training" / f"seed_{seed}" / "model.pt"
        actor = HistoryAlignedPolicy()
        actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
        actor.eval()
        scenes, items = [], []
        with AlignedContactSortingScene() as env:
            for scenario in TEST:
                sc, rows = run_history_scene_with_frame(env, scenario, actor, max_retries=1)
                scenes.append(sc)
                items.extend(rows)
        folder = output / "retry" / f"seed_{seed}"
        write_csv(folder / "scenes.csv", scenes)
        write_csv(folder / "items.csv", items)
        summary = {
            "seed": seed, "checkpoint_sha256": sha(weights), "scenes": len(scenes),
            "all_sorted": sum(r["all_sorted"] for r in scenes),
            "sorted_items": sum(r["sorted"] for r in items),
            "retries_used": sum(r["retries_used"] for r in scenes),
            "cube_sorted": sum(r["shape"] == "cube" and r["sorted"] for r in items),
            "brick_sorted": sum(r["shape"] == "brick" and r["sorted"] for r in items),
            "cylinder_sorted": sum(r["shape"] == "cylinder" and r["sorted"] for r in items),
        }
        write_json(folder / "summary.json", summary)
        summary_rows.append(summary)
        print("Retry:", summary, flush=True)
    write_csv(output / "retry_summary.csv", summary_rows)
    tot_all = sum(r["all_sorted"] for r in summary_rows)
    tot_items = sum(r["sorted_items"] for r in summary_rows)
    print(f"Retry total: {tot_all}/90 ({100.0 * tot_all / 90:.1f}%), items {tot_items}/270 ({100.0 * tot_items / 270:.1f}%)",
          flush=True)


def render_videos(output):
    import imageio_ffmpeg

    # Pick 1 rescued 3/3 success (where baseline failed on seed 0) and 1 remaining failure on seed 0
    base_scenes = {int(r["scenario_seed"]): r for r in read_rows(SCHEDULE_OUTPUT / "test/force_in_transport/seed_0/scenes.csv")}
    hist_scenes = read_rows(output / "test/seed_0/scenes.csv")
    hist_items = read_rows(output / "test/seed_0/items.csv")

    rescued = next((r for r in hist_scenes if r["all_sorted"] == "True" and base_scenes[int(r["scenario_seed"])]["all_sorted"] == "False"), hist_scenes[0])
    failed = next((r for r in hist_scenes if r["all_sorted"] == "False"), None)

    weights = output / "training/seed_0/model.pt"
    actor = HistoryAlignedPolicy()
    actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    actor.eval()

    folder = output / "videos"
    folder.mkdir(exist_ok=True)
    clips = []
    for tag, row in (("rescued_success", rescued), ("remaining_failure", failed)):
        if row is None:
            continue
        scene_id = int(row["scenario_seed"])
        path = folder / f"history3_ema_seed_0_scene_{scene_id}_{row['sorted_items']}of3.mp4"
        encoder = imageio_ffmpeg.write_frames(str(path), (800, 480), fps=20, codec="libx264", macro_block_size=1)
        try:
            encoder.send(None)
            with AlignedContactSortingScene(render_mode="rgb_array") as env:
                replay, items = run_history_scene_with_frame(env, scene_id, actor, frame=encoder.send, max_retries=0)
            if int(replay["sorted_items"]) != int(row["sorted_items"]) or str(replay["all_sorted"]) != row["all_sorted"]:
                raise ValueError(f"Replay mismatch for {path}")
            archived = [r for r in hist_items if int(r["scenario_seed"]) == scene_id]
            for got, ref in zip(items, archived, strict=True):
                if str(got["sorted"]) != ref["sorted"] or got["attempt_steps"] != int(ref["attempt_steps"]):
                    raise ValueError(f"Item replay mismatch for {path}")
        finally:
            encoder.close()
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        clips.append({"tag": tag, "path": path.as_posix(), "seed": 0, "frames": frames, "seconds": seconds, **replay, "items": items})
        print(f"Rendered {path} ({frames} frames, {seconds:.1f}s)", flush=True)
    write_json(folder / "criteria.json", clips)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HISTORY_OUTPUT)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    overall, by_seed, _ = summarize(args.out)
    plot_comparison(args.out, overall, by_seed)
    if args.retry:
        evaluate_retry(args.out)
    if args.videos:
        render_videos(args.out)


if __name__ == "__main__":
    main()
