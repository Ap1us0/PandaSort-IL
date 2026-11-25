"""Summarize DR+DAgger (6 benchmarks) and Residual SAC OOD adaptation, plot charts, and render verified videos."""

import argparse
from collections import deque
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .experiment import write_csv, write_json
from .place_objects import SHAPES
from .place_sorting import COLOR_BY_SHAPE, TRAYS
from .place_sorting_aligned import AlignedContactSortingScene
from .place_sorting_contact_schedule import OUTPUT as SCHEDULE_OUTPUT
from .place_sorting_dr import DRHistoryPolicy, OUTPUT as DR_OUTPUT
from .place_sorting_history import HISTORY, OUTPUT as HISTORY_OUTPUT
from .place_sorting_ood import CONDITIONS, OODSortingScene, OUTPUT as OOD_OUTPUT
from .place_sorting_residual_sac import OUTPUT as RL_OUTPUT

BENCHMARKS = (
    ("nominal", "Nominal\n(70% fric, 150g)"),
    ("size_108", "Size 1.08x\n(wider)"),
    ("jitter_12mm", "Jitter ±12mm\n(4.8x spawn)"),
    ("size_90", "Size 0.90x\n(smaller)"),
    ("friction_60", "Friction 60%\n(slippery)"),
    ("mass_175", "Mass 175g\n(heavier)"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_history_nominal():
    scenes, items = [], []
    per_seed = []
    for s in (0, 1, 2):
        sc = read_rows(HISTORY_OUTPUT / f"test/seed_{s}/scenes.csv")
        it = read_rows(HISTORY_OUTPUT / f"test/seed_{s}/items.csv")
        scenes.extend(sc)
        items.extend(it)
        per_seed.append(sum(r["all_sorted"] == "True" for r in sc))
    cubes = [r for r in items if r["shape"] == "cube"]
    return {
        "condition": "nominal",
        "scenes": len(scenes),
        "all_sorted": sum(r["all_sorted"] == "True" for r in scenes),
        "all_sorted_pct": round(100.0 * sum(r["all_sorted"] == "True" for r in scenes) / len(scenes), 1),
        "per_seed": "/".join(str(x) for x in per_seed),
        "sorted_items": sum(r["sorted"] == "True" for r in items),
        "sorted_items_pct": round(100.0 * sum(r["sorted"] == "True" for r in items) / len(items), 1),
        "cube_sorted": sum(r["sorted"] == "True" for r in cubes),
        "brick_sorted": sum(r["sorted"] == "True" for r in items if r["shape"] == "brick"),
        "cylinder_sorted": sum(r["sorted"] == "True" for r in items if r["shape"] == "cylinder"),
        "cube_drops": sum(r["transport_drop"] == "True" for r in cubes),
        "cube_lifted": sum(r["lifted"] == "True" for r in cubes),
    }


def summarize_dr(output):
    rows = []
    for cond, _ in BENCHMARKS:
        h = load_history_nominal() if cond == "nominal" else read_json(OOD_OUTPUT / cond / "overall.json")
        d = read_json(output / "test" / cond / "overall.json")
        rows.append({
            "condition": cond,
            "history_all_sorted": h["all_sorted"],
            "history_all_pct": h["all_sorted_pct"],
            "history_per_seed": h["per_seed"],
            "history_items_pct": h["sorted_items_pct"],
            "history_cube_drops": h["cube_drops"],
            "dr_all_sorted": d["all_sorted"],
            "dr_all_pct": d["all_sorted_pct"],
            "dr_per_seed": d["per_seed"],
            "dr_items_pct": d["sorted_items_pct"],
            "dr_cube_sorted": d["cube_sorted"],
            "dr_brick_sorted": d["brick_sorted"],
            "dr_cylinder_sorted": d["cylinder_sorted"],
            "dr_cube_drops": d["cube_drops"],
            "dr_cube_lifted": d["cube_lifted"],
            "gain_all_pp": round(d["all_sorted_pct"] - h["all_sorted_pct"], 1),
        })
    write_csv(output / "ood_comparison.csv", rows)
    return rows


def summarize_rl(rl_root):
    results = {}
    for mode, subdir in (("selected", "test_selected"), ("final", "test_final")):
        cond_dir = rl_root / "friction_60" / subdir
        if not cond_dir.exists():
            continue
        seed_rows = []
        for s in (0, 1, 2):
            p = cond_dir / f"seed_{s}" / "summary.json"
            if not p.exists():
                seed_rows = []
                break
            seed_rows.append(read_json(p))
        if not seed_rows:
            continue
        write_csv(rl_root / "friction_60" / f"by_seed_{mode}.csv", seed_rows)
        agg = {
            "condition": "friction_60", "checkpoint_mode": mode,
            "scenes": sum(r["scenes"] for r in seed_rows),
            "all_sorted": sum(r["all_sorted"] for r in seed_rows),
            "all_sorted_pct": round(100.0 * sum(r["all_sorted"] for r in seed_rows) / sum(r["scenes"] for r in seed_rows), 1),
            "per_seed": "/".join(str(r["all_sorted"]) for r in seed_rows),
            "sorted_items": sum(r["sorted_items"] for r in seed_rows),
            "sorted_items_pct": round(100.0 * sum(r["sorted_items"] for r in seed_rows) / (3 * sum(r["scenes"] for r in seed_rows)), 1),
            "cube_sorted": sum(r["cube_sorted"] for r in seed_rows),
            "brick_sorted": sum(r["brick_sorted"] for r in seed_rows),
            "cylinder_sorted": sum(r["cylinder_sorted"] for r in seed_rows),
            "cube_drops": sum(r["cube_transport_drops"] for r in seed_rows),
            "cube_lifted": sum(r["cube_lifted"] for r in seed_rows),
        }
        write_json(rl_root / "friction_60" / f"overall_{mode}.json", agg)
        results[mode] = agg
    return results.get("final")


def plot_ood(output, rows, rl_agg=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), layout="constrained")
    x = np.arange(len(rows))
    labels = [lbl for _, lbl in BENCHMARKS]
    w = 0.34

    # Left: All-3 Sorted Rate (%) across all 6 benchmarks
    ax = axes[0]
    h_pct = [r["history_all_pct"] for r in rows]
    d_pct = [r["dr_all_pct"] for r in rows]
    ax.bar(x - w / 2, h_pct, width=w, color="#4C78A8", label="Single-Domain History3+EMA")
    ax.bar(x + w / 2, d_pct, width=w, color="#1D9A6C", label="DR + V3 Expert + DAgger (384w)")
    for i in range(len(rows)):
        ax.text(i - w / 2, h_pct[i] + 1.5, f"{h_pct[i]:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w / 2, d_pct[i] + 1.5, f"{d_pct[i]:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0F5F42")
    if rl_agg is not None:
        f_idx = next(i for i, (c, _) in enumerate(BENCHMARKS) if c == "friction_60")
        ax.scatter([f_idx], [rl_agg["all_sorted_pct"]], color="#E45756", marker="D", s=65, zorder=6,
                   label=f"Residual SAC (final 8k): {rl_agg['all_sorted_pct']:.1f}%")
    ax.set_xticks(x, labels, fontsize=8.2)
    ax.set_ylim(0, 105)
    ax.set_ylabel("All-3 Sorted Rate (%)")
    ax.set_title("Zero-Shot Generalization Across Nominal & 5 OOD Benchmarks (90 Tables Each)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Right: Cube Transport Drop Count (lower is better)
    ax2 = axes[1]
    h_drp = [r["history_cube_drops"] for r in rows]
    d_drp = [r["dr_cube_drops"] for r in rows]
    ax2.bar(x - w / 2, h_drp, width=w, color="#E45756", alpha=0.85, label="Single-Domain History3+EMA")
    ax2.bar(x + w / 2, d_drp, width=w, color="#2E9E5B", alpha=0.9, label="DR + V3 Expert + DAgger (384w)")
    for i in range(len(rows)):
        ax2.text(i - w / 2, h_drp[i] + 1.0, str(h_drp[i]), ha="center", va="bottom", fontsize=8)
        ax2.text(i + w / 2, d_drp[i] + 1.0, str(d_drp[i]), ha="center", va="bottom", fontsize=8, fontweight="bold")
    if rl_agg is not None:
        f_idx = next(i for i, (c, _) in enumerate(BENCHMARKS) if c == "friction_60")
        ax2.scatter([f_idx], [rl_agg["cube_drops"]], color="#111111", marker="D", s=65, zorder=6,
                    label=f"Online Residual SAC drops: {rl_agg['cube_drops']}")
    ax2.set_xticks(x, labels, fontsize=8.2)
    ax2.set_ylim(0, max(h_drp) + 10)
    ax2.set_ylabel("Cube Transport Drops (out of 90 attempts)")
    ax2.set_title("Cube Transport Drops Across All 6 Benchmarks (Lower is Better)")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.savefig(output / "ood_comparison.png", dpi=160)
    plt.close(fig)


def render_dr_videos(output):
    import imageio_ffmpeg

    # Render 1 nominal 3/3 (seed 0 where baseline failed) and 1 friction_60 3/3 (seed 1 where single-domain failed)
    base_nom = {int(r["scenario_seed"]): r for r in read_rows(SCHEDULE_OUTPUT / "test/force_in_transport/seed_0/scenes.csv")}
    dr_nom = read_rows(output / "test/nominal/seed_0/scenes.csv")
    pick_nom = next((r for r in dr_nom if r["all_sorted"] == "True" and base_nom[int(r["scenario_seed"])]["all_sorted"] == "False"), dr_nom[0])

    h_f60 = {int(r["scenario_seed"]): r for r in read_rows(OOD_OUTPUT / "friction_60/seed_1/scenes.csv")}
    dr_f60 = read_rows(output / "test/friction_60/seed_1/scenes.csv")
    pick_f60 = next((r for r in dr_f60 if r["all_sorted"] == "True" and h_f60[int(r["scenario_seed"])]["all_sorted"] == "False"), dr_f60[0])

    folder = output / "videos"
    folder.mkdir(exist_ok=True)
    clips = []

    for tag, cond, seed, row in (
        ("nominal_3of3", "nominal", 0, pick_nom),
        ("friction60_rescued_3of3", "friction_60", 1, pick_f60),
    ):
        scene_id = int(row["scenario_seed"])
        weights = output / "training" / f"seed_{seed}" / "model.pt"
        actor = DRHistoryPolicy()
        actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
        actor.eval()
        env_factory = AlignedContactSortingScene if cond == "nominal" else (lambda c=cond, **kw: OODSortingScene(CONDITIONS[c], **kw))
        path = folder / f"dr_{cond}_seed_{seed}_scene_{scene_id}_{row['sorted_items']}of3.mp4"
        encoder = imageio_ffmpeg.write_frames(str(path), (800, 480), fps=20, codec="libx264", macro_block_size=1)
        try:
            encoder.send(None)
            with env_factory(render_mode="rgb_array") as env:
                obs, _ = env.reset(seed=scene_id)
                encoder.send(env.render())
                items = []
                homing_steps = 0
                for idx, name in enumerate(env.order):
                    if idx:
                        steps, _ = env.return_home(encoder.send)
                        homing_steps += steps
                        obs = env.select(name)
                    buf = deque([obs.copy() for _ in range(HISTORY)], maxlen=HISTORY)
                    while True:
                        act = actor.action(np.concatenate(list(buf), axis=0))
                        obs, _, term, trunc, info = env.step(act)
                        buf.append(obs.copy())
                        encoder.send(env.render())
                        if term or trunc:
                            break
                    items.append({
                        "shape": name, "attempt_steps": env.episode.steps,
                        "stable_in_target": bool(info["stable_placement_success"]),
                        "sorted": bool(info["stable_placement_success"]) and env.final_tray(name) == env.plan[name].slot,
                        "failure_stage": info["failure_stage"],
                    })
                sorted_n = sum(x["sorted"] for x in items)
                if sorted_n != int(row["sorted_items"]):
                    raise ValueError(f"Video replay mismatch for {path}: got {sorted_n} vs {row['sorted_items']}")
        finally:
            encoder.close()
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        clips.append({
            "tag": tag, "condition": cond, "seed": seed, "scenario_seed": scene_id,
            "path": path.as_posix(), "frames": frames, "seconds": seconds,
            "sorted_items": sorted_n, "all_sorted": sorted_n == 3, "items": items,
        })
        print(f"Rendered {path} ({frames} frames, {seconds:.1f}s)", flush=True)
    write_json(folder / "criteria.json", clips)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DR_OUTPUT)
    parser.add_argument("--rl", type=Path, default=RL_OUTPUT)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = summarize_dr(args.out)
    rl_agg = summarize_rl(args.rl)
    plot_ood(args.out, rows, rl_agg)
    if args.videos:
        render_dr_videos(args.out)


if __name__ == "__main__":
    main()
