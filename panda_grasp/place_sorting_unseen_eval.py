"""Held-out evaluation of frozen policies on brand-new layouts (1568000-1568029) and unseen parameter combinations."""

import argparse
import json
from pathlib import Path

import torch

from .experiment import write_csv, write_json
from .place_objects_train import sha
from .place_sorting_aligned import AlignedContactSortingScene
from .place_sorting_dr import DRHistoryPolicy, OUTPUT as DR_OUTPUT, RobustSortingTeacherV3
from .place_sorting_history import HistoryAlignedPolicy, OUTPUT as HISTORY_OUTPUT, run_history_scene
from .place_sorting_ood import OODSortingScene, OODSpec

OUTPUT = Path("results/place_sorting_unseen")
UNSEEN_TEST = range(1_568_000, 1_568_030)

# Strictly unseen parameter combinations (none appeared in DR_BUCKETS)
UNSEEN_CONDITIONS = {
    "unseen_nominal": OODSpec("unseen_nominal"),
    "unseen_feasible_combo": OODSpec(
        "unseen_feasible_combo",
        friction_mult=66.0 / 70.0,
        mass_mult=156.0 / 150.0,
        pickup_jitter_m=0.008,
        xy_scale=0.97,
    ),
    "unseen_combo_4way": OODSpec(
        "unseen_combo_4way",
        friction_mult=62.0 / 70.0,
        mass_mult=162.0 / 150.0,
        pickup_jitter_m=0.009,
        xy_scale=0.95,
    ),
    "unseen_extrap_fric57": OODSpec(
        "unseen_extrap_fric57",
        friction_mult=57.0 / 70.0,
        mass_mult=155.0 / 150.0,
        pickup_jitter_m=0.007,
    ),
    "unseen_combo_heavy_small": OODSpec(
        "unseen_combo_heavy_small",
        friction_mult=66.0 / 70.0,
        mass_mult=168.0 / 150.0,
        pickup_jitter_m=0.011,
        xy_scale=0.92,
    ),
}


def load_frozen_policy(family, seed):
    if family == "history3_ema":
        root = HISTORY_OUTPUT / "training" / f"seed_{seed}"
        weights = root / "model.pt"
        expected = json.loads((root / "config.json").read_text(encoding="utf-8"))["checkpoint_sha256"]
        actual = sha(weights)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {weights}")
        actor = HistoryAlignedPolicy()
        actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
        actor.eval()
        return actor, actual
    if family == "dr_history_dagger":
        root = DR_OUTPUT / "training" / f"seed_{seed}"
        weights = root / "model.pt"
        expected = json.loads((root / "config.json").read_text(encoding="utf-8"))["checkpoint_sha256"]
        actual = sha(weights)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {weights}")
        actor = DRHistoryPolicy()
        actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
        actor.eval()
        return actor, actual
    raise ValueError(f"Unknown policy family: {family}")


def eval_v3_expert(cond_name, output):
    cfg = UNSEEN_CONDITIONS[cond_name]
    env_factory = AlignedContactSortingScene if cond_name == "unseen_nominal" else (lambda: OODSortingScene(cfg))
    scenes, items = [], []
    with env_factory() as env:
        for scenario in UNSEEN_TEST:
            obs, _ = env.reset(seed=scenario)
            rows = []
            for idx, name in enumerate(env.order):
                if idx:
                    env.return_home()
                    obs = env.select(name)
                teacher = RobustSortingTeacherV3(env, brick_long_axis=True, perturb=False)
                while True:
                    _, act = teacher(obs)
                    obs, _, term, trunc, info = env.step(act)
                    if term or trunc:
                        break
                ok = bool(info["stable_placement_success"]) and env.final_tray(name) == env.plan[name].slot
                rows.append({
                    "scenario_seed": scenario, "shape": name, "sorted": ok,
                    "lifted": info["ever_carried"], "transport_drop": info["transport_drop"],
                    "failure_stage": info["failure_stage"],
                })
            items.extend(rows)
            scenes.append({
                "scenario_seed": scenario, "sorted_items": sum(r["sorted"] for r in rows),
                "all_sorted": all(r["sorted"] for r in rows),
            })
    folder = output / cond_name / "v3_expert"
    write_csv(folder / "scenes.csv", scenes)
    write_csv(folder / "items.csv", items)
    cubes = [r for r in items if r["shape"] == "cube"]
    agg = {
        "family": "v3_expert", "condition": cond_name, "scenes": len(scenes),
        "all_sorted": sum(r["all_sorted"] for r in scenes),
        "all_sorted_pct": round(100.0 * sum(r["all_sorted"] for r in scenes) / len(scenes), 1),
        "sorted_items": sum(r["sorted"] for r in items),
        "sorted_items_pct": round(100.0 * sum(r["sorted"] for r in items) / len(items), 1),
        "cube_sorted": sum(r["sorted"] for r in cubes),
        "cube_lifted": sum(r["lifted"] for r in cubes),
        "cube_drops": sum(r["transport_drop"] for r in cubes),
        "brick_sorted": sum(r["shape"] == "brick" and r["sorted"] for r in items),
        "cylinder_sorted": sum(r["shape"] == "cylinder" and r["sorted"] for r in items),
    }
    write_json(folder / "overall.json", agg)
    print("UNSEEN-EXPERT:", agg, flush=True)


def eval_family(family, cond_name, output):
    if family == "v3_expert":
        return eval_v3_expert(cond_name, output)
    cfg = UNSEEN_CONDITIONS[cond_name]
    env_factory = AlignedContactSortingScene if cond_name == "unseen_nominal" else (lambda: OODSortingScene(cfg))
    all_scenes, all_items, seed_rows = [], [], []
    with env_factory() as env:
        for seed in (0, 1, 2):
            actor, identity = load_frozen_policy(family, seed)
            scenes, items = [], []
            for scenario in UNSEEN_TEST:
                sc, rows = run_history_scene(env, scenario, actor)
                scenes.append(sc)
                items.extend(rows)
            folder = output / cond_name / family / f"seed_{seed}"
            write_csv(folder / "scenes.csv", scenes)
            write_csv(folder / "items.csv", items)
            cubes = [r for r in items if r["shape"] == "cube"]
            summary = {
                "family": family, "condition": cond_name, "seed": seed, "checkpoint_sha256": identity,
                "scenes": len(scenes), "all_sorted": sum(r["all_sorted"] for r in scenes),
                "sorted_items": sum(r["sorted"] for r in items),
                "cube_sorted": sum(r["sorted"] for r in cubes),
                "cube_lifted": sum(r["lifted"] for r in cubes),
                "cube_drops": sum(r["transport_drop"] for r in cubes),
                "brick_sorted": sum(r["shape"] == "brick" and r["sorted"] for r in items),
                "cylinder_sorted": sum(r["shape"] == "cylinder" and r["sorted"] for r in items),
            }
            write_json(folder / "summary.json", summary)
            seed_rows.append(summary)
            all_scenes.extend(scenes)
            all_items.extend(items)
    cubes_all = [r for r in all_items if r["shape"] == "cube"]
    agg = {
        "family": family, "condition": cond_name, "scenes": len(all_scenes),
        "all_sorted": sum(r["all_sorted"] for r in all_scenes),
        "all_sorted_pct": round(100.0 * sum(r["all_sorted"] for r in all_scenes) / len(all_scenes), 1),
        "per_seed": "/".join(str(r["all_sorted"]) for r in seed_rows),
        "sorted_items": sum(r["sorted"] for r in all_items),
        "sorted_items_pct": round(100.0 * sum(r["sorted"] for r in all_items) / len(all_items), 1),
        "cube_sorted": sum(r["sorted"] for r in cubes_all),
        "cube_lifted": sum(r["lifted"] for r in cubes_all),
        "cube_drops": sum(r["transport_drop"] for r in cubes_all),
        "brick_sorted": sum(r["shape"] == "brick" and r["sorted"] for r in all_items),
        "cylinder_sorted": sum(r["shape"] == "cylinder" and r["sorted"] for r in all_items),
    }
    write_json(output / cond_name / family / "overall.json", agg)
    print("UNSEEN:", agg, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("history3_ema", "dr_history_dagger", "v3_expert"), required=True)
    parser.add_argument("--condition", choices=tuple(UNSEEN_CONDITIONS), required=True)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    eval_family(args.family, args.condition, args.out)


if __name__ == "__main__":
    main()
