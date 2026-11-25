"""Localize transport changes induced by the previous landing-only update."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import write_csv, write_json
from .place_boundary import CANDIDATES
from .place_transport_diagnosis import LOW, TransportTrace
from .place_low_friction import digest
from .place_robustness import load_clone

OUTPUT = Path("results/place_transport_anchor")
PROBES = ("reference_grip", "reference_z", "reference_xyz", "reference_carry", "reference_prefix")
OUTCOME = ("is_success", "stable_placement_success", "transport_drop", "transported", "ever_carried", "failure_stage")
KEYS = ("x", "y", "z", "grip")


def archive(edge_root, variant):
    directory = edge_root / "seed_1" / variant
    case = directory / ("test" if variant == "frozen_previous" else "fixed_250_test")
    report = json.loads((case / "summary.json").read_text(encoding="utf-8"))
    path = Path(report["checkpoint"])
    if digest(path) != report["checkpoint_sha256"]:
        raise ValueError(f"Saved weight changed: {path}")
    with (case / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = {int(row["scenario_seed"]): row for row in csv.DictReader(stream)}
    return load_clone(path), rows


def replay(env, actor, reference, scene, mode):
    obs, _ = env.reset(seed=scene)
    trace = []
    while True:
        env.raw = actor.action(obs)
        env.applied = env.raw.copy()
        original = reference.action(obs)
        if not env.transported and not env.transport_drop:
            if mode == "reference_prefix":
                env.applied[:] = original
            elif env.episode.ever_carried:
                selection = {"reference_grip": slice(3, 4), "reference_z": slice(2, 3),
                             "reference_xyz": slice(0, 3), "reference_carry": slice(0, 4)}.get(mode)
                if selection is not None:
                    env.applied[selection] = original[selection]
        trace.append({
            "step": env.episode.steps + 1, "t_s": float(env.data.time),
            "height_m": float(env.cube_xyz[2] - env.rest_z), "goal_distance_m": env.goal_distance,
            "carry_phase": env.episode.ever_carried and not env.transported and not env.transport_drop,
            **{f"updated_{key}": float(value) for key, value in zip(KEYS, env.raw)},
            **{f"reference_{key}": float(value) for key, value in zip(KEYS, original)},
            **{f"applied_{key}": float(value) for key, value in zip(KEYS, env.applied)},
        })
        obs, _, term, trunc, info = env.step(env.applied)
        if term or trunc:
            break
    return {"scenario_seed": scene, "steps": env.episode.steps, "final_error_m": info["goal_distance_m"],
            **{key: info[key] for key in OUTCOME}}, trace


def events(samples, trace):
    drop = next((row for row in samples if row["drop"]), None)
    gripped = [row for row in samples if row["gripping"] and not row["drop"]]
    relaxed = next((row for row in trace if row["carry_phase"] and row["applied_grip"] > -0.5), None)
    closest = min((r for r in gripped if r["height_m"] >= 0.05), key=lambda r: r["goal_distance_m"], default=None)
    near = [row for row in trace if row["carry_phase"] and row["goal_distance_m"] < 0.045]
    delta_z = [row["updated_z"] - row["reference_z"] for row in near]
    return {
        "min_qualified_goal_distance_m": closest["goal_distance_m"] if closest else None,
        "drop_step": drop["step"] if drop else None, "drop_goal_distance_m": drop["goal_distance_m"] if drop else None,
        "height_at_last_grip_m": gripped[-1]["height_m"] if gripped else None,
        "slip_at_last_grip_m": gripped[-1]["slip_z_m"] if gripped else None,
        "relax_step": relaxed["step"] if relaxed else None,
        "relax_goal_distance_m": relaxed["goal_distance_m"] if relaxed else None,
        "median_near_goal_z_delta_on_same_obs": float(np.median(delta_z)) if delta_z else None,
    }


def diagnose(results, output):
    edge = results / "place_edge"
    reference, prior = archive(edge, "frozen_previous")
    ordinary, updated = archive(edge, "generic_landing")
    targeted, targeted_rows = archive(edge, "edge_landing")
    drops = [scene for scene, row in updated.items() if row["transport_drop"] == "True"]
    other = [scene for scene, row in targeted_rows.items() if row["transport_drop"] == "True" and scene not in drops]
    controls = [scene for scene, row in updated.items()
                if row["stable_placement_success"] == prior[scene]["stable_placement_success"] == "True"
                and scene not in other][:len(drops)]
    groups = {"ordinary_drops": drops, "edge_only_drop": other, "stable_controls": controls}
    runs = [("reference", reference, prior), ("ordinary", ordinary, updated), ("edge", targeted, targeted_rows)]
    runs.extend((mode, ordinary, None) for mode in PROBES)
    all_rows, steps, physics, totals = [], [], [], []
    for mode, actor, expected in runs:
        with TransportTrace(CANDIDATES[LOW]) as env:
            for group, scenes in groups.items():
                records = []
                for scene in scenes:
                    row, trace = replay(env, actor, reference, scene, mode)
                    if expected and any(str(row[key]) != expected[scene][key] for key in (*OUTCOME, "steps")):
                        raise ValueError(f"Replay differs from archived result: {mode}, {scene}")
                    labels = {"mode": mode, "group": group, "scenario_seed": scene}
                    records.append({**labels, **row, **events(env.samples, trace)})
                    if mode in ("reference", "ordinary", "edge"):
                        steps.extend({**labels, **point} for point in trace)
                        physics.extend({**labels, **point} for point in env.samples)
                totals.append({"mode": mode, "group": group, "episodes": len(records),
                               "drops": sum(row["transport_drop"] for row in records),
                               "stable": sum(row["stable_placement_success"] for row in records),
                               "arrived": sum(row["transported"] for row in records)})
                all_rows.extend(records)
        write_csv(output / "summary.csv", totals)
        print(f"{mode}: {[(r['group'], r['drops'], r['stable']) for r in totals if r['mode'] == mode]}", flush=True)
    write_csv(output / "episodes.csv", all_rows)
    write_csv(output / "policy_trace.csv", steps)
    write_csv(output / "physics_trace.csv", physics)
    write_json(output / "scenes.json", groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    diagnose(args.results, args.out / "diagnosis")


if __name__ == "__main__":
    main()
