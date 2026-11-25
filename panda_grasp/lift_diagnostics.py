"""Replay saved lift policies and measure breaks in the physical hold criterion."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np
import torch

from .cli import load_policy
from .env import ANGULAR_LIMIT, LIFT_M, LINEAR_LIMIT, PandaGraspEnv
from .experiment import TEST_SEEDS, write_csv, write_json


class LiftTrace(PandaGraspEnv):
    def reset(self, **kwargs):
        self.failures = Counter()
        self.breaks = Counter()
        return super().reset(**kwargs)

    def _sample_hold(self, height, contacts):
        previous = self.episode.hold_ticks
        super()._sample_hold(height, contacts)
        if self.episode.ever_carried:
            linear, angular = np.linalg.norm(self.cube_velocity.reshape(2, 3), axis=1)
            conditions = {
                "low_height": height < LIFT_M,
                "no_bilateral": not contacts.gripping,
                "table_contact": bool(contacts.table_count),
                "linear_speed": linear > LINEAR_LIMIT,
                "angular_speed": angular > ANGULAR_LIMIT,
            }
            self.failures["sampled"] += 1
            for reason, active in conditions.items():
                if active:
                    self.failures[reason] += 1
                    if previous and self.episode.hold_ticks == 0:
                        self.breaks[reason] += 1


def replay(env, policy, scenario, mode):
    obs, _ = env.reset(seed=scenario)
    predict = policy(env, scenario)
    intervention = False
    actions = []
    trace = []
    while True:
        action = predict(obs)
        if mode == "close_near" and (intervention or np.linalg.norm(env.tcp - env.cube_xyz) < 0.010):
            intervention = True
            action = action.copy()
            action[3] = -1
        if env.episode.ever_carried:
            actions.append(action.copy())
            if mode in ("freeze_xyz", "freeze_close"):
                intervention = True
                action = action.copy()
                action[:3] = 0
                if mode == "freeze_close":
                    action[3] = -1
        obs, _, term, trunc, info = env.step(action)
        if scenario == TEST_SEEDS[0]:
            trace.append({
                "step": env.episode.steps,
                "lift_m": float(env.cube_xyz[2] - env.rest_z),
                "linear_speed_m_s": float(np.linalg.norm(env.cube_velocity[:3])),
                "angular_speed_rad_s": float(np.linalg.norm(env.cube_velocity[3:])),
                "action_x": float(action[0]), "action_y": float(action[1]),
                "action_z": float(action[2]), "action_gripper": float(action[3]),
                "left_force_n": info["left_force_n"], "right_force_n": info["right_force_n"],
                "hold_s": env.episode.hold_ticks * env.dt,
            })
        if term or trunc:
            break
    reasons = ("low_height", "no_bilateral", "table_contact", "linear_speed", "angular_speed")
    return {
        "scenario_seed": scenario, "mode": mode, "steps": env.episode.steps,
        "intervention": intervention, **{k: info[k] for k in (
            "is_success", "failure_reason", "ever_bilateral", "ever_carried", "dropped", "max_hold_s",
            "best_bilateral_lift_m", "ik_failures",
        )},
        "sampled_ticks_after_lift": env.failures["sampled"],
        **{f"ticks_{key}": env.failures[key] for key in reasons},
        **{f"breaks_{key}": env.breaks[key] for key in reasons},
        "post_lift_xyz_rms": float(np.sqrt(np.mean(np.square(np.asarray(actions)[:, :3])))) if actions else None,
        "post_lift_mean_gripper": float(np.mean(np.asarray(actions)[:, 3])) if actions else None,
    }, trace


def analyze(source, output):
    policy = load_policy(source.name, source)
    with (source / "test_episodes.csv").open(encoding="utf-8", newline="") as file:
        archived = {int(row["scenario_seed"]): row for row in csv.DictReader(file)}
    rows = []
    traces = []
    with LiftTrace() as env:
        for mode, seeds in (("policy", TEST_SEEDS), ("freeze_xyz", TEST_SEEDS[::4]),
                            ("freeze_close", TEST_SEEDS[::4])):
            for scenario in seeds:
                row, trace = replay(env, policy, scenario, mode)
                if mode == "policy":
                    reference = archived[scenario]
                    if (str(row["is_success"]) != reference["is_success"]
                            or row["failure_reason"] != reference["failure_reason"]
                            or row["steps"] != int(reference["steps"])
                            or not np.isclose(row["max_hold_s"], float(reference["max_hold_s"]), atol=1e-12, rtol=0)):
                        raise ValueError(f"Archived result differs for {source}, scenario {scenario}")
                rows.append(row)
                traces.extend({"mode": mode, **sample} for sample in trace)
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "trace_900000.csv", traces)
    metrics = {}
    for mode in ("policy", "policy_paired", "freeze_xyz", "freeze_close"):
        selection = ([row for row in rows if row["mode"] == "policy" and row["scenario_seed"] in TEST_SEEDS[::4]]
                     if mode == "policy_paired" else [row for row in rows if row["mode"] == mode])
        lifted = [row for row in selection if row["ever_carried"]]
        sampled = sum(row["sampled_ticks_after_lift"] for row in selection)
        metrics[mode] = {
            "episodes": len(selection), "successes": sum(row["is_success"] for row in selection),
            "lifted": len(lifted),
            "mean_max_hold_s": float(np.mean([row["max_hold_s"] for row in selection])),
            "mean_post_lift_xyz_rms": float(np.mean([row["post_lift_xyz_rms"] for row in lifted])) if lifted else None,
            "failure_reasons": dict(Counter(row["failure_reason"] for row in selection if not row["is_success"])),
            "hold_breaks": {key: sum(row[key] for row in selection) for key in selection[0] if key.startswith("breaks_")},
            "post_lift_violation_fraction": {
                key: sum(row[key] for row in selection) / sampled if sampled else None
                for key in selection[0] if key.startswith("ticks_")
            },
        }
    write_json(output / "summary.json", metrics)
    print(f"{source}: {[row['successes'] for row in metrics.values()]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--close-probe", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for profile, method in (("pilot", "sac"), ("pilot", "bc_sac"), ("contact_trigger", "bc_sac")):
        relative = Path(profile) / f"seed_{args.seed}" / method
        source = args.results / relative
        destination = args.results / "lift_diagnostics" / relative
        if args.close_probe:
            if method != "bc_sac":
                continue
            policy = load_policy(method, source)
            with LiftTrace() as env:
                rows = [replay(env, policy, scene, "close_near")[0] for scene in TEST_SEEDS[::4]]
            write_csv(destination / "close_probe.csv", rows)
            print(f"{source}: forced closure, "
                  f"{sum(row['ever_bilateral'] for row in rows)}/{len(rows)} bilateral, "
                  f"{sum(row['ever_carried'] for row in rows)}/{len(rows)} lifts", flush=True)
        else:
            analyze(source, destination)


if __name__ == "__main__":
    main()
