"""Trace and intervene on archived low-friction transportation failures."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .env import ACTION_STEP
from .experiment import write_csv, write_json
from .place_boundary import BoundaryEvaluation, CANDIDATES, REFERENCE
from .place_robustness import load_clone

OUTPUT = Path("results/place_low_friction")
LOW = "friction_70_mass_150"
MODES = ("policy", "full_close", "half_xy", "lift_clearance")


def checkpoint(results, seed):
    return results / "place_touchdown" / "training" / f"seed_{seed}" / "model.pt"


class TransportTrace(BoundaryEvaluation):
    def _reset_task(self, options):
        super()._reset_task(options)
        self.samples = []
        self.lift_offset = None
        self.raw = self.applied = np.zeros(4)

    def _sample_hold(self, height, contacts):
        super()._sample_hold(height, contacts)
        if not self.episode.ever_carried or self.transported or (self.samples and self.samples[-1]["drop"]):
            return
        relative = self.cube_xyz - self.tcp
        if self.lift_offset is None:
            self.lift_offset = relative.copy()
        self.samples.append({
            "t_s": float(self.data.time), "step": self.episode.steps,
            "height_m": height, "goal_distance_m": self.goal_distance,
            "left_n": contacts.left_normal_n, "right_n": contacts.right_normal_n,
            "gripping": contacts.gripping, "drop": self.transport_drop,
            "gap_m": float(self.data.qpos[list(self.gripper.fingers)].sum()),
            "slip_z_m": float(relative[2] - self.lift_offset[2]),
            "slip_xy_m": float(np.linalg.norm(relative[:2] - self.lift_offset[:2])),
            "relative_z_m": float(relative[2]),
            **{f"raw_{key}": float(x) for key, x in zip(("x", "y", "z", "grip"), self.raw)},
            **{f"applied_{key}": float(x) for key, x in zip(("x", "y", "z", "grip"), self.applied)},
        })


def archived_scene_rows(results, seed, condition):
    path = results / "place_boundary" / "test" / f"seed_{seed}" / condition / "episodes.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        return {int(row["scenario_seed"]): row for row in csv.DictReader(stream)}


def replay(env, actor, mode, scene):
    obs, _ = env.reset(seed=scene)
    while True:
        env.raw = actor.action(obs)
        env.applied = env.raw.copy()
        if env.episode.ever_carried and not env.transported and not env.transport_drop:
            if mode == "full_close":
                env.applied[3] = -1.0
            elif mode == "half_xy":
                env.applied[:2] *= 0.5
            elif mode == "lift_clearance":
                height = env.cube_xyz[2] - env.rest_z
                floor = np.clip((0.072 - height) / ACTION_STEP, 0.0, 0.6)
                if height < 0.072:
                    env.applied[2] = max(env.applied[2], floor)
        obs, _, term, trunc, info = env.step(env.applied)
        if term or trunc:
            break
    keys = ("is_success", "stable_placement_success", "transport_drop", "transported",
            "ever_carried", "failure_stage", "goal_distance_m")
    return {"scenario_seed": scene, "steps": env.episode.steps, **{key: info[key] for key in keys}}


def trace_summary(samples):
    if not samples:
        return {}
    lost = next((row for row in samples if not row["gripping"]), None)
    drop = next((row for row in samples if row["drop"]), None)
    before_loss = [row for row in samples if not lost or row["t_s"] < lost["t_s"]]
    return {
        "lift_t_s": samples[0]["t_s"], "lift_gap_m": samples[0]["gap_m"],
        "initial_relative_z_m": samples[0]["relative_z_m"],
        "min_pre_loss_slip_z_m": min((row["slip_z_m"] for row in before_loss), default=None),
        "max_pre_loss_slip_xy_m": max((row["slip_xy_m"] for row in before_loss), default=None),
        "min_pre_loss_grip_action": min((row["raw_grip"] for row in before_loss), default=None),
        "max_pre_loss_grip_action": max((row["raw_grip"] for row in before_loss), default=None),
        "weakest_force_at_lift_n": min(samples[0]["left_n"], samples[0]["right_n"]),
        "first_loss_t_s": lost["t_s"] if lost else None,
        "first_loss_height_m": lost["height_m"] if lost else None,
        "first_loss_slip_z_m": lost["slip_z_m"] if lost else None,
        "first_loss_relative_z_m": lost["relative_z_m"] if lost else None,
        "first_loss_grip_action": lost["raw_grip"] if lost else None,
        "drop_t_s": drop["t_s"] if drop else None,
    }


def diagnose(results, output):
    seed1 = archived_scene_rows(results, 1, LOW)
    dropped = [scene for scene, row in seed1.items() if row["transport_drop"] == "True"]
    good = [scene for scene, row in seed1.items() if row["stable_placement_success"] == "True"][:len(dropped)]
    scenarios = dropped + good
    rows, traces = [], []
    runs = [
        *((1, LOW, mode) for mode in MODES),
        (1, REFERENCE, "policy"), (2, LOW, "policy"),
    ]
    for seed, condition, mode in runs:
        actor = load_clone(checkpoint(results, seed))
        archived = archived_scene_rows(results, seed, condition)
        with TransportTrace(CANDIDATES[condition]) as env:
            for scene in scenarios:
                result = replay(env, actor, mode, scene)
                if mode == "policy":
                    for key in ("steps", "is_success", "stable_placement_success", "transport_drop", "failure_stage"):
                        if str(result[key]) != archived[scene][key]:
                            raise ValueError(f"Replay differs from archive for {seed}, {condition}, {scene}: {key}")
                group = "original_drop" if scene in dropped else "original_stable"
                label = {"seed": seed, "condition": condition, "mode": mode, "group": group}
                summary = trace_summary(env.samples)
                rows.append({**label, **result, **summary})
                traces.extend({**label, "scenario_seed": scene, **row} for row in env.samples)
        print(f"Traced {len(scenarios)} scenes: seed {seed}, {condition}, {mode}", flush=True)
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "physics_trace.csv", traces)
    metrics = []
    for seed, condition, mode in runs:
        for group in ("original_drop", "original_stable"):
            group_rows = [r for r in rows if (r["seed"], r["condition"], r["mode"], r["group"])
                          == (seed, condition, mode, group)]
            metrics.append({"seed": seed, "condition": condition, "mode": mode, "group": group,
                            "episodes": len(group_rows), "drops": sum(r["transport_drop"] for r in group_rows),
                            "stable": sum(r["stable_placement_success"] for r in group_rows),
                            "transported": sum(r["transported"] for r in group_rows)})
    write_csv(output / "summary.csv", metrics)
    write_json(output / "protocol.json", {"drop_scenes": dropped, "control_scenes": good,
               "physics_sample_s": 0.002, "translation_scale": 0.5, "clearance_m": 0.072})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    diagnose(args.results, args.out / "diagnosis")


if __name__ == "__main__":
    main()
