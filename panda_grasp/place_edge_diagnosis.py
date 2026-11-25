"""Replay paired seed-2 landing regressions and intervene on one action component."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .env import ACTION_STEP
from .experiment import write_csv, write_json
from .place import GOAL_RADIUS, GRIP_OPEN_M
from .place_boundary import BoundaryEvaluation, CANDIDATES
from .place_low_friction import digest
from .place_robustness import load_clone
from .place_transport_diagnosis import LOW

OUTPUT = Path("results/place_edge")
METHODS = ("frozen_bc", "extra_demo_bc")
PROBES = ("center_xy", "slow_z", "hold_close")
PLACEMENT_KEYS = ("is_success", "stable_placement_success", "transport_drop", "transported", "failure_stage")


def prior_run(results, seed, method):
    directory = results / "place_low_friction" / f"seed_{seed}" / method
    receipt = json.loads((directory / "cost.json").read_text(encoding="utf-8"))
    path = Path(receipt["selected_checkpoint"])
    if digest(path) != receipt["checkpoint_sha256"]:
        raise ValueError(f"Previous checkpoint differs from its record: {path}")
    with (directory / "test" / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        return path, {int(r["scenario_seed"]): r for r in csv.DictReader(stream)}


class LandingTrace(BoundaryEvaluation):
    def _reset_task(self, options):
        super()._reset_task(options)
        self.samples = []
        self.touched_table = False
        self.raw_action = self.applied_action = np.zeros(4)

    def _sample_hold(self, height, contacts):
        super()._sample_hold(height, contacts)
        if not self.transported:
            return
        supported = contacts.supported
        self.touched_table |= supported
        dx, dy = self.cube_xyz[:2] - self.target[:2]
        finger_gap = float(self.data.qpos[list(self.gripper.fingers)].sum())
        self.samples.append({
            "t_s": float(self.data.time), "step": self.episode.steps, "height_m": height,
            "x_error_m": float(dx), "y_error_m": float(dy), "error_m": self.goal_distance,
            "cube_vxy_m_s": float(np.linalg.norm(self.cube_velocity[:2])),
            "left_force_n": contacts.left_normal_n, "right_force_n": contacts.right_normal_n,
            "gripping": contacts.gripping, "supported": supported, "finger_gap_m": finger_gap,
            "open_clear_of_cube": finger_gap >= GRIP_OPEN_M and not (contacts.left_count or contacts.right_count),
            "outcome_hold_s": self.outcome_ticks * self.dt,
            **{f"raw_{key}": float(value) for key, value in zip(("x", "y", "z", "grip"), self.raw_action)},
            **{f"applied_{key}": float(value) for key, value in zip(("x", "y", "z", "grip"), self.applied_action)},
        })


def replay(env, actor, scene, mode):
    obs, _ = env.reset(seed=scene)
    while True:
        env.raw_action = actor.action(obs)
        env.applied_action = env.raw_action.copy()
        if env.transported and not env.touched_table:
            if mode == "center_xy":
                delta = (env.target[:2] - env.cube_xyz[:2]) / ACTION_STEP
                env.applied_action[:2] = delta * min(1.0, 0.45 / max(np.linalg.norm(delta), 0.45))
            elif mode == "slow_z" and env.applied_action[2] < 0:
                env.applied_action[2] *= 0.5
            elif mode == "hold_close":
                env.applied_action[3] = -1.0
        obs, _, term, trunc, info = env.step(env.applied_action)
        if term or trunc:
            break
    return {"scenario_seed": scene, "steps": env.episode.steps, "final_error_m": info["goal_distance_m"],
            **{key: info[key] for key in PLACEMENT_KEYS}}


def landmarks(samples):
    arrive = samples[0] if samples else None
    touch = next((row for row in samples if row["supported"]), None)
    opened = next((row for row in samples if row["open_clear_of_cube"]), None)
    crossing = next((row for row in samples if row["error_m"] > GOAL_RADIUS), None)
    return {key: value for name, row in (("arrival", arrive), ("first_support", touch),
                                           ("first_open", opened), ("first_exit", crossing))
            for key, value in ((f"{name}_step", row["step"] if row else None),
                               (f"{name}_time_s", row["t_s"] if row else None),
                               (f"{name}_error_m", row["error_m"] if row else None))}


def diagnose(results, output):
    frozen, older = prior_run(results, 2, METHODS[0])
    adapted, newer = prior_run(results, 2, METHODS[1])
    def stable(row):
        return row["stable_placement_success"] == "True"
    groups = {
        "new_failure": [scene for scene in older if stable(older[scene]) and not stable(newer[scene])],
        "persistent_failure": [scene for scene in older if not stable(older[scene]) and not stable(newer[scene])],
        "recovered": [scene for scene in older if not stable(older[scene]) and stable(newer[scene])],
        "stable_control": [scene for scene in older if stable(older[scene]) and stable(newer[scene])][:7],
    }
    runs = [(METHODS[0], "policy", frozen, older), (METHODS[1], "policy", adapted, newer)]
    runs += [(METHODS[1], mode, adapted, newer) for mode in PROBES]
    summaries, traces = [], []
    for method, mode, path, archived in runs:
        actor = load_clone(path)
        counts = Counter()
        with LandingTrace(CANDIDATES[LOW]) as env:
            for group, scenes in groups.items():
                for scene in scenes:
                    result = replay(env, actor, scene, mode)
                    if mode == "policy" and any(str(result[key]) != archived[scene][key]
                                                for key in (*PLACEMENT_KEYS, "steps")):
                        raise ValueError(f"Archived behavior changed: {method} scene {scene}")
                    meta = {"method": method, "mode": mode, "group": group}
                    summaries.append({**meta, **result, **landmarks(env.samples)})
                    traces.extend({**meta, "scenario_seed": scene, **point} for point in env.samples)
                    counts[group] += result["stable_placement_success"]
        print(method, mode, dict(counts), flush=True)
    write_csv(output / "episodes.csv", summaries)
    write_csv(output / "physics_trace.csv", traces)
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
