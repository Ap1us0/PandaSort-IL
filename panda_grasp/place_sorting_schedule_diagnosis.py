"""Replay the transport and placement regressions of the two-stage tactile BC."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .env import ACTION_STEP
from .experiment import write_csv
from .place import GOAL_RADIUS
from .place_objects import OBS_SIZE
from .place_sorting import run_scene
from .place_sorting_analysis import read_rows, record_cases
from .place_sorting_contact_ablation import ARRIVED_INDEX, LIFTED_INDEX, SWITCHED, scene_type
from .place_sorting_contact_phase import OUTPUT as PRIOR, load as load_prior
from .place_sorting_contact_schedule import OUTPUT
from .place_contact_signals import ContactSortingScene

OUTCOMES = ("attempt_steps", "lifted", "transported", "transport_drop", "failure_stage",
            "stable_in_target", "hold_s", "final_color", "sorted")
PHASES = {1: "prelift", 2: "place"}


def archive(method, seed):
    path = PRIOR / "test" / method / f"seed_{seed}"
    actor, identity = load_prior(method, seed, PRIOR)
    receipt = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if receipt["checkpoint_sha256"] != identity:
        raise ValueError(f"Unexpected archive checkpoint: {path}")
    scenes = {int(row["scenario_seed"]): row for row in read_rows(path / "scenes.csv")}
    items = defaultdict(list)
    for row in read_rows(path / "items.csv"):
        items[int(row["scenario_seed"])].append(row)
    return actor, scenes, items


def discordant(seed, old, new):
    _, before, old_cases = old
    _, after, new_cases = new
    for scene, before_scene in before.items():
        baseline = {row["shape"]: row for row in old_cases[scene]}
        switched = {row["shape"]: row for row in new_cases[scene]}
        if seed == 1 and baseline["cube"]["transport_drop"] != switched["cube"]["transport_drop"]:
            transition = "drop_added" if switched["cube"]["transport_drop"] == "True" else "drop_avoided"
            yield scene, "cube", transition
        if seed == 2 and before_scene["all_sorted"] != after[scene]["all_sorted"]:
            lost = after[scene]["all_sorted"] == "False"
            affected = switched if lost else baseline
            for shape, row in affected.items():
                if row["sorted"] == "False":
                    yield scene, shape, "full_scene_lost" if lost else "full_scene_gained"


class FailureTrace(ContactSortingScene):
    def __init__(self, focus):
        super().__init__()
        self.focus = focus

    def _reset_task(self, options):
        self.samples = []
        self.commands = []
        super()._reset_task(options)

    def _sample_hold(self, height, contacts):
        super()._sample_hold(height, contacts)
        if self.shape_name != self.focus or not self.episode.ever_carried:
            return
        slot = self.plan[self.focus].slot
        others = set()
        for contact in self.data.contact:
            if self.cube not in contact.geom or contact.efc_address < 0:
                continue
            other = int(contact.geom[1] if contact.geom[0] == self.cube else contact.geom[0])
            if other != self.table and self.model.geom_bodyid[other] not in self.gripper.bodies:
                others.add(self.model.geom(other).name)
        self.samples.append({
            "t_s": float(self.data.time), "step": self.episode.steps, "height_m": height,
            "goal_distance_m": self.goal_distance, "target_inside": self.contains(self.focus, slot),
            "left_n": contacts.left_normal_n, "right_n": contacts.right_normal_n, "gripping": contacts.gripping,
            "finger_contact": bool(contacts.left_count or contacts.right_count),
            "width_m": float(self.data.qpos[list(self.gripper.fingers)].sum()),
            "supported": contacts.supported, "gripped_touchdown": self.supported_at_target,
            "picked": self.episode.ever_carried, "arrived": self.transported, "dropped": self.transport_drop,
            "hold_s": self.outcome_ticks * self.dt,
            "object_x": float(self.cube_xyz[0]), "object_y": float(self.cube_xyz[1]),
            "tool_x": float(self.tcp[0]), "tool_y": float(self.tcp[1]), "tool_z": float(self.tcp[2]),
            "grip_action": self.applied_grip, "other_geoms": "/".join(sorted(others)),
        })


class DiagnosticPolicy:
    def __init__(self, env, method, baseline, switched, intervention_phase):
        self.env = env
        self.method = method
        self.baseline = baseline
        self.switched = switched
        self.intervention_phase = intervention_phase

    def action(self, obs):
        lifted = obs[LIFTED_INDEX] > 0.5
        arrived = obs[ARRIVED_INDEX] > 0.5
        focus = self.env.shape_name == self.env.focus
        window = focus and (not lifted if self.intervention_phase == "prelift" else arrived)
        swap_transport = self.method == "swap_transport" and focus and lifted and not arrived
        swap_before_arrival = self.method == "swap_before_arrival" and focus and not arrived
        if self.method == "binary" or (self.method == "swap_binary" and window) or swap_transport or swap_before_arrival:
            action = self.baseline.action(obs[:OBS_SIZE])
        else:
            if self.method == "hide_slip" and window:
                obs = obs.copy()
                obs[OBS_SIZE + 2:] = 0
            action = self.switched.action(obs)
        if (self.method == "raise_clearance" and focus and lifted and not arrived
                and not self.env.transport_drop and self.env.goal_distance <= 0.095):
            height = self.env.cube_xyz[2] - self.env.rest_z
            if height < 0.072:
                action[2] = max(action[2], np.clip((0.072 - height) / ACTION_STEP, 0.0, 0.6))
        self.env.applied_grip = float(action[3])
        if focus:
            self.env.commands.append({
                "t_s": float(self.env.data.time), "step": self.env.episode.steps + 1,
                "lifted_obs": lifted, "arrived_obs": arrived, "goal_distance_m": self.env.goal_distance,
                "grip": float(action[3]), "dx": float(action[0]), "dy": float(action[1]),
                "dz": float(action[2]), "dyaw": float(action[4]),
                "slip_v_norm": float(np.linalg.norm(obs[OBS_SIZE + 2:OBS_SIZE + 5])),
                "slip_d_norm": float(np.linalg.norm(obs[OBS_SIZE + 5:])),
            })
        return action


def summarize(trace, commands):
    first = trace[0] if trace else {}
    arrival = next((row for row in trace if row["arrived"]), {})
    drop = next((row for row in trace if row["dropped"]), {})
    end = trace.index(drop) if drop else len(trace)
    last_contact = max((i for i in range(end) if trace[i]["finger_contact"]), default=-1)
    final_detach = trace[last_contact + 1] if drop and last_contact >= 0 else {}
    post_arrival = [row for row in trace if row["arrived"]]
    off_hand = next((row for row in post_arrival if not row["finger_contact"]), {})
    pre_contact = [row for row in post_arrival if not off_hand or row["t_s"] <= off_hand["t_s"]]
    after_arrival_cmd = [row for row in commands if row["arrived_obs"]]
    before_drop = trace[:end]
    return {
        "lift_t_s": first.get("t_s"), "lift_force_weaker_n": min(first["left_n"], first["right_n"]) if first else None,
        "arrival_t_s": arrival.get("t_s"), "arrival_goal_error_m": arrival.get("goal_distance_m"),
        "drop_t_s": drop.get("t_s"), "drop_goal_error_m": drop.get("goal_distance_m"),
        "drop_detach_t_s": final_detach.get("t_s"), "drop_detach_grip": final_detach.get("grip_action"),
        "drop_detach_height_m": final_detach.get("height_m"),
        "obstacles_before_drop": "/".join(sorted({g for row in before_drop for g in row["other_geoms"].split("/") if g})),
        "first_post_arrival_detach_t_s": off_hand.get("t_s"),
        "first_post_arrival_detach_goal_error_m": off_hand.get("goal_distance_m"),
        "first_post_arrival_detach_height_m": off_hand.get("height_m"),
        "first_post_arrival_detach_grip": off_hand.get("grip_action"),
        "first_post_arrival_detach_inside": off_hand.get("target_inside"),
        "gripped_touchdown_before_detach": any(row["gripped_touchdown"] for row in pre_contact),
        "ever_gripped_touchdown": any(row["gripped_touchdown"] for row in trace),
        "positive_grip_after_arrival": any(row["grip"] > 0 for row in after_arrival_cmd),
        "last_goal_error_m": trace[-1]["goal_distance_m"] if trace else None,
        "post_arrival_min_height_m": min((row["height_m"] for row in post_arrival), default=None),
        "post_arrival_max_hold_s": max((row["hold_s"] for row in post_arrival), default=None),
    }


def diagnose(seed, output, extra=None):
    old = archive("binary", seed)
    new = archive(SWITCHED, seed)
    events, substeps, actions = [], [], []
    for scene, focus, transition in discordant(seed, old, new):
        intervention_phase = PHASES[seed]
        probes = transition == "drop_added" or (
            transition == "full_scene_lost" and any(
                row["shape"] == focus and row["transported"] == "True" for row in new[2][scene]
            )
        )
        if extra == "arrival_probes":
            if not probes:
                continue
            modes = ("swap_transport", "swap_before_arrival")
        elif extra == "height_probe":
            if transition != "drop_added":
                continue
            modes = ("raise_clearance",)
        else:
            modes = ("binary", SWITCHED, "swap_binary", "hide_slip") if probes else ("binary", SWITCHED)
        with FailureTrace(focus) as env:
            for mode in modes:
                policy = DiagnosticPolicy(env, mode, old[0], new[0], intervention_phase)
                scene_result, items = run_scene(env, scene, policy)
                if mode in ("binary", SWITCHED):
                    _, archived_scene, archived_items = old if mode == "binary" else new
                    if any(str(value) != archived_scene[scene][key] for key, value in scene_result.items()) or any(
                        str(item[key]) != saved[key] for item, saved in zip(items, archived_items[scene], strict=True)
                        for key in (*OUTCOMES, "shape")
                    ):
                        raise ValueError(f"Archived scene did not reproduce: {seed=}, {scene=}, {mode=}")
                item = next(row for row in items if row["shape"] == focus)
                probed_phase = {"swap_binary": intervention_phase, "hide_slip": intervention_phase,
                                "swap_transport": "transport", "swap_before_arrival": "before_arrival",
                                "raise_clearance": "transport_near_target"}
                meta = {"seed": seed, "scenario_seed": scene, "shape": focus, "transition": transition,
                        "mode": mode, "probed_phase": probed_phase.get(mode, "")}
                events.append({**meta, **{key: item[key] for key in OUTCOMES}, "scene_sorted_items": scene_result["sorted_items"],
                               "all_sorted": scene_result["all_sorted"], **summarize(env.samples, env.commands)})
                substeps.extend({**meta, **row} for row in env.samples)
                actions.extend({**meta, **row} for row in env.commands)
    folder = output / "diagnosis" / f"seed_{seed}"
    if extra:
        folder /= extra
    write_csv(folder / "events.csv", events)
    write_csv(folder / "physics_trace.csv", substeps)
    write_csv(folder / "policy_trace.csv", actions)
    print(f"Seed {seed}: {len(events)} diagnostic replays", flush=True)


def record(output):
    seed = 2
    old, new = archive("binary", seed), archive(SWITCHED, seed)
    scene = next(scene for scene, _, _ in discordant(seed, old, new)
                 if scene == 1_566_001)
    picks = [("binary", seed, old[1][scene]), (SWITCHED, seed, new[1][scene])]
    cases = {(method, seed): ([], data[2][scene]) for method, data in (("binary", old), (SWITCHED, new))}
    folder = output / "diagnosis"
    folder.mkdir(parents=True, exist_ok=True)
    record_cases(folder, cases, picks, lambda method, _: (old if method == "binary" else new)[0],
                 scene_type=scene_type)


def mechanism_summary(output):
    folder = output / "diagnosis"
    rows = []
    for seed in (1, 2):
        root = folder / f"seed_{seed}"
        first_inside = {}
        with (root / "physics_trace.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["mode"] in ("binary", SWITCHED) and float(row["goal_distance_m"]) <= GOAL_RADIUS:
                    first_inside.setdefault((row["scenario_seed"], row["shape"], row["mode"]), row)
        for row in read_rows(root / "events.csv"):
            if row["mode"] in ("binary", SWITCHED):
                inside = first_inside.get((row["scenario_seed"], row["shape"], row["mode"]), {})
                rows.append({**row, "first_inside_height_m": inside.get("height_m"),
                             "first_inside_gripping": inside.get("gripping"),
                             "first_inside_qualified_arrival": inside.get("arrived")})
    write_csv(folder / "mechanisms.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=(1, 2), type=int)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument("--arrival-probes", action="store_const", dest="extra", const="arrival_probes")
    probes.add_argument("--height-probe", action="store_const", dest="extra", const="height_probe")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.videos:
        record(args.out)
    elif args.summarize:
        mechanism_summary(args.out)
    elif args.seed is None:
        parser.error("Provide --seed, --videos, or --summarize")
    elif args.extra == "arrival_probes" and args.seed != 2:
        parser.error("Arrival probes use the seed 2 touchdown regressions")
    elif args.extra == "height_probe" and args.seed != 1:
        parser.error("The height probe uses the seed 1 added drops")
    else:
        diagnose(args.seed, args.out, args.extra)


if __name__ == "__main__":
    main()
