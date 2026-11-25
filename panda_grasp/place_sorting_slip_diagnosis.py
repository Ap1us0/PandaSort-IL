"""Replay paired cube drops from the previous contact ablation and test mechanisms."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .experiment import write_csv
from .place import GOAL_RADIUS
from .place_contact_signals import ContactSortingScene
from .place_objects import OBS_SIZE
from .place_sorting import run_scene
from .place_sorting_analysis import read_rows
from .place_sorting_contact_ablation import ARRIVED_INDEX, LIFTED_INDEX, OUTPUT as PRIOR
from .place_sorting_contact_phase import OUTPUT, load
from .place_transport_diagnosis import trace_summary

MODES = ("binary", "slip", "force", "full_close", "half_xy")
PRELIFT_MODES = ("prelift_binary", "prelift_no_slip")
OUTCOMES = ("lifted", "transported", "transport_drop", "stable_in_target", "final_color", "sorted", "attempt_steps")


class TracedScene(ContactSortingScene):
    """Record transport physics with the learned actions left intact unless probing."""

    def _reset_task(self, options):
        self.samples = []
        self.commands = []
        self.lift_offset = None
        super()._reset_task(options)

    def step(self, action):
        self.applied = np.clip(action, -1, 1)
        return super().step(action)

    def _sample_hold(self, height, contacts):
        super()._sample_hold(height, contacts)
        if self.shape_name != "cube" or not self.episode.ever_carried or (self.samples and self.samples[-1]["drop"]):
            return
        if self.transported and self.samples and self.samples[-1]["arrived"]:
            return
        relative = self.data.xmat[self.arm.hand].reshape(3, 3).T @ (self.cube_xyz - self.tcp)
        if self.lift_offset is None:
            self.lift_offset = relative.copy()
        obstacle = set()
        for contact in self.data.contact:
            if self.cube in contact.geom and contact.efc_address >= 0:
                other = contact.geom[1] if contact.geom[0] == self.cube else contact.geom[0]
                if other != self.table and self.model.geom_bodyid[other] not in self.gripper.bodies:
                    obstacle.add(self.model.geom(other).name)
        self.samples.append({
            "t_s": float(self.data.time), "step": self.episode.steps,
            "height_m": height, "goal_distance_m": self.goal_distance,
            "left_n": contacts.left_normal_n, "right_n": contacts.right_normal_n,
            "gripping": contacts.gripping, "drop": self.transport_drop, "arrived": self.transported,
            "any_finger_contact": bool(contacts.left_count or contacts.right_count),
            "gap_m": float(self.data.qpos[list(self.gripper.fingers)].sum()),
            "supported": contacts.supported, "table_contacts": contacts.table_count,
            "obstacle": "/".join(sorted(obstacle)),
            "slip_z_m": float(relative[2] - self.lift_offset[2]),
            "slip_xy_m": float(np.linalg.norm(relative[:2] - self.lift_offset[:2])),
            "relative_z_m": float(relative[2]), "raw_grip": self.raw_grip,
            "applied_grip": float(self.applied[3]),
            "applied_dx": float(self.applied[0]), "applied_dy": float(self.applied[1]),
            "applied_dz": float(self.applied[2]), "applied_dyaw": float(self.applied[4]),
        })


class DiagnosticActor:
    def __init__(self, actor, mode, env, binary_actor):
        self.actor, self.mode, self.env, self.binary_actor = actor, mode, env, binary_actor

    def action(self, observation):
        cube = self.env.shape_name == "cube"
        lifted = observation[LIFTED_INDEX] > 0.5
        if self.mode == "binary" or (self.mode == "prelift_binary" and cube and not lifted):
            raw = self.binary_actor.action(observation[:OBS_SIZE])
        else:
            if self.mode == "prelift_no_slip" and cube and not lifted:
                observation = observation.copy()
                observation[OBS_SIZE + 2:] = 0
            raw = self.actor.action(observation)
        action = raw.copy()
        self.env.raw_grip = float(raw[3])
        transport = lifted and observation[ARRIVED_INDEX] < 0.5
        if cube and transport and not self.env.transport_drop:
            if self.mode == "full_close":
                action[3] = -1
            elif self.mode == "half_xy":
                action[:2] *= 0.5
            self.env.commands.append({
                "t_s": float(self.env.data.time), "step": self.env.episode.steps + 1,
                "goal_distance_m": self.env.goal_distance,
                **{f"input_{name}": float(value)
                   for name, value in zip(("left_n_over_3", "right_n_over_3", "slip_vx", "slip_vy", "slip_vz",
                                           "slip_dx", "slip_dy", "slip_dz"), observation[OBS_SIZE:], strict=True)},
                **{f"raw_{name}": float(value) for name, value in zip(("dx", "dy", "dz", "grip", "dyaw"), raw)},
                **{f"applied_{name}": float(value) for name, value in zip(("dx", "dy", "dz", "grip", "dyaw"), action)},
            })
        return action


def event_summary(samples):
    summary = trace_summary(samples)
    loss = next((row for row in samples if not row["gripping"] and not row["arrived"]), {})
    departure = next((row for row in samples if not row["any_finger_contact"] and not row["arrived"]), {})
    arrival = next((row for row in samples if row["arrived"]), {})
    drop = next((row for row in samples if row["drop"]), {})
    prior = [row for row in samples if not loss or row["t_s"] < loss["t_s"]]
    peak = max((np.hypot(row["applied_dx"], row["applied_dy"]) for row in prior), default=None)
    return {
        **summary,
        "loss_goal_distance_m": loss.get("goal_distance_m"),
        "loss_finger_gap_m": loss.get("gap_m"),
        "loss_slip_xy_m": loss.get("slip_xy_m"),
        "loss_applied_dz": loss.get("applied_dz"),
        "loss_applied_dyaw": loss.get("applied_dyaw"),
        "full_detach_t_s": departure.get("t_s"),
        "arrival_t_s": arrival.get("t_s"),
        "drop_goal_distance_m": drop.get("goal_distance_m"),
        "max_pre_loss_xy_action": peak,
        "prior_obstacles": "/".join(sorted({name for row in prior for name in row["obstacle"].split("/") if name})),
        "loss_obstacle": loss.get("obstacle"),
        "loss_table_contacts": loss.get("table_contacts"),
        "placed_goal_radius_m": GOAL_RADIUS,
    }


def diagnose(seed, output, prelift=False):
    archive = {}
    for method in ("binary", "force", "slip"):
        p = PRIOR / "test" / method / f"seed_{seed}" / "items.csv"
        archive[method] = {int(r["scenario_seed"]): r for r in read_rows(p) if r["shape"] == "cube"}
    actors = {method: load(method, seed, output)[0] for method in archive}
    events, physics, commands = [], [], []
    for scene, baseline in archive["binary"].items():
        slip = archive["slip"][scene]
        if baseline["transport_drop"] == slip["transport_drop"]:
            continue
        added = slip["transport_drop"] == "True"
        if prelift and not added:
            continue
        modes = PRELIFT_MODES if prelift else (MODES if added else MODES[:3])
        with TracedScene() as env:
            for mode in modes:
                method = mode if mode in actors else "slip"
                actor = DiagnosticActor(actors[method], mode, env, actors["binary"])
                _, items = run_scene(env, scene, actor)
                cube = next(row for row in items if row["shape"] == "cube")
                if mode in actors and any(str(cube[key]) != archive[method][scene][key] for key in OUTCOMES):
                    raise ValueError(f"Replay changed the result for {seed=} {scene=} {mode=}")
                labels = {"seed": seed, "scenario_seed": scene, "mode": mode, "added_slip_drop": added,
                          "binary_lifted": baseline["lifted"] == "True"}
                events.append({**labels, **{key: cube[key] for key in OUTCOMES},
                               "order": cube["order"], "pre_attempt_shift_m": cube["pickup_shift_before_attempt_m"],
                               "ik_failures": cube["ik_failures"], **event_summary(env.samples)})
                physics.extend({**labels, **row} for row in env.samples)
                commands.extend({**labels, **row} for row in env.commands)
    root = output / "diagnosis" / f"seed_{seed}"
    if prelift:
        root /= "prelift"
    fields = dict.fromkeys(key for event in events for key in event)
    write_csv(root / "events.csv", [fields | event for event in events])
    write_csv(root / "physics_trace.csv", physics)
    write_csv(root / "policy_trace.csv", commands)
    print(f"Seed {seed}: {len(events)} verified diagnostic rollouts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(1, 2))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--prelift", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    diagnose(args.seed, args.out, args.prelift)


if __name__ == "__main__":
    main()
