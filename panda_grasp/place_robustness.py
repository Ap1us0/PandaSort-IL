"""Frozen placement-policy evaluation under single-variable physics shifts."""

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ._vendor.pickup import State
from .experiment import evaluate, write_csv, write_json
from .learning import Clone
from .physics_shift import CONDITIONS, apply_physics_condition
from .place import GOAL_RADIUS, GRIP_OPEN_M, PLACE_OBS_SIZE, REST_TOLERANCE_M, PandaPickPlaceEnv
from .place_cli import clone_policy, metrics
from .place_expert import PlacementExpert

STRESS_CONDITIONS = ("nominal", "friction_50", "mass_250")
SCREEN_SCENES = range(1_410_000, 1_410_012)
TEST_SCENES = range(1_420_000, 1_420_100)
OUTPUT = Path("results/place_robustness")


class ShiftedPlacementEnv(PandaPickPlaceEnv):
    """Keep the success predicate and log failures at physical milestones."""

    def __init__(self, condition, **kwargs):
        super().__init__(**kwargs)
        self.condition = condition
        self.mass_kg, self.cube_sliding = apply_physics_condition(self, condition)

    def _reset_task(self, options):
        super()._reset_task(options)
        self.transport_drop = False
        self.table_supported_at_goal = False
        self.open_without_gripped_touchdown = False
        self.transport_lost_contact = False

    def _sample_hold(self, height, contacts):
        super()._sample_hold(height, contacts)
        if not self.episode.ever_carried:
            return
        no_finger_contact = not (contacts.left_count or contacts.right_count)
        if not self.transported:
            self.transport_lost_contact |= not contacts.gripping
            self.transport_drop |= bool(no_finger_contact and (contacts.supported or height < 0.025))
        at_goal_table = (self.goal_distance <= GOAL_RADIUS and contacts.supported
                         and abs(height) <= REST_TOLERANCE_M)
        self.table_supported_at_goal |= at_goal_table
        if self.transported and not self.supported_at_target and no_finger_contact:
            opened = np.sum(self.data.qpos[list(self.gripper.fingers)]) >= GRIP_OPEN_M
            self.open_without_gripped_touchdown |= bool(opened)

    def _info(self, contacts, status):
        data = super()._info(contacts, status)
        milestones = (
            ("grip", self.episode.ever_bilateral), ("lift", self.episode.ever_carried),
            ("transport", self.transported), ("touchdown", self.supported_at_target),
            ("release", self.released_at_target),
        )
        stage = next((name for name, reached in milestones if not reached), "settle")
        data.update(
            failure_stage="success" if data["is_success"] else stage,
            transport_drop=bool(self.transport_drop),
            transport_lost_contact=bool(self.transport_lost_contact),
            table_supported_at_goal=bool(self.table_supported_at_goal),
            open_without_gripped_touchdown=bool(self.open_without_gripped_touchdown),
        )
        return data


class TransportTeacher(PlacementExpert):
    """Allow slower lift and transport demonstrators when planning data collection."""

    def __init__(self, env, transit_scale=1.0):
        super().__init__(env)
        self.transit_scale = transit_scale

    def __call__(self, obs):
        label, action = super().__call__(obs)
        if self.state in (State.LIFT, State.TRANSPORT):
            label[:3] *= self.transit_scale
            action[:3] *= self.transit_scale
        return label, action


def teacher_policy(transit_scale):
    def factory(env, seed):
        teacher = TransportTeacher(env, transit_scale)
        return lambda obs: teacher(obs)[1]
    return factory


def load_clone(path):
    model = Clone(PLACE_OBS_SIZE)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def placement_metrics(rows):
    lifted = sum(row["ever_carried"] for row in rows)
    dropped = sum(row["transport_drop"] for row in rows)
    result = metrics(rows)
    result.update(
        touchdown_rate=float(np.mean([row["supported_at_target"] for row in rows])),
        transport_drop_rate=dropped / len(rows),
        transport_drops_per_lift=dropped / lifted if lifted else None,
        transport_grip_loss_rate=float(np.mean([row["transport_lost_contact"] for row in rows])),
        table_at_goal_rate=float(np.mean([row["table_supported_at_goal"] for row in rows])),
        early_open_rate=float(np.mean([row["open_without_gripped_touchdown"] for row in rows])),
        median_goal_error_m=float(np.median([row["goal_distance_m"] for row in rows])),
        failure_stages=dict(Counter(row["failure_stage"] for row in rows if not row["is_success"])),
    )
    return result


def run_evaluation(env, policy, scenes, directory, extra=None):
    rows = evaluate(env, policy, scenes)
    result = {**(extra or {}), **placement_metrics(rows)}
    write_csv(directory / "episodes.csv", rows)
    write_json(directory / "summary.json", result)
    print(
        f"{directory}: {result['successes']}/{len(rows)}, transport={result['transport_rate']:.0%}, "
        f"drop={result['transport_drop_rate']:.0%}, stages={result['failure_stages']}",
        flush=True,
    )
    return result


def grip_probe(policy):
    """On arrival, retain the grip until a physically verified touchdown."""

    def factory(env, seed):
        predict = policy(env, seed)

        def choose(observation):
            action = predict(observation)
            if env.transported and not env.supported_at_target:
                action = action.copy()
                action[3] = -1.0
            return action

        return choose

    return factory



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("screen", "test"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--conditions", nargs="+", choices=STRESS_CONDITIONS, default=STRESS_CONDITIONS)
    parser.add_argument("--teacher-speed", type=float, default=1.0)
    parser.add_argument("--probe-grip", action="store_true")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoints", type=Path, default=Path("results/pick_place"))
    args = parser.parse_args()
    if not 0 < args.teacher_speed <= 1 or (args.seed is not None and args.teacher_speed != 1):
        parser.error("Teacher speed must be in (0, 1] and applies only to the expert")
    if args.probe_grip and (args.seed is None or args.split != "test"
                            or args.conditions != ["friction_50"]):
        parser.error("Use --probe-grip with test, a training seed and --conditions friction_50")
    torch.set_num_threads(1)
    name = f"seed_{args.seed}" if args.seed is not None else (
        "expert" if args.teacher_speed == 1 else f"expert_speed_{args.teacher_speed:g}"
    )
    if args.seed is None:
        policy = teacher_policy(args.teacher_speed)
    else:
        path = args.checkpoints / name / "model.pt"
        policy = clone_policy(load_clone(path))
    split = "grip_probe" if args.probe_grip else args.split
    scenes = SCREEN_SCENES if args.split == "screen" else TEST_SCENES
    if args.probe_grip:
        scenes = TEST_SCENES[::4]
        policy = grip_probe(policy)
    for label in args.conditions:
        with ShiftedPlacementEnv(CONDITIONS[label]) as env:
            run_evaluation(env, policy, scenes, args.out / split / name / label, {
                "condition": label, "seed": args.seed, "teacher_speed": args.teacher_speed,
                **asdict(CONDITIONS[label]), "mass_kg": env.mass_kg, "sliding_coefficient": env.cube_sliding,
            })


if __name__ == "__main__":
    main()
