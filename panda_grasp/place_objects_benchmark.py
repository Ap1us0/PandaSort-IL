"""Screen the geometry/yaw benchmark using archived BC and physical teachers."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ._vendor.pickup import State, measure_contacts

from .experiment import evaluate, write_csv, write_json
from .place_cli import clone_policy
from .place_objects import ObjectPlacementEnv, SHAPES, YawTeacher
from .place_robustness import load_clone
from .place_touchdown import build_teacher

OUTPUT = Path("results/place_objects")
SCREEN = range(1_550_000, 1_550_016)
TEST = {shape: range(1_552_000 + 1000 * i, 1_552_036 + 1000 * i)
        for i, shape in enumerate(SHAPES)}
SCREEN_GATE = 13
BRICK_CONFIRM = range(1_550_500, 1_550_516)
SCREEN_CASES = ("frozen_zero", "frozen_random", "fixed_teacher_random", "yaw_teacher_random")


def source_policy(results, seed):
    root = results / "place_transport_anchor" / f"seed_{seed}"
    weights = root / "model.pt"
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    identity = hashlib.sha256(weights.read_bytes()).hexdigest()
    if config["model_sha256"] != identity:
        raise ValueError(f"Checkpoint does not match archived hash: {weights}")
    return weights, identity


def yaw_teacher(env, scene):
    actor = YawTeacher(env)
    return lambda obs: actor(obs)[1]


def brick_long_teacher(env, scene):
    actor = YawTeacher(env, brick_long_axis=True)
    return lambda obs: actor(obs)[1]


def screen_brick(output):
    for name, scenes in (("paired_screen", SCREEN), ("confirmation", BRICK_CONFIRM)):
        evaluate_cell(output / "brick_axis" / name, "brick", True, True, brick_long_teacher, scenes,
                      method="brick_long_axis_teacher")


def probe_grasp_force(output):
    rows = []
    for long_axis in (False, True):
        with ObjectPlacementEnv("brick", True, True) as env:
            for scene in SCREEN:
                obs, _ = env.reset(seed=scene)
                expert = YawTeacher(env, brick_long_axis=long_axis)
                for _ in range(150):
                    if expert.state == State.GRASP and measure_contacts(env, env.data).gripping:
                        break
                    _, action = expert(obs)
                    obs, _, term, trunc, _ = env.step(action)
                    if term or trunc:
                        break
                forces, gaps = [], []
                closed = expert.state == State.GRASP and measure_contacts(env, env.data).gripping
                if closed:
                    for count in range(8):
                        env.step([0, 0, 0, -1, 0])
                        contact = measure_contacts(env, env.data)
                        if count >= 4:
                            forces.append(min(contact.left_normal_n, contact.right_normal_n))
                            gaps.append(float(env.data.qpos[list(env.gripper.fingers)].sum()))
                rows.append({"scenario_seed": scene, "long_axis": long_axis, "bilateral_at_probe": closed,
                             "median_weaker_finger_force_n": float(np.median(forces)) if forces else None,
                             "median_finger_gap_m": float(np.median(gaps)) if gaps else None})
    write_csv(output / "brick_axis" / "static_grip_probe.csv", rows)


def fixed_teacher(env, scene):
    actor = build_teacher(env, "verified_touch")
    return lambda obs: actor(obs)[1]


def evaluate_cell(directory, shape, random_yaw, yaw_action, policy, scenes, **method):
    with ObjectPlacementEnv(shape, random_yaw, yaw_action) as env:
        rows = evaluate(env, policy, scenes)
        if any(row["is_success"] and not row["stable_placement_success"] for row in rows):
            raise ValueError(f"Inconsistent strict success in {directory}")
        summary = {
            **method, "shape": shape, "random_initial_yaw": random_yaw, "wrist_yaw_action": yaw_action,
            "episodes": len(rows), "stable": sum(row["stable_placement_success"] for row in rows),
            "strict": sum(row["is_success"] for row in rows),
            "gripped": sum(row["ever_bilateral"] for row in rows),
            "lifted": sum(row["ever_carried"] for row in rows),
            "transported": sum(row["transported"] for row in rows),
            "transport_drops": sum(row["transport_drop"] for row in rows),
            "failure_stages": dict(Counter(row["failure_stage"] for row in rows if not row["stable_placement_success"])),
            "mass_g": round(1000 * env.mass_kg, 6), "finger_sliding_coefficient": env.cube_sliding,
        }
    write_csv(directory / "episodes.csv", rows)
    write_json(directory / "summary.json", summary)
    print(f"{directory}: stable {summary['stable']}/{summary['episodes']}, "
          f"lift {summary['lifted']}, drops {summary['transport_drops']}", flush=True)


def screen(results, output, shape):
    weights, identity = source_policy(results, 1)
    frozen = clone_policy(load_clone(weights))
    policies = ((frozen, False, False), (frozen, True, False), (fixed_teacher, True, False), (yaw_teacher, True, True))
    for name, (policy, random_yaw, yaw_action) in zip(SCREEN_CASES, policies):
        metadata = {"method": name}
        if name.startswith("frozen"):
            metadata.update(seed=1, checkpoint_sha256=identity)
        evaluate_cell(output / "screen" / shape / name, shape, random_yaw, yaw_action, policy, SCREEN, **metadata)


def select(output):
    summary = []
    for shape in SHAPES:
        for method in SCREEN_CASES:
            path = output / "screen" / shape / method / "summary.json"
            summary.append(json.loads(path.read_text(encoding="utf-8")))
    shapes = [r["shape"] for r in summary if r["method"] == "yaw_teacher_random" and r["stable"] >= SCREEN_GATE]
    teacher_modes = {shape: "default" for shape in SHAPES}
    confirmation = json.loads((output / "brick_axis" / "confirmation" / "summary.json").read_text(encoding="utf-8"))
    if "brick" not in shapes and confirmation["stable"] >= SCREEN_GATE:
        shapes.insert(list(SHAPES).index("brick"), "brick")
        teacher_modes["brick"] = "long_axis"
    write_json(output / "screen_summary.json", summary)
    write_json(output / "selection.json", {
        "minimum_expert_stable": SCREEN_GATE, "screen_scenarios": list(SCREEN),
        "test_scenarios": {shape: list(scenes) for shape, scenes in TEST.items()}, "selected_shapes": shapes,
        "brick_confirmation_scenarios": list(BRICK_CONFIRM), "teacher_modes": teacher_modes,
    })
    print(f"Eligible for mixed BC: {shapes}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "select", "screen-brick", "probe-grip"))
    parser.add_argument("--shape", choices=tuple(SHAPES))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "screen":
        if args.shape is None:
            parser.error("screen requires --shape")
        screen(args.results, args.out, args.shape)
    elif args.command == "screen-brick":
        screen_brick(args.out)
    elif args.command == "probe-grip":
        probe_grasp_force(args.out)
    else:
        select(args.out)


if __name__ == "__main__":
    main()
