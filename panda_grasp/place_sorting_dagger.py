"""Step 1: Consistent prefix-takeover DAgger on top of Round-1 AlignedPolicy."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ._vendor.pickup import State, measure_contacts
from .experiment import save_demonstrations, write_csv, write_json
from .place_objects import SHAPES
from .place_objects_train import sha
from .place_sorting import COLOR_BY_SHAPE, TRAYS
from .place_sorting_aligned import (
    AlignedContactSortingScene,
    AlignedPolicy,
    AlignedSortingTeacher,
    OUTPUT as ALIGNED_OUTPUT,
    VALIDATION,
    _load_examples,
    run_aligned_scene,
    validate,
)
from .place_sorting_contact_schedule import TEST

OUTPUT = Path("results/place_sorting_dagger")
DAGGER_SCENES = 60
DAGGER_UPDATES = 3000
CHECKPOINTS = (1000, 2000, 3000)
LEARNING_RATE = 2e-4


def make_takeover_teacher(env):
    """Initialize the exact AlignedSortingTeacher aligned to the live physical state."""
    teacher = AlignedSortingTeacher(env, brick_long_axis=True, perturb=False)
    contacts = measure_contacts(env, env.data)
    height = float(env.cube_xyz[2] - env.rest_z)
    if env.supported_at_target:
        teacher.state = State.RETREAT
    elif contacts.gripping and not contacts.table_count:
        if env.transported:
            teacher.state = State.PLACE
        elif height >= 0.052 or env.episode.ever_carried:
            teacher.state = State.TRANSPORT
        else:
            teacher.state = State.LIFT
    elif contacts.gripping:
        teacher.state = State.LIFT
    else:
        teacher.state = State.APPROACH
    return teacher


def find_policy_failures(env, seed, actor):
    """Run Round-1 policy on a training scene and record which items fail and key phase steps."""
    obs, _ = env.reset(seed=seed)
    plan_info = {}
    for index, name in enumerate(env.order):
        if index:
            env.return_home()
            obs = env.select(name)
        lift_step = None
        arrive_step = None
        for step in range(env.max_steps):
            action = actor.action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if lift_step is None and info["ever_carried"]:
                lift_step = step + 1
            if arrive_step is None and info["transported"]:
                arrive_step = step + 1
            if terminated or truncated:
                break
        ok = bool(info["stable_placement_success"]) and env.final_tray(name) == env.plan[name].slot
        plan_info[name] = {
            "sorted": ok,
            "failure_stage": info["failure_stage"],
            "steps": env.episode.steps,
            "lift_step": lift_step,
            "arrive_step": arrive_step,
        }
    return plan_info


def choose_takeover_step(meta, rng):
    """Choose a takeover step right before the policy's failure phase (or random phase for coverage)."""
    stage = meta["failure_stage"]
    if not meta["sorted"]:
        if stage in ("grip", "lift"):
            return int(rng.integers(5, 25))
        if stage == "transport":
            if meta["lift_step"] is not None:
                return max(5, meta["lift_step"] + int(rng.integers(2, 18)))
            return int(rng.integers(30, 60))
        if stage in ("touchdown", "release", "settle"):
            if meta["arrive_step"] is not None:
                return max(10, meta["arrive_step"] - int(rng.integers(5, 20)))
            if meta["lift_step"] is not None:
                return max(10, meta["lift_step"] + 20)
            return 80
        return max(10, meta["steps"] // 2)
    # For already-succeeding items, occasionally collect a mid-transport/pre-place recovery prefix
    if meta["lift_step"] is not None:
        return meta["lift_step"] + int(rng.integers(5, 25))
    return int(rng.integers(15, 45))


def collect_prefix_takeover(seed, actor, directory):
    """Collect verified expert recovery suffixes from Round-1 policy prefixes."""
    rng = np.random.default_rng(seed * 7919 + 87_300_000)
    start_seed = 87_300_000 + 100_000 * seed
    groups = {name: [] for name in SHAPES}
    records_rows = []
    started = perf_counter()
    scanned_fails = 0
    rescued = 0

    with AlignedContactSortingScene() as env:
        for scenario in range(start_seed, start_seed + DAGGER_SCENES):
            probe = find_policy_failures(env, scenario, actor)
            # Determine which items to collect takeover suffixes for:
            # 1) ALL failed items, plus 2) 20% of succeeded items for stability
            targets = {}
            for name, meta in probe.items():
                if not meta["sorted"]:
                    scanned_fails += 1
                    targets[name] = (choose_takeover_step(meta, rng), True)
                elif rng.random() < 0.20:
                    targets[name] = (choose_takeover_step(meta, rng), False)
            if not targets:
                continue

            # Replay the scene: run policy until takeover step on target items, then exact teacher
            obs, _ = env.reset(seed=scenario)
            for index, name in enumerate(env.order):
                if index:
                    env.return_home()
                    obs = env.select(name)
                if name not in targets:
                    # Fast-forward with expert so subsequent items see clean table state
                    teacher = AlignedSortingTeacher(env, brick_long_axis=True)
                    while True:
                        _, act = teacher(obs)
                        obs, _, term, trunc, _ = env.step(act)
                        if term or trunc:
                            break
                    continue

                takeover_step, was_fail = targets[name]
                teacher = None
                suffix = []
                for step in range(env.max_steps):
                    if teacher is None and step >= takeover_step:
                        teacher = make_takeover_teacher(env)
                    if teacher is None:
                        executed = actor.action(obs)
                        following, reward, term, trunc, info = env.step(executed)
                    else:
                        label, executed = teacher(obs)
                        phase = teacher.state.name
                        following, reward, term, trunc, info = env.step(executed)
                        suffix.append((obs, executed, label, following, reward, term, trunc, phase))
                    obs = following
                    if term or trunc:
                        break
                ok = bool(info["stable_placement_success"]) and env.final_tray(name) == env.plan[name].slot
                if was_fail and ok:
                    rescued += 1
                if ok and suffix:
                    groups[name].append((suffix, {"is_success": True, "scenario_seed": scenario, "shape": name}))
                records_rows.append({
                    "scenario_seed": scenario, "shape": name, "was_fail": was_fail,
                    "takeover_step": takeover_step, "suffix_steps": len(suffix),
                    "rescued_ok": ok, "failure_stage": info["failure_stage"],
                })

    paths, meta_shapes = [], []
    for name, pairs in groups.items():
        if not pairs:
            continue
        trajectories, outcomes = zip(*pairs)
        path = directory / name / "dagger.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_demonstrations(path, trajectories, outcomes)
        paths.append(path)
        meta_shapes.append({
            "shape": name, "trajectories": len(pairs),
            "steps": sum(len(t) for t in trajectories), "sha256": sha(path),
        })
    write_csv(directory / "takeover_log.csv", records_rows)
    meta = {
        "seed": seed, "scanned_scenes": DAGGER_SCENES, "scanned_policy_failures": scanned_fails,
        "rescued_failures": rescued, "collection_s": perf_counter() - started,
        "shape_data": meta_shapes,
    }
    write_json(directory / "dataset.json", meta)
    print(f"Seed {seed} [Takeover-DAgger]: scanned {DAGGER_SCENES} scenes, rescued {rescued}/{scanned_fails} fails, "
          f"collected {sum(s['trajectories'] for s in meta_shapes)} suffixes ({sum(s['steps'] for s in meta_shapes)} steps)",
          flush=True)
    return paths, meta


def train_dagger(seed, aligned_root, output):
    r1_dir = aligned_root / "training" / f"seed_{seed}"
    r1_weights = r1_dir / "model_round1.pt"
    r1_paths = [r1_dir / "round1_demos" / name / "expert.npz" for name in SHAPES]

    model = AlignedPolicy()
    model.load_state_dict(torch.load(r1_weights, weights_only=True, map_location="cpu"))
    model.eval()

    root = output / "training" / f"seed_{seed}"
    dagger_paths, d_meta = collect_prefix_takeover(seed, model, root / "dagger_demos")

    r1_datasets = [_load_examples(p) for p in r1_paths]
    dg_datasets = [_load_examples(p) for p in dagger_paths]

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DAGGER_UPDATES, eta_min=LEARNING_RATE * 0.1)
    gen = torch.Generator().manual_seed(seed + 368270)

    evaluations = []
    with AlignedContactSortingScene() as env:
        base_res = validate(env, model)
        best = (base_res["all_sorted"], base_res["items_sorted"], -base_res["transport_drops"], 0)
        torch.save(model.state_dict(), root / "model.pt")
        evaluations.append({"updates": 0, "xyz_mse": None, "gripper_bce": None, **base_res})
        print(f"Seed {seed} [DAgger u=0]: {base_res['all_sorted']}/{len(VALIDATION)} full, "
              f"{base_res['items_sorted']}/{3 * len(VALIDATION)} items, drops {base_res['transport_drops']} "
              f"(c={base_res['cube_sorted']}, b={base_res['brick_sorted']}, y={base_res['cylinder_sorted']})",
              flush=True)

        for step in range(1, DAGGER_UPDATES + 1):
            model.train()
            obs_list, tgt_list = [], []
            # 75% batch from Round-1 full trajectories (3 shapes x 96 = 288)
            for x, y, groups in r1_datasets:
                count, extra = divmod(64, len(groups))
                bal = [g[torch.randint(len(g), (count + int(i < extra),), generator=gen)] for i, g in enumerate(groups)]
                ids = torch.cat([torch.randint(len(x), (32,), generator=gen), *bal])
                obs_list.append(x[ids])
                tgt_list.append(y[ids])
            # 25% batch from verified DAgger takeover suffixes (up to 32 per shape = 96)
            for x, y, groups in dg_datasets:
                count, extra = divmod(24, len(groups))
                bal = [g[torch.randint(len(g), (count + int(i < extra),), generator=gen)] for i, g in enumerate(groups)]
                ids = torch.cat([torch.randint(len(x), (8,), generator=gen), *bal])
                obs_list.append(x[ids])
                tgt_list.append(y[ids])
            obs_b = torch.cat(obs_list)
            tgt_b = torch.cat(tgt_list)
            logits = model.logits(obs_b)
            xyz = F.mse_loss(logits[:, :3].tanh(), tgt_b[:, :3])
            grip = F.binary_cross_entropy_with_logits(2 * logits[:, 3], (1 + tgt_b[:, 3]) / 2)
            yaw = F.mse_loss(logits[:, 4:].tanh(), tgt_b[:, 4:])
            loss = xyz + 0.18 * grip + 0.35 * yaw
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step in CHECKPOINTS:
                res = validate(env, model)
                rank = (res["all_sorted"], res["items_sorted"], -res["transport_drops"], -step)
                if rank > best:
                    best = rank
                    torch.save(model.state_dict(), root / "model.pt")
                evaluations.append({"updates": step, "xyz_mse": float(xyz.detach()), "gripper_bce": float(grip.detach()), **res})
                write_csv(root / "validation.csv", evaluations)
                print(f"Seed {seed} [DAgger u={step}]: {res['all_sorted']}/{len(VALIDATION)} full, "
                      f"{res['items_sorted']}/{3 * len(VALIDATION)} items, drops {res['transport_drops']} "
                      f"(c={res['cube_sorted']}, b={res['brick_sorted']}, y={res['cylinder_sorted']})",
                      flush=True)

    write_json(root / "config.json", {
        "seed": seed, "initial_sha256": sha(r1_weights), "checkpoint_sha256": sha(root / "model.pt"),
        "selected_updates": -best[-1], "dagger_meta": d_meta,
        "validation_scenarios": list(VALIDATION), "test_scenarios": list(TEST),
    })


def evaluate_dagger(seed, output):
    root = output / "training" / f"seed_{seed}"
    weights = root / "model.pt"
    identity = sha(weights)
    actor = AlignedPolicy()
    actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    actor.eval()
    directory = output / "test" / f"seed_{seed}"
    scenes, objects = [], []
    with AlignedContactSortingScene() as env:
        for scenario in TEST:
            row, items = run_aligned_scene(env, scenario, actor)
            scenes.append(row)
            objects.extend(items)
    summary = {
        "method": "dagger", "seed": seed, "checkpoint_sha256": identity,
        "scenes": len(scenes), "all_sorted": sum(r["all_sorted"] for r in scenes),
        "items": len(objects), "sorted_items": sum(r["sorted"] for r in objects),
        "wrong_color_items": sum(r["wrong_color"] for r in objects),
        "homing_failures": sum(r["homing_failures"] for r in scenes),
        "per_shape": {name: sum(row["shape"] == name and row["sorted"] for row in objects) for name in SHAPES},
        "cube_lifted": sum(row["shape"] == "cube" and row["lifted"] for row in objects),
        "cube_transport_drops": sum(row["shape"] == "cube" and row["transport_drop"] for row in objects),
    }
    write_csv(directory / "scenes.csv", scenes)
    write_csv(directory / "items.csv", objects)
    write_json(directory / "summary.json", summary)
    print(directory, summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--aligned", type=Path, default=ALIGNED_OUTPUT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        train_dagger(args.seed, args.aligned, args.out)
    else:
        evaluate_dagger(args.seed, args.out)


if __name__ == "__main__":
    main()
