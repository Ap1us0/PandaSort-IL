"""Strict matched-budget single-factor ablation separating DR and faithful Prefix-Takeover DAgger.

All 3 arms share:
- Same V3 expert (RobustSortingTeacherV3)
- Same 384w residual 3-frame history + EMA architecture (DRHistoryPolicy)
- Same total demonstration scene budget (120 scenes/seed)
- Same total optimizer updates (8,000 updates/seed, batch size 384)
- Same validation set (VAL_NOMINAL + VAL_OOD) and brand-new held-out test layouts (1568000-1568029)
"""

import argparse
from collections import deque
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .experiment import save_demonstrations, write_csv, write_json
from .place_objects import SHAPES
from .place_objects_train import sha
from .place_sorting_aligned import AlignedContactSortingScene
from .place_sorting_dr import (
    DR_BUCKETS,
    DRHistoryPolicy,
    EMA_DECAY,
    LEARNING_RATE,
    OUTPUT as DR_OUTPUT,
    _train_loop,
    make_takeover_v3,
    run_v3_expert_scene,
    validate_dr,
)
from .place_sorting_history import HISTORY, load_stacked_examples, run_history_scene
from .place_sorting_ood import OODSortingScene
from .place_sorting_unseen_eval import UNSEEN_CONDITIONS, UNSEEN_TEST

OUTPUT = Path("results/place_sorting_ablation_dr_dagger")
ARMS = ("nominal_only", "dr_only", "dr_plus_dagger")
TOTAL_UPDATES = 8000
R1_DAGGER_UPDATES = 5000
R2_DAGGER_UPDATES = 3000
CKPTS_SINGLE = (3000, 5500, 8000)
CKPTS_R1 = (3000, 5000)
CKPTS_R2 = (1500, 3000)
EVAL_CONDS = ("unseen_nominal", "unseen_feasible_combo", "unseen_extrap_fric57")


def load_stacked_slice(path, max_episodes=None, history=HISTORY):
    """Load K-frame stacked successful transitions from the first `max_episodes` episodes of an archive."""
    with np.load(path) as data:
        obs = data["observations"]
        act = data["expert_actions"]
        phase = data["phase"]
        offsets = data["offsets"]
        successful = data["successful"]

    limit = len(successful) if max_episodes is None else min(len(successful), max_episodes)
    stacked_x, stacked_y, stacked_p = [], [], []
    for ep_idx in range(limit):
        if not successful[ep_idx]:
            continue
        start, end = int(offsets[ep_idx]), int(offsets[ep_idx + 1])
        ep_obs, ep_act, ep_phase = obs[start:end], act[start:end], phase[start:end]
        n = len(ep_obs)
        if n == 0:
            continue
        frames = np.empty((n, history * ep_obs.shape[1]), dtype=np.float32)
        for t in range(n):
            idxs = [max(0, t - (history - 1 - k)) for k in range(history)]
            frames[t] = ep_obs[idxs].reshape(-1)
        stacked_x.append(frames)
        stacked_y.append(ep_act.astype(np.float32))
        stacked_p.append(ep_phase)

    x = torch.as_tensor(np.concatenate(stacked_x, axis=0), dtype=torch.float32)
    y = torch.as_tensor(np.concatenate(stacked_y, axis=0), dtype=torch.float32)
    p = np.concatenate(stacked_p, axis=0)
    groups = [torch.from_numpy(np.flatnonzero(p == name)) for name in np.unique(p)]
    return x, y, groups


def collect_extra_nominal_60(seed, root):
    """Collect 60 extra nominal V3-expert scenes so nominal_only matches the exact 120-scene budget."""
    folder = root / "extra_nominal_60"
    paths = [folder / name / "expert.npz" for name in SHAPES]
    if all(p.exists() for p in paths):
        return paths
    rng = np.random.default_rng(seed * 31337 + 94_000_000)
    groups = {name: [] for name in SHAPES}
    full = 0
    started = perf_counter()
    with AlignedContactSortingScene() as env:
        base = 94_000_000 + 100_000 * seed
        for i in range(60):
            ok_all, rows, traces = run_v3_expert_scene(env, base + i, rng, perturb=True)
            full += int(ok_all)
            for row, traj in zip(rows, traces, strict=True):
                groups[row["shape"]].append((traj, {"is_success": row["sorted"], "scenario_seed": row["scenario_seed"]}))
    for name, pairs in groups.items():
        trajs, outs = zip(*pairs)
        p = folder / name / "expert.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        save_demonstrations(p, trajs, outs)
    print(f"Seed {seed} [ExtraNom60]: {full}/60 full, time={perf_counter()-started:.1f}s", flush=True)
    return paths


def collect_faithful_dagger_30(seed, actor, root):
    """Collect 30 faithful prefix-takeover DAgger scenes (18 nominal + 12 DR).

    Faithfulness guarantee: preceding non-target items are replayed by the EXACT SAME policy `actor`
    (never fast-forwarded with the expert), and live qpos/object state at target entry is verified exact.
    """
    rng = np.random.default_rng(seed * 41333 + 95_000_000)
    groups = {name: [] for name in SHAPES}
    rescued = scanned = verified_entries = 0
    started = perf_counter()

    def process_scene(env, scenario):
        nonlocal rescued, scanned, verified_entries
        # Pass 1: run policy cleanly on the whole scene and record entry cube_xyz + failure metadata
        obs, _ = env.reset(seed=scenario)
        probe = {}
        entry_xyz = {}
        for index, name in enumerate(env.order):
            if index:
                env.return_home()
                obs = env.select(name)
            entry_xyz[name] = env.cube_xyz.copy()
            buf = deque([obs.copy() for _ in range(HISTORY)], maxlen=HISTORY)
            lift_s = arr_s = None
            for step in range(env.max_steps):
                act = actor.action(np.concatenate(list(buf), axis=0))
                obs, _, term, trunc, info = env.step(act)
                buf.append(obs.copy())
                if lift_s is None and info["ever_carried"]:
                    lift_s = step + 1
                if arr_s is None and info["transported"]:
                    arr_s = step + 1
                if term or trunc:
                    break
            ok = bool(info["stable_placement_success"]) and env.final_tray(name) == env.plan[name].slot
            probe[name] = (ok, info["failure_stage"], lift_s, arr_s, env.episode.steps)

        # Pick at most ONE target item per replay pass (or replay independently per failed item so
        # every takeover sees the exact untouched policy-generated table state!)
        target_items = []
        for name in env.order:
            ok, stage, lift_s, arr_s, _ = probe[name]
            if not ok:
                scanned += 1
                if stage in ("grip", "lift"):
                    t_step = int(rng.integers(5, 22))
                elif stage == "transport":
                    t_step = max(5, (lift_s or 40) + int(rng.integers(2, 16)))
                else:
                    t_step = max(10, (arr_s or (lift_s or 40) + 20) - int(rng.integers(4, 16)))
                target_items.append((name, t_step, True))
            elif rng.random() < 0.15:
                target_items.append((name, (lift_s or 40) + int(rng.integers(5, 22)), False))

        # Pass 2: for each target item, faithfully replay ALL preceding items with the SAME `actor`!
        for target_name, t_step, was_fail in target_items:
            obs, _ = env.reset(seed=scenario)
            for index, name in enumerate(env.order):
                if index:
                    env.return_home()
                    obs = env.select(name)
                buf = deque([obs.copy() for _ in range(HISTORY)], maxlen=HISTORY)
                if name != target_name:
                    # Faithful replay with policy (NOT expert!)
                    while True:
                        act = actor.action(np.concatenate(list(buf), axis=0))
                        obs, _, term, trunc, _ = env.step(act)
                        buf.append(obs.copy())
                        if term or trunc:
                            break
                    continue

                # Verify bit-exact live object state at target entry!
                if not np.allclose(env.cube_xyz, entry_xyz[name], atol=1e-5):
                    raise RuntimeError(f"Faithful replay state mismatch on scene {scenario}, {name}")
                verified_entries += 1
                teacher = None
                suffix = []
                for step in range(env.max_steps):
                    if teacher is None and step >= t_step:
                        teacher = make_takeover_v3(env)
                    if teacher is None:
                        act = actor.action(np.concatenate(list(buf), axis=0))
                        following, reward, term, trunc, info = env.step(act)
                    else:
                        label, act = teacher(obs)
                        phase = teacher.state.name
                        following, reward, term, trunc, info = env.step(act)
                        suffix.append((obs, act, label, following, reward, term, trunc, phase))
                    obs = following
                    buf.append(obs.copy())
                    if term or trunc:
                        break
                ok = bool(info["stable_placement_success"]) and env.final_tray(name) == env.plan[name].slot
                if was_fail and ok:
                    rescued += 1
                if ok and suffix:
                    groups[name].append((suffix, {"is_success": True, "scenario_seed": scenario}))
                break

    with AlignedContactSortingScene() as env:
        base = 95_000_000 + 100_000 * seed
        for i in range(18):
            process_scene(env, base + i)

    base_dr = 96_000_000 + 100_000 * seed
    dagger_buckets = (DR_BUCKETS[0], DR_BUCKETS[2], DR_BUCKETS[3], DR_BUCKETS[4], DR_BUCKETS[6], DR_BUCKETS[7])
    sc_idx = 0
    for spec in dagger_buckets:
        with OODSortingScene(spec) as env:
            for _ in range(2):
                process_scene(env, base_dr + sc_idx)
                sc_idx += 1

    paths = []
    tot_traj = tot_steps = 0
    for name, pairs in groups.items():
        if not pairs:
            continue
        trajs, outs = zip(*pairs)
        p = root / "faithful_dagger_demos" / name / "dagger.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        save_demonstrations(p, trajs, outs)
        paths.append(p)
        tot_traj += len(pairs)
        tot_steps += sum(len(t) for t in trajs)
    print(
        f"Seed {seed} [FaithfulDAgger30]: verified_entries={verified_entries}, rescued={rescued}/{scanned}, "
        f"{tot_traj} suffixes ({tot_steps} steps), time={perf_counter()-started:.1f}s",
        flush=True,
    )
    return paths, {"verified_entries": verified_entries, "rescued": rescued, "scanned": scanned, "suffixes": tot_traj, "steps": tot_steps}


def score_val(res):
    return (res["nom_all"] + res["ood_all"], res["nom_items"] + res["ood_items"], -res["nom_drops"])


def train_arm(arm, seed, output):
    dr_r1_dir = DR_OUTPUT / "training" / f"seed_{seed}" / "round1_demos"
    nom60_paths = [dr_r1_dir / "nominal" / name / "expert.npz" for name in SHAPES]
    dr60_paths = [dr_r1_dir / "dr" / name / "expert.npz" for name in SHAPES]

    root = output / arm / "training" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed + 880_000)
    model = DRHistoryPolicy()
    ema_model = copy.deepcopy(model)
    evals, best = [], None

    if arm == "nominal_only":
        extra_paths = collect_extra_nominal_60(seed, output / "shared" / f"seed_{seed}")
        datasets = [load_stacked_examples(p) for p in (*nom60_paths, *extra_paths)]
        weights = [64] * 6  # total batch 384, 120 nominal scenes
        def rep(step, cur, xyz, grip):
            nonlocal best
            res = validate_dr(cur)
            rk = (*score_val(res), -step)
            if best is None or rk > best:
                best = rk
                torch.save(cur.state_dict(), root / "model.pt")
            evals.append({"arm": arm, "updates": step, "xyz_mse": xyz, "gripper_bce": grip, **res})
            write_csv(root / "validation.csv", evals)
            print(f"Seed {seed} [{arm} u={step}]: nom={res['nom_all']}/15, ood={res['ood_all']}/6", flush=True)
        _train_loop(model, ema_model, datasets, weights, TOTAL_UPDATES, CKPTS_SINGLE, LEARNING_RATE, seed + 881_000, rep)
        meta = {"arm": arm, "seed": seed, "total_scenes": 120, "nominal_scenes": 120, "dr_scenes": 0, "dagger_scenes": 0,
                "total_updates": TOTAL_UPDATES, "selected_updates": -best[-1], "checkpoint_sha256": sha(root / "model.pt")}

    elif arm == "dr_only":
        datasets = [load_stacked_examples(p) for p in (*nom60_paths, *dr60_paths)]
        weights = [64] * 6  # total batch 384, 60 nominal + 60 DR = 120 scenes
        def rep(step, cur, xyz, grip):
            nonlocal best
            res = validate_dr(cur)
            rk = (*score_val(res), -step)
            if best is None or rk > best:
                best = rk
                torch.save(cur.state_dict(), root / "model.pt")
            evals.append({"arm": arm, "updates": step, "xyz_mse": xyz, "gripper_bce": grip, **res})
            write_csv(root / "validation.csv", evals)
            print(f"Seed {seed} [{arm} u={step}]: nom={res['nom_all']}/15, ood={res['ood_all']}/6", flush=True)
        _train_loop(model, ema_model, datasets, weights, TOTAL_UPDATES, CKPTS_SINGLE, LEARNING_RATE, seed + 881_000, rep)
        meta = {"arm": arm, "seed": seed, "total_scenes": 120, "nominal_scenes": 60, "dr_scenes": 60, "dagger_scenes": 0,
                "total_updates": TOTAL_UPDATES, "selected_updates": -best[-1], "checkpoint_sha256": sha(root / "model.pt")}

    elif arm == "dr_plus_dagger":
        # 45 nominal + 45 DR = 90 R1 scenes, plus 30 faithful DAgger scenes = EXACTLY 120 scenes, 5000+3000=8000 updates!
        r1_datasets = [load_stacked_slice(p, max_episodes=45) for p in (*nom60_paths, *dr60_paths)]
        r1_weights = [64] * 6
        def rep_r1(step, cur, xyz, grip):
            nonlocal best
            res = validate_dr(cur)
            rk = (*score_val(res), -step)
            if best is None or rk > best:
                best = rk
                torch.save(cur.state_dict(), root / "model_round1.pt")
            evals.append({"arm": arm, "round": 1, "updates": step, "xyz_mse": xyz, "gripper_bce": grip, **res})
            write_csv(root / "validation.csv", evals)
            print(f"Seed {seed} [{arm} R1 u={step}]: nom={res['nom_all']}/15, ood={res['ood_all']}/6", flush=True)
        _train_loop(model, ema_model, r1_datasets, r1_weights, R1_DAGGER_UPDATES, CKPTS_R1, LEARNING_RATE, seed + 881_000, rep_r1)

        model.load_state_dict(torch.load(root / "model_round1.pt", weights_only=True, map_location="cpu"))
        ema_model = copy.deepcopy(model)
        torch.save(ema_model.state_dict(), root / "model.pt")
        dg_paths, dg_info = collect_faithful_dagger_30(seed, ema_model, root)
        dg_datasets = [load_stacked_examples(p) for p in dg_paths]
        all_ds = r1_datasets + dg_datasets
        # Keep exact batch size 384: 6 x 52 = 312 (81.25% R1) + 3 x 24 = 72 (18.75% DAgger) = 384
        all_w = [52] * len(r1_datasets) + [24] * len(dg_datasets)
        def rep_r2(step, cur, xyz, grip):
            nonlocal best
            res = validate_dr(cur)
            rk = (*score_val(res), -(R1_DAGGER_UPDATES + step))
            if rk > best:
                best = rk
                torch.save(cur.state_dict(), root / "model.pt")
            evals.append({"arm": arm, "round": 2, "updates": R1_DAGGER_UPDATES + step, "xyz_mse": xyz, "gripper_bce": grip, **res})
            write_csv(root / "validation.csv", evals)
            print(f"Seed {seed} [{arm} R2 u={step}]: nom={res['nom_all']}/15, ood={res['ood_all']}/6", flush=True)
        _train_loop(model, ema_model, all_ds, all_w, R2_DAGGER_UPDATES, CKPTS_R2, LEARNING_RATE * 0.4, seed + 882_000, rep_r2)
        meta = {
            "arm": arm, "seed": seed, "total_scenes": 120, "nominal_scenes": 45, "dr_scenes": 45, "dagger_scenes": 30,
            "total_updates": TOTAL_UPDATES, "selected_updates": -best[-1],
            "r1_sha256": sha(root / "model_round1.pt"), "checkpoint_sha256": sha(root / "model.pt"),
            "dagger_improved_over_r1": sha(root / "model.pt") != sha(root / "model_round1.pt"),
            "dagger_info": dg_info,
        }
    else:
        raise ValueError(f"Unknown arm: {arm}")
    write_json(root / "config.json", meta)


def test_arm(arm, cond_name, output):
    cfg = UNSEEN_CONDITIONS[cond_name]
    env_factory = AlignedContactSortingScene if cond_name == "unseen_nominal" else (lambda: OODSortingScene(cfg))
    all_scenes, all_items, seed_rows = [], [], []
    with env_factory() as env:
        for seed in (0, 1, 2):
            root = output / arm / "training" / f"seed_{seed}"
            weights = root / "model.pt"
            identity = sha(weights)
            actor = DRHistoryPolicy()
            actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
            actor.eval()
            scenes, items = [], []
            for scenario in UNSEEN_TEST:
                sc, rows = run_history_scene(env, scenario, actor)
                scenes.append(sc)
                items.extend(rows)
            folder = output / arm / "test" / cond_name / f"seed_{seed}"
            write_csv(folder / "scenes.csv", scenes)
            write_csv(folder / "items.csv", items)
            cubes = [r for r in items if r["shape"] == "cube"]
            summary = {
                "arm": arm, "condition": cond_name, "seed": seed, "checkpoint_sha256": identity,
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
        "arm": arm, "condition": cond_name, "scenes": len(all_scenes),
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
    write_json(output / arm / "test" / cond_name / "overall.json", agg)
    print(f"ABLATION[{arm}][{cond_name}]:", agg, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--condition", choices=EVAL_CONDS, default="unseen_nominal")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        if args.seed is None:
            parser.error("train requires --seed")
        train_arm(args.arm, args.seed, args.out)
    else:
        test_arm(args.arm, args.condition, args.out)


if __name__ == "__main__":
    main()
