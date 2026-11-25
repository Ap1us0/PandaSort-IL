"""Step 2: 3-frame short-history stacking + EMA policy on the same Round-1 + DAgger datasets."""

import argparse
from collections import deque
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .experiment import write_csv, write_json
from .place_objects import ACTION_NAMES, OBS_SIZE, SHAPES
from .place_objects_train import sha
from .place_contact_signals import TACTILE_SIZE
from .place_sorting import COLOR_BY_SHAPE, TRAYS
from .place_sorting_aligned import (
    ALIGNED_OBS_SIZE,
    AlignedContactSortingScene,
    OUTPUT as ALIGNED_OUTPUT,
    VALIDATION,
)
from .place_sorting_contact_ablation import ARRIVED_INDEX, LIFTED_INDEX
from .place_sorting_contact_schedule import TEST
from .place_sorting_dagger import OUTPUT as DAGGER_OUTPUT

OUTPUT = Path("results/place_sorting_history")
HISTORY = 3
UPDATES = 5000
CHECKPOINTS = (2000, 3500, 5000)
LEARNING_RATE = 5e-4
EMA_DECAY = 0.995


class HistoryAlignedPolicy(nn.Module):
    """256x256 LayerNorm+SiLU policy over K=3 stacked frames with per-frame force_in_transport gating."""

    def __init__(self, frame_size=ALIGNED_OBS_SIZE, history=HISTORY, action_size=len(ACTION_NAMES), width=256):
        super().__init__()
        self.frame_size = frame_size
        self.history = history
        self.net = nn.Sequential(
            nn.Linear(frame_size * history, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, action_size),
        )

    def _mask_frame(self, frame):
        lifted = frame[..., LIFTED_INDEX:LIFTED_INDEX + 1] > 0.5
        arrived = frame[..., ARRIVED_INDEX:ARRIVED_INDEX + 1] > 0.5
        transport = lifted & ~arrived
        tactile = frame[..., OBS_SIZE:TACTILE_SIZE]
        gated = torch.cat((tactile[..., :2] * transport, tactile[..., 2:] * ~transport), dim=-1)
        return torch.cat((frame[..., :OBS_SIZE], gated, frame[..., TACTILE_SIZE:]), dim=-1)

    def logits(self, stacked_obs):
        # stacked_obs: (..., history, frame_size) or (..., history * frame_size)
        if stacked_obs.ndim == 2 and stacked_obs.shape[-1] == self.history * self.frame_size:
            frames = stacked_obs.view(-1, self.history, self.frame_size)
        elif stacked_obs.ndim == 1 and stacked_obs.shape[-1] == self.history * self.frame_size:
            frames = stacked_obs.view(1, self.history, self.frame_size)
        else:
            frames = stacked_obs
        masked = self._mask_frame(frames).reshape(frames.shape[0], self.history * self.frame_size)
        return self.net(masked)

    def forward(self, stacked_obs):
        return self.logits(stacked_obs).tanh()

    @torch.no_grad()
    def action(self, stacked_obs):
        out = self(torch.as_tensor(stacked_obs, dtype=torch.float32))
        return out[0].numpy() if out.ndim == 2 else out.numpy()


def load_stacked_examples(path, history=HISTORY):
    """Build K-frame stacked observations per episode (zero-padding via first-frame repeat at t=0)."""
    with np.load(path) as data:
        obs = data["observations"]
        act = data["expert_actions"]
        phase = data["phase"]
        offsets = data["offsets"]
        successful = data["successful"]

    stacked_x, stacked_y, stacked_p = [], [], []
    for ep_idx, ok in enumerate(successful):
        if not ok:
            continue
        start, end = int(offsets[ep_idx]), int(offsets[ep_idx + 1])
        ep_obs = obs[start:end]
        ep_act = act[start:end]
        ep_phase = phase[start:end]
        n = len(ep_obs)
        if n == 0:
            continue
        # Build sliding window of length `history` with edge-repeat at start
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


def run_history_scene(env, seed, actor, history=HISTORY):
    obs, _ = env.reset(seed=seed)
    items = []
    homing_steps = 0
    for index, name in enumerate(env.order):
        returned_home = True
        if index:
            steps, returned_home = env.return_home()
            homing_steps += steps
            obs = env.select(name)
        initial_shift = float(np.linalg.norm(env.cube_xyz[:2] - np.asarray(env.plan[name].xy)))
        buf = deque([obs.copy() for _ in range(history)], maxlen=history)
        while True:
            stacked = np.concatenate(list(buf), axis=0)
            action = actor.action(stacked)
            obs, _, terminated, truncated, info = env.step(action)
            buf.append(obs.copy())
            if terminated or truncated:
                break
        assignment = env.plan[name]
        items.append({
            "scenario_seed": seed, "order": index + 1, "shape": name, "expected_color": COLOR_BY_SHAPE[name],
            "target_slot": assignment.slot, "pickup_x": assignment.xy[0], "pickup_y": assignment.xy[1],
            "target_x": float(TRAYS[assignment.slot, 0]), "target_y": float(TRAYS[assignment.slot, 1]),
            "initial_yaw_deg": math.degrees(assignment.yaw), "returned_home": returned_home,
            "pickup_shift_before_attempt_m": initial_shift,
            "attempt_steps": env.episode.steps, "stable_in_target": info["stable_placement_success"],
            "hold_s": info["outcome_stable_s"], "failure_stage": info["failure_stage"],
            "lifted": info["ever_carried"], "transported": info["transported"], "transport_drop": info["transport_drop"],
            "terminal_error_m": info["goal_distance_m"], "ik_failures": info["ik_failures"],
        })
    slot_colors = {assignment.slot: COLOR_BY_SHAPE[name] for name, assignment in env.plan.items()}
    for row in items:
        slot = env.final_tray(row["shape"])
        row["final_color"] = slot_colors.get(slot, "outside")
        row["wrong_color"] = slot is not None and row["final_color"] != row["expected_color"]
        row["sorted"] = row["stable_in_target"] and row["final_color"] == row["expected_color"]
    scene = {
        "scenario_seed": seed, "order": "/".join(env.order),
        "tray_colors": "/".join(slot_colors[i] for i in range(len(TRAYS))),
        "sorted_items": sum(row["sorted"] for row in items), "all_sorted": all(row["sorted"] for row in items),
        "wrong_color_items": sum(row["wrong_color"] for row in items), "homing_steps": homing_steps,
        "homing_failures": sum(not row["returned_home"] for row in items),
    }
    return scene, items


def validate_history(env, actor):
    actor.eval()
    scenes = items = drops = lifts = 0
    sorted_by_shape = dict.fromkeys(SHAPES, 0)
    for seed in VALIDATION:
        result, rows = run_history_scene(env, seed, actor)
        scenes += result["all_sorted"]
        items += result["sorted_items"]
        drops += sum(row["transport_drop"] for row in rows)
        lifts += sum(row["lifted"] for row in rows)
        for row in rows:
            sorted_by_shape[row["shape"]] += row["sorted"]
    return {
        "all_sorted": scenes, "items_sorted": items, "transport_drops": drops, "lifted": lifts,
        **{f"{name}_sorted": value for name, value in sorted_by_shape.items()},
    }


def train_history(seed, aligned_root, dagger_root, output):
    r1_paths = [aligned_root / "training" / f"seed_{seed}" / "round1_demos" / name / "expert.npz" for name in SHAPES]
    dg_paths = [
        p for name in SHAPES
        if (p := dagger_root / "training" / f"seed_{seed}" / "dagger_demos" / name / "dagger.npz").exists()
    ]
    r1_datasets = [load_stacked_examples(p) for p in r1_paths]
    dg_datasets = [load_stacked_examples(p) for p in dg_paths]

    torch.manual_seed(seed + 568270)
    model = HistoryAlignedPolicy()
    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES, eta_min=LEARNING_RATE * 0.1)
    gen = torch.Generator().manual_seed(seed + 573270)

    root = output / "training" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    evaluations, best = [], None

    with AlignedContactSortingScene() as env:
        for step in range(1, UPDATES + 1):
            model.train()
            obs_list, tgt_list = [], []
            # 85% from Round-1 full trajectories (3 x 96 = 288)
            for x, y, groups in r1_datasets:
                count, extra = divmod(64, len(groups))
                bal = [g[torch.randint(len(g), (count + int(i < extra),), generator=gen)] for i, g in enumerate(groups)]
                ids = torch.cat([torch.randint(len(x), (32,), generator=gen), *bal])
                obs_list.append(x[ids])
                tgt_list.append(y[ids])
            # 15% from verified DAgger takeover suffixes (3 x 16 = 48)
            for x, y, groups in dg_datasets:
                count, extra = divmod(12, len(groups))
                bal = [g[torch.randint(len(g), (count + int(i < extra),), generator=gen)] for i, g in enumerate(groups)]
                ids = torch.cat([torch.randint(len(x), (4,), generator=gen), *bal])
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

            with torch.no_grad():
                for p_ema, p_cur in zip(ema_model.parameters(), model.parameters(), strict=True):
                    p_ema.mul_(EMA_DECAY).add_(p_cur, alpha=1.0 - EMA_DECAY)

            if step in CHECKPOINTS:
                res = validate_history(env, ema_model)
                rank = (res["all_sorted"], res["items_sorted"], -res["transport_drops"], -step)
                if best is None or rank > best:
                    best = rank
                    torch.save(ema_model.state_dict(), root / "model.pt")
                evaluations.append({"updates": step, "xyz_mse": float(xyz.detach()), "gripper_bce": float(grip.detach()), **res})
                write_csv(root / "validation.csv", evaluations)
                print(f"Seed {seed} [Hist3+EMA u={step}]: {res['all_sorted']}/{len(VALIDATION)} full, "
                      f"{res['items_sorted']}/{3 * len(VALIDATION)} items, drops {res['transport_drops']} "
                      f"(c={res['cube_sorted']}, b={res['brick_sorted']}, y={res['cylinder_sorted']})",
                      flush=True)

    write_json(root / "config.json", {
        "seed": seed, "history": HISTORY, "ema_decay": EMA_DECAY,
        "checkpoint_sha256": sha(root / "model.pt"), "selected_updates": -best[-1],
        "new_demonstration_scenes": 0,
        "validation_scenarios": list(VALIDATION), "test_scenarios": list(TEST),
    })


def evaluate_history(seed, output):
    root = output / "training" / f"seed_{seed}"
    weights = root / "model.pt"
    identity = sha(weights)
    actor = HistoryAlignedPolicy()
    actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    actor.eval()
    directory = output / "test" / f"seed_{seed}"
    scenes, objects = [], []
    with AlignedContactSortingScene() as env:
        for scenario in TEST:
            row, items = run_history_scene(env, scenario, actor)
            scenes.append(row)
            objects.extend(items)
    summary = {
        "method": "history3_ema", "seed": seed, "checkpoint_sha256": identity,
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
    parser.add_argument("--dagger", type=Path, default=DAGGER_OUTPUT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        train_history(args.seed, args.aligned, args.dagger, args.out)
    else:
        evaluate_history(args.seed, args.out)


if __name__ == "__main__":
    main()
