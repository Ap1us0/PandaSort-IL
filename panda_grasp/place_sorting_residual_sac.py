"""Online Residual SAC adaptation (0 new expert demos) on OOD low-friction 3-object sorting."""

import argparse
from collections import deque
import json
import math
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch

from .experiment import write_csv, write_json
from .place_objects_train import sha
from .place_sorting import COLOR_BY_SHAPE, TRAYS
from .place_sorting_aligned import ALIGNED_OBS_SIZE
from .place_sorting_contact_ablation import ARRIVED_INDEX, LIFTED_INDEX
from .place_sorting_contact_schedule import TEST
from .place_sorting_history import HISTORY, HistoryAlignedPolicy, OUTPUT as HISTORY_OUTPUT
from .place_sorting_ood import CONDITIONS, OODSortingScene
from .sac_ablation import Ablation, AblationSAC

OUTPUT = Path("results/place_sorting_residual_sac")
RESIDUAL_SCALE = np.array([0.12, 0.12, 0.25, 0.08], dtype=np.float32)  # surgical transport-phase correction
VAL_SCENES = range(1_562_000, 1_562_012)
ONLINE_STEPS = 8_000
CRITIC_WARMUP = 1_500
EVAL_EVERY = 2_000


def apply_residual(base_action, res_action, obs_frame=None, transport_only=True):
    if transport_only and obs_frame is not None:
        in_transport = (float(obs_frame[LIFTED_INDEX]) > 0.5) and (float(obs_frame[ARRIVED_INDEX]) <= 0.5)
        if not in_transport:
            return base_action.copy()
    cmd = base_action.copy()
    cmd[0] += RESIDUAL_SCALE[0] * res_action[0]
    cmd[1] += RESIDUAL_SCALE[1] * res_action[1]
    cmd[2] += RESIDUAL_SCALE[2] * res_action[2]
    cmd[4] += RESIDUAL_SCALE[3] * res_action[3]
    return np.clip(cmd, -1.0, 1.0).astype(np.float32)


class ResidualCubeSortingEnv(gym.Env):
    """Live 3-object OOD sorting scene exposing the cube attempt for transport-gated residual RL."""

    def __init__(self, condition_name, base_actor, train_seed_base, transport_only=True):
        super().__init__()
        self.base_env = OODSortingScene(CONDITIONS[condition_name])
        self.base_actor = base_actor
        self.train_seed_base = train_seed_base
        self.transport_only = transport_only
        self.ep_count = 0
        self.buf = None
        self.last_obs = np.zeros(ALIGNED_OBS_SIZE, dtype=np.float32)
        self.last_base_action = np.zeros(5, dtype=np.float32)
        self.prev_dropped = False
        # Compact 69-dim obs: current 64-dim frame + frozen 3-frame base policy's 5-dim action
        obs_dim = ALIGNED_OBS_SIZE + 5
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def _pack_obs(self):
        stacked = np.concatenate(list(self.buf), axis=0)
        self.last_obs = self.buf[-1]
        self.last_base_action = self.base_actor.action(stacked).astype(np.float32)
        return np.concatenate((self.last_obs, self.last_base_action), axis=0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = self.train_seed_base + self.ep_count
        self.ep_count += 1
        obs, _ = self.base_env.reset(seed=scenario)
        for index, name in enumerate(self.base_env.order):
            if index:
                self.base_env.return_home()
                obs = self.base_env.select(name)
            if name == "cube":
                self.buf = deque([obs.copy() for _ in range(HISTORY)], maxlen=HISTORY)
                self.prev_dropped = False
                return self._pack_obs(), {"scenario_seed": scenario}
            tmp_buf = deque([obs.copy() for _ in range(HISTORY)], maxlen=HISTORY)
            while True:
                act = self.base_actor.action(np.concatenate(list(tmp_buf), axis=0))
                obs, _, term, trunc, _ = self.base_env.step(act)
                tmp_buf.append(obs.copy())
                if term or trunc:
                    break
        raise RuntimeError("Cube not found in sorting scene order")

    def step(self, res_action):
        res_action = np.clip(np.asarray(res_action, dtype=np.float32), -1.0, 1.0)
        cmd = apply_residual(self.last_base_action, res_action, self.last_obs, transport_only=self.transport_only)
        obs, reward, terminated, truncated, info = self.base_env.step(cmd)
        self.buf.append(obs.copy())

        # Pure task reward (no hover-farming): native potential + terminal sorted bonus - drop penalty - L2 anchor
        shaped = float(reward)
        if info["transport_drop"] and not self.prev_dropped:
            shaped -= 4.0
            self.prev_dropped = True
        if terminated and info["stable_placement_success"] and self.base_env.contains("cube", self.base_env.plan["cube"].slot):
            shaped += 10.0
        in_transport = (float(self.last_obs[LIFTED_INDEX]) > 0.5) and (float(self.last_obs[ARRIVED_INDEX]) <= 0.5)
        if in_transport or not self.transport_only:
            shaped -= 0.05 * float(np.sum(res_action ** 2))
        return self._pack_obs(), shaped, terminated, truncated, info

    def close(self):
        self.base_env.close()


def run_full_scene_with_residual(env, seed, base_actor, sac_model=None, transport_only=True):
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
        buf = deque([obs.copy() for _ in range(HISTORY)], maxlen=HISTORY)
        while True:
            stacked = np.concatenate(list(buf), axis=0)
            base_act = base_actor.action(stacked).astype(np.float32)
            if sac_model is not None and name == "cube":
                rl_obs = np.concatenate((buf[-1], base_act), axis=0).astype(np.float32)
                res_act, _ = sac_model.predict(rl_obs, deterministic=True)
                action = apply_residual(base_act, res_act, buf[-1], transport_only=transport_only)
            else:
                action = base_act
            obs, _, term, trunc, info = env.step(action)
            buf.append(obs.copy())
            if term or trunc:
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


def evaluate_val(eval_env, base_actor, sac_model, transport_only=True):
    all_ok = cube_ok = drops = 0
    for sc in VAL_SCENES:
        scene, rows = run_full_scene_with_residual(eval_env, sc, base_actor, sac_model, transport_only=transport_only)
        all_ok += int(scene["all_sorted"])
        c = next(r for r in rows if r["shape"] == "cube")
        cube_ok += int(c["sorted"])
        drops += int(c["transport_drop"])
    return {"val_all": all_ok, "val_cube": cube_ok, "val_cube_drops": drops}


class ResidualEvalCallback(BaseCallback):
    def __init__(self, eval_env, base_actor, root, seed, transport_only=True):
        super().__init__()
        self.eval_env = eval_env
        self.base_actor = base_actor
        self.root = root
        self.seed = seed
        self.transport_only = transport_only
        self.rows = []
        self.best = None
        self.next_eval = 0

    def _check(self):
        step = self.num_timesteps
        res = evaluate_val(self.eval_env, self.base_actor, self.model if step > 0 else None, transport_only=self.transport_only)
        rank = (res["val_cube"], res["val_all"], -res["val_cube_drops"], -step)
        if self.best is None or rank > self.best:
            self.best = rank
            self.model.save(self.root / "best_model")
        self.rows.append({"online_steps": step, **res})
        write_csv(self.root / "validation.csv", self.rows)
        print(f"Seed {self.seed} [GatedResSAC step={step}]: val_all={res['val_all']}/{len(VAL_SCENES)}, "
              f"val_cube={res['val_cube']}/{len(VAL_SCENES)}, drops={res['val_cube_drops']}", flush=True)

    def _on_training_start(self):
        self._check()
        self.next_eval = EVAL_EVERY

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self._check()
            self.next_eval += EVAL_EVERY
        return True


def train_residual(seed, condition_name, history_root, output):
    weights = history_root / "training" / f"seed_{seed}" / "model.pt"
    base_actor = HistoryAlignedPolicy()
    base_actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    base_actor.eval()

    root = output / condition_name / "training" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    with ResidualCubeSortingEnv(condition_name, base_actor, 93_000_000 + 100_000 * seed, transport_only=True) as env, \
            OODSortingScene(CONDITIONS[condition_name]) as eval_env:
        model = AblationSAC(
            "MlpPolicy", env, seed=seed + 930_000, device="cpu",
            learning_rate=2e-4, buffer_size=20_000, batch_size=128, learning_starts=400,
            train_freq=4, gradient_steps=2, gamma=0.99, tau=0.005, ent_coef=0.02,
            policy_kwargs={"net_arch": [128, 128]},
        )
        # Critic warmup for 1,500 steps + gentle exploration scale 0.35 so early rollouts stay near base policy
        model.ablation = Ablation(critic_warmup_steps=CRITIC_WARMUP, behavior_std_scale=0.35)
        with torch.no_grad():
            model.actor.mu.weight.mul_(0.01)
            model.actor.mu.bias.zero_()
            model.actor.log_std.weight.zero_()
            model.actor.log_std.bias.fill_(-2.5)
        model.set_logger(configure(str(root), ["csv"]))
        cb = ResidualEvalCallback(eval_env, base_actor, root, seed, transport_only=True)
        model.learn(total_timesteps=ONLINE_STEPS, callback=cb)
        model.save(root / "final_model")

    write_json(root / "config.json", {
        "seed": seed, "condition": condition_name, "transport_only": True,
        "critic_warmup": CRITIC_WARMUP, "base_sha256": sha(weights),
        "online_steps": ONLINE_STEPS, "new_expert_demos": 0,
        "selected_step": -cb.best[-1], "training_s": perf_counter() - started,
    })


def test_residual(seed, condition_name, history_root, output, checkpoint="selected"):
    weights = history_root / "training" / f"seed_{seed}" / "model.pt"
    base_actor = HistoryAlignedPolicy()
    base_actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    base_actor.eval()

    train_dir = output / condition_name / "training" / f"seed_{seed}"
    cfg = json.loads((train_dir / "config.json").read_text(encoding="utf-8"))
    if checkpoint == "selected":
        if cfg["selected_step"] == 0:
            sac_model = None
            evaluated_label = "val_selected_step_0_frozen_base"
        else:
            sac_model = AblationSAC.load(train_dir / "best_model", device="cpu")
            evaluated_label = f"val_selected_step_{cfg['selected_step']}"
        folder = output / condition_name / "test_selected" / f"seed_{seed}"
    else:
        sac_model = AblationSAC.load(train_dir / "final_model", device="cpu")
        evaluated_label = f"final_step_{cfg['online_steps']}"
        folder = output / condition_name / "test_final" / f"seed_{seed}"

    scenes, items = [], []
    with OODSortingScene(CONDITIONS[condition_name]) as env:
        for sc in TEST:
            scene, rows = run_full_scene_with_residual(env, sc, base_actor, sac_model, transport_only=True)
            scenes.append(scene)
            items.extend(rows)

    write_csv(folder / "scenes.csv", scenes)
    write_csv(folder / "items.csv", items)
    cubes = [r for r in items if r["shape"] == "cube"]
    summary = {
        "method": "gated_residual_sac", "condition": condition_name, "seed": seed,
        "checkpoint_mode": checkpoint, "evaluated_checkpoint": evaluated_label,
        "selected_step": cfg["selected_step"], "scenes": len(scenes),
        "all_sorted": sum(r["all_sorted"] for r in scenes),
        "sorted_items": sum(r["sorted"] for r in items),
        "cube_sorted": sum(r["sorted"] for r in cubes),
        "cube_lifted": sum(r["lifted"] for r in cubes),
        "cube_transport_drops": sum(r["transport_drop"] for r in cubes),
        "brick_sorted": sum(r["shape"] == "brick" and r["sorted"] for r in items),
        "cylinder_sorted": sum(r["shape"] == "cylinder" and r["sorted"] for r in items),
    }
    write_json(folder / "summary.json", summary)
    print(f"GatedResSAC-TEST[{checkpoint}]:", summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), default="friction_60")
    parser.add_argument("--checkpoint", choices=("selected", "final"), default="selected")
    parser.add_argument("--history", type=Path, default=HISTORY_OUTPUT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        train_residual(args.seed, args.condition, args.history, args.out)
    else:
        test_residual(args.seed, args.condition, args.history, args.out, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
