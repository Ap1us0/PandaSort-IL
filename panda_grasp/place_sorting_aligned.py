"""Aligned sorting expert, 256x256 force-in-transport policy, and 1-round DAgger on 3-object sorting."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._vendor.pickup import State, measure_contacts
from .env import ACTION_STEP
from .experiment import save_demonstrations, write_csv, write_json
from .place_contact_signals import CONTACTS, ContactSortingScene, TACTILE_SIZE
from .place_objects import ACTION_NAMES, OBS_SIZE, SHAPES, YAW_STEP, YawTeacher
from .place_objects_train import sha
from .place_sorting import COLOR_BY_SHAPE, TRAYS
from .place_sorting_contact_ablation import ARRIVED_INDEX, LIFTED_INDEX
from .place_sorting_contact_schedule import TEST

OUTPUT = Path("results/place_sorting_aligned")
ROUND1_SCENES = 60
DAGGER_SCENES = 36
VALIDATION = range(1_562_000, 1_562_015)
ROUND1_UPDATES = 4500
ROUND2_UPDATES = 4500
CHECKPOINTS = (1500, 3000, 4500)
LEARNING_RATE = 5e-4
CONTEXT_SIZE = 4
ALIGNED_OBS_SIZE = TACTILE_SIZE + CONTEXT_SIZE


class AlignedSortingTeacher(YawTeacher):
    """Deeper grasp (-3mm), gentle lift, fast transport, tray-aligned yaw, with optional perturbation."""

    def __init__(self, env, brick_long_axis=True, rng=None, perturb=False):
        self.grasp_z_offset = -0.003
        self.transport_h = 0.078
        self.transport_limit = 1.0
        self.descend_steps = 0
        self.perturb_rng = rng
        self.perturb_active = perturb
        super().__init__(env, brick_long_axis=brick_long_axis)

    def _set_targets(self):
        super()._set_targets()
        self.descend = self.descend.copy()
        self.descend[2] += self.grasp_z_offset

    def _target_yaw_for_state(self, state):
        if state in (State.TRANSPORT, State.PLACE, State.RETREAT):
            if self.env.shape_name == "cube":
                return 0.0
            if self.env.shape_name == "brick":
                return 0.0 if abs(self.grasp_yaw) <= math.pi / 4 else math.copysign(math.pi / 2, self.grasp_yaw)
        return self.grasp_yaw

    def _perturb_action(self, label, state):
        if not self.perturb_active or self.perturb_rng is None:
            return label.copy()
        if state not in (State.APPROACH, State.DESCEND, State.TRANSPORT, State.PLACE):
            return label.copy()
        executed = label.copy()
        scale = 0.09 if state in (State.APPROACH, State.TRANSPORT) else 0.05
        executed[:3] += self.perturb_rng.normal(0.0, scale, size=3)
        executed[4] += self.perturb_rng.normal(0.0, 0.05)
        return np.clip(executed, -1.0, 1.0).astype(np.float32)

    def __call__(self, observation):
        env = self.env
        tcp = env.tcp
        target_yaw = self._target_yaw_for_state(self.state)
        yaw_error = target_yaw - env.tool_yaw
        yaw_action = float(np.clip(yaw_error / YAW_STEP, -1, 1))

        if self.state in (State.APPROACH, State.DESCEND, State.GRASP, State.LIFT):
            if self.state == State.APPROACH and abs(yaw_error) > 0.05:
                delta = (self.approach - tcp) / ACTION_STEP
                delta *= min(1.0, 1.0 / max(1.0, np.linalg.norm(delta)))
                label = np.asarray([*delta, 1.0, yaw_action], dtype=np.float32)
                return label, self._perturb_action(label, self.state)
            if self.state == State.DESCEND:
                self.descend_steps += 1
                if np.linalg.norm(self.descend - tcp) < 0.0035 or self.descend_steps >= 45:
                    self.state = State.GRASP
            if self.state == State.LIFT and (
                env.cube_xyz[2] - env.rest_z >= 0.058
                or tcp[2] >= env.rest_z + self.transport_h - 0.010
            ):
                self.state = State.TRANSPORT
            else:
                label, _ = super(YawTeacher, self).__call__(observation)
                out = np.concatenate((label, np.asarray([yaw_action], dtype=np.float32)))
                return out, self._perturb_action(out, self.state)

        if self.state == State.TRANSPORT and np.linalg.norm(env.target[:2] - env.cube_xyz[:2]) < 0.006:
            self.state = State.PLACE
        if self.state == State.PLACE and self._ready_to_retreat():
            self.state = State.RETREAT

        height = {State.TRANSPORT: self.transport_h, State.PLACE: -0.001, State.RETREAT: 0.10}[self.state]
        target = env.target + np.array([0.0, 0.0, height])
        if self.state != State.RETREAT:
            target[:2] += tcp[:2] - env.cube_xyz[:2]
        delta = (target - tcp) / ACTION_STEP
        limit = 0.70 if self.state == State.PLACE else self.transport_limit
        delta *= min(1.0, limit / max(limit, np.linalg.norm(delta)))
        label = np.array([*delta, 1.0 if self.state == State.RETREAT else -1.0, yaw_action], dtype=np.float32)
        return label, self._perturb_action(label, self.state)

    def relabel(self):
        """Stateless Markovian expert label for DAgger on arbitrary policy-visited states."""
        env = self.env
        tcp = env.tcp
        contacts = measure_contacts(env, env.data)
        height = float(env.cube_xyz[2] - env.rest_z)
        goal_xy = float(np.linalg.norm(env.target[:2] - env.cube_xyz[:2]))
        obj_xy = float(np.linalg.norm(env.cube_xyz[:2] - tcp[:2]))

        if env.supported_at_target:
            state = State.RETREAT
        elif contacts.gripping and not contacts.table_count:
            if env.transported or (height >= 0.052 and goal_xy < 0.007):
                state = State.PLACE
            elif height >= 0.055 or env.episode.ever_carried:
                state = State.TRANSPORT
            else:
                state = State.LIFT
        elif contacts.gripping:
            state = State.LIFT
        else:
            # Refresh grasp_yaw from live object_yaw in case object was nudged
            jaw_axis = env.home_rotation[:, 1]
            home_angle = math.atan2(jaw_axis[1], jaw_axis[0])
            period = math.pi / 2 if env.shape_name == "cube" else math.pi
            axis = env.object_yaw
            self.grasp_yaw = 0.0 if env.shape.cylinder else (axis - home_angle + period / 2) % period - period / 2
            yaw_err_pre = abs(self.grasp_yaw - env.tool_yaw)
            descend_z = env.cube_xyz[2] + self.grasp_z_offset
            if obj_xy < 0.014 and abs(tcp[2] - descend_z) < 0.009 and yaw_err_pre < 0.14:
                state = State.GRASP
            elif obj_xy < 0.016 and yaw_err_pre < 0.12:
                state = State.DESCEND
            else:
                state = State.APPROACH

        target_yaw = self._target_yaw_for_state(state)
        yaw_action = float(np.clip((target_yaw - env.tool_yaw) / YAW_STEP, -1.0, 1.0))

        if state == State.APPROACH:
            target = env.cube_xyz + np.array([0.0, 0.0, 0.085])
            limit, grip = 0.85, 1.0
        elif state == State.DESCEND:
            target = env.cube_xyz + np.array([0.0, 0.0, self.grasp_z_offset])
            limit, grip = 0.60, 1.0
        elif state == State.GRASP:
            target = env.cube_xyz + np.array([0.0, 0.0, self.grasp_z_offset])
            limit, grip = 0.60, -1.0
        elif state == State.LIFT:
            target = np.array([env.cube_xyz[0], env.cube_xyz[1], env.rest_z + self.transport_h + 0.005])
            limit, grip = 0.60, -1.0
        elif state == State.TRANSPORT:
            target = env.target + np.array([0.0, 0.0, self.transport_h])
            target[:2] += tcp[:2] - env.cube_xyz[:2]
            limit, grip = self.transport_limit, -1.0
        elif state == State.PLACE:
            target = env.target + np.array([0.0, 0.0, -0.001])
            target[:2] += tcp[:2] - env.cube_xyz[:2]
            limit, grip = 0.70, -1.0
        else:  # RETREAT
            target = env.target + np.array([0.0, 0.0, 0.10])
            limit, grip = 0.85, 1.0

        delta = (target - tcp) / ACTION_STEP
        delta *= min(1.0, limit / max(limit, np.linalg.norm(delta)))
        label = np.array([*delta, grip, yaw_action], dtype=np.float32)
        return label, state.name


class AlignedContactSortingScene(ContactSortingScene):
    """64-dim observation: 52 base + 8 contact signals + unfolded (cos, sin) object yaw + nearest obstacle XY."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(ALIGNED_OBS_SIZE,), dtype=np.float32)

    def _context_features(self):
        tcp_xy = self.tcp[:2]
        nearest = np.array([0.5, 0.5], dtype=np.float64)
        best_dist = float("inf")
        for name, (geom, _) in self.objects.items():
            if name == self.shape_name:
                continue
            rel = self.data.geom_xpos[geom, :2] - tcp_xy
            dist = float(np.linalg.norm(rel))
            if dist < best_dist:
                best_dist = dist
                nearest = rel
        return np.asarray([
            math.cos(self.object_yaw),
            math.sin(self.object_yaw),
            *np.clip(nearest / 0.25, -2.0, 2.0),
        ], dtype=np.float32)

    def _observation(self, contacts):
        return np.concatenate((super()._observation(contacts), self._context_features()))


class AlignedPolicy(nn.Module):
    """256x256 LayerNorm+SiLU policy with force_in_transport tactile gating."""

    def __init__(self, obs_size=ALIGNED_OBS_SIZE, action_size=len(ACTION_NAMES), width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, action_size),
        )

    def _mask_obs(self, observation):
        lifted = observation[..., LIFTED_INDEX:LIFTED_INDEX + 1] > 0.5
        arrived = observation[..., ARRIVED_INDEX:ARRIVED_INDEX + 1] > 0.5
        transport = lifted & ~arrived
        tactile = observation[..., OBS_SIZE:TACTILE_SIZE]
        gated = torch.cat((tactile[..., :2] * transport, tactile[..., 2:] * ~transport), dim=-1)
        return torch.cat((observation[..., :OBS_SIZE], gated, observation[..., TACTILE_SIZE:]), dim=-1)

    def logits(self, observation):
        return self.net(self._mask_obs(observation))

    def forward(self, observation):
        return self.logits(observation).tanh()

    @torch.no_grad()
    def action(self, observation):
        return self(torch.as_tensor(observation, dtype=torch.float32)).numpy()


def run_aligned_scene(env, seed, actor=None, frame=None, demonstrations=None, perturb=False, rng=None, dagger_beta=0.0):
    obs, _ = env.reset(seed=seed)
    if frame is not None:
        frame(env.render())
    items = []
    homing_steps = 0
    for index, name in enumerate(env.order):
        returned_home = True
        if index:
            steps, returned_home = env.return_home(frame)
            homing_steps += steps
            obs = env.select(name)
        initial_shift = float(np.linalg.norm(env.cube_xyz[:2] - np.asarray(env.plan[name].xy)))
        expert = AlignedSortingTeacher(env, brick_long_axis=True, rng=rng, perturb=perturb) if (actor is None or demonstrations is not None) else None
        trajectory = [] if demonstrations is not None else None
        while True:
            if actor is None:
                label, executed = expert(obs)
                phase = expert.state.name
            elif demonstrations is not None:
                # DAgger: relabel on-policy state with stateless Markovian expert
                label, phase = expert.relabel()
                if rng is not None and rng.random() < dagger_beta:
                    executed = label.copy()
                else:
                    executed = actor.action(obs)
            else:
                executed = actor.action(obs)
                label, phase = executed, "POLICY"
            following, reward, terminated, truncated, info = env.step(executed)
            if trajectory is not None:
                trajectory.append((obs, executed, label, following, reward, terminated, truncated, phase))
            obs = following
            if frame is not None:
                frame(env.render())
            if terminated or truncated:
                break
        if trajectory is not None:
            demonstrations.append(trajectory)
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


def collect_round(seed, directory, start_seed, count, *, actor=None, perturb=True, dagger_beta=0.0, keep_all=False):
    rng = np.random.default_rng(seed * 9973 + start_seed)
    scenes, items = [], []
    groups = {name: [] for name in SHAPES}
    started = perf_counter()
    with AlignedContactSortingScene() as env:
        for scenario in range(start_seed, start_seed + count):
            traces = []
            scene, rows = run_aligned_scene(
                env, scenario, actor=actor, demonstrations=traces, perturb=perturb, rng=rng, dagger_beta=dagger_beta
            )
            scenes.append(scene)
            items.extend(rows)
            for row, trajectory in zip(rows, traces, strict=True):
                # In DAgger, keep ALL relabeled on-policy trajectories (failed rollouts carry the most valuable corrections!)
                ok = True if keep_all else row["sorted"]
                groups[row["shape"]].append((trajectory, {**row, "is_success": ok}))
    paths, records = [], []
    for name, pairs in groups.items():
        trajectories, outcomes = zip(*pairs)
        path = directory / name / "expert.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_demonstrations(path, trajectories, outcomes)
        paths.append(path)
        records.append({
            "shape": name, "attempts": len(pairs),
            "successful": sum(row["is_success"] for row in outcomes),
            "expert_action_steps": sum(len(t) for t in trajectories),
            "sha256": sha(path),
        })
    write_csv(directory / "scenes.csv", scenes)
    write_csv(directory / "items.csv", items)
    meta = {
        "scenes": count, "all_sorted": sum(r["all_sorted"] for r in scenes),
        "sorted_items": sum(r["sorted_items"] for r in scenes),
        "expert_action_steps": sum(r["expert_action_steps"] for r in records),
        "collection_s": perf_counter() - started, "shape_data": records,
    }
    write_json(directory / "dataset.json", meta)
    tag = "DAgger" if actor is not None else "Expert"
    print(f"Seed {seed} [{tag}]: {count} scenes, {meta['all_sorted']}/{count} full, "
          f"{meta['sorted_items']}/{3 * count} items, {meta['expert_action_steps']} steps", flush=True)
    return paths, meta


def _load_examples(path):
    with np.load(path) as data:
        counts = np.diff(data["offsets"])
        mask = np.repeat(data["successful"], counts)
        x = torch.as_tensor(data["observations"][mask], dtype=torch.float32)
        y = torch.as_tensor(data["expert_actions"][mask], dtype=torch.float32)
        phase = data["phase"][mask]
    groups = [torch.from_numpy(np.flatnonzero(phase == name)) for name in np.unique(phase)]
    return x, y, groups


def train_policy(paths, seed, updates, checkpoints, report_fn, model=None, lr=LEARNING_RATE):
    torch.manual_seed(seed)
    model = AlignedPolicy() if model is None else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=lr * 0.1)
    datasets = [_load_examples(p) for p in paths]
    gen = torch.Generator().manual_seed(seed + 5000)
    for step in range(1, updates + 1):
        model.train()
        obs_list, tgt_list = [], []
        for x, y, groups in datasets:
            count, extra = divmod(64, len(groups))
            bal = [g[torch.randint(len(g), (count + int(i < extra),), generator=gen)] for i, g in enumerate(groups)]
            ids = torch.cat([torch.randint(len(x), (32,), generator=gen), *bal])
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
        if step in checkpoints:
            model.eval()
            report_fn(step, model, float(xyz.detach()), float(grip.detach()))
    return model


def validate(env, actor):
    actor.eval()
    scenes = items = drops = lifts = 0
    sorted_by_shape = dict.fromkeys(SHAPES, 0)
    for seed in VALIDATION:
        result, rows = run_aligned_scene(env, seed, actor)
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


def train(seed, output):
    root = output / "training" / f"seed_{seed}"
    # Round 1: 60 perturbed aligned expert scenes
    r1_paths, r1_cost = collect_round(
        seed, root / "round1_demos", 86_300_000 + 100_000 * seed, ROUND1_SCENES, perturb=True, keep_all=False
    )
    model = AlignedPolicy()
    evaluations, best_r1 = [], None
    with AlignedContactSortingScene() as env:
        def report_r1(step, cur, xyz, grip):
            nonlocal best_r1
            res = validate(env, cur)
            rank = (res["all_sorted"], res["items_sorted"], -res["transport_drops"], -step)
            if best_r1 is None or rank > best_r1:
                best_r1 = rank
                torch.save(cur.state_dict(), root / "model_round1.pt")
            evaluations.append({"round": 1, "updates": step, "xyz_mse": xyz, "gripper_bce": grip, **res})
            write_csv(root / "validation.csv", evaluations)
            print(f"Seed {seed} [R1 u={step}]: {res['all_sorted']}/{len(VALIDATION)} full, "
                  f"{res['items_sorted']}/{3 * len(VALIDATION)} items, drops {res['transport_drops']} "
                  f"(c={res['cube_sorted']}, b={res['brick_sorted']}, y={res['cylinder_sorted']})", flush=True)

        train_policy(r1_paths, seed + 168270, ROUND1_UPDATES, CHECKPOINTS, report_r1, model=model, lr=LEARNING_RATE)

    # Load best Round 1 model for DAgger rollout
    model.load_state_dict(torch.load(root / "model_round1.pt", weights_only=True, map_location="cpu"))
    model.eval()

    # Round 2: DAgger on 36 fresh training scenes (beta=0.35 expert mix, relabeled at every step)
    r2_paths, r2_cost = collect_round(
        seed, root / "round2_dagger", 87_300_000 + 100_000 * seed, DAGGER_SCENES,
        actor=model, perturb=False, dagger_beta=0.35, keep_all=True
    )

    all_paths = r1_paths + r2_paths
    best_r2 = best_r1
    torch.save(model.state_dict(), root / "model.pt")
    with AlignedContactSortingScene() as env:
        def report_r2(step, cur, xyz, grip):
            nonlocal best_r2
            res = validate(env, cur)
            rank = (res["all_sorted"], res["items_sorted"], -res["transport_drops"], -step)
            if rank > best_r2:
                best_r2 = rank
                torch.save(cur.state_dict(), root / "model.pt")
            evaluations.append({"round": 2, "updates": step, "xyz_mse": xyz, "gripper_bce": grip, **res})
            write_csv(root / "validation.csv", evaluations)
            print(f"Seed {seed} [R2-DAgger u={step}]: {res['all_sorted']}/{len(VALIDATION)} full, "
                  f"{res['items_sorted']}/{3 * len(VALIDATION)} items, drops {res['transport_drops']} "
                  f"(c={res['cube_sorted']}, b={res['brick_sorted']}, y={res['cylinder_sorted']})", flush=True)

        train_policy(all_paths, seed + 268270, ROUND2_UPDATES, CHECKPOINTS, report_r2, model=model, lr=LEARNING_RATE * 0.5)

    write_json(root / "config.json", {
        "seed": seed, "checkpoint_sha256": sha(root / "model.pt"),
        "round1_scenes": ROUND1_SCENES, "dagger_scenes": DAGGER_SCENES,
        "round1_steps": r1_cost["expert_action_steps"], "dagger_steps": r2_cost["expert_action_steps"],
        "validation_scenarios": list(VALIDATION), "test_scenarios": list(TEST),
    })


def evaluate_aligned(method, seed, output):
    if method == "expert":
        actor, identity = None, None
        directory = output / "test" / "expert"
    else:
        root = output / "training" / f"seed_{seed}"
        weights = root / ("model_round1.pt" if method == "round1" else "model.pt")
        identity = sha(weights)
        actor = AlignedPolicy()
        actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
        actor.eval()
        directory = output / "test" / method / f"seed_{seed}"
    scenes, objects = [], []
    with AlignedContactSortingScene() as env:
        for scenario in TEST:
            row, items = run_aligned_scene(env, scenario, actor)
            scenes.append(row)
            objects.extend(items)
    summary = {
        "method": method, "seed": seed, "checkpoint_sha256": identity,
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
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--method", choices=("expert", "round1", "aligned"), default="aligned")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        if args.seed is None:
            parser.error("train requires --seed")
        train(args.seed, args.out)
    else:
        if args.method != "expert" and args.seed is None:
            parser.error(f"test --method {args.method} requires --seed")
        evaluate_aligned(args.method, args.seed, args.out)


if __name__ == "__main__":
    main()
