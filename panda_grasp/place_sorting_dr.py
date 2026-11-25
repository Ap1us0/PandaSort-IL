"""Domain-Randomized V3-Expert + Prefix-Takeover DAgger + 3-Frame History EMA sorting policy."""

import argparse
from collections import deque
import copy
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._vendor.pickup import State, measure_contacts
from .env import ACTION_STEP
from .experiment import save_demonstrations, write_csv, write_json
from .place_contact_signals import TACTILE_SIZE
from .place_objects import ACTION_NAMES, OBS_SIZE, SHAPES, YAW_STEP, YawTeacher
from .place_objects_train import sha
from .place_sorting import COLOR_BY_SHAPE, TRAYS
from .place_sorting_aligned import ALIGNED_OBS_SIZE, AlignedContactSortingScene
from .place_sorting_contact_ablation import ARRIVED_INDEX, LIFTED_INDEX
from .place_sorting_contact_schedule import TEST
from .place_sorting_history import HISTORY, load_stacked_examples, run_history_scene
from .place_sorting_ood import CONDITIONS, OODSortingScene, OODSpec

OUTPUT = Path("results/place_sorting_dr")
NOMINAL_SCENES = 60
DR_SCENES = 60
DAGGER_NOMINAL_SCENES = 30
DAGGER_DR_SCENES = 24
VAL_NOMINAL = range(1_562_000, 1_562_015)
VAL_OOD = range(1_562_015, 1_562_021)  # 6 held-out validation layouts tested under friction_60 / size_90 / jitter
ROUND1_UPDATES = 6000
ROUND1_CKPTS = (2500, 4500, 6000)
ROUND2_UPDATES = 4000
ROUND2_CKPTS = (1500, 3000, 4000)
LEARNING_RATE = 5e-4
EMA_DECAY = 0.995
WIDTH = 384


class RobustSortingTeacherV3(YawTeacher):
    """V3 expert (29/30 nominal): neighbor-avoiding cylinder yaw, soft cylinder touchdown, bounded cube creep compensation."""

    def __init__(self, env, brick_long_axis=True, rng=None, perturb=False):
        self.grasp_z_offset = -0.006 if env.shape_name == "cube" else -0.003
        self.transport_h = 0.086 if env.shape_name == "cube" else 0.078
        self.transport_limit = 1.0
        self.descend_steps = 0
        self.perturb_rng = rng
        self.perturb_active = perturb
        super().__init__(env, brick_long_axis=brick_long_axis)
        if env.shape.cylinder:
            self.grasp_yaw = self._cylinder_clear_yaw()

    def _cylinder_clear_yaw(self):
        env = self.env
        obj_xy = env.cube_xyz[:2]
        jaw_axis = env.home_rotation[:, 1]
        home_angle = math.atan2(jaw_axis[1], jaw_axis[0])
        nearest_vec = None
        best_dist = float("inf")
        for name, (geom, _) in env.objects.items():
            if name == env.shape_name:
                continue
            rel = env.data.geom_xpos[geom, :2] - obj_xy
            d = float(np.linalg.norm(rel))
            if d < best_dist:
                best_dist = d
                nearest_vec = rel
        if nearest_vec is None or best_dist > 0.14:
            return 0.0
        neighbor_angle = math.atan2(nearest_vec[1], nearest_vec[0])
        desired_jaw_angle = neighbor_angle + math.pi / 2
        grasp_yaw = (desired_jaw_angle - home_angle + math.pi / 2) % math.pi - math.pi / 2
        return float(np.clip(grasp_yaw, -math.pi / 2, math.pi / 2))

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
            cube_h = float(env.cube_xyz[2] - env.rest_z)
            if self.state == State.LIFT and (
                (env.shape_name == "cube" and cube_h >= 0.064)
                or (env.shape_name != "cube" and (cube_h >= 0.056 or tcp[2] >= env.rest_z + self.transport_h - 0.008))
            ):
                self.state = State.TRANSPORT
            else:
                label, _ = super(YawTeacher, self).__call__(observation)
                if self.state == State.LIFT and env.shape_name == "cube":
                    lift_tgt = np.array([self.lift[0], self.lift[1], env.rest_z + 0.092])
                    delta = (lift_tgt - tcp) / ACTION_STEP
                    delta *= min(1.0, 0.85 / max(0.85, np.linalg.norm(delta)))
                    label = np.array([*delta, -1.0], dtype=np.float32)
                out = np.concatenate((label, np.asarray([yaw_action], dtype=np.float32)))
                return out, self._perturb_action(out, self.state)

        if self.state == State.TRANSPORT and np.linalg.norm(env.target[:2] - env.cube_xyz[:2]) < 0.007 and env.transported:
            self.state = State.PLACE
        if self.state == State.PLACE and self._ready_to_retreat():
            self.state = State.RETREAT

        if self.state == State.TRANSPORT:
            if env.shape_name == "cube":
                slip_comp = min(0.022, max(0.0, float(tcp[2] - env.cube_xyz[2])))
                h = self.transport_h + slip_comp
            else:
                h = self.transport_h
            target = env.target + np.array([0.0, 0.0, h])
            target[:2] += tcp[:2] - env.cube_xyz[:2]
            limit = self.transport_limit
        elif self.state == State.PLACE:
            target = env.target + np.array([0.0, 0.0, -0.001])
            target[:2] += tcp[:2] - env.cube_xyz[:2]
            limit = 0.45 if env.shape.cylinder else 0.72
        else:
            target = env.target + np.array([0.0, 0.0, 0.10])
            limit = 0.85

        delta = (target - tcp) / ACTION_STEP
        delta *= min(1.0, limit / max(limit, np.linalg.norm(delta)))
        label = np.array([*delta, 1.0 if self.state == State.RETREAT else -1.0, yaw_action], dtype=np.float32)
        return label, self._perturb_action(label, self.state)


class DRHistoryPolicy(nn.Module):
    """384x384 residual LayerNorm+SiLU policy over 3 stacked frames with per-frame force_in_transport gating."""

    def __init__(self, frame_size=ALIGNED_OBS_SIZE, history=HISTORY, action_size=len(ACTION_NAMES), width=WIDTH):
        super().__init__()
        self.frame_size = frame_size
        self.history = history
        self.in_proj = nn.Sequential(
            nn.Linear(frame_size * history, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.act = nn.SiLU()
        self.head = nn.Linear(width, action_size)

    def _mask_frame(self, frame):
        lifted = frame[..., LIFTED_INDEX:LIFTED_INDEX + 1] > 0.5
        arrived = frame[..., ARRIVED_INDEX:ARRIVED_INDEX + 1] > 0.5
        transport = lifted & ~arrived
        tactile = frame[..., OBS_SIZE:TACTILE_SIZE]
        gated = torch.cat((tactile[..., :2] * transport, tactile[..., 2:] * ~transport), dim=-1)
        return torch.cat((frame[..., :OBS_SIZE], gated, frame[..., TACTILE_SIZE:]), dim=-1)

    def logits(self, stacked_obs):
        if stacked_obs.ndim == 2 and stacked_obs.shape[-1] == self.history * self.frame_size:
            frames = stacked_obs.view(-1, self.history, self.frame_size)
        elif stacked_obs.ndim == 1 and stacked_obs.shape[-1] == self.history * self.frame_size:
            frames = stacked_obs.view(1, self.history, self.frame_size)
        else:
            frames = stacked_obs
        masked = self._mask_frame(frames).reshape(frames.shape[0], self.history * self.frame_size)
        h = self.in_proj(masked)
        h = self.act(h + self.block1(h))
        return self.head(h)

    def forward(self, stacked_obs):
        return self.logits(stacked_obs).tanh()

    @torch.no_grad()
    def action(self, stacked_obs):
        out = self(torch.as_tensor(stacked_obs, dtype=torch.float32))
        return out[0].numpy() if out.ndim == 2 else out.numpy()


DR_BUCKETS = (
    OODSpec("dr_fric_60", friction_mult=60.0 / 70.0, pickup_jitter_m=0.008),
    OODSpec("dr_fric_64", friction_mult=64.0 / 70.0, pickup_jitter_m=0.010),
    OODSpec("dr_mass_165", mass_mult=165.0 / 150.0, pickup_jitter_m=0.006),
    OODSpec("dr_mass_175", mass_mult=175.0 / 150.0, pickup_jitter_m=0.004),
    OODSpec("dr_size_90", xy_scale=0.90, pickup_jitter_m=0.008),
    OODSpec("dr_size_94", xy_scale=0.94, pickup_jitter_m=0.010),
    OODSpec("dr_size_108", xy_scale=1.08, pickup_jitter_m=0.008),
    OODSpec("dr_jitter_12", pickup_jitter_m=0.012),
    OODSpec("dr_combo_a", friction_mult=64.0 / 70.0, mass_mult=160.0 / 150.0, pickup_jitter_m=0.008),
    OODSpec("dr_combo_b", friction_mult=65.0 / 70.0, xy_scale=0.94, pickup_jitter_m=0.010),
)


def run_v3_expert_scene(env, seed, rng, perturb=True):
    obs, _ = env.reset(seed=seed)
    traces, items = [], []
    for index, name in enumerate(env.order):
        if index:
            env.return_home()
            obs = env.select(name)
        teacher = RobustSortingTeacherV3(env, brick_long_axis=True, rng=rng, perturb=perturb)
        traj = []
        while True:
            label, executed = teacher(obs)
            phase = teacher.state.name
            following, reward, term, trunc, info = env.step(executed)
            traj.append((obs, executed, label, following, reward, term, trunc, phase))
            obs = following
            if term or trunc:
                break
        traces.append(traj)
        assignment = env.plan[name]
        items.append({
            "scenario_seed": seed, "shape": name, "expected_color": COLOR_BY_SHAPE[name],
            "target_slot": assignment.slot, "stable_in_target": info["stable_placement_success"],
            "failure_stage": info["failure_stage"], "lifted": info["ever_carried"],
            "transport_drop": info["transport_drop"],
        })
    slot_colors = {assignment.slot: COLOR_BY_SHAPE[name] for name, assignment in env.plan.items()}
    for row in items:
        slot = env.final_tray(row["shape"])
        row["final_color"] = slot_colors.get(slot, "outside")
        row["sorted"] = row["stable_in_target"] and row["final_color"] == row["expected_color"]
    return all(r["sorted"] for r in items), items, traces


def collect_round1(seed, root):
    rng = np.random.default_rng(seed * 13337 + 88_000_000)
    nom_groups = {name: [] for name in SHAPES}
    dr_groups = {name: [] for name in SHAPES}
    started = perf_counter()

    # 1) 60 nominal perturbed scenes
    nom_full = 0
    with AlignedContactSortingScene() as env:
        base = 88_000_000 + 100_000 * seed
        for i in range(NOMINAL_SCENES):
            ok_all, rows, traces = run_v3_expert_scene(env, base + i, rng, perturb=True)
            nom_full += int(ok_all)
            for row, traj in zip(rows, traces, strict=True):
                nom_groups[row["shape"]].append((traj, {"is_success": row["sorted"], "scenario_seed": row["scenario_seed"]}))

    # 2) 60 Domain-Randomized perturbed scenes (10 compiled buckets x 6 scenes each)
    dr_full = 0
    base_dr = 89_000_000 + 100_000 * seed
    per_bucket = DR_SCENES // len(DR_BUCKETS)
    sc_idx = 0
    for spec in DR_BUCKETS:
        with OODSortingScene(spec) as env:
            for _ in range(per_bucket):
                ok_all, rows, traces = run_v3_expert_scene(env, base_dr + sc_idx, rng, perturb=True)
                sc_idx += 1
                dr_full += int(ok_all)
                for row, traj in zip(rows, traces, strict=True):
                    dr_groups[row["shape"]].append((traj, {"is_success": row["sorted"], "scenario_seed": row["scenario_seed"]}))

    paths = []
    for tag, grp in (("nominal", nom_groups), ("dr", dr_groups)):
        for name, pairs in grp.items():
            trajs, outs = zip(*pairs)
            p = root / "round1_demos" / tag / name / "expert.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            save_demonstrations(p, trajs, outs)
            paths.append(p)
    print(f"Seed {seed} [R1 Collect]: nom={nom_full}/{NOMINAL_SCENES}, dr={dr_full}/{DR_SCENES}, time={perf_counter()-started:.1f}s", flush=True)
    return paths


def make_takeover_v3(env):
    teacher = RobustSortingTeacherV3(env, brick_long_axis=True, perturb=False)
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


def collect_dagger_v3(seed, actor, root):
    rng = np.random.default_rng(seed * 21313 + 90_000_000)
    groups = {name: [] for name in SHAPES}
    rescued = scanned = 0
    started = perf_counter()

    def process_env(env, scenario):
        nonlocal rescued, scanned
        # First probe policy rollout
        obs, _ = env.reset(seed=scenario)
        probe = {}
        for index, name in enumerate(env.order):
            if index:
                env.return_home()
                obs = env.select(name)
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

        targets = {}
        for name, (ok, stage, lift_s, arr_s, steps) in probe.items():
            if not ok:
                scanned += 1
                if stage in ("grip", "lift"):
                    t_step = int(rng.integers(5, 22))
                elif stage == "transport":
                    t_step = max(5, (lift_s or 40) + int(rng.integers(2, 16)))
                else:
                    t_step = max(10, (arr_s or (lift_s or 40) + 20) - int(rng.integers(4, 16)))
                targets[name] = (t_step, True)
            elif rng.random() < 0.15:
                targets[name] = ((lift_s or 40) + int(rng.integers(5, 22)), False)
        if not targets:
            return

        # Replay scene with policy prefix -> V3 teacher takeover
        obs, _ = env.reset(seed=scenario)
        for index, name in enumerate(env.order):
            if index:
                env.return_home()
                obs = env.select(name)
            if name not in targets:
                t = RobustSortingTeacherV3(env, brick_long_axis=True)
                while True:
                    _, act = t(obs)
                    obs, _, term, trunc, _ = env.step(act)
                    if term or trunc:
                        break
                continue
            t_step, was_fail = targets[name]
            buf = deque([obs.copy() for _ in range(HISTORY)], maxlen=HISTORY)
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

    with AlignedContactSortingScene() as env:
        base = 90_000_000 + 100_000 * seed
        for i in range(DAGGER_NOMINAL_SCENES):
            process_env(env, base + i)

    base_dr = 91_000_000 + 100_000 * seed
    dagger_buckets = (DR_BUCKETS[0], DR_BUCKETS[2], DR_BUCKETS[3], DR_BUCKETS[4], DR_BUCKETS[6], DR_BUCKETS[7])
    per_b = DAGGER_DR_SCENES // len(dagger_buckets)
    sc_idx = 0
    for spec in dagger_buckets:
        with OODSortingScene(spec) as env:
            for _ in range(per_b):
                process_env(env, base_dr + sc_idx)
                sc_idx += 1

    paths = []
    tot_traj = tot_steps = 0
    for name, pairs in groups.items():
        if not pairs:
            continue
        trajs, outs = zip(*pairs)
        p = root / "dagger_demos" / name / "dagger.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        save_demonstrations(p, trajs, outs)
        paths.append(p)
        tot_traj += len(pairs)
        tot_steps += sum(len(t) for t in trajs)
    print(f"Seed {seed} [DR-DAgger]: rescued {rescued}/{scanned} fails, {tot_traj} suffixes ({tot_steps} steps), time={perf_counter()-started:.1f}s", flush=True)
    return paths


def validate_dr(actor):
    actor.eval()
    nom_all = nom_items = nom_drops = 0
    with AlignedContactSortingScene() as env:
        for sc in VAL_NOMINAL:
            res, rows = run_history_scene(env, sc, actor)
            nom_all += res["all_sorted"]
            nom_items += res["sorted_items"]
            nom_drops += sum(r["transport_drop"] for r in rows)
    ood_all = ood_items = 0
    for cond_name, sc_slice in (("friction_60", (0, 2)), ("size_90", (2, 4)), ("jitter_12mm", (4, 6))):
        with OODSortingScene(CONDITIONS[cond_name]) as env:
            for sc in list(VAL_OOD)[sc_slice[0]:sc_slice[1]]:
                res, _ = run_history_scene(env, sc, actor)
                ood_all += res["all_sorted"]
                ood_items += res["sorted_items"]
    return {
        "nom_all": nom_all, "nom_items": nom_items, "nom_drops": nom_drops,
        "ood_all": ood_all, "ood_items": ood_items,
    }


def _train_loop(model, ema_model, datasets, weights_per_ds, updates, ckpts, lr, seed, report_fn):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=lr * 0.1)
    gen = torch.Generator().manual_seed(seed)
    for step in range(1, updates + 1):
        model.train()
        obs_list, tgt_list = [], []
        for (x, y, groups), n_batch in zip(datasets, weights_per_ds, strict=True):
            n_uni = max(4, n_batch // 3)
            n_bal = n_batch - n_uni
            count, extra = divmod(n_bal, len(groups))
            bal = [g[torch.randint(len(g), (count + int(i < extra),), generator=gen)] for i, g in enumerate(groups)]
            ids = torch.cat([torch.randint(len(x), (n_uni,), generator=gen), *bal])
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
        if step in ckpts:
            report_fn(step, ema_model, float(xyz.detach()), float(grip.detach()))


def train(seed, output):
    root = output / "training" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    r1_paths = collect_round1(seed, root)
    r1_datasets = [load_stacked_examples(p) for p in r1_paths]
    # 3 nominal shapes (64 each = 192) + 3 DR shapes (64 each = 192) = 384 batch
    r1_weights = [64] * len(r1_datasets)

    torch.manual_seed(seed + 768270)
    model = DRHistoryPolicy()
    ema_model = copy.deepcopy(model)
    evals, best = [], None

    def score(res):
        # Jointly optimize nominal (primary) + OOD generalization
        return (res["nom_all"] + res["ood_all"], res["nom_items"] + res["ood_items"], -res["nom_drops"])

    def rep_r1(step, cur, xyz, grip):
        nonlocal best
        res = validate_dr(cur)
        rk = (*score(res), -step)
        if best is None or rk > best:
            best = rk
            torch.save(cur.state_dict(), root / "model_round1.pt")
        evals.append({"round": 1, "updates": step, "xyz_mse": xyz, "gripper_bce": grip, **res})
        write_csv(root / "validation.csv", evals)
        print(f"Seed {seed} [DR-R1 u={step}]: nom={res['nom_all']}/15 ({res['nom_items']}/45, d={res['nom_drops']}), "
              f"ood={res['ood_all']}/6 ({res['ood_items']}/18)", flush=True)

    _train_loop(model, ema_model, r1_datasets, r1_weights, ROUND1_UPDATES, ROUND1_CKPTS, LEARNING_RATE, seed + 773270, rep_r1)

    # Load best R1 EMA weights for DAgger rollout & R2 fine-tuning
    model.load_state_dict(torch.load(root / "model_round1.pt", weights_only=True, map_location="cpu"))
    ema_model = copy.deepcopy(model)
    torch.save(ema_model.state_dict(), root / "model.pt")

    dg_paths = collect_dagger_v3(seed, ema_model, root)
    dg_datasets = [load_stacked_examples(p) for p in dg_paths]
    all_datasets = r1_datasets + dg_datasets
    # 80% R1 (384) + 20% DAgger (96 total -> 32 per shape)
    all_weights = [64] * len(r1_datasets) + [32] * len(dg_datasets)

    def rep_r2(step, cur, xyz, grip):
        nonlocal best
        res = validate_dr(cur)
        rk = (*score(res), -(ROUND1_UPDATES + step))
        if rk > best:
            best = rk
            torch.save(cur.state_dict(), root / "model.pt")
        evals.append({"round": 2, "updates": step, "xyz_mse": xyz, "gripper_bce": grip, **res})
        write_csv(root / "validation.csv", evals)
        print(f"Seed {seed} [DR-R2 u={step}]: nom={res['nom_all']}/15 ({res['nom_items']}/45, d={res['nom_drops']}), "
              f"ood={res['ood_all']}/6 ({res['ood_items']}/18)", flush=True)

    _train_loop(model, ema_model, all_datasets, all_weights, ROUND2_UPDATES, ROUND2_CKPTS, LEARNING_RATE * 0.4, seed + 873270, rep_r2)
    write_json(root / "config.json", {
        "seed": seed, "width": WIDTH, "history": HISTORY, "ema_decay": EMA_DECAY,
        "checkpoint_sha256": sha(root / "model.pt"),
        "nominal_scenes": NOMINAL_SCENES, "dr_scenes": DR_SCENES,
        "dagger_nominal": DAGGER_NOMINAL_SCENES, "dagger_dr": DAGGER_DR_SCENES,
        "test_scenarios": list(TEST),
    })


def evaluate_condition(cond_name, output):
    env_factory = AlignedContactSortingScene if cond_name == "nominal" else (lambda: OODSortingScene(CONDITIONS[cond_name]))
    all_scenes, all_items, seed_rows = [], [], []
    with env_factory() as env:
        for seed in (0, 1, 2):
            weights = output / "training" / f"seed_{seed}" / "model.pt"
            identity = sha(weights)
            actor = DRHistoryPolicy()
            actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
            actor.eval()
            scenes, items = [], []
            for scenario in TEST:
                sc, rows = run_history_scene(env, scenario, actor)
                scenes.append(sc)
                items.extend(rows)
            folder = output / "test" / cond_name / f"seed_{seed}"
            write_csv(folder / "scenes.csv", scenes)
            write_csv(folder / "items.csv", items)
            cubes = [r for r in items if r["shape"] == "cube"]
            summary = {
                "condition": cond_name, "seed": seed, "checkpoint_sha256": identity,
                "scenes": len(scenes), "all_sorted": sum(r["all_sorted"] for r in scenes),
                "sorted_items": sum(r["sorted"] for r in items),
                "cube_sorted": sum(r["sorted"] for r in cubes),
                "cube_lifted": sum(r["lifted"] for r in cubes),
                "cube_transport_drops": sum(r["transport_drop"] for r in cubes),
                "brick_sorted": sum(r["shape"] == "brick" and r["sorted"] for r in items),
                "cylinder_sorted": sum(r["shape"] == "cylinder" and r["sorted"] for r in items),
            }
            write_json(folder / "summary.json", summary)
            seed_rows.append(summary)
            all_scenes.extend(scenes)
            all_items.extend(items)
    cubes_all = [r for r in all_items if r["shape"] == "cube"]
    agg = {
        "condition": cond_name,
        "scenes": len(all_scenes),
        "all_sorted": sum(r["all_sorted"] for r in all_scenes),
        "all_sorted_pct": round(100.0 * sum(r["all_sorted"] for r in all_scenes) / len(all_scenes), 1),
        "per_seed": "/".join(str(r["all_sorted"]) for r in seed_rows),
        "sorted_items": sum(r["sorted"] for r in all_items),
        "sorted_items_pct": round(100.0 * sum(r["sorted"] for r in all_items) / len(all_items), 1),
        "cube_sorted": sum(r["sorted"] for r in cubes_all),
        "brick_sorted": sum(r["shape"] == "brick" and r["sorted"] for r in all_items),
        "cylinder_sorted": sum(r["shape"] == "cylinder" and r["sorted"] for r in all_items),
        "cube_drops": sum(r["transport_drop"] for r in cubes_all),
        "cube_lifted": sum(r["lifted"] for r in cubes_all),
    }
    write_json(output / "test" / cond_name / "overall.json", agg)
    print("DR-TEST:", agg, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "test"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--condition", choices=("nominal", *CONDITIONS), default="nominal")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "train":
        if args.seed is None:
            parser.error("train requires --seed")
        train(args.seed, args.out)
    else:
        evaluate_condition(args.condition, args.out)


if __name__ == "__main__":
    main()
