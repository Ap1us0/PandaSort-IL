"""Zero-shot OOD generalization stress test for frozen history3_ema sorting policies."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .experiment import write_csv, write_json
from .place_objects import ObjectShape, SHAPES, YAW_LIMIT
from .place_objects_train import sha
from .place_sorting import Assignment, COLOR_BY_SHAPE, PICKUP, TRAY_HALF, TRAYS
from .place_sorting_aligned import AlignedContactSortingScene
from .place_sorting_contact_schedule import TEST
from .place_sorting_history import HistoryAlignedPolicy, OUTPUT as HISTORY_OUTPUT, run_history_scene

OUTPUT = Path("results/place_sorting_ood")


@dataclass(frozen=True)
class OODSpec:
    name: str
    friction_mult: float = 1.0   # relative to training's 70% (0.98 -> 60% is 60/70)
    mass_mult: float = 1.0       # relative to training's 150g (175g is 175/150)
    pickup_jitter_m: float = 0.0025
    xy_scale: float = 1.0


CONDITIONS = {
    "friction_60": OODSpec("friction_60", friction_mult=60.0 / 70.0),
    "mass_175": OODSpec("mass_175", mass_mult=175.0 / 150.0),
    "jitter_12mm": OODSpec("jitter_12mm", pickup_jitter_m=0.012),
    "size_90": OODSpec("size_90", xy_scale=0.90),
    "size_108": OODSpec("size_108", xy_scale=1.08),
}


class OODSortingScene(AlignedContactSortingScene):
    """Apply controlled OOD physics, spawn jitter, or XY geometry scaling to all 3 sorting objects."""

    def __init__(self, spec_cfg: OODSpec, **kwargs):
        self.ood = spec_cfg
        self.scaled_shapes = {
            name: ObjectShape(
                (sh.half_extents[0] * spec_cfg.xy_scale, sh.half_extents[1] * spec_cfg.xy_scale, sh.half_extents[2]),
                cylinder=sh.cylinder,
                rgba=sh.rgba,
            )
            for name, sh in SHAPES.items()
        }
        super().__init__(**kwargs)

    def _scene_spec(self):
        spec = super()._scene_spec()
        for name, body_name, geom_name in (
            ("cube", "cube", "cube"),
            ("brick", "sort_brick_body", "sort_brick"),
            ("cylinder", "sort_cylinder_body", "sort_cylinder"),
        ):
            geom = spec.geom(geom_name)
            body = spec.body(body_name)
            sh = self.scaled_shapes[name]
            x, y, z = sh.half_extents
            if sh.cylinder:
                geom.size = [x, z, 0.0]
            else:
                geom.size = [x, y, z]
            # Scale sliding friction only (table pairs keep original friction)
            fric = list(geom.friction)
            fric[0] = float(fric[0]) * self.ood.friction_mult
            geom.friction = fric
            # Scale mass and inertia proportionally
            if geom_name == "cube":
                body.mass = float(body.mass) * self.ood.mass_mult
                if sh.cylinder:
                    trans = body.mass * (3 * x**2 + 4 * z**2) / 12
                    body.inertia = [trans, trans, body.mass * x**2 / 2]
                else:
                    body.inertia = [body.mass * (b * b + c * c) / 3 for b, c in ((y, z), (x, z), (x, y))]
            else:
                geom.mass = float(geom.mass) * self.ood.mass_mult
        return spec

    def _set_object(self, name):
        super()._set_object(name)
        self.shape = self.scaled_shapes[name]

    def contains(self, name, slot):
        geom, _ = self.objects[name]
        shape = self.scaled_shapes[name]
        axes = self.data.geom_xmat[geom].reshape(3, 3)[:2]
        if shape.cylinder:
            radius, _, height = shape.half_extents
            z_axis = np.abs(axes[:, 2])
            radius_xy = radius * np.sqrt(np.maximum(0.0, 1 - z_axis**2)) + height * z_axis
        else:
            radius_xy = np.abs(axes) @ shape.half_extents
        offset = np.abs(self.data.geom_xpos[geom, :2] - TRAYS[slot])
        return bool(np.all(offset + radius_xy <= TRAY_HALF + 1e-5))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.ood.pickup_jitter_m == 0.0025:
            return obs, info
        # Re-sample pickup XY with wider jitter while preserving exact order, slots, and yaw
        rng = np.random.default_rng(int(seed) + 424242)
        pickup_order = np.random.default_rng(int(seed)).permutation(len(SHAPES))
        pick_xy = PICKUP + rng.uniform(-self.ood.pickup_jitter_m, self.ood.pickup_jitter_m, PICKUP.shape)
        new_plan = {}
        for i, name in enumerate(SHAPES):
            old = self.plan[name]
            new_plan[name] = Assignment(tuple(pick_xy[pickup_order[i]]), old.yaw, old.slot)
        self.plan = new_plan
        for name, assignment in self.plan.items():
            _, joint = self.objects[name]
            addr = int(self.model.jnt_qposadr[joint])
            self.data.qpos[addr:addr + 7] = (*assignment.xy, self.rest_z, *self.data.qpos[addr + 3:addr + 7])
        self._begin_attempt()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.frame_skip * 4):
            self._physics(self.start_q)
        mujoco.mj_forward(self.model, self.data)
        from ._vendor.pickup import measure_contacts
        contacts = measure_contacts(self, self.data)
        self.potential = self._potential(contacts)
        return self._observation(contacts), self._info(contacts, "running")


def evaluate_condition(cond_name, history_root, output):
    cfg = CONDITIONS[cond_name]
    all_scenes, all_items, seed_rows = [], [], []
    with OODSortingScene(cfg) as env:
        for seed in (0, 1, 2):
            weights = history_root / "training" / f"seed_{seed}" / "model.pt"
            identity = sha(weights)
            actor = HistoryAlignedPolicy()
            actor.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
            actor.eval()
            scenes, items = [], []
            for scenario in TEST:
                sc, rows = run_history_scene(env, scenario, actor)
                scenes.append(sc)
                items.extend(rows)
            folder = output / cond_name / f"seed_{seed}"
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
    write_json(output / cond_name / "overall.json", agg)
    print("OOD:", agg, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--history", type=Path, default=HISTORY_OUTPUT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluate_condition(args.condition, args.history, args.out)


if __name__ == "__main__":
    main()
