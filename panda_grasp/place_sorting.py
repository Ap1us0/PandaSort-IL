"""Physical multi-object tray sorting with a shared, externally specified route."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from ._vendor.pickup import measure_contacts
from .env import ACTION_STEP, ANGULAR_LIMIT, Episode, LINEAR_LIMIT, START
from .experiment import write_csv, write_json
from .learning import Clone
from .place import REST_TOLERANCE_M
from .place_objects import ACTION_NAMES, OBS_SIZE, SHAPES, YAW_LIMIT, YAW_STEP, ObjectPlacementEnv, YawTeacher, yaw_quaternion
from .place_objects_benchmark import OUTPUT as OBJECT_RESULTS
from .place_objects_train import sha

OUTPUT = Path("results/place_sorting")
PILOT = range(1_560_000, 1_560_004)
TEST = range(1_561_000, 1_561_030)
TRAYS = np.array([[0.452, 0.041], [0.548, 0.041], [0.500, 0.133]])
TRAY_HALF = np.array([0.043, 0.042])
WALL = 0.003
WALL_HEIGHT = 0.008
TABLE_TOP = 0.26
COLOR_BY_SHAPE = {"cube": "red", "brick": "blue", "cylinder": "green"}
COLORS = {"red": (0.76, 0.18, 0.20, 1.0), "blue": (0.12, 0.43, 0.80, 1.0),
          "green": (0.12, 0.64, 0.33, 1.0)}
OBJECT_COLORS = {"red": (0.98, 0.28, 0.31, 1.0), "blue": (0.27, 0.62, 1.0, 1.0),
                 "green": (0.30, 0.91, 0.46, 1.0)}
PICKUP = np.array([[0.459, -0.141], [0.541, -0.141], [0.500, -0.060]])


@dataclass(frozen=True)
class Assignment:
    xy: tuple[float, float]
    yaw: float
    slot: int


def route(seed):
    rng = np.random.default_rng(seed)
    pickup_order = rng.permutation(len(SHAPES))
    tray_order = rng.permutation(len(SHAPES))
    pick_xy = PICKUP + rng.uniform(-0.0025, 0.0025, PICKUP.shape)
    assignments = {name: Assignment(tuple(pick_xy[pickup_order[i]]), float(rng.uniform(-YAW_LIMIT, YAW_LIMIT)),
                                    int(tray_order[i])) for i, name in enumerate(SHAPES)}
    order = [str(name) for name in rng.permutation(list(SHAPES))]
    return assignments, order


class SortingScene(ObjectPlacementEnv):
    """Keep three dynamic geoms in one scene and score each assigned trial in turn."""

    def __init__(self, **kwargs):
        super().__init__(shape="cube", random_yaw=True, yaw_action=True, **kwargs)
        self.objects = {}
        for name, geom, joint in (("cube", "cube", "cube_free"),
                                  ("brick", "sort_brick", "sort_brick_free"),
                                  ("cylinder", "sort_cylinder", "sort_cylinder_free")):
            joint_id = self.model.joint(joint).id
            self.objects[name] = (self.model.geom(geom).id, joint_id)

    def _scene_spec(self):
        spec = super()._scene_spec()
        spec.site("place_goal").rgba = [0, 0, 0, 0]
        spec.body("cube").pos = [*PICKUP[0], 0.29]
        spec.geom("cube").rgba = OBJECT_COLORS[COLOR_BY_SHAPE["cube"]]
        source_pair = spec.pair("cube_table")
        for name, point in (("brick", PICKUP[1]), ("cylinder", PICKUP[2])):
            shape = SHAPES[name]
            body = spec.worldbody.add_body(name=f"sort_{name}_body", pos=[*point, 0.29])
            body.add_freejoint(name=f"sort_{name}_free")
            x, y, z = shape.half_extents
            kind = mujoco.mjtGeom.mjGEOM_CYLINDER if shape.cylinder else mujoco.mjtGeom.mjGEOM_BOX
            size = [x, z, 0.0] if shape.cylinder else [x, y, z]
            geom = f"sort_{name}"
            body.add_geom(name=geom, type=kind, size=size, mass=0.15, friction=[0.98, 0.006, 0.001],
                          condim=4, solref=[0.008, 1], rgba=OBJECT_COLORS[COLOR_BY_SHAPE[name]])
            spec.add_pair(geomname1="table", geomname2=geom, condim=source_pair.condim,
                          friction=source_pair.friction, solref=source_pair.solref, solimp=source_pair.solimp)
        for index, center in enumerate(TRAYS):
            self._add_tray(spec, index, center)
        return spec

    @staticmethod
    def _add_tray(spec, index, center):
        x, y = center
        hx, hy = TRAY_HALF
        box = mujoco.mjtGeom.mjGEOM_BOX
        appearance = dict(type=box, rgba=[0.4, 0.4, 0.4, 1], group=0)
        # The perfectly flat common tabletop is the contact floor of all three trays.
        spec.worldbody.add_geom(name=f"tray_{index}_floor", pos=[x, y, TABLE_TOP + 0.0002],
                                size=[hx + WALL, hy + WALL, 0.0002], contype=0, conaffinity=0, **appearance)
        z = TABLE_TOP + WALL_HEIGHT / 2
        for side, sign in (("minus", -1), ("plus", 1)):
            spec.worldbody.add_geom(name=f"tray_{index}_x_{side}", pos=[x + sign * (hx + WALL / 2), y, z],
                                    size=[WALL / 2, hy + WALL, WALL_HEIGHT / 2], **appearance)
            spec.worldbody.add_geom(name=f"tray_{index}_y_{side}", pos=[x, y + sign * (hy + WALL / 2), z],
                                    size=[hx, WALL / 2, WALL_HEIGHT / 2], **appearance)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Sorting layouts are derived from the scene seed")
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31))
        self.plan, self.order = route(seed)
        for name, assignment in self.plan.items():
            color = COLORS[COLOR_BY_SHAPE[name]]
            for part in ("floor", "x_minus", "x_plus", "y_minus", "y_plus"):
                self.model.geom(f"tray_{assignment.slot}_{part}").rgba = color
        self._set_object(self.order[0])
        return super().reset(seed=seed, options={"cube_xy": self.plan[self.shape_name].xy})

    def _set_object(self, name):
        self.shape_name, self.shape = name, SHAPES[name]
        self.cube, self.cube_joint = self.objects[name]
        self.cube_qpos = int(self.model.jnt_qposadr[self.cube_joint])
        self.cube_dof = int(self.model.jnt_dofadr[self.cube_joint])

    def _reset_task(self, options):
        # These are the only writes to object qpos in the sorting scene, before the initial settle.
        for name, assignment in self.plan.items():
            _, joint = self.objects[name]
            addr = int(self.model.jnt_qposadr[joint])
            self.data.qpos[addr:addr + 7] = (*assignment.xy, self.rest_z, *yaw_quaternion(assignment.yaw))
        self._begin_attempt()

    def _begin_attempt(self):
        assignment = self.plan[self.shape_name]
        self.spawn_xy = np.asarray(assignment.xy)
        self.target = np.array([*TRAYS[assignment.slot], self.rest_z])
        self.goal_site.pos[:2] = self.target[:2]
        self.initial_yaw = assignment.yaw
        self.episode = Episode()
        self.place_ticks = self.max_place_ticks = 0
        self.outcome_ticks = self.max_outcome_ticks = 0
        self.transported = self.supported_at_target = self.released_at_target = False
        self.transport_drop = self.table_supported_at_goal = self.open_without_gripped_touchdown = False
        self.transport_lost_contact = False
        self.close_alignment_deg = None
        self.finished = False

    def select(self, name):
        """Retarget sensing and counters in the live simulation."""
        self._set_object(name)
        self._begin_attempt()
        contacts = measure_contacts(self, self.data)
        self.potential = self._potential(contacts)
        return self._observation(contacts)

    def contains(self, name, slot):
        geom, _ = self.objects[name]
        shape = SHAPES[name]
        axes = self.data.geom_xmat[geom].reshape(3, 3)[:2]
        if shape.cylinder:
            radius, _, height = shape.half_extents
            z_axis = np.abs(axes[:, 2])
            radius_xy = radius * np.sqrt(np.maximum(0.0, 1 - z_axis**2)) + height * z_axis
        else:
            radius_xy = np.abs(axes) @ shape.half_extents
        offset = np.abs(self.data.geom_xpos[geom, :2] - TRAYS[slot])
        return bool(np.all(offset + radius_xy <= TRAY_HALF + 1e-5))

    def _sample_hold(self, height, contacts):
        previous_outcome, previous_strict = self.max_outcome_ticks, self.max_place_ticks
        super()._sample_hold(height, contacts)
        if (self.outcome_ticks or self.place_ticks) and not self.contains(self.shape_name, self.plan[self.shape_name].slot):
            self.outcome_ticks = self.place_ticks = 0
            self.max_outcome_ticks, self.max_place_ticks = previous_outcome, previous_strict

    def _task_success(self):
        return self.outcome_ticks >= self.hold_ticks

    def final_tray(self, name):
        geom, joint = self.objects[name]
        view = SimpleNamespace(cube=geom, table=self.table, gripper=self.gripper, model=self.model)
        contacts = measure_contacts(view, self.data)
        start = int(self.model.jnt_dofadr[joint])
        velocity = self.data.qvel[start:start + 6]
        height_ok = abs(self.data.geom_xpos[geom, 2] - self.rest_z) <= REST_TOLERANCE_M
        if (not height_ok or not contacts.supported or contacts.left_count or contacts.right_count
                or np.linalg.norm(velocity[:3]) > LINEAR_LIMIT or np.linalg.norm(velocity[3:]) > ANGULAR_LIMIT):
            return None
        return next((slot for slot in range(len(TRAYS)) if self.contains(name, slot)), None)

    def return_home(self, frame=None):
        """Open and return through real joint-servo motion before the next learned attempt."""
        current = self.tcp.copy()
        height = max(current[2], 0.445)
        waypoints = ([*current[:2], height], [*START[:2], height], START)
        steps = 0
        for index, waypoint in enumerate(waypoints):
            for _ in range(80):
                delta = waypoint - self.tcp
                turn = 0.0 if index == 0 else float(np.clip(-self.tool_yaw / YAW_STEP, -1, 1))
                if np.linalg.norm(delta) < 0.005 and (index == 0 or abs(self.tool_yaw) < 0.04):
                    break
                direction = delta * min(1.0, 0.011 / max(np.linalg.norm(delta), 0.011))
                orientation = self._orientation_target([0, 0, 0, 1, turn])
                ref, error = self.ik.iterate(self.tcp + direction, self.data.qpos[self.arm.qpos], orientation, 5, 5e-5)
                self._servo_interval(ref if error <= 0.005 else self.data.qpos[self.arm.qpos], frame)
                steps += 1
            else:
                return steps, False
        for _ in range(4):
            self._servo_interval(self.start_q, frame)
            steps += 1
        return steps, True

    def _servo_interval(self, reference, frame):
        self.gripper.command(self.data, 1.0)
        for _ in range(self.frame_skip):
            self._physics(reference)
        mujoco.mj_forward(self.model, self.data)
        if frame is not None:
            frame(self.render())


def load_mixed(seed, source):
    folder = source / "training" / f"seed_{seed}"
    weights = folder / "model.pt"
    receipt = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    identity = sha(weights)
    if identity != receipt["checkpoint_sha256"]:
        raise ValueError(f"Mixed-object weights differ from the archive: {weights}")
    model = Clone(OBS_SIZE, len(ACTION_NAMES))
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    model.eval()
    return model, identity


def run_scene(env, seed, actor=None, frame=None, demonstrations=None):
    if demonstrations is not None and actor is not None:
        raise ValueError("Only expert actions can be recorded as demonstrations")
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
        expert = YawTeacher(env, brick_long_axis=True) if actor is None else None
        trajectory = [] if demonstrations is not None else None
        while True:
            action = expert(obs)[1] if expert is not None else actor.action(obs)
            following, reward, terminated, truncated, info = env.step(action)
            if trajectory is not None:
                trajectory.append((obs, action, action, following, reward, terminated, truncated, expert.state.name))
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
    scene = {"scenario_seed": seed, "order": "/".join(env.order),
             "tray_colors": "/".join(slot_colors[i] for i in range(len(TRAYS))),
             "sorted_items": sum(row["sorted"] for row in items), "all_sorted": all(row["sorted"] for row in items),
             "wrong_color_items": sum(row["wrong_color"] for row in items), "homing_steps": homing_steps,
             "homing_failures": sum(not row["returned_home"] for row in items)}
    return scene, items


def evaluate(method, training_seed, source, output, scenarios):
    actor, identity = load_mixed(training_seed, source) if method == "mixed" else (None, None)
    evaluate_actor(method, training_seed, actor, identity, output, scenarios)


def evaluate_actor(method, training_seed, actor, identity, output, scenarios, scene_type=SortingScene):
    directory = output / (method if training_seed is None else f"{method}/seed_{training_seed}")
    scenes, objects = [], []
    with scene_type() as env:
        for seed in scenarios:
            row, items = run_scene(env, seed, actor)
            scenes.append(row)
            objects.extend(items)
    summary = {
        "method": method, "seed": training_seed, "checkpoint_sha256": identity,
        "scenes": len(scenes), "all_sorted": sum(r["all_sorted"] for r in scenes),
        "items": len(objects), "sorted_items": sum(r["sorted"] for r in objects),
        "wrong_color_items": sum(r["wrong_color"] for r in objects),
        "homing_failures": sum(r["homing_failures"] for r in scenes),
        "per_shape": {name: sum(row["shape"] == name and row["sorted"] for row in objects) for name in SHAPES},
    }
    write_csv(directory / "scenes.csv", scenes)
    write_csv(directory / "items.csv", objects)
    write_json(directory / "summary.json", summary)
    print(directory, summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "test"))
    parser.add_argument("--method", choices=("expert", "mixed"), default="expert")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--source", type=Path, default=OBJECT_RESULTS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if (args.method == "mixed") != (args.seed is not None):
        parser.error("Provide a training seed for mixed BC")
    torch.set_num_threads(1)
    if args.command == "pilot":
        evaluate(args.method, args.seed, args.source, args.out / "pilot", PILOT)
    else:
        evaluate(args.method, args.seed, args.source, args.out / "test", TEST)


if __name__ == "__main__":
    main()
