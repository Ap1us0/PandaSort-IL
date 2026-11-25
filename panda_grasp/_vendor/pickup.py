"""Selected gripper, contact and waypoint logic from 夹取/src/panda_pick_place.py."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

import mujoco
import numpy as np

MIN_NORMAL_N = 0.15
MAX_CONTACT_DEPTH_M = 0.001


@dataclass(frozen=True)
class Gripper:
    actuator: int
    fingers: tuple[int, int]
    bodies: tuple[int, int]
    closed: float
    opened: float

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "Gripper":
        actuator = model.actuator("actuator8")
        tendon = model.tendon("split")
        left = model.joint("finger_joint1")
        right = model.joint("finger_joint2")
        fingers = left.id, right.id
        start = model.tendon_adr[tendon.id]
        stop = start + model.tendon_num[tendon.id]
        if not (
            actuator.trntype[0] == mujoco.mjtTrn.mjTRN_TENDON
            and actuator.trnid[0] == tendon.id
            and actuator.ctrllimited[0] and actuator.forcelimited[0]
            and all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE for j in fingers)
            and np.array_equal(model.wrap_objid[start:stop], fingers)
            and np.allclose(model.wrap_prm[start:stop], 0.5)
        ):
            raise ValueError("Expected actuator8 to drive both sliding fingers through the split tendon")
        return cls(
            actuator.id, (int(left.qposadr[0]), int(right.qposadr[0])),
            (model.body("left_finger").id, model.body("right_finger").id),
            *map(float, actuator.ctrlrange),
        )

    def command(self, data: mujoco.MjData, opening: float) -> None:
        """Set normalized opening, where 0 closes, 1 opens, and a constant holds."""
        if not 0.0 <= opening <= 1.0:
            raise ValueError("Gripper opening must be within [0, 1]")
        data.ctrl[self.actuator] = self.closed + opening * (self.opened - self.closed)


class State(Enum):
    SETTLE = auto()
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    HOLD = auto()
    TRANSPORT = auto()
    PLACE = auto()
    RETREAT = auto()


class Contacts(NamedTuple):
    left_count: int
    right_count: int
    table_count: int
    left_normal_n: float
    right_normal_n: float
    table_normal_n: float
    finger_depth_m: float
    table_depth_m: float
    unexpected: str

    @property
    def gripping(self) -> bool:
        return min(self.left_normal_n, self.right_normal_n) >= MIN_NORMAL_N

    @property
    def supported(self) -> bool:
        return self.table_normal_n >= MIN_NORMAL_N


def measure_contacts(task, data: mujoco.MjData) -> Contacts:
    counts = [0, 0, 0]
    normal = [0.0, 0.0, 0.0]
    finger_depth = table_depth = 0.0
    unexpected = ""
    force = np.empty(6)
    model = task.model
    for i, contact in enumerate(data.contact):
        a, b = map(int, contact.geom)
        other = b if a == task.cube else a
        body = model.geom_bodyid[other]
        depth = max(0.0, -contact.dist)
        if task.cube in (a, b) and other == task.table:
            index = 2
            table_depth = max(table_depth, depth)
        elif task.cube in (a, b) and body in task.gripper.bodies:
            index = task.gripper.bodies.index(body)
            finger_depth = max(finger_depth, depth)
        else:
            if contact.efc_address >= 0 or depth > 1e-6:
                labels = [model.geom(g).name or model.body(int(model.geom_bodyid[g])).name for g in (a, b)]
                unexpected = f"{labels[0]} / {labels[1]} ({1000 * depth:.3f} mm overlap)"
            continue
        if contact.efc_address >= 0:
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > 0:
                counts[index] += 1
                normal[index] += force[0]
    return Contacts(*counts, *normal, finger_depth, table_depth, unexpected)


def pickup_targets(pickup):
    """Approach, descend and lift waypoints from the original controller."""
    overhead = np.array([0.0, 0.0, 0.115])
    carried = np.array([0.0, 0.0, 0.09])
    return pickup + overhead, pickup, pickup + carried
