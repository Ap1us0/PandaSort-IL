"""A controlled geometry and yaw extension of the original placement benchmark."""

from dataclasses import dataclass
import math

import gymnasium as gym
import mujoco
import numpy as np

from ._vendor.pickup import State
from .env import ACTION_STEP
from .place import PLACE_OBS_SIZE
from .place_boundary import BoundaryEvaluation, CANDIDATES
from .place_touchdown import TouchdownTeacher
from .place_transport_diagnosis import LOW

YAW_STEP = 0.08
YAW_LIMIT = math.pi / 2
OBS_SIZE = PLACE_OBS_SIZE + 8
ACTION_NAMES = ("dx", "dy", "dz", "gripper", "dyaw")


@dataclass(frozen=True)
class ObjectShape:
    half_extents: tuple[float, float, float]
    cylinder: bool = False
    rgba: tuple[float, float, float, float] = (0.98, 0.43, 0.12, 1.0)


SHAPES = {
    "cube": ObjectShape((0.0225, 0.0225, 0.0225)),
    "brick": ObjectShape((0.029, 0.016, 0.0225), rgba=(0.91, 0.50, 0.17, 1.0)),
    "cylinder": ObjectShape((0.0225, 0.0225, 0.0225), cylinder=True, rgba=(0.8, 0.33, 0.17, 1.0)),
}


class ObjectPlacementEnv(BoundaryEvaluation):
    """Hold mass/contact parameters fixed while varying shape and initial yaw."""

    def __init__(self, shape="cube", random_yaw=True, yaw_action=True, **kwargs):
        if shape not in SHAPES:
            raise ValueError(f"Unknown object shape: {shape}")
        self.shape_name = shape
        self.shape = SHAPES[shape]
        self.random_yaw = random_yaw
        self.yaw_action = yaw_action
        super().__init__(CANDIDATES[LOW], **kwargs)
        self.home_rotation = quaternion_matrix(self.ik.home_orientation)
        if yaw_action:
            self.action_space = gym.spaces.Box(-1, 1, shape=(len(ACTION_NAMES),), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)

    def _scene_spec(self):
        spec = super()._scene_spec()
        if self.shape_name == "cube":
            return spec
        geom = spec.geom("cube")
        body = spec.body("cube")
        x, y, z = self.shape.half_extents
        if self.shape.cylinder:
            geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            geom.size = [x, z, 0.0]
            transverse = body.mass * (3 * x**2 + 4 * z**2) / 12
            body.inertia = [transverse, transverse, body.mass * x**2 / 2]
        else:
            geom.size = [x, y, z]
            body.inertia = [body.mass * (b * b + c * c) / 3 for b, c in ((y, z), (x, z), (x, y))]
        geom.rgba = self.shape.rgba
        return spec

    def _object_half_height(self):
        return self.shape.half_extents[2]

    def _reset_task(self, options):
        super()._reset_task(options)
        if "object_yaw" in options:
            yaw = float(options["object_yaw"])
        else:
            yaw = self.np_random.uniform(-YAW_LIMIT, YAW_LIMIT) if self.random_yaw else 0.0
        if not math.isfinite(yaw) or not -YAW_LIMIT <= yaw <= YAW_LIMIT:
            raise ValueError("object_yaw must be finite and lie between -pi/2 and pi/2")
        self.initial_yaw = yaw
        self.close_alignment_deg = None
        self.data.qpos[self.cube_qpos + 3:self.cube_qpos + 7] = yaw_quaternion(yaw)

    @property
    def tool_yaw(self):
        relative = self.data.xmat[self.arm.hand].reshape(3, 3) @ self.home_rotation.T
        return math.atan2(relative[1, 0], relative[0, 0])

    @property
    def object_yaw(self):
        axes = self.data.geom_xmat[self.cube].reshape(3, 3)
        return math.atan2(axes[1, 0], axes[0, 0])

    def _orientation_target(self, action):
        if not self.yaw_action:
            return super()._orientation_target(action)
        command_yaw = float(np.clip(self.tool_yaw + YAW_STEP * action[4], -YAW_LIMIT, YAW_LIMIT))
        orientation = np.empty(4)
        mujoco.mju_mulQuat(orientation, yaw_quaternion(command_yaw), self.ik.home_orientation)
        return orientation

    def _observation(self, contacts):
        obs = super()._observation(contacts)
        if not self.yaw_action:
            return obs
        # Square, rectangular and circular cross-sections have different angular symmetries.
        symmetric_yaw = 0.0 if self.shape.cylinder else self.object_yaw * (4 if self.shape_name == "cube" else 2)
        extra = [
            math.cos(symmetric_yaw), math.sin(symmetric_yaw), math.cos(self.tool_yaw), math.sin(self.tool_yaw),
            *(size / 0.03 for size in self.shape.half_extents), float(self.shape.cylinder),
        ]
        return np.concatenate((obs, np.asarray(extra, dtype=np.float32)))

    def step(self, action):
        result = super().step(action)
        if (self.close_alignment_deg is None and not self.shape.cylinder and float(action[3]) <= -0.5
                and np.linalg.norm(self.tcp - self.cube_xyz) <= 0.03):
            period = math.pi / 2 if self.shape_name == "cube" else math.pi
            jaw = self.data.xmat[self.arm.hand].reshape(3, 3)[:, 1]
            delta = self.object_yaw - math.atan2(jaw[1], jaw[0])
            self.close_alignment_deg = math.degrees(abs((delta + period / 2) % period - period / 2))
        result[-1]["close_alignment_deg"] = self.close_alignment_deg
        return result

    def _info(self, contacts, status):
        return {**super()._info(contacts, status), "shape": self.shape_name,
                "initial_yaw_deg": math.degrees(self.initial_yaw)}


def yaw_quaternion(angle):
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


def quaternion_matrix(quaternion):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    return rotation.reshape(3, 3)


class YawTeacher(TouchdownTeacher):
    """Rotate the jaws to a grasp axis before descent while collecting demonstrations."""

    def __init__(self, env, brick_long_axis=False):
        super().__init__(env, verify=True, descent_scale=1.0)
        jaw_axis = env.home_rotation[:, 1]
        home_angle = math.atan2(jaw_axis[1], jaw_axis[0])
        period = math.pi / 2 if env.shape_name == "cube" else math.pi
        axis = env.object_yaw
        if env.shape_name == "brick" and not brick_long_axis:
            axis += math.pi / 2
        self.grasp_yaw = (axis - home_angle + period / 2) % period - period / 2
        if env.shape.cylinder:
            self.grasp_yaw = 0.0

    def __call__(self, observation):
        yaw_error = self.grasp_yaw - self.env.tool_yaw
        yaw_action = np.clip(yaw_error / YAW_STEP, -1, 1)
        if self.state == State.APPROACH and abs(yaw_error) > 0.05:
            delta = (self.approach - self.env.tcp) / ACTION_STEP
            delta *= min(1.0, 0.85 / max(0.85, np.linalg.norm(delta)))
            action = np.asarray([*delta, 1.0, yaw_action], dtype=np.float32)
            return action, action.copy()
        label, action = super().__call__(observation)
        yaw = np.asarray([yaw_action], dtype=np.float32)
        return np.concatenate((label, yaw)), np.concatenate((action, yaw))
