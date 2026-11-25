"""Physical cube lift with a Cartesian delta and an independent gripper action."""

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from ._vendor.cartesian import InverseKinematics, joint_controls, read_arm, tcp_position
from ._vendor.pickup import Gripper, measure_contacts
from ._vendor.position import SCENE, load_panda_spec


ACTION_STEP = 0.015
START = np.array([0.50, -0.10, 0.41])
SPAWN_LOW = np.array([0.455, -0.145])
SPAWN_HIGH = np.array([0.545, -0.055])
WORK_LOW = np.array([0.385, -0.225, 0.280])
WORK_HIGH = np.array([0.615, 0.025, 0.52])
LIFT_M = 0.05
HOLD_S = 1.0
LINEAR_LIMIT = 0.03
ANGULAR_LIMIT = 0.5
GRIP_FORCE = 0.15
CONTROL_HZ = 20
GAMMA = 0.99

# Physical state, scaled by characteristic magnitudes without fitted statistics.
OBS_FIELDS = {
    "joint_position_rad": (7, 1.0),
    "joint_velocity_rad_s": (7, 2.0),
    "tcp_position_m": (3, 0.5),
    "tcp_velocity_m_s": (3, 0.3),
    "cube_position_m": (3, 0.5),
    "cube_velocity_m_s": (3, 0.3),
    "cube_angular_velocity_rad_s": (3, 2.0),
    "finger_position_m": (2, 0.04),
    "tcp_to_cube_m": (3, 0.12),
    "grasp_force_indicator": (2, 1.0),
    "hold_fraction": (1, 1.0),
}
OBS_SIZE = sum(size for size, _ in OBS_FIELDS.values())
OBS_SCALE = np.repeat([scale for _, scale in OBS_FIELDS.values()],
                      [size for size, _ in OBS_FIELDS.values()])


@dataclass
class Episode:
    steps: int = 0
    hold_ticks: int = 0
    max_hold_ticks: int = 0
    max_lift: float = 0.0
    best_bilateral_lift: float = 0.0
    min_distance: float = float("inf")
    ever_bilateral: bool = False
    ever_carried: bool = False
    dropped: bool = False
    ik_failures: int = 0


class PandaGraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}
    work_high = WORK_HIGH
    render_goal = False

    def __init__(self, render_mode=None, max_steps=120):
        super().__init__()
        if render_mode not in (None, "rgb_array") or max_steps < 1:
            raise ValueError("Expected rgb_array or None and a positive episode horizon")
        self.render_mode = render_mode
        self.max_steps = max_steps
        spec = self._scene_spec()
        if not self.render_goal:
            spec.delete(spec.site("place_goal"))
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.frame_skip = round(1 / (CONTROL_HZ * self.dt))
        self.hold_ticks = round(HOLD_S / self.dt)
        if not np.isclose(self.frame_skip * self.dt, 1 / CONTROL_HZ):
            raise ValueError("Control interval must divide into whole physics steps")
        if np.any(self.model.eq_type == mujoco.mjtEq.mjEQ_WELD):
            raise ValueError("The lift scene must have no weld constraints")
        self.arm = read_arm(self.model)
        self.gripper = Gripper.from_model(self.model)
        self.cube = self.model.geom("cube").id
        self.table = self.model.geom("table").id
        self.cube_joint = self.model.joint("cube_free").id
        self.cube_qpos = int(self.model.jnt_qposadr[self.cube_joint])
        self.cube_dof = int(self.model.jnt_dofadr[self.cube_joint])
        self.home_key = self.model.key("home").id
        self.rest_z = float(self.model.geom_pos[self.table, 2] +
                            self.model.geom_size[self.table, 2] + self._object_half_height())
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        mujoco.mj_forward(self.model, self.data)
        self.ik = InverseKinematics(self.model, self.data, self.arm)
        self.start_q = self.ik.solve(START, self.data.qpos[self.arm.qpos], stage="reset arm")
        self.stationary = np.zeros(7)
        self.jacobian = np.zeros((3, self.model.nv))
        self.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.renderer = None
        self.finished = True

    def _scene_spec(self):
        return load_panda_spec(SCENE.with_name("pick_place.xml"))

    def _object_half_height(self):
        return self.model.geom_size[self.cube, 2]

    def _orientation_target(self, action):
        return self.ik.home_orientation

    @property
    def tcp(self):
        return tcp_position(self.data, self.arm)

    @property
    def cube_xyz(self):
        return self.data.geom_xpos[self.cube]

    @property
    def cube_velocity(self):
        return self.data.qvel[self.cube_dof:self.cube_dof + 6]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "cube_xy" in options:
            xy = np.asarray(options["cube_xy"], dtype=np.float64)
            if xy.shape != (2,) or not np.isfinite(xy).all() or np.any(xy < SPAWN_LOW) or np.any(xy > SPAWN_HIGH):
                raise ValueError("cube_xy must lie in the benchmark spawn region")
        else:
            xy = self.np_random.uniform(SPAWN_LOW, SPAWN_HIGH)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        self.data.qpos[self.arm.qpos] = self.start_q
        self.data.ctrl[self.arm.actuators] = self.start_q
        self.data.qpos[self.cube_qpos:self.cube_qpos + 7] = (*xy, self.rest_z, 1, 0, 0, 0)
        self.gripper.command(self.data, 1.0)
        self._reset_task(options)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.frame_skip * 4):
            self._physics(self.start_q)
        mujoco.mj_forward(self.model, self.data)
        self.spawn_xy = xy.copy()
        self.episode = Episode()
        self.finished = False
        contacts = measure_contacts(self, self.data)
        self.potential = self._potential(contacts)
        return self._observation(contacts), self._info(contacts, "running")

    def _reset_task(self, options):
        pass

    def _physics(self, reference):
        mujoco.mj_step1(self.model, self.data)
        self.data.ctrl[self.arm.actuators] = joint_controls(self.data, self.arm, reference, self.stationary)
        mujoco.mj_step2(self.model, self.data)

    def _observation(self, contacts):
        p = self.tcp
        mujoco.mj_jac(self.model, self.data, self.jacobian, None, p, self.arm.hand)
        values = np.concatenate((
            self.data.qpos[self.arm.qpos], self.data.qvel[self.arm.dofs],
            p, self.jacobian @ self.data.qvel,
            self.cube_xyz, self.cube_velocity,
            self.data.qpos[list(self.gripper.fingers)],
            self.cube_xyz - p,
            [contacts.left_normal_n >= GRIP_FORCE, contacts.right_normal_n >= GRIP_FORCE,
             min(self.episode.hold_ticks / self.hold_ticks, 1.0)],
        ))
        return (values / OBS_SCALE).astype(np.float32)

    def _sample_hold(self, height, contacts):
        ep = self.episode
        ep.ever_bilateral |= contacts.gripping
        carrying = contacts.gripping and not contacts.table_count and height >= LIFT_M
        ep.ever_carried |= carrying
        if ep.ever_carried and (contacts.table_count or height < 0.025):
            ep.dropped = True
        stable = (carrying and np.linalg.norm(self.cube_velocity[:3]) <= LINEAR_LIMIT
                  and np.linalg.norm(self.cube_velocity[3:]) <= ANGULAR_LIMIT)
        ep.hold_ticks = ep.hold_ticks + 1 if stable else 0
        ep.max_hold_ticks = max(ep.max_hold_ticks, ep.hold_ticks)

    def step(self, action):
        if self.finished:
            raise gym.error.ResetNeeded("Call reset before stepping the environment")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Expected {self.action_space.shape[0]} finite action components")
        action = np.clip(action, -1, 1)
        ep = self.episode
        ep.steps += 1
        current = self.data.qpos[self.arm.qpos]
        goal = np.clip(self.tcp + ACTION_STEP * action[:3], WORK_LOW, self.work_high)
        reference, error = self.ik.iterate(goal, current, self._orientation_target(action), 5, 5e-5)
        if error > 0.005:
            reference = current
            ep.ik_failures += 1
        self.gripper.command(self.data, (1 + action[3]) / 2)

        before = self.data.time
        for tick in range(1, self.frame_skip + 1):
            self._physics(reference)
            height = float(self.cube_xyz[2]) - self.rest_z
            if height >= LIFT_M or ep.ever_carried:
                self._sample_hold(height, measure_contacts(self, self.data))
            else:
                ep.hold_ticks = 0
            if self._task_success():
                break
        mujoco.mj_forward(self.model, self.data)
        contacts = measure_contacts(self, self.data)
        height = float(self.cube_xyz[2]) - self.rest_z
        distance = float(np.linalg.norm(self.tcp - self.cube_xyz))
        ep.min_distance = min(ep.min_distance, distance)
        ep.max_lift = max(ep.max_lift, height)
        ep.ever_bilateral |= contacts.gripping
        if contacts.gripping and not contacts.table_count:
            ep.best_bilateral_lift = max(ep.best_bilateral_lift, height)

        valid = (abs(self.data.time - before - tick * self.dt) < self.dt / 2
                 and not self.data.warning.number.any()
                 and np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        success = valid and self._task_success()
        fallen = height < -0.045
        terminated = bool(success or fallen or not valid)
        truncated = bool(ep.steps >= self.max_steps and not terminated)
        status = "running"
        if success:
            status = "success"
        elif not valid:
            status = "invalid_dynamics"
        elif fallen:
            status = "cube_off_table"
        elif truncated:
            status = self._timeout_reason()

        potential = self._potential(contacts) if not terminated else dict.fromkeys(self.potential, 0.0)
        reward_parts = {key: GAMMA * value - self.potential[key] for key, value in potential.items()}
        self.potential = potential
        reward_parts.update(
            success=12.0 * success,
            effort=-0.003 * float(action[:3] @ action[:3]) - 0.01,
            failure=-3.0 * (fallen or not valid),
        )
        self.finished = terminated or truncated
        info = self._info(contacts, status)
        info["reward_parts"] = {k: float(v) for k, v in reward_parts.items()}
        obs = self._observation(contacts)
        if not valid:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(sum(reward_parts.values())), terminated, truncated, info

    def _task_success(self):
        return self.episode.hold_ticks >= self.hold_ticks

    def _timeout_reason(self):
        ep = self.episode
        if ep.dropped:
            return "dropped_after_lift"
        if ep.ever_carried:
            return "unstable_hold"
        if ep.ever_bilateral:
            return "grasp_without_lift"
        return "no_bilateral_grasp"

    def _potential(self, contacts):
        distance = np.linalg.norm(self.tcp - self.cube_xyz)
        height = self.cube_xyz[2] - self.rest_z
        return {
            "approach": 2.0 * np.exp(-distance / 0.045),
            "contact": 0.7 * contacts.gripping,
            "lift": 2.0 * np.clip(height / LIFT_M, 0, 1) * (contacts.gripping and not contacts.table_count),
            "hold": 2.0 * self.episode.hold_ticks / self.hold_ticks,
        }

    def _info(self, contacts, status):
        ep = self.episode
        return {
            "is_success": status == "success",
            "failure_reason": status,
            "cube_x": float(self.spawn_xy[0]),
            "cube_y": float(self.spawn_xy[1]),
            "max_lift_m": ep.max_lift,
            "best_bilateral_lift_m": ep.best_bilateral_lift,
            "max_hold_s": ep.max_hold_ticks * self.dt,
            "min_tcp_distance_m": ep.min_distance,
            "ever_bilateral": bool(ep.ever_bilateral),
            "ever_carried": bool(ep.ever_carried),
            "dropped": bool(ep.dropped),
            "ik_failures": ep.ik_failures,
            "left_force_n": float(contacts.left_normal_n),
            "right_force_n": float(contacts.right_normal_n),
        }

    def render(self):
        if self.render_mode != "rgb_array":
            raise ValueError("Initialize with render_mode='rgb_array' to render frames")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=800)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.44, 0.0 if self.render_goal else -0.06, 0.33]
        camera.distance = 1.35
        camera.azimuth = 128
        camera.elevation = -25
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
