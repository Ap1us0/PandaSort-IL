"""Target-conditioned Panda pick, carry and place task."""

import gymnasium as gym
import numpy as np

from .env import (
    ANGULAR_LIMIT, LINEAR_LIMIT, LIFT_M, OBS_SIZE, PandaGraspEnv,
)

GOAL_LOW = np.array([0.45, 0.035])
GOAL_HIGH = np.array([0.55, 0.135])
GOAL_RADIUS = 0.022
RETRACT_M = 0.055
REST_TOLERANCE_M = 0.006
GRIP_OPEN_M = 0.062
PLACE_OBS_SIZE = OBS_SIZE + 7


class PandaPickPlaceEnv(PandaGraspEnv):
    """The action controls an ordinary Cartesian servo and the tendon gripper."""

    work_high = np.array([0.615, 0.18, 0.52])
    render_goal = True

    def __init__(self, render_mode=None, max_steps=240):
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (PLACE_OBS_SIZE,), dtype=np.float32)
        self.goal_site = self.model.site("place_goal")
        self.goal_site.size[0] = GOAL_RADIUS

    def _reset_task(self, options):
        if "goal_xy" in options:
            xy = np.asarray(options["goal_xy"], dtype=np.float64)
            if xy.shape != (2,) or not np.isfinite(xy).all() or np.any(xy < GOAL_LOW) or np.any(xy > GOAL_HIGH):
                raise ValueError("goal_xy must lie in the placement region")
        else:
            xy = self.np_random.uniform(GOAL_LOW, GOAL_HIGH)
        self.target = np.array([*xy, self.rest_z])
        self.goal_site.pos[:2] = xy
        self.place_ticks = 0
        self.max_place_ticks = 0
        self.transported = False
        self.supported_at_target = False
        self.released_at_target = False

    def _observation(self, contacts):
        goal_features = np.array([
            *((self.target[:2] - self.cube_xyz[:2]) / 0.25),
            *((self.target[:2] - self.tcp[:2]) / 0.25),
            self.episode.ever_carried, self.transported,
            self.place_ticks / self.hold_ticks,
        ], dtype=np.float32)
        return np.concatenate((super()._observation(contacts), goal_features))

    @property
    def goal_distance(self):
        return float(np.linalg.norm(self.cube_xyz[:2] - self.target[:2]))

    def _sample_hold(self, height, contacts):
        ep = self.episode
        ep.ever_bilateral |= contacts.gripping
        carrying = contacts.gripping and not contacts.table_count and height >= LIFT_M
        ep.ever_carried |= carrying
        near_goal = self.goal_distance <= GOAL_RADIUS
        self.transported |= carrying and near_goal
        if ep.ever_carried and not (self.transported and near_goal):
            ep.dropped |= bool(contacts.table_count or height < 0.025)

        stable = (np.linalg.norm(self.cube_velocity[:3]) <= LINEAR_LIMIT
                  and np.linalg.norm(self.cube_velocity[3:]) <= ANGULAR_LIMIT)
        ep.hold_ticks = ep.hold_ticks + 1 if carrying and stable else 0
        ep.max_hold_ticks = max(ep.max_hold_ticks, ep.hold_ticks)
        supported = self.transported and near_goal and contacts.supported and abs(height) <= REST_TOLERANCE_M
        self.supported_at_target |= supported and contacts.gripping
        opened = np.sum(self.data.qpos[list(self.gripper.fingers)]) >= GRIP_OPEN_M
        released = self.supported_at_target and supported and opened and not (contacts.left_count or contacts.right_count)
        self.released_at_target |= released
        clear = self.tcp[2] >= self.cube_xyz[2] + RETRACT_M
        self.place_ticks = self.place_ticks + 1 if released and stable and clear else 0
        self.max_place_ticks = max(self.max_place_ticks, self.place_ticks)

    def _task_success(self):
        return self.place_ticks >= self.hold_ticks

    def _timeout_reason(self):
        if self.released_at_target:
            return "placement_unstable_or_hand_not_clear"
        if self.supported_at_target:
            return "not_released_at_target"
        if self.transported:
            return "no_gripped_touchdown_at_target"
        if self.episode.ever_carried and not self.episode.dropped:
            return "not_transported_to_target"
        return super()._timeout_reason()

    def _potential(self, contacts):
        potential = super()._potential(contacts)
        potential.update(
            transport=1.5 * self.episode.ever_carried * np.exp(-self.goal_distance / 0.08),
            supported=0.8 * self.supported_at_target,
            released=0.8 * self.released_at_target,
            settled=2.0 * self.place_ticks / self.hold_ticks,
        )
        return potential

    def _info(self, contacts, status):
        return {
            **super()._info(contacts, status),
            "goal_x": float(self.target[0]), "goal_y": float(self.target[1]),
            "goal_distance_m": self.goal_distance,
            "transported": bool(self.transported),
            "supported_at_target": bool(self.supported_at_target),
            "released_at_target": bool(self.released_at_target),
            "settled_at_target_s": self.max_place_ticks * self.dt,
        }
