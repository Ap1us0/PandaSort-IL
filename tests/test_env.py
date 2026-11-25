"""Gymnasium and physics regressions for the lift benchmark."""

import unittest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np

from panda_grasp.env import GAMMA, HOLD_S, LIFT_M, PandaGraspEnv
from panda_grasp.expert import PickupExpert


SEED = 909_000


class GraspTests(unittest.TestCase):
    def setUp(self):
        self.env = PandaGraspEnv()
        self.addCleanup(self.env.close)

    def test_registration_and_gymnasium_contract(self):
        with gym.make("PandaGrasp-v0") as registered:
            check_env(registered.unwrapped, skip_render_check=True)

    def test_seeded_reset_and_action_bounds(self):
        first, _ = self.env.reset(seed=SEED)
        second, _ = self.env.reset(seed=SEED)
        np.testing.assert_array_equal(first, second)
        with self.assertRaises(ValueError):
            self.env.step([0, 0, 0, np.nan])
        self.env.step([2, 2, 2, -2])
        self.assertEqual(self.env.data.ctrl[self.env.gripper.actuator], self.env.gripper.closed)

    def run_sequence(self, mode):
        env = self.env
        obs, _ = env.reset(seed=SEED)
        teacher = PickupExpert(env)
        returns = 0.0
        sum_shaping = 0.0
        initial_phi = sum(env.potential.values())
        released = False
        while True:
            _, act = teacher(obs)
            if mode == "open":
                act[3] = 1
            elif mode == "closed_idle":
                act[:] = [0, 0, 0, -1]
            elif mode == "open_idle":
                act[:] = [0, 0, 0, 1]
            elif mode == "contact_only" and teacher.closed_steps >= 8:
                act[:] = [0, 0, 0, -1]
            elif mode == "short_lift" and env.tcp[2] > env.rest_z + 0.030:
                if act[2] > 0 and act[3] < 0:
                    act[2] = 0
            elif mode == "release" and (released or env.episode.hold_ticks * env.dt >= 0.30):
                released = True
                act[:] = [0, 0, 0, 1]
            obs, rew, term, trunc, info = env.step(act)
            weight = GAMMA ** (env.episode.steps - 1)
            returns += weight * rew
            sum_shaping += weight * sum(info["reward_parts"][k] for k in env.potential)
            if term or trunc:
                break
        self.assertAlmostEqual(sum_shaping, (GAMMA ** env.episode.steps) * sum(env.potential.values()) - initial_phi)
        boundary = (GAMMA ** env.episode.steps) * sum(env.potential.values())
        return info, returns - boundary, term, trunc

    def test_physical_success_and_cube_qpos_only_moves_in_physics(self):
        env = self.env
        env.reset(seed=SEED)
        original_step = env._physics
        after_last = None
        cube = slice(env.cube_qpos, env.cube_qpos + 7)

        def watch(reference):
            nonlocal after_last
            if after_last is not None:
                np.testing.assert_array_equal(after_last, env.data.qpos[cube])
            original_step(reference)
            after_last = env.data.qpos[cube].copy()

        env._physics = watch
        self.assertFalse(np.any(env.model.eq_type == mujoco.mjtEq.mjEQ_WELD))
        info, reward, terminated, truncated = self.run_sequence("expert")
        np.testing.assert_array_equal(after_last, env.data.qpos[cube])
        self.assertTrue(terminated and not truncated and info["is_success"])
        self.assertGreaterEqual(info["best_bilateral_lift_m"], LIFT_M)
        self.assertGreaterEqual(info["max_hold_s"], HOLD_S)
        self.assertGreater(reward, 0)
        self.assertEqual(int(env.data.eq_active.sum()), 1)

    def test_reward_and_success_shortcuts(self):
        for mode in ("open", "closed_idle", "open_idle", "contact_only", "short_lift", "release"):
            with self.subTest(mode=mode):
                info, reward, terminated, truncated = self.run_sequence(mode)
                self.assertFalse(info["is_success"])
                self.assertGreaterEqual(info["max_hold_s"], 0)
                self.assertLess(info["max_hold_s"], HOLD_S)
                # A time limit leaves a value to bootstrap. Remove that boundary
                # term when testing whether a loop can accumulate shaping.
                self.assertLess(reward, 0)
                if mode in ("closed_idle", "open_idle", "contact_only", "short_lift"):
                    self.assertTrue(truncated and not terminated)
                    self.assertFalse(info["ever_carried"])
                if mode == "contact_only":
                    self.assertTrue(info["ever_bilateral"])
                if mode == "short_lift":
                    self.assertTrue(info["ever_bilateral"])
                    self.assertGreater(info["best_bilateral_lift_m"], 0.02)
                if mode == "release":
                    self.assertTrue(info["ever_carried"] and info["dropped"])
                    self.assertEqual(self.env.episode.hold_ticks, 0)


if __name__ == "__main__":
    unittest.main()
