"""Regression checks for geometry/yaw sampling and legacy compatibility."""

import unittest

import gymnasium as gym
import numpy as np

from panda_grasp.place_boundary import BoundaryEvaluation, CANDIDATES
from panda_grasp.place_objects import ObjectPlacementEnv, YAW_LIMIT
from panda_grasp.place_transport_diagnosis import LOW


class ObjectPlacementTest(unittest.TestCase):
    def test_original_path_and_paired_scene_sampling(self):
        with BoundaryEvaluation(CANDIDATES[LOW]) as old, ObjectPlacementEnv("cube", False, False) as fixed:
            a, old_info = old.reset(seed=1550005)
            b, fixed_info = fixed.reset(seed=1550005)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(old_info["goal_x"], fixed_info["goal_x"])
            for _ in range(5):
                action = [0.08, -0.05, -0.15, 1.0]
                x, ar, at, ax, _ = old.step(action)
                y, br, bt, bx, _ = fixed.step(action)
                np.testing.assert_array_equal(x, y)
                self.assertEqual((ar, at, ax), (br, bt, bx))
        layouts = []
        for name in ("cube", "brick", "cylinder"):
            with gym.make("PandaObjectPlace-v0", shape=name) as env:
                obs, info = env.reset(seed=1550005)
                self.assertEqual(obs.shape, (52,))
                self.assertEqual(env.action_space.shape, (5,))
                self.assertAlmostEqual(env.unwrapped.mass_kg, 0.15)
                self.assertLessEqual(abs(info["initial_yaw_deg"]), 90)
                layouts.append([info[key] for key in ("cube_x", "cube_y", "goal_x", "goal_y", "initial_yaw_deg")])
        np.testing.assert_array_equal(layouts[0], layouts[1])
        np.testing.assert_array_equal(layouts[0], layouts[2])

    def test_wrist_action_moves_wrist_through_physics(self):
        with ObjectPlacementEnv("brick", True, True) as env:
            env.reset(seed=1550006, options={"object_yaw": -0.9})
            start = env.tool_yaw
            for _ in range(12):
                env.step([0, 0, 0, 1, 1])
            self.assertGreater(env.tool_yaw - start, 0.12)
            self.assertLess(abs(env.tool_yaw), YAW_LIMIT + 0.05)
            self.assertAlmostEqual(env.object_yaw, -0.9, places=5)
