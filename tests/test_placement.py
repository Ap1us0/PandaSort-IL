"""Physical placement regressions and lift-policy transfer."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import gymnasium as gym
import numpy as np
import torch

from panda_grasp.env import OBS_SIZE
from panda_grasp.learning import Clone
from panda_grasp.place import PandaPickPlaceEnv
from panda_grasp.place_cli import expand_lift_clone
from panda_grasp.place_expert import PlacementExpert


class PlacementTest(unittest.TestCase):
    def test_seeded_target_observation_and_marker(self):
        with gym.make("PandaPickPlace-v0") as wrapped:
            env = wrapped.unwrapped
            obs, info = wrapped.reset(seed=141)
            following, repeated = wrapped.reset(seed=141)
            np.testing.assert_array_equal(obs, following)
            np.testing.assert_array_equal(env.goal_site.pos[:2], [info["goal_x"], info["goal_y"]])
            self.assertEqual(info["goal_y"], repeated["goal_y"])
            self.assertTrue(env.observation_space.contains(obs))
            _, another = wrapped.reset(seed=142)
            self.assertNotEqual(another["goal_y"], info["goal_y"])

    def test_teacher_places_and_early_release_is_not_success(self):
        with PandaPickPlaceEnv() as env:
            for early_release in (False, True):
                obs, _ = env.reset(seed=1_200_000)
                expert = PlacementExpert(env)
                while True:
                    _, action = expert(obs)
                    if early_release and env.transported:
                        # Drop the cube over the target, then retract the open hand.
                        action[:] = [0, 0, 0.7, 1]
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                self.assertTrue(info["transported"])
                self.assertEqual(info["is_success"], not early_release)
                self.assertEqual(info["supported_at_target"], not early_release)
                if early_release:
                    self.assertLessEqual(info["goal_distance_m"], 0.022)
                    self.assertEqual(info["settled_at_target_s"], 0.0)
                else:
                    self.assertTrue(info["released_at_target"])
                    self.assertAlmostEqual(info["settled_at_target_s"], 1.0)

    def test_expansion_preserves_lift_policy_outputs(self):
        lift = Clone()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lift.pt"
            torch.save(lift.state_dict(), path)
            placement = expand_lift_clone(path)
        observation = torch.randn(16, placement.features[0].in_features)
        torch.testing.assert_close(placement(observation), lift(observation[:, :OBS_SIZE]))


if __name__ == "__main__":
    unittest.main()
