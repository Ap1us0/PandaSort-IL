"""Verify placement diagnostics with physically executed trajectories."""

import unittest

from panda_grasp.physics_shift import CONDITIONS
from panda_grasp.place_robustness import ShiftedPlacementEnv, teacher_policy


class PlacementRobustnessTest(unittest.TestCase):
    def execute(self, mode):
        with ShiftedPlacementEnv(CONDITIONS["nominal"]) as env:
            obs, _ = env.reset(seed=1_200_000)
            act = teacher_policy(1.0)(env, 1_200_000)
            while True:
                release_early = mode == "after_transport" and env.transported
                drop_in_transit = mode == "after_lift" and env.episode.ever_carried
                action = [0, 0, 0.7, 1] if release_early or drop_in_transit else act(obs)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    return info

    def test_teacher_has_no_false_transport_drop_or_early_open(self):
        info = self.execute("normal")
        self.assertTrue(info["is_success"])
        self.assertTrue(info["table_supported_at_goal"])
        self.assertFalse(info["transport_drop"])
        self.assertFalse(info["open_without_gripped_touchdown"])

    def test_release_failure_is_distinct_from_transport_drop(self):
        after_transport = self.execute("after_transport")
        self.assertFalse(after_transport["is_success"])
        self.assertEqual(after_transport["failure_stage"], "touchdown")
        self.assertTrue(after_transport["table_supported_at_goal"])
        self.assertTrue(after_transport["open_without_gripped_touchdown"])
        self.assertFalse(after_transport["transport_drop"])

        after_lift = self.execute("after_lift")
        self.assertFalse(after_lift["is_success"])
        self.assertEqual(after_lift["failure_stage"], "transport")
        self.assertTrue(after_lift["transport_drop"])


if __name__ == "__main__":
    unittest.main()
