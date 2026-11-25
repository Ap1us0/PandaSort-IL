"""Check that placement probes isolate actions and demonstrations verify contact."""

from types import SimpleNamespace
import unittest

import numpy as np

from panda_grasp._vendor.pickup import State
from panda_grasp.place_robustness import ShiftedPlacementEnv, grip_probe
from panda_grasp.place_touchdown import DOMAINS, MODERATE, VALIDATION_SCENES, build_teacher, slow_descent


class TouchdownStudyTest(unittest.TestCase):
    def test_probes_only_change_intended_action_after_arrival(self):
        action = np.array([0.23, -0.17, -0.63, 0.72], dtype=np.float32)
        original = action.copy()
        for wrapper, altered in (
            (grip_probe, [0.23, -0.17, -0.63, -1.0]),
            (slow_descent, [0.23, -0.17, -0.315, 0.72]),
        ):
            env = SimpleNamespace(transported=False, supported_at_target=False)
            call = wrapper(lambda env, seed: lambda obs: action)(env, 0)
            np.testing.assert_array_equal(call(None), original)
            env.transported = True
            np.testing.assert_allclose(call(None), altered)
            np.testing.assert_array_equal(action, original)
            env.supported_at_target = True
            np.testing.assert_array_equal(call(None), original)

    def test_verified_teacher_waits_for_gripped_touchdown(self):
        with ShiftedPlacementEnv(DOMAINS[MODERATE]) as env:
            obs, _ = env.reset(seed=VALIDATION_SCENES[0])
            expert = build_teacher(env, "verified_touch")
            reached_target = False
            while True:
                _, action = expert(obs)
                if expert.state == State.RETREAT:
                    self.assertTrue(env.supported_at_target)
                obs, _, terminated, truncated, info = env.step(action)
                reached_target |= info["transported"]
                if terminated or truncated:
                    break
            self.assertTrue(reached_target)
            self.assertFalse(info["open_without_gripped_touchdown"])


if __name__ == "__main__":
    unittest.main()
