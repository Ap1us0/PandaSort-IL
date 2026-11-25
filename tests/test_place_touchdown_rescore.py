"""A stable placement can satisfy the outcome while missing gripped touchdown."""

import unittest

from panda_grasp.physics_shift import CONDITIONS
from panda_grasp.place_robustness import teacher_policy
from panda_grasp.place_touchdown_rescore import OutcomeEvaluation


class OutcomeEvaluationTest(unittest.TestCase):
    def check_placement(self, early_release):
        with OutcomeEvaluation(CONDITIONS["nominal"]) as env:
            scene = 1_200_000
            obs, _ = env.reset(seed=scene)
            teacher = teacher_policy(1.0)(env, scene)
            while True:
                action = [0, 0, 0.7, 1] if early_release and env.transported else teacher(obs)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            return info, env.max_outcome_ticks >= env.hold_ticks

    def test_outcome_and_placement_sequence_are_scored_independently(self):
        good, outcome = self.check_placement(False)
        self.assertTrue(good["is_success"] and outcome)
        early, outcome = self.check_placement(True)
        self.assertFalse(early["is_success"])
        self.assertTrue(outcome)
        self.assertEqual(early["failure_stage"], "touchdown")


if __name__ == "__main__":
    unittest.main()
