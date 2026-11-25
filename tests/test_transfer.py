"""Verify that cloning initializes exactly the supported SAC parameters."""

import unittest

from stable_baselines3 import SAC
import torch

from panda_grasp.env import PandaGraspEnv
from panda_grasp.place import PandaPickPlaceEnv
from panda_grasp.learning import ACTIVATION, LOG_STD_INIT, WIDTHS, Clone, initialize_actor


class TransferTest(unittest.TestCase):
    def test_actor_output_matches_and_critics_stay_fresh(self):
        torch.set_num_threads(1)
        for environment in (PandaGraspEnv, PandaPickPlaceEnv):
            with self.subTest(environment=environment.__name__), environment() as env:
                observation, _ = env.reset(seed=81)
                clone = Clone(env.observation_space.shape[0])
                model = SAC("MlpPolicy", env, device="cpu", seed=81, policy_kwargs={
                    "net_arch": WIDTHS, "activation_fn": ACTIVATION,
                })
                before = {key: value.clone() for key, value in model.critic.state_dict().items()}
                initialize_actor(model.actor, clone)
                action, _ = model.predict(observation, deterministic=True)
                torch.testing.assert_close(torch.tensor(action), torch.tensor(clone.action(observation)))
                for key, value in model.critic.state_dict().items():
                    torch.testing.assert_close(value, before[key], atol=0, rtol=0)
                torch.testing.assert_close(model.actor.log_std.weight, torch.zeros_like(model.actor.log_std.weight))
                torch.testing.assert_close(model.actor.log_std.bias, torch.full_like(model.actor.log_std.bias, LOG_STD_INIT))
                self.assertEqual(model.replay_buffer.size(), 0)
                self.assertFalse(model.actor.optimizer.state)


if __name__ == "__main__":
    unittest.main()
