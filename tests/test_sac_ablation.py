"""Tests for isolated SAC changes used in the retention experiment."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from panda_grasp.env import OBS_SIZE, PandaGraspEnv
from panda_grasp.learning import initialize_actor
from panda_grasp.sac_ablation import Ablation, AblationSAC, DemonstrationPenalty


class AblationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = PandaGraspEnv()
        self.addCleanup(self.env.close)
        self.model = AblationSAC(
            "MlpPolicy", self.env, device="cpu", seed=19, learning_starts=4,
            buffer_size=64, batch_size=4, train_freq=4, gradient_steps=2,
            ent_coef="auto_0.08", policy_kwargs={"net_arch": [16, 16]},
        )
        initialize_actor(self.model.actor)

    def test_warming_critic_preserves_actor_and_temperature(self):
        model = self.model
        model.ablation = Ablation(critic_warmup_steps=12)
        actor_before = {key: tensor.clone() for key, tensor in model.actor.state_dict().items()}
        critic_before = {key: tensor.clone() for key, tensor in model.critic.state_dict().items()}
        entropy_before = model.log_ent_coef.detach().clone()
        model.learn(total_timesteps=12)
        self.assertEqual(model._n_updates, 4)
        self.assertFalse(model.actor.optimizer.state)
        self.assertFalse(model.ent_coef_optimizer.state)
        torch.testing.assert_close(model.log_ent_coef, entropy_before, rtol=0, atol=0)
        for key, current in model.actor.state_dict().items():
            torch.testing.assert_close(current, actor_before[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(current, critic_before[key])
                            for key, current in model.critic.state_dict().items()))
        model.learn(total_timesteps=4, reset_num_timesteps=False)
        self.assertEqual(model.actor_gradient_updates, 2)
        self.assertTrue(model.actor.optimizer.state)
        self.assertTrue(model.ent_coef_optimizer.state)

    def test_reduced_exploration_changes_only_rollout_deviation(self):
        model = self.model
        obs, _ = self.env.reset(seed=42)
        model._last_obs = obs[None]
        model.num_timesteps = 12
        deterministic, _ = model.predict(obs, deterministic=True)
        model.ablation = Ablation()
        torch.manual_seed(28)
        _, standard = model._sample_action(learning_starts=4)
        model.ablation = Ablation(behavior_std_scale=0.25)
        torch.manual_seed(28)
        _, reduced = model._sample_action(learning_starts=4)
        center = np.arctanh(deterministic)
        np.testing.assert_allclose(np.arctanh(reduced) - center, 0.25 * (np.arctanh(standard) - center),
                                   atol=1e-6, rtol=1e-5)
        actual, _ = model.predict(obs, deterministic=True)
        np.testing.assert_array_equal(actual, deterministic)

    def test_demonstration_penalty_updates_mean_and_leaves_variance_head(self):
        actor = self.model.actor
        with TemporaryDirectory() as temp:
            dataset = Path(temp) / "expert.npz"
            np.savez(dataset, observations=np.zeros((3, OBS_SIZE), dtype=np.float32),
                     expert_actions=np.ones((3, 4), dtype=np.float32), offsets=[0, 3], successful=[True],
                     phase=["GRASP", "LIFT", "HOLD"])
            penalty = DemonstrationPenalty(actor, dataset, 20, 7)
        mean = actor.mu.bias.detach().clone()
        variance = actor.log_std.bias.detach().clone()
        with actor.optimizer.register_step_pre_hook(penalty):
            actor.optimizer.step()
        self.assertFalse(torch.equal(actor.mu.bias, mean))
        torch.testing.assert_close(actor.log_std.bias, variance, rtol=0, atol=0)
        self.assertGreater(penalty.drain(), 0)


if __name__ == "__main__":
    unittest.main()
