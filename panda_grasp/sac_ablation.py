"""Single-factor SAC modifications for retaining a pretrained grasp actor."""

from dataclasses import dataclass
import math

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Ablation:
    critic_warmup_steps: int = 0
    behavior_std_scale: float = 1.0
    bc_weight: float = 0.0


ABLATIONS = {
    "critic_warmup": Ablation(critic_warmup_steps=2000),
    "low_exploration": Ablation(behavior_std_scale=0.25),
    "bc_regularized": Ablation(bc_weight=20.0),
}


class AblationSAC(SAC):
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        scale = getattr(self, "ablation", Ablation()).behavior_std_scale
        if scale == 1 or self.num_timesteps < learning_starts:
            return super()._sample_action(learning_starts, action_noise, n_envs)
        if action_noise is not None:
            raise ValueError("This ablation models exploration through the Gaussian policy")
        self.policy.set_training_mode(False)
        with torch.no_grad():
            observation, _ = self.policy.obs_to_tensor(self._last_obs)
            mean, log_std, distribution_args = self.actor.get_action_dist_params(observation)
            scaled = self.actor.action_dist.actions_from_params(
                mean, log_std + math.log(scale), **distribution_args,
            ).cpu().numpy()
        return self.policy.unscale_action(scaled), scaled

    def train(self, gradient_steps, batch_size=64):
        warmup_steps = getattr(self, "ablation", Ablation()).critic_warmup_steps
        if self.num_timesteps > warmup_steps:
            self.actor_gradient_updates = getattr(self, "actor_gradient_updates", 0) + gradient_steps
            return super().train(gradient_steps, batch_size)
        # Perform the SB3 SAC critic and target updates with a fixed actor and temperature.
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.critic.optimizer])
        alpha = self.log_ent_coef.detach().exp() if self.ent_coef_optimizer is not None else self.ent_coef_tensor
        losses = []
        for step in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discount = batch.discounts if batch.discounts is not None else self.gamma
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(batch.next_observations)
                target_q = torch.cat(self.critic_target(batch.next_observations, next_actions), dim=1)
                target_q = target_q.min(dim=1, keepdim=True).values - alpha * next_log_prob.reshape(-1, 1)
                target_q = batch.rewards + (1 - batch.dones) * discount * target_q
            values = self.critic(batch.observations, batch.actions)
            loss = 0.5 * sum(F.mse_loss(q, target_q) for q in values)
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            losses.append(loss.detach().item())
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/critic_loss", np.mean(losses))
        self.logger.record("train/ent_coef", alpha.item())


class DemonstrationPenalty:
    """Add a phase-balanced supervised loss to the deterministic actor mean."""

    def __init__(self, actor, dataset, weight, seed):
        self.actor = actor
        self.weight = weight
        with np.load(dataset) as data:
            eligible = np.repeat(data["successful"], np.diff(data["offsets"]))
            if not eligible.any():
                raise ValueError("BC regularization requires successful demonstrations")
            self.obs = torch.as_tensor(data["observations"][eligible], device=actor.device)
            self.actions = torch.as_tensor(data["expert_actions"][eligible], device=actor.device)
            phase = data["phase"][eligible]
            self.groups = [torch.from_numpy(np.flatnonzero(phase == stage)) for stage in np.unique(phase)]
        self.generator = torch.Generator().manual_seed(seed + 100_000)
        self.parameters = tuple(actor.latent_pi.parameters()) + tuple(actor.mu.parameters())
        self.loss_sum = 0.0
        self.num_updates = 0

    def __call__(self, optimizer, args, kwargs):
        with torch.enable_grad():
            batch = torch.cat([group[torch.randint(len(group), (24,), generator=self.generator)]
                               for group in self.groups])
            logits, _, _ = self.actor.get_action_dist_params(self.obs[batch])
            targets = self.actions[batch]
            motion = F.mse_loss(logits[:, :3].tanh(), targets[:, :3])
            grip = F.binary_cross_entropy_with_logits(2 * logits[:, 3], (1 + targets[:, 3]) / 2)
            penalty = motion + 0.18 * grip
            gradients = torch.autograd.grad(self.weight * penalty, self.parameters)
        for param, gradient in zip(self.parameters, gradients):
            param.grad = gradient if param.grad is None else param.grad + gradient
        self.loss_sum += penalty.detach().item()
        self.num_updates += 1

    def drain(self):
        average = self.loss_sum / self.num_updates if self.num_updates else None
        self.loss_sum = 0.0
        self.num_updates = 0
        return average
