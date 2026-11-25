"""PyTorch cloning and a checked mapping into the SB3 SAC actor."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .env import OBS_SIZE

WIDTHS = [128, 128]
ACTIVATION = nn.ReLU
LOG_STD_INIT = -2.5


class Clone(nn.Module):
    def __init__(self, obs_size=OBS_SIZE, action_size=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_size, WIDTHS[0]), ACTIVATION(),
            nn.Linear(WIDTHS[0], WIDTHS[1]), ACTIVATION(),
        )
        self.mean = nn.Linear(WIDTHS[-1], action_size)

    def logits(self, observation):
        return self.mean(self.features(observation))

    def forward(self, observation):
        return self.logits(observation).tanh()

    @torch.no_grad()
    def action(self, observation):
        return self(torch.as_tensor(observation)).numpy()


def _cloning_examples(path):
    with np.load(path) as data:
        counts = np.diff(data["offsets"])
        successful = np.repeat(data["successful"], counts)
        if not successful.any():
            raise ValueError(f"Behavior cloning requires a successful demonstration in {path}")
        x = torch.as_tensor(data["observations"][successful])
        y = torch.as_tensor(data["expert_actions"][successful])
        phase = data["phase"][successful]
    groups = [torch.from_numpy(np.flatnonzero(phase == name)) for name in np.unique(phase)]
    return x, y, groups


def _condition_batch(size, groups, generator):
    """Draw 32 uniform and 64 phase-balanced samples from one condition."""
    count, extra = divmod(64, len(groups))
    balanced = [
        group[torch.randint(len(group), (count + int(index < extra),), generator=generator)]
        for index, group in enumerate(groups)
    ]
    return torch.cat([torch.randint(size, (32,), generator=generator), *balanced])


def train_clone(dataset_path, seed, updates, checkpoint, log, *, model=None, additional_datasets=(),
                learning_rate=0.001, regularizer=None):
    torch.manual_seed(seed)
    model = Clone() if model is None else model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    datasets = [_cloning_examples(path) for path in (dataset_path, *additional_datasets)]
    generator = torch.Generator().manual_seed(seed + 5000)
    for step in range(1, updates + 1):
        if len(datasets) == 1:
            x, y, groups = datasets[0]
            # Give the brief closure phase supervision comparable to longer motion phases.
            balanced = [group[torch.randint(len(group), (32,), generator=generator)] for group in groups]
            ids = torch.cat([torch.randint(len(x), (96,), generator=generator), *balanced])
            observations, targets = x[ids], y[ids]
        else:
            batches = []
            for x, y, groups in datasets:
                ids = _condition_batch(len(x), groups, generator)
                batches.append((x[ids], y[ids]))
            observations = torch.cat([obs for obs, _ in batches])
            targets = torch.cat([label for _, label in batches])
        logits = model.logits(observations)
        target = targets
        xyz = F.mse_loss(logits[:, :3].tanh(), target[:, :3])
        grip = F.binary_cross_entropy_with_logits(2 * logits[:, 3], (1 + target[:, 3]) / 2)
        loss = xyz + 0.18 * grip
        if logits.shape[1] > 4:
            loss = loss + 0.25 * F.mse_loss(logits[:, 4:].tanh(), target[:, 4:])
        if regularizer is not None:
            loss = loss + regularizer(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in checkpoint:
            log(step, model, float(xyz.detach()), float(grip.detach()))
    return model


@torch.no_grad()
def initialize_actor(actor, clone=None):
    actor.log_std.weight.zero_()
    actor.log_std.bias.fill_(LOG_STD_INIT)
    if clone is None:
        return
    actor.latent_pi.load_state_dict(clone.features.state_dict(), strict=True)
    actor.mu.load_state_dict(clone.mean.state_dict(), strict=True)
    obs_size = clone.features[0].in_features
    probe = torch.linspace(-1, 1, 20 * obs_size, device=actor.device).reshape(20, obs_size)
    torch.testing.assert_close(actor(probe, deterministic=True), clone(probe), atol=1e-6, rtol=1e-6)
