"""Continuous simulated finger forces and hand-relative object translation."""

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .learning import Clone, WIDTHS
from .place_objects import ACTION_NAMES, OBS_SIZE, ObjectPlacementEnv
from .place_sorting import SortingScene

CONTACTS = (
    "left_normal_n_over_3", "right_normal_n_over_3",
    "tool_slip_vx_over_0p08", "tool_slip_vy_over_0p08", "tool_slip_vz_over_0p08",
    "tool_slip_dx_over_0p02", "tool_slip_dy_over_0p02", "tool_slip_dz_over_0p02",
)
TACTILE_SIZE = OBS_SIZE + len(CONTACTS)


class ContactSignals:
    """Sample contact at the control boundary, with no dependence on following steps."""

    def __init__(self):
        self.time = None
        self.position = None
        self.origin = None
        self.features = np.zeros(len(CONTACTS), dtype=np.float32)

    def observe(self, time, loads, position):
        if time == self.time:
            return self.features
        force = np.clip(loads, 0, 6) / 3.0
        velocity = displacement = np.zeros(3)
        if position is not None:
            if self.position is None:
                self.origin = position.copy()
            else:
                velocity = (position - self.position) / (time - self.time)
            displacement = position - self.origin
        else:
            self.origin = None
        self.features = np.asarray([
            *force, *np.clip(velocity / 0.08, -4, 4), *np.clip(displacement / 0.02, -4, 4),
        ], dtype=np.float32)
        self.position = None if position is None else position.copy()
        self.time = time
        return self.features


class ContinuousContact:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(TACTILE_SIZE,), dtype=np.float32)

    def _reset_task(self, options):
        self.contact_signals = ContactSignals()
        super()._reset_task(options)

    def _begin_attempt(self):
        super()._begin_attempt()
        self.contact_signals = ContactSignals()

    def _observation(self, contacts):
        position = None
        if contacts.left_count or contacts.right_count:
            rotation = self.data.xmat[self.arm.hand].reshape(3, 3)
            position = rotation.T @ (self.cube_xyz - self.tcp)
        extra = self.contact_signals.observe(self.data.time, (contacts.left_normal_n, contacts.right_normal_n), position)
        return np.concatenate((super()._observation(contacts), extra))


class ContactSortingScene(ContinuousContact, SortingScene):
    pass


class ContactObjectScene(ContinuousContact, ObjectPlacementEnv):
    pass


class ContactClone(Clone):
    """Inject new sensor features without changing the original input projection."""

    def __init__(self):
        super().__init__(OBS_SIZE, len(ACTION_NAMES))
        self.contact_projection = nn.Linear(len(CONTACTS), WIDTHS[0], bias=False)
        nn.init.zeros_(self.contact_projection.weight)

    def initialize(self, source):
        missing, unexpected = self.load_state_dict(source.state_dict(), strict=False)
        if missing != ["contact_projection.weight"] or unexpected:
            raise ValueError("Incompatible binary-contact actor")
        self.eval()
        base = torch.linspace(-1.0, 1.0, 13 * OBS_SIZE).reshape(13, OBS_SIZE)
        features = torch.linspace(-1.0, 2.0, 13 * len(CONTACTS)).reshape(13, len(CONTACTS))
        with torch.no_grad():
            torch.testing.assert_close(self(torch.cat((base, features), dim=1)), source(base), rtol=0, atol=0)
        return self

    def contact_inputs(self, observation):
        return observation[..., OBS_SIZE:]

    def logits(self, observation):
        base = self.features[0](observation[..., :OBS_SIZE])
        first = base + self.contact_projection(self.contact_inputs(observation))
        activated = self.features[1](first)
        hidden = self.features[3](self.features[2](activated))
        return self.mean(hidden)
