"""Physical demonstration teacher for the placement benchmark."""

import numpy as np

from ._vendor.pickup import State, measure_contacts
from .env import ACTION_STEP
from .expert import PickupExpert


class PlacementExpert(PickupExpert):
    def __init__(self, env, rng=None, perturb=False):
        super().__init__(env, contact_trigger=True)
        self.placement_rng = rng
        self.placement_perturb = perturb

    def __call__(self, obs):
        env = self.env
        tcp = env.tcp
        if self.state in (State.APPROACH, State.DESCEND, State.GRASP, State.LIFT):
            # Join the carrying path just before the pickup waypoint to avoid a full stop.
            if self.state == State.LIFT and tcp[2] >= self.lift[2] - 0.012:
                self.state = State.TRANSPORT
            else:
                label, _ = super().__call__(obs)
                return label, self._perturb(label)

        if self.state == State.TRANSPORT and np.linalg.norm(env.target[:2] - env.cube_xyz[:2]) < 0.007:
            self.state = State.PLACE
        if self.state == State.PLACE and self._ready_to_retreat():
            self.state = State.RETREAT

        height = {State.TRANSPORT: 0.09, State.PLACE: -0.001, State.RETREAT: 0.10}[self.state]
        target = env.target + [0, 0, height]
        if self.state != State.RETREAT:
            # Correct small in-gripper XY offsets while carrying or lowering.
            target[:2] += tcp[:2] - env.cube_xyz[:2]
        delta = (target - tcp) / ACTION_STEP
        limit = 0.64 if self.state == State.PLACE else 0.85
        delta *= min(1.0, limit / max(limit, np.linalg.norm(delta)))
        label = np.array([*delta, 1.0 if self.state == State.RETREAT else -1.0], dtype=np.float32)
        return label, self._perturb(label)

    def _ready_to_retreat(self):
        return measure_contacts(self.env, self.env.data).supported

    def _perturb(self, label):
        if not self.placement_perturb:
            return label.copy()
        if self.state not in (State.APPROACH, State.DESCEND, State.TRANSPORT, State.PLACE):
            return label.copy()
        action = label.copy()
        action[:3] += self.placement_rng.normal(0, 0.07, size=3)
        return np.clip(action, -1, 1)
