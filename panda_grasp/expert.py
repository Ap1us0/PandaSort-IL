"""Cartesian demonstration adapter for the existing pick-and-place controller."""

import numpy as np

from ._vendor.pickup import State, measure_contacts, pickup_targets
from .env import ACTION_STEP


class PickupExpert:
    """Replay approach, descent, closure, lift and hold through env.step."""

    def __init__(self, env, rng=None, perturb=False, contact_trigger=False):
        self.env = env
        self.rng = rng
        self.perturb = perturb
        self.contact_trigger = contact_trigger
        self.state = State.APPROACH
        self.closed_steps = 0
        self._set_targets()

    def _set_targets(self):
        self.approach, self.descend, self.lift = pickup_targets(self.env.cube_xyz.copy())

    def __call__(self, obs):
        del obs
        env = self.env
        tcp = env.tcp
        if self.state == State.APPROACH and np.linalg.norm(self.approach - tcp) < 0.006:
            self._set_targets()
            self.state = State.DESCEND
        if self.state == State.DESCEND and np.linalg.norm(self.descend - tcp) < 0.0025:
            self.state = State.GRASP
        if self.state == State.GRASP:
            ready = self.closed_steps >= 8
            if self.contact_trigger:
                ready = self.closed_steps >= 1 and measure_contacts(env, env.data).gripping
            if ready:
                self.state = State.LIFT
        if self.state == State.LIFT and np.linalg.norm(self.lift - tcp) < 0.003:
            self.state = State.HOLD

        targets = {
            State.APPROACH: self.approach, State.DESCEND: self.descend,
            State.GRASP: self.descend, State.LIFT: self.lift, State.HOLD: self.lift,
        }
        delta = (targets[self.state] - tcp) / ACTION_STEP
        limit = 0.60 if self.state in (State.DESCEND, State.LIFT) else 0.85
        delta *= min(1.0, limit / max(limit, np.linalg.norm(delta)))
        closed = self.state in (State.GRASP, State.LIFT, State.HOLD)
        if self.state == State.GRASP:
            self.closed_steps += 1
        label = np.array([*delta, -1 if closed else 1], dtype=np.float32)
        executed = label.copy()
        if self.perturb:
            scale = 0.16 if self.state in (State.APPROACH, State.DESCEND) else 0.04
            executed[:3] += self.rng.normal(0, scale, size=3)
            executed = np.clip(executed, -1, 1)
        return label, executed
