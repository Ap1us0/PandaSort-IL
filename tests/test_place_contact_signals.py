"""Verify causal contact history and the exact binary-policy warm start."""

import unittest

import numpy as np
import torch

from panda_grasp.learning import Clone
from panda_grasp.place_contact_signals import ContactClone, ContactSignals, ContactSortingScene
from panda_grasp.place_objects import ACTION_NAMES, OBS_SIZE
from panda_grasp.place_sorting import SortingScene
from panda_grasp.place_sorting_contact_ablation import ARRIVED_INDEX, LIFTED_INDEX, SWITCHED, THREE_PHASE, MaskedContactClone


class ContactSignalTest(unittest.TestCase):
    def test_slip_starts_at_contact_and_restarts_after_separation(self):
        signals = ContactSignals()
        at_contact = np.array([0, 0, 0.02])
        first = signals.observe(0, [1.5, 3], at_contact)
        np.testing.assert_allclose(first, [0.5, 1, 0, 0, 0, 0, 0, 0])
        slid = signals.observe(0.05, [1.5, 3], at_contact + [0, 0, 0.004])
        np.testing.assert_allclose(slid, [0.5, 1, 0, 0, 1, 0, 0, 0.2])
        np.testing.assert_array_equal(slid, signals.observe(0.05, [0, 0], None))
        np.testing.assert_array_equal(signals.observe(0.1, [0, 0], None), np.zeros(8))
        renewed = signals.observe(0.15, [3, 3], at_contact + [0, 0, 0.009])
        np.testing.assert_array_equal(renewed, [1, 1, 0, 0, 0, 0, 0, 0])

    def test_original_observations_and_initial_actions_are_exact(self):
        base = Clone(OBS_SIZE, len(ACTION_NAMES)).eval()
        continuous = ContactClone().initialize(base)
        with SortingScene() as old, ContactSortingScene() as new:
            a, _ = old.reset(seed=1564000)
            b, _ = new.reset(seed=1564000)
            for _ in range(4):
                np.testing.assert_array_equal(a, b[:OBS_SIZE])
                np.testing.assert_array_equal(base.action(a), continuous.action(b))
                u = [0.11, 0.04, 0, 1, -0.07]
                a, *_ = old.step(u)
                b, *_ = new.step(u)

    def test_force_slip_and_transport_masks(self):
        original = torch.zeros(3, OBS_SIZE + 8)
        original[:, OBS_SIZE:] = torch.arange(1, 9, dtype=torch.float32)
        original[1:, LIFTED_INDEX] = 1
        original[2, ARRIVED_INDEX] = 1
        retained = original.clone()
        force = MaskedContactClone("force").contact_inputs(original)
        slip = MaskedContactClone("slip").contact_inputs(original)
        transport = MaskedContactClone("transport").contact_inputs(original)
        switched = MaskedContactClone(SWITCHED)
        three_phase = MaskedContactClone(THREE_PHASE)
        torch.testing.assert_close(force[:, :2], original[:, OBS_SIZE:OBS_SIZE + 2])
        self.assertEqual(torch.count_nonzero(force[:, 2:]), 0)
        torch.testing.assert_close(slip[:, 2:], original[:, OBS_SIZE + 2:])
        self.assertEqual(torch.count_nonzero(slip[:, :2]), 0)
        torch.testing.assert_close(transport[1], original[1, OBS_SIZE:])
        self.assertEqual(torch.count_nonzero(transport[[0, 2]]), 0)
        torch.testing.assert_close(switched.contact_inputs(original), torch.stack((slip[0], force[1], slip[2])))
        torch.testing.assert_close(switched.contact_inputs(original[1]), force[1])
        torch.testing.assert_close(three_phase.contact_inputs(original), torch.stack((torch.zeros_like(force[0]), force[1], slip[2])))
        torch.testing.assert_close(three_phase.contact_inputs(original[2]), slip[2])
        torch.testing.assert_close(original, retained, rtol=0, atol=0)
