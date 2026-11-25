"""Unit tests for aligned sorting context, takeover teacher, and 3-frame history policy."""

import math
import unittest

import numpy as np
import torch

from panda_grasp.place_contact_signals import TACTILE_SIZE
from panda_grasp.place_objects import OBS_SIZE
from panda_grasp.place_sorting_aligned import (
    ALIGNED_OBS_SIZE,
    AlignedContactSortingScene,
    AlignedPolicy,
    AlignedSortingTeacher,
)
from panda_grasp.place_sorting_contact_ablation import ARRIVED_INDEX, LIFTED_INDEX
from panda_grasp.place_sorting_dagger import make_takeover_teacher
from panda_grasp.place_sorting_history import HISTORY, HistoryAlignedPolicy


class AlignedHistorySortingTest(unittest.TestCase):
    def test_aligned_scene_observation_and_unfolded_yaw(self):
        with AlignedContactSortingScene() as env:
            obs, _ = env.reset(seed=1567000)
            self.assertEqual(obs.shape, (ALIGNED_OBS_SIZE,))
            self.assertEqual(ALIGNED_OBS_SIZE, TACTILE_SIZE + 4)
            self.assertAlmostEqual(float(obs[TACTILE_SIZE]), math.cos(env.object_yaw), places=5)
            self.assertAlmostEqual(float(obs[TACTILE_SIZE + 1]), math.sin(env.object_yaw), places=5)
            self.assertTrue(np.all(np.abs(obs[TACTILE_SIZE + 2:]) <= 2.0))
            teacher = AlignedSortingTeacher(env, brick_long_axis=True)
            label, executed = teacher(obs)
            self.assertEqual(label.shape, (5,))
            self.assertEqual(executed.shape, (5,))
            takeover = make_takeover_teacher(env)
            self.assertEqual(takeover.state.name, "APPROACH")

    def test_history_policy_per_frame_tactile_masking(self):
        policy = HistoryAlignedPolicy()
        policy.eval()
        # Two stacked observations differing only in masked-out tactile channels should give identical outputs
        frame = torch.zeros(ALIGNED_OBS_SIZE)
        frame[LIFTED_INDEX] = 1.0
        frame[ARRIVED_INDEX] = 0.0  # transport phase -> slip (indices 2..7 of tactile) must be masked out
        f1 = frame.clone()
        f2 = frame.clone()
        f1[OBS_SIZE + 2:TACTILE_SIZE] = 1.5
        f2[OBS_SIZE + 2:TACTILE_SIZE] = -2.0
        s1 = torch.cat([f1] * HISTORY)
        s2 = torch.cat([f2] * HISTORY)
        with torch.no_grad():
            torch.testing.assert_close(policy(s1), policy(s2), rtol=0, atol=1e-6)
        # Verify single-frame AlignedPolicy output bounds
        single = AlignedPolicy()
        out = single.action(f1.numpy())
        self.assertEqual(out.shape, (5,))
        self.assertTrue(np.all(np.abs(out) <= 1.0))
