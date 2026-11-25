"""Unit tests for OOD sorting environments, V3 expert, DR residual history policy, and Residual SAC env."""

import unittest

import numpy as np
import torch

from panda_grasp.place_contact_signals import TACTILE_SIZE
from panda_grasp.place_objects import OBS_SIZE
from panda_grasp.place_sorting_aligned import ALIGNED_OBS_SIZE
from panda_grasp.place_sorting_contact_ablation import ARRIVED_INDEX, LIFTED_INDEX
from panda_grasp.place_sorting_dr import DRHistoryPolicy, RobustSortingTeacherV3
from panda_grasp.place_sorting_history import HISTORY, HistoryAlignedPolicy
from panda_grasp.place_sorting_ood import CONDITIONS, OODSortingScene
from panda_grasp.place_sorting_residual_sac import ResidualCubeSortingEnv, apply_residual
from panda_grasp.place_sorting_unseen_eval import UNSEEN_CONDITIONS, UNSEEN_TEST
from panda_grasp.place_sorting_contact_schedule import TEST as SCHEDULE_TEST


class DRSortingAndResidualRLTest(unittest.TestCase):
    def test_ood_scene_scales_physics_and_geometry_without_touching_table_pair(self):
        with OODSortingScene(CONDITIONS["friction_60"]) as f_env, \
                OODSortingScene(CONDITIONS["mass_175"]) as m_env, \
                OODSortingScene(CONDITIONS["size_90"]) as s_env:
            # Finger-object sliding friction scales on all 3 objects; table contact pair stays at 0.8
            for geom_name in ("cube", "sort_brick", "sort_cylinder"):
                self.assertAlmostEqual(float(f_env.model.geom(geom_name).friction[0]), 0.98 * (60.0 / 70.0), places=4)
            self.assertAlmostEqual(float(f_env.model.pair_friction[0, 0]), 0.8, places=5)
            # Mass scales to 175g
            self.assertAlmostEqual(float(m_env.model.body("cube").mass[0]), 0.175, places=4)
            # XY size scales by 0.90 while Z half-height stays 0.0225
            self.assertAlmostEqual(float(s_env.model.geom("cube").size[0]), 0.0225 * 0.90, places=5)
            self.assertAlmostEqual(float(s_env.model.geom("cube").size[2]), 0.0225, places=5)

    def test_v3_teacher_and_dr_history_policy_contract(self):
        with OODSortingScene(CONDITIONS["jitter_12mm"]) as env:
            obs, _ = env.reset(seed=1567000)
            self.assertEqual(obs.shape, (ALIGNED_OBS_SIZE,))
            teacher = RobustSortingTeacherV3(env, brick_long_axis=True)
            label, executed = teacher(obs)
            self.assertEqual(label.shape, (5,))
            self.assertEqual(executed.shape, (5,))
        policy = DRHistoryPolicy()
        policy.eval()
        frame = torch.zeros(ALIGNED_OBS_SIZE)
        frame[LIFTED_INDEX] = 1.0
        frame[ARRIVED_INDEX] = 0.0
        f1, f2 = frame.clone(), frame.clone()
        f1[OBS_SIZE + 2:TACTILE_SIZE] = 2.0
        f2[OBS_SIZE + 2:TACTILE_SIZE] = -2.0
        with torch.no_grad():
            torch.testing.assert_close(
                policy(torch.cat([f1] * HISTORY)),
                policy(torch.cat([f2] * HISTORY)),
                rtol=0, atol=1e-6,
            )

    def test_residual_sac_keeps_gripper_untouched_and_steps_cleanly(self):
        base_act = np.array([0.2, -0.3, 0.5, -1.0, 0.1], dtype=np.float32)
        res_act = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        cmd = apply_residual(base_act, res_act)
        self.assertEqual(cmd[3], -1.0)
        self.assertTrue(np.all(np.abs(cmd) <= 1.0))
        dummy_actor = HistoryAlignedPolicy()
        dummy_actor.eval()
        with ResidualCubeSortingEnv("friction_60", dummy_actor, 99_000_000) as env:
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (ALIGNED_OBS_SIZE + 5,))
            next_obs, reward, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            self.assertEqual(next_obs.shape, obs.shape)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)

    def test_unseen_layouts_are_disjoint_from_schedule_test_and_transport_gating_works(self):
        self.assertTrue(set(UNSEEN_TEST).isdisjoint(set(SCHEDULE_TEST)))
        self.assertIn("unseen_feasible_combo", UNSEEN_CONDITIONS)
        # Outside transport phase, transport_only residual must return exact untouched base_action
        base_act = np.array([0.2, -0.3, 0.5, -1.0, 0.1], dtype=np.float32)
        res_act = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        pre_lift_frame = np.zeros(ALIGNED_OBS_SIZE, dtype=np.float32)
        pre_lift_frame[LIFTED_INDEX] = 0.0
        np.testing.assert_array_equal(apply_residual(base_act, res_act, pre_lift_frame, transport_only=True), base_act)
        post_arrive_frame = np.zeros(ALIGNED_OBS_SIZE, dtype=np.float32)
        post_arrive_frame[LIFTED_INDEX] = 1.0
        post_arrive_frame[ARRIVED_INDEX] = 1.0
        np.testing.assert_array_equal(apply_residual(base_act, res_act, post_arrive_frame, transport_only=True), base_act)
