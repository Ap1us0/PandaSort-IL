"""Check the multi-object scene and live retargeting contract."""

import unittest

import mujoco
import numpy as np

from panda_grasp.place_sorting import COLOR_BY_SHAPE, COLORS, OBJECT_COLORS, SortingScene, run_scene


class SortingSceneTest(unittest.TestCase):
    def test_all_objects_share_one_live_scene(self):
        with SortingScene() as env:
            obs, _ = env.reset(seed=1560000)
            self.assertEqual(obs.shape, (52,))
            self.assertEqual(env.action_space.shape, (5,))
            self.assertEqual(len(env.objects), 3)
            for name, (geom, joint) in env.objects.items():
                self.assertEqual(env.model.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
                self.assertAlmostEqual(env.model.body_mass[env.model.geom_bodyid[geom]], 0.15)
                slot = env.plan[name].slot
                color = COLOR_BY_SHAPE[name]
                np.testing.assert_allclose(env.model.geom(f"tray_{slot}_floor").rgba, COLORS[color])
                np.testing.assert_allclose(env.model.geom_rgba[geom], OBJECT_COLORS[color])
                wall = env.model.geom(f"tray_{slot}_x_plus")
                self.assertTrue(wall.contype[0] and wall.conaffinity[0])
                self.assertIsNone(env.final_tray(name))
            self.assertFalse(np.any(env.model.eq_type == mujoco.mjtEq.mjEQ_WELD))
            positions = env.data.qpos.copy()
            velocities = env.data.qvel.copy()
            end = env.data.time
            env.select(env.order[1])
            np.testing.assert_array_equal(env.data.qpos, positions)
            np.testing.assert_array_equal(env.data.qvel, velocities)
            self.assertEqual(env.data.time, end)
            self.assertEqual(env.shape_name, env.order[1])

    def test_demonstrations_keep_one_complete_trajectory_per_item(self):
        with SortingScene() as env:
            demonstrations = []
            scene, items = run_scene(env, 1560000, demonstrations=demonstrations)
        self.assertEqual(scene["sorted_items"], sum(row["sorted"] for row in items))
        self.assertEqual(len(demonstrations), 3)
        for transitions, item in zip(demonstrations, items):
            self.assertEqual(len(transitions), item["attempt_steps"])
            self.assertTrue(any(transitions[-1][5:7]))
            obs, action, label, following, _, *_ = transitions[0]
            self.assertEqual(obs.shape, (52,))
            self.assertEqual(action.shape, (5,))
            np.testing.assert_array_equal(action, label)
            np.testing.assert_array_equal(following, transitions[1][0])
