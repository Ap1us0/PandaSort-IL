"""Check that stress conditions change the intended physical parameters."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from panda_grasp.experiment import collect_demonstrations, rollout
from panda_grasp.expert import PickupExpert
from panda_grasp.physics_shift import CONDITIONS, ShiftedGraspEnv, teacher_policy


class PhysicsShiftTest(unittest.TestCase):
    def finger_and_table_sliding(self, condition):
        with ShiftedGraspEnv(CONDITIONS[condition]) as env:
            obs, _ = env.reset(seed=1_310_000)
            act = teacher_policy(env, 1_310_000)
            finger, table = set(), set()
            while True:
                obs, _, terminated, truncated, _ = env.step(act(obs))
                for contact in env.data.contact:
                    pair = (contact.geom1, contact.geom2)
                    if env.cube not in pair:
                        continue
                    other = pair[int(pair[0] == env.cube)]
                    if env.model.geom_bodyid[other] in env.gripper.bodies:
                        finger.add(tuple(contact.friction))
                    if other == env.table:
                        table.add(tuple(contact.friction))
                if terminated or truncated:
                    return finger, table

    def test_sliding_changes_in_finger_contacts_and_not_table_pair(self):
        original_finger, original_table = self.finger_and_table_sliding("nominal")
        shifted_finger, shifted_table = self.finger_and_table_sliding("friction_27")
        self.assertEqual(len(original_finger), 1)
        self.assertEqual(len(original_table), 1)
        self.assertEqual(len(shifted_finger), 1)
        self.assertEqual(original_table, shifted_table)
        original = next(iter(original_finger))
        shifted = next(iter(shifted_finger))
        np.testing.assert_allclose(shifted[:2], 0.27 * np.array(original[:2]))
        np.testing.assert_array_equal(shifted[2:], original[2:])

    def test_exact_demonstration_budget_marks_partial_episode(self):
        with TemporaryDirectory() as temp, ShiftedGraspEnv(CONDITIONS["friction_50"]) as env:
            output = Path(temp)
            collect_demonstrations(
                output, env, 0, range(3), lambda rng, perturb: PickupExpert(env, contact_trigger=True),
                {}, interaction_budget=7,
            )
            with np.load(output / "expert.npz") as dataset:
                np.testing.assert_array_equal(dataset["offsets"], [0, 7])
                self.assertFalse(dataset["terminated"].any())
                np.testing.assert_array_equal(np.flatnonzero(dataset["truncated"]), [6])
                self.assertFalse(dataset["successful"].any())
            self.assertIn("collection_budget_cutoff", (output / "expert_episodes.csv").read_text())

    def test_mass_and_inertia_scale_one_body(self):
        with ShiftedGraspEnv(CONDITIONS["nominal"]) as original, ShiftedGraspEnv(CONDITIONS["mass_200"]) as shifted:
            index = original.model.geom_bodyid[original.cube]
            mask = np.arange(original.model.nbody) != index
            np.testing.assert_array_equal(original.model.body_mass[mask], shifted.model.body_mass[mask])
            np.testing.assert_array_equal(original.model.body_inertia[mask], shifted.model.body_inertia[mask])
            np.testing.assert_allclose(shifted.model.body_inertia[index], 2 * original.model.body_inertia[index])
            self.assertAlmostEqual(shifted.mass_kg, 0.150)
            np.testing.assert_array_equal(original.model.geom_friction, shifted.model.geom_friction)
            self.assertTrue(rollout(shifted, teacher_policy, 1_310_000)["is_success"])


if __name__ == "__main__":
    unittest.main()
