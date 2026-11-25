"""Check mixed BC data balance and the policy's unmodified input interface."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from panda_grasp.learning import Clone, train_clone
from panda_grasp.place import PLACE_OBS_SIZE


class AuditedClone(Clone):
    def logits(self, observation):
        self.training_batch = observation.detach().clone()
        return super().logits(observation)


class MixedPlacementTest(unittest.TestCase):
    def test_equal_source_samples_without_condition_input(self):
        with TemporaryDirectory() as folder:
            paths = []
            for source, count in enumerate((100, 8, 4)):
                path = Path(folder) / f"source_{source}.npz"
                obs = np.zeros((count + 1, PLACE_OBS_SIZE), dtype=np.float32)
                obs[:count, 0] = source
                obs[count, 0] = 9
                labels = np.zeros((len(obs), 4), dtype=np.float32)
                np.savez(path, observations=obs, expert_actions=labels, offsets=[0, count, count + 1],
                         successful=[True, False],
                         phase=["APPROACH"] * (count - 1) + ["GRASP", "RETREAT"])
                paths.append(path)
            actor = AuditedClone(PLACE_OBS_SIZE)
            train_clone(paths[0], 31, 1, set(), None, model=actor, additional_datasets=paths[1:])
        self.assertEqual(actor.training_batch.shape, (288, PLACE_OBS_SIZE))
        sources, totals = torch.unique(actor.training_batch[:, 0], return_counts=True)
        torch.testing.assert_close(sources, torch.tensor([0.0, 1.0, 2.0]))
        torch.testing.assert_close(totals, torch.tensor([96, 96, 96]))


if __name__ == "__main__":
    unittest.main()
