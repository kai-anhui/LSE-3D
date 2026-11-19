"""Build scene-preserving loaders; a pair never spans two physical blocks."""

import random

import torch
from torch.utils.data import DataLoader, IterableDataset

from .dataset import SceneDataset


class SceneBatchStream(IterableDataset):
    """Infinite balanced training stream, or finite whole-scene evaluation stream.

    A shared RNG is exposed to the trainer's checkpoint machinery. One worker
    preserves exact sampling state and avoids duplicate scene preprocessing.
    """

    def __init__(self, dataset, training, rng):
        self.scenes = dataset
        self.training = training
        self.rng = rng

    def __iter__(self):
        if self.training:
            while True:
                yield self.scenes.sample(self.rng)
        else:
            yield from self.scenes.evaluation_batches()


def _identity(batch):
    return batch


def build_dataloader(
    config, root=None, split="train", allow_unverified=False, dataset=None, rng=None
):
    """Each item already contains a complete scene and one or more queries.

    Training length is controlled by the trainer's step budget. Evaluation emits
    each scene once. num_workers>0 is rejected until worker RNG resume is supported.
    """
    workers = config.get("data", {}).get("num_workers", 0)
    if workers != 0:
        raise ValueError(
            "Scene-shared resumable sampling currently requires num_workers=0"
        )
    if dataset is None:
        root = root or config.get("data", {}).get("root")
        if root is None:
            raise ValueError("A dataset root is required")
        dataset = SceneDataset(
            root, split, config.get("voxel_size", 1.0), allow_unverified
        )
    seed = config.get("train", {}).get("seed", 0)
    stream = SceneBatchStream(dataset, split == "train", rng or random.Random(seed))
    # DataLoader iterator seeding must not consume the model's global RNG.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        stream,
        batch_size=None,
        num_workers=0,
        collate_fn=_identity,
        generator=generator,
    )
