# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging
from functools import total_ordering
from itertools import pairwise
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Sampler

_logger = logging.getLogger(__name__)


@total_ordering
class IdxWithData(tuple):
    """Index with additional field that behaves like a regular dataset index (for the most part).

    This is used to propogate the  additional data to the dataset __getitem__ function through the dataloader.
    """

    def __new__(self, idx: int | tuple[int, int], data: dict[str, int] | None = None):
        if isinstance(idx, tuple) and data is None:
            return tuple.__new__(IdxWithData, idx)
        return tuple.__new__(IdxWithData, (idx, data))

    def __eq__(self, value: object) -> bool:
        if isinstance(value, tuple):
            return self[0] == value[0]
        if isinstance(value, int):
            return self[0] == value
        raise NotImplementedError()

    def __lt__(self, value: object) -> bool:
        if isinstance(value, tuple):
            return self[0] < value[0]
        if isinstance(value, int):
            return self[0] < value
        raise NotImplementedError()

    def __sub__(self, value: object) -> bool:
        if isinstance(value, int):
            return IdxWithData(self[0] - value, self[1])
        raise NotImplementedError()


class WeightedBatchSampler(Sampler[list[IdxWithData]]):
    """Samples batches randomly from multiple datasets with weighted proportions.

    Also provides secondary data that is the same within each batch which useful to, e.g., synchonize resolution and
    aspect ratio for a batch.

    Args:
        data_source: Datasets to sample from.
        num_batches: Number of batches to draw.
        micro_batch_size: Number of image collections to draw in each batch.
        num_imgs_range: Range of number of images to draw in each image collection.
        aspect_ratio_range: Range of aspect ratios to draw to draw from. Consistent within each batch.
        seed: Seed for the sampler.
        weights: Weights for each dataset. Do not need to sum to 1.
        world_size: Number of GPUs.
        world_rank: Rank of the current GPU.
        shuffle: Whether to change the samples depending on the epoch.
    """

    def __init__(
        self,
        data_source: ConcatDataset,
        num_batches: int | None,
        micro_batch_size: int,
        num_imgs_range: tuple[int, int],
        aspect_ratio_range: tuple[float, float],
        seed: int,
        world_size: int,
        world_rank: int,
        weights: list[float] | None = None,
        shuffle: bool = True,
    ) -> None:
        num_batches = num_batches or len(data_source) // micro_batch_size

        if not isinstance(num_batches, int) or num_batches <= 0:
            raise ValueError(f"num_batches should be a positive integer value, but got num_batches={num_batches}")

        assert num_imgs_range[1] >= num_imgs_range[0]

        # Setup per dataset weights
        if weights is None:
            weights = torch.ones(len(data_source.datasets))
        else:
            weights = torch.Tensor(weights)
        self.weights = weights / weights.sum()

        self.data_source = data_source

        # Round number of batches to nearest multiple of world_size so training does not get stuck
        self.num_batches = max(world_size, int((num_batches // world_size) * world_size))

        self.seed = seed
        self.shuffle = shuffle
        self.world_rank = world_rank
        self.world_size = world_size
        self.epoch = 0
        self.num_imgs_range = (num_imgs_range[0], num_imgs_range[1])
        self.max_total_imgs = num_imgs_range[1] * micro_batch_size
        self.aspect_ratio_range = aspect_ratio_range

        # Downstream code needs a batch size attribute so we set one here that is approximately correct
        self.batch_size = self.max_total_imgs

    def __iter__(self) -> Iterator[list[IdxWithData]]:
        """Create batches of sample indices with a second index that is constant for each batch."""
        seed = self.seed + self.epoch
        generator = torch.Generator()
        generator.manual_seed(seed)
        np_generator = np.random.default_rng(seed)

        _logger.info(
            "[RANK %s] Preparing batch sampling indices using parameters: epoch=%s, seed=%s",
            self.world_rank,
            self.epoch,
            seed,
        )

        # Aspect ratio consistent within each batch and synchronized across ranks
        global_batch_count = len(self)
        base_aspect_ratios = torch.empty(global_batch_count, dtype=torch.float32)
        base_aspect_ratios.uniform_(self.aspect_ratio_range[0], self.aspect_ratio_range[1], generator=generator)
        batch_aspect_ratios = base_aspect_ratios.repeat_interleave(self.world_size)

        # Compute number of image collections per batch and total number of collections
        size_collection_per_step = torch.randint(
            self.num_imgs_range[0], self.num_imgs_range[1] + 1, size=(global_batch_count,), generator=generator
        )
        size_collection_per_batch = size_collection_per_step.repeat_interleave(self.world_size)
        n_collections_per_batch = (self.max_total_imgs // size_collection_per_batch).to(torch.long)
        assert (size_collection_per_batch * n_collections_per_batch <= self.max_total_imgs).all()
        n_total_collections = n_collections_per_batch.sum()

        # Weighted sampling for dataset index for each sample
        n_collections_per_dataset = torch.floor(self.weights * n_total_collections).to(torch.long)
        n_missing_collections = n_total_collections - n_collections_per_dataset.sum()
        n_collections_per_dataset[:n_missing_collections] += 1

        sample_idxs_global = []
        per_ds_sample_bounds = [(start, end) for start, end in pairwise([0] + self.data_source.cumulative_sizes)]
        for num_collections, (start, end) in zip(n_collections_per_dataset.tolist(), per_ds_sample_bounds):
            sample_idxs_global.extend(np_generator.choice(np.arange(start, end), size=(num_collections,), replace=True))

        # Shuffle the global sample indices
        sample_idxs_global = torch.tensor(sample_idxs_global, dtype=torch.long)
        sample_idxs_global = sample_idxs_global[torch.randperm(n_total_collections, generator=generator)]

        # Get the sample indices for each batch
        per_batch_sample_ranges = pairwise([0] + n_collections_per_batch.cumsum(0).tolist())
        sample_idxs_per_batch = [sample_idxs_global[start:end] for start, end in per_batch_sample_ranges]

        # Image collection sampling seed to be passed to the dataset
        per_collection_seeds = torch.randint(0, 1000000, size=(n_total_collections,), generator=generator).tolist()

        for batch_idx in range(self.world_rank, self.num_batches, self.world_size):
            n_imgs_per_collection = size_collection_per_batch[batch_idx].item()
            aspect_ratio = batch_aspect_ratios[batch_idx].item()

            batch = []
            for sample_idx in sample_idxs_per_batch[batch_idx].tolist():
                batch.append(
                    IdxWithData(
                        sample_idx,
                        data={
                            "aspect_ratio": aspect_ratio,
                            "n_imgs": n_imgs_per_collection,
                            "epoch": self.epoch,
                            "seed": per_collection_seeds.pop(),
                        },
                    )
                )
            yield batch

    def __len__(self) -> int:
        """Get number of batches."""
        return self.num_batches // self.world_size

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch. Otherwise,
        the next iteration of this sampler will yield the same ordering.
        """
        # Only change the subsets when shuffling is enabled
        if self.shuffle:
            self.epoch = epoch
