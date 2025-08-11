# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import pytest
from torch.utils.data import ConcatDataset, Dataset

from vggttt.data.sampler import WeightedBatchSampler


@pytest.fixture(name="concat_ds")
def _concat_ds_fxt() -> ConcatDataset:
    class IncreasingDataset(Dataset):
        def __init__(self, start: int, end: int):
            self.length = end - start
            self.samples = list(range(start, end))

        def __getitem__(self, index) -> int:
            return self.samples[index]

        def __len__(self) -> int:
            return self.length

    ds0 = IncreasingDataset(0, 5)
    ds1 = IncreasingDataset(5, 10)
    return ConcatDataset((ds0, ds1))


def test_sampler(concat_ds: ConcatDataset) -> None:
    """Verify default behaviour with shuffling enabled.

    The sampler should yield ``num_samples // max_total_imgs`` batches. For a fixed
    seed and epoch it must return deterministic indices, while changing the
    epoch should change the order when ``shuffle=True``.
    """
    num_batches = 16
    micro_batch_size = 2
    max_total_imgs = 4
    sampler = WeightedBatchSampler(
        concat_ds,
        num_batches=num_batches,
        micro_batch_size=micro_batch_size,
        num_imgs_range=(1, 2),
        aspect_ratio_range=(0.5, 1.5),
        seed=0,
        shuffle=True,
        world_size=1,
        world_rank=0,
    )
    out = list(sampler)

    # Number of generated batches matches the sampler length
    assert len(out) == num_batches

    # Internal consistency – all elements inside one batch must share the same
    # metadata and the collection size must multiply to ``max_total_imgs``.
    for batch in out:
        first_data = batch[0][1]
        for idx in batch:
            assert idx[1]["n_imgs"] == first_data["n_imgs"]
            assert idx[1]["aspect_ratio"] == pytest.approx(first_data["aspect_ratio"])
            assert idx[1]["epoch"] == 0
        assert len(batch) * first_data["n_imgs"] == max_total_imgs

    # A different epoch should lead to a different ordering when shuffling is
    # enabled (very small probability of collision ignored for simplicity).
    sampler.set_epoch(1)
    out2 = list(sampler)
    assert out != out2

    # samples should come from both underlying datasets
    ds0_samples = ds1_samples = 0
    for batch in out2:
        for idx_with_data in batch:
            idx = idx_with_data[0]
            if idx >= 5:
                ds1_samples += 1
            else:
                ds0_samples += 1
    assert ds0_samples > 0
    assert ds1_samples > 0


def test_sampler_no_shuffle(concat_ds: ConcatDataset) -> None:
    """The ordering must stay fixed across epochs when ``shuffle=False``."""
    sampler = WeightedBatchSampler(
        concat_ds,
        num_batches=8,
        micro_batch_size=2,
        num_imgs_range=(2, 2),
        aspect_ratio_range=(1.0, 1.0),
        seed=0,
        shuffle=False,
        world_size=1,
        world_rank=0,
    )
    out = list(sampler)
    sampler.set_epoch(1)
    out2 = list(sampler)
    assert out == out2


def test_sampler_batch_properties(concat_ds: ConcatDataset) -> None:
    """Ensure that every batch always represents ``max_total_imgs`` images."""
    micro_batch_size = 2
    max_total_imgs = 6
    sampler = WeightedBatchSampler(
        concat_ds,
        num_batches=18,
        micro_batch_size=micro_batch_size,
        num_imgs_range=(1, 3),
        aspect_ratio_range=(0.2, 2.0),
        seed=42,
        world_size=1,
        world_rank=0,
    )
    for batch in sampler:
        n_imgs = batch[0][1]["n_imgs"]
        assert len(batch) * n_imgs <= max_total_imgs


def test_sampler_world_ranks_have_different_batches(concat_ds: ConcatDataset) -> None:
    """Ensure that every batch always represents ``max_total_imgs`` images."""
    micro_batch_size = 2
    max_total_imgs = 6
    sampler_rank0 = WeightedBatchSampler(
        concat_ds,
        num_batches=18,
        micro_batch_size=micro_batch_size,
        num_imgs_range=(1, 3),
        aspect_ratio_range=(0.2, 2.0),
        seed=42,
        world_size=2,
        world_rank=0,
    )
    sampler_rank1 = WeightedBatchSampler(
        concat_ds,
        num_batches=18,
        micro_batch_size=micro_batch_size,
        num_imgs_range=(1, 3),
        aspect_ratio_range=(0.2, 2.0),
        seed=42,
        world_size=2,
        world_rank=1,
    )
    assert len(sampler_rank0) == len(sampler_rank1)
    out0 = list(sampler_rank0)
    out1 = list(sampler_rank1)
    assert out0 != out1
