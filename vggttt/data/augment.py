# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging
from typing import Callable, Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2

_logger = logging.getLogger(__name__)


def get_augmentations(split: Literal["train", "val", "test"]) -> Callable:
    """Get the augmentations for the given split."""
    if split == "train":
        return A.Compose(
            [
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.9),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2(),
            ],
            additional_targets={"mask1": "mask"},
            strict=False,
        )

    if split in ("test", "val"):
        return A.Compose(
            [
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2(),
            ],
            additional_targets={"mask1": "mask"},
            strict=False,
        )
