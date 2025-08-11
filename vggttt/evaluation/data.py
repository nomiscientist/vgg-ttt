# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

from typing import TypedDict

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from vggttt.geometry import inverse_se3
from vggttt.nets.vggt.img import get_target_shape, process_one_image


class Frame(TypedDict):
    image: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    pose: np.ndarray


def resize_imgs_to_patch_size(imgs: torch.Tensor | list[torch.Tensor], patch_size: int, target_size: int):
    B, C, H, W = imgs.shape
    aspect_ratio = W / H
    target_image_shape = get_target_shape(aspect_ratio, target_size, patch_size).tolist()
    return torch.nn.functional.interpolate(
        imgs, size=target_image_shape, mode="bilinear", antialias=True, align_corners=False
    )


def preproc_fun(x: Frame, target_size: int, patch_size: int):
    # Preserve aspect ratio of the image, resizing the longer side to the target size
    original_size = np.array(x["image"].shape[:2])
    aspect_ratio = original_size[1] / original_size[0]
    target_image_shape = get_target_shape(aspect_ratio, target_size, patch_size)

    (
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        _,
    ) = process_one_image(
        image=x["image"],
        depth_map=x["depth"],
        extri_opencv=inverse_se3(x["pose"]),
        intri_opencv=x["intrinsics"],
        target_image_shape=target_image_shape,
        test_mode=True,
    )
    out = {
        "image": to_tensor(image),
        "intrinsics": torch.from_numpy(intri_opencv),
        "pose": torch.from_numpy(inverse_se3(extri_opencv)),
    }
    if depth_map is not None:
        out["depth"] = torch.from_numpy(depth_map)
    return out
