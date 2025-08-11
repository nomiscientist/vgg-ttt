# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

from functools import lru_cache
from typing import Literal

import numpy as np
import torch


def Rt_to_4x4(
    R: torch.Tensor | np.ndarray, t: torch.Tensor | np.ndarray, s: float | torch.Tensor | np.ndarray = 1.0
) -> torch.Tensor | np.ndarray:
    """Convert rotation and translation to 4x4 transformation matrix.

    Args:
        R: Rotation matrix with shape (..., 3, 3)
        t: Translation vector with shape (..., 3)
        s: Optional scaling factor, either scalar or broadcastable to R's batch shape

    Returns:
        4x4 transformation matrix with shape (..., 4, 4) that combines rotation,
        translation and optional scaling. The matrix has the form:
        [sR  t]
        [0   1]
    """
    if isinstance(R, np.ndarray):
        assert R.shape[-2:] == (3, 3)
        trf = np.eye(4, dtype=np.float32)
        trf = np.broadcast_to(trf, (*R.shape[:-2], 4, 4)).copy()
        trf[..., :3, :3] = R * s
        trf[..., :3, 3] = t
        return trf

    s = s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=R.dtype, device=R.device)
    trf = torch.eye(4, device=R.device, dtype=R.dtype)
    trf = trf.expand(*R.shape[:-2], 4, 4).clone()
    trf[..., :3, :3] = torch.einsum("... i j, ... -> ... i j", R, s)
    trf[..., :3, 3] = t
    return trf


def inverse_se3(se3: torch.Tensor | np.ndarray):
    """Compute the inverse of an SE(3) transformation matrix.

    Args:
        se3: (..., 4, 4) SE(3) transformation matrix (torch.Tensor or np.ndarray)

    Returns:
        (..., 4, 4) Inverse SE(3) transformation matrix (same type as input)
    """
    R = se3[..., :3, :3]
    T = se3[..., :3, [3]]
    if isinstance(se3, np.ndarray):
        R_inv = np.swapaxes(R, -2, -1)
        T_inv = -np.matmul(R_inv, T)[..., 0]
        return Rt_to_4x4(R_inv, T_inv)

    R_inv = R.mT
    T_inv = -torch.matmul(R_inv, T).squeeze(-1)
    return Rt_to_4x4(R_inv, T_inv)


def rotmat_angle_diff(
    R1: np.ndarray | torch.Tensor, R2: np.ndarray | torch.Tensor, unit: Literal["rad", "deg"] = "rad"
) -> float | torch.Tensor | np.ndarray:
    """Angle difference between 2 rotation matrices in radians."""
    assert R1.shape[-1] == R2.shape[-1] == 3, "Expecting rotation matrices of shape (3, 3)"

    if isinstance(R1, np.ndarray) and isinstance(R2, np.ndarray):
        cos = (np.trace((np.swapaxes(R1, -2, -1) @ R2), axis1=-2, axis2=-1) - 1) / 2
        cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
        diff = np.abs(np.arccos(cos))
        return np.rad2deg(diff) if unit == "deg" else diff

    # Also works for batched matrices
    cos = (torch.einsum("...ii", R1.mT @ R2) - 1) / 2
    cos = torch.clip(cos, -1.0, 1.0)
    diff = torch.abs(torch.arccos(cos))
    return torch.rad2deg(diff) if unit == "deg" else diff


def transform_pc(pc, T):
    if len(T.shape) == 2:
        return pc @ T[:3, :3].T + T[:3, 3]

    if isinstance(pc, np.ndarray):
        # Handle batch case: pc (b, ..., 3) and T (b, 4, 4)
        # Flatten pc to (b, n_points, 3) for batch matrix multiplication
        batch_size = pc.shape[0]
        original_shape = pc.shape
        flat_points = pc.reshape(batch_size, -1, 3)

        # Apply transformation: (b, n_points, 3) @ (b, 3, 3) + (b, 3)
        rotated = np.matmul(flat_points, np.swapaxes(T[..., :3, :3], -2, -1))
        translated = rotated + T[..., :3, 3][:, np.newaxis, :]

        # Reshape back to original shape
        return translated.reshape(original_shape)

    if isinstance(pc, torch.Tensor):
        t_leading_dims = len(T.shape) - 2  # exclude the 4x4 matrix dimensions
        pc_leading_dims = len(pc.shape) - 1  # exclude the coordinate dimension

        if t_leading_dims <= pc_leading_dims:
            shared_dims = T.shape[:-2]  # leading dimensions of T

            # Check that the shared dimensions match
            assert pc.shape[:t_leading_dims] == shared_dims, (
                f"Leading dimensions must match: T {shared_dims} vs pc {pc.shape[:t_leading_dims]}"
            )

            extra_spatial_dims = pc.shape[t_leading_dims:-1]  # dimensions between shared and coordinates
            n_extra_points = 1
            for dim in extra_spatial_dims:
                n_extra_points *= dim

            # Flatten pc: [shared_dims..., n_extra_points, 3]
            flat_shape = shared_dims + (n_extra_points, pc.size(-1))
            flat_points = pc.view(flat_shape)

            # Apply transformation
            rotated = flat_points @ T[..., :3, :3].mT
            translated = rotated + T[..., :3, 3].unsqueeze(-2)

            # Reshape back to original shape
            return translated.reshape(*pc.shape)
        else:
            # Original case: T has more leading dimensions than pc
            assert T.size(0) == pc.size(0), "Need to be equal batch size."
            flat_points = pc.view(pc.size(0), -1, pc.size(-1))
            return (flat_points @ T[..., :3, :3].mT + T[..., :3, 3].unsqueeze(1)).reshape(*pc.shape)

    raise ValueError()


@lru_cache(maxsize=10)
def uv_grid(height: int, width: int, device: str, dtype: torch.dtype = torch.float32):
    u, v = torch.meshgrid(
        torch.arange(width, dtype=dtype, device=device),
        torch.arange(height, dtype=dtype, device=device),
        indexing="xy",
    )
    uv_h = torch.stack([u + 0.5, v + 0.5, torch.ones_like(u)], axis=-1)
    return uv_h


ArrType = torch.Tensor | np.ndarray


def unproject_depthmap(depth: ArrType, K: ArrType) -> tuple[ArrType, ArrType]:
    is_valid = depth > 0.0
    *other, height, width = depth.shape

    if isinstance(depth, torch.Tensor):
        is_valid &= (~torch.isnan(depth)) & torch.isfinite(depth)
        depth = depth.view(-1, height, width)

        # Caches the grid to reduce memory allocations
        uv_h = uv_grid(height, width, device=depth.device)[None]
        pc = torch.einsum("bhwj,bkj -> bhwk", uv_h * depth[..., None], torch.linalg.inv(K).view(-1, 3, 3))
        pc = pc.reshape(*other, height, width, 3)
        is_valid &= ~pc.isnan().any(-1)
        pc[~is_valid] = 0.0

        return pc, is_valid

    is_valid &= (~np.isnan(depth)) & np.isfinite(depth)
    depth = depth.reshape(-1, height, width)
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    uv_h = np.stack([u, v, np.ones_like(u)], axis=-1)[None]

    pc: np.ndarray = np.einsum("b h w j, b k j -> b h w k", uv_h * depth[..., None], np.linalg.inv(K.reshape(-1, 3, 3)))
    pc = pc.reshape(*other, height, width, 3)

    is_valid = is_valid & (~np.isnan(pc).any(-1))
    return pc, is_valid


def unproject_depthmap_to_point_map(depths: ArrType, intrinsics: ArrType, poses: ArrType) -> ArrType:
    """Unproject depths to point maps.

    Args:
        depths: Batch of depths of shape (b, ..., h, w, 1).
        intrinsics: Batch of intrinsics of shape (b, ..., 3, 3).
        poses: Batch of poses of shape (b, ..., 4, 4).

    Returns:
        Pointmaps in global reference frame of shape (b, ..., h, w, 3).
    """
    cam_points, mask = unproject_depthmap(depths, intrinsics)
    cam_points = transform_pc(cam_points, poses)
    return cam_points, mask
