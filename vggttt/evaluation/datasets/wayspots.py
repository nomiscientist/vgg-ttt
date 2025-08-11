# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Wayspots dataset.

https://nianticlabs.github.io/ace/wayspots.html
"""

from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

WAYSPOTS_SCENES = [
    "bears",
    "cubes",
    "inscription",
    "lawn",
    "map",
    "squarebench",
    "statue",
    "tendrils",
    "therock",
    "wintersign",
]


def rot90_pinhole_intrinsics(intrinsics: np.ndarray, hw: tuple[int, int], factor: int) -> np.ndarray:
    """Rotate pinhole intrinsics by multiples of 90 degrees counterclockwise.

    Args:
        intrinsics: 1D array-like of shape (4,) with `[fx, fy, cx, cy]`.
        hw: Tuple `(height, width)` of the image before rotation.
        factor: Integer number of 90° CCW rotations. Can be any integer.

    Returns:
        Rotated intrinsics.
    """
    factor %= 4

    if factor == 0:
        return intrinsics

    # For intrinsics change on rotating images,
    # see https://answers.opencv.org/question/221903/rotating-camera-intrinsics-matrix/
    h, w = hw
    fx, fy, cx, cy = intrinsics
    if factor == 1:  # 90 degrees CCW
        # cx=cy, cy=xmax - cx
        return np.array([fy, fx, cy, w - cx], dtype=intrinsics.dtype)
    if factor == 2:  # 180 degrees CCW
        return np.array([fx, fy, w - cx, h - cy], dtype=intrinsics.dtype)
    if factor == 3:  # 270 degrees CCW
        # cx=xmax-cy and cy=cx
        return np.array([fy, fx, h - cy, cx], dtype=intrinsics.dtype)


def rot_z_pose(
    pose: np.ndarray | torch.Tensor, degree: float | None = None, rad: float | None = None
) -> np.ndarray | torch.Tensor:
    """Rotate a 4x4 pose matrix CCW about the Z axis.

    Args:
        pose: A 4x4 homogeneous pose as a NumPy array or PyTorch tensor.
        degree: Rotation angle in degrees. Mutually exclusive with `rad`.
        rad: Rotation angle in radians. Mutually exclusive with `degree`.

    Returns:
        Rotated pose.
    """
    if (degree is None) == (rad is None):
        raise ValueError("Either degree or rad must be provided.")

    RT = np.eye(4)
    RT[:3, :3] = Rotation.from_euler("z", degree or rad, degrees=degree is not None).as_matrix()

    if isinstance(pose, torch.Tensor):
        return pose @ torch.from_numpy(RT).to(pose)
    else:
        return pose @ RT.astype(pose.dtype)


class Wayspots:
    """Interface for loading frames from Wayspots dataset."""

    def __init__(
        self, root_dir: Path | str, seq_id: str, split: Literal["train", "test"], preproc_fun: Callable | None = None
    ):
        assert seq_id in WAYSPOTS_SCENES, f"Scene {seq_id} not in Wayspots dataset"
        self.preproc_fun = preproc_fun if preproc_fun is not None else lambda x: x
        self.scene_name = seq_id
        self.split = split

        self.root_dir = Path(root_dir)
        self.scene_dir = self.root_dir / f"wayspots_{seq_id}" / split

        # File names are `frame_00000.jpg`
        self.image_paths = sorted((self.scene_dir / "rgb").glob("*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        id = image_path.stem.split("_")[1]  # frame_00000.jpg -> 00000

        img = np.asarray(Image.open(image_path))

        calib_path = self.scene_dir / "calibration" / f"calibration_{id}.txt"
        calib = np.loadtxt(calib_path, dtype=np.float32).item()  # Just contains the focal length

        pose_path = self.scene_dir / "poses" / f"pose_{id}.txt"
        pose = np.loadtxt(pose_path, dtype=np.float32)

        # Apply 90-degree clockwise rotation
        img = np.rot90(img, k=-1)  # k=-1 for clockwise rotation
        h, w = hw = img.shape[:2]

        # Rotate intrinsics (clockwise = -1 = 3 CCW rotations)
        intrinsics_1d = np.array([calib, calib, w / 2, h / 2], dtype=np.float32)
        rotated_intrinsics_1d = rot90_pinhole_intrinsics(intrinsics_1d, hw, factor=-1)
        fx, fy, cx, cy = rotated_intrinsics_1d
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # Rotate pose clockwise around Z axis
        pose = rot_z_pose(pose, degree=-90)

        return self.preproc_fun(
            {
                "image": img,
                "depth": np.zeros((h, w), dtype=np.float32),
                "intrinsics": intrinsics,
                "pose": pose,
            }
        )
