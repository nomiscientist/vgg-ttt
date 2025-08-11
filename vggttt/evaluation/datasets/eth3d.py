# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

from pathlib import Path

import numpy as np
import torchvision.transforms as tvf
from PIL import Image

from vggttt.geometry import inverse_se3

to_tensor = tvf.ToTensor()

ETH3D_SCENES = [
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",
]


class ETH3DScene:
    def __init__(self, root_dir: str | Path, seq_id: str):
        self.root_dir = Path(root_dir)
        self.scene_dir = self.root_dir / seq_id
        seq_image_root = self.root_dir / seq_id / "images" / "custom_undistorted"
        image_list = [imgname for imgname in seq_image_root.iterdir() if imgname.name.endswith(".JPG")]
        self.image_list = sorted(image_list)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int):
        impath = self.image_list[idx]
        # npz file but .jpg extension :D
        depthpath = self.scene_dir / "ground_truth_depth" / "custom_undistorted" / impath.name
        cam_path = self.scene_dir / "custom_undistorted_cam" / (impath.stem + ".npz")

        cam = np.load(cam_path)
        intrinsic = cam["intrinsics"]
        extrinsic = cam["extrinsics"]

        # load image and depth
        rgb_image: Image.Image = Image.open(impath)
        width, height = rgb_image.size
        depthmap: np.ndarray = np.fromfile(depthpath, dtype=np.float32).reshape(height, width)
        depthmap[~np.isfinite(depthmap)] = -1

        return {
            "image": np.asarray(rgb_image),
            "depth": depthmap,
            "intrinsics": intrinsic,
            "pose": inverse_se3(extrinsic),
        }
