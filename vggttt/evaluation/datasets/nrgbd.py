# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import cv2
import numpy as np
from pathlib import Path


NRGBD = [
    "breakfast_room",
    "complete_kitchen",
    "green_room",
    "grey_white_room",
    "kitchen",
    "morning_apartment",
    "staircase",
    "thin_geometry",
    "whiteroom",
]


class NRGBDScene:
    def __init__(self, root_dir: Path | str, seq_id: str):
        self.scene_dir = Path(root_dir) / seq_id
        self.camera_poses = np.loadtxt(self.scene_dir / "poses.txt").reshape(-1, 4, 4)

    def __len__(self):
        return len(self.camera_poses)

    def __name__(self):
        return self.scene_dir.name

    def __getitem__(self, idx: int):
        impath = self.scene_dir / "images" / f"img{idx}.png"
        depthpath = self.scene_dir / "depth" / f"depth{idx}.png"

        fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        rgb_image = cv2.cvtColor(cv2.imread(impath.as_posix()), cv2.COLOR_BGR2RGB)
        depthmap = cv2.imread(depthpath.as_posix(), cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

        depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
        depthmap[depthmap > 10] = 0
        depthmap[depthmap < 1e-3] = 0

        rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

        camera_pose = self.camera_poses[int(idx)]
        # gl to cv
        camera_pose[:, 1:3] *= -1.0

        return {
            "image": rgb_image,
            "depth": depthmap,
            "intrinsics": intrinsics_.copy(),
            "pose": camera_pose,
        }

    def __str__(self):
        return self.scene_dir.name
