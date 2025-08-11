# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

from pathlib import Path

import cv2
import numpy as np

from vggttt.geometry import inverse_se3

DTU_TEST_SCENES = [1, 10, 11, 110, 114, 118, 12, 13, 15, 23, 24, 29, 32, 33, 34, 4, 48, 49, 62, 75, 77, 9]
DTU_SCENE_IDS = [f"scan{i}" for i in DTU_TEST_SCENES]


def load_cam_mvsnet(file, interval_scale=1):
    """read camera txt file"""
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = 192
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0
    extrinsic = cam[0].astype(np.float32)
    intrinsic = cam[1].astype(np.float32)

    return intrinsic, extrinsic


class DTUScene:
    def __init__(self, root_dir: Path | str, seq_id: str):
        self.root_dir = Path(root_dir)
        self.scene_id = seq_id
        self.scene_dir = self.root_dir / seq_id

        self.length = len(list((self.scene_dir / "images").iterdir()))

    def __getitem__(self, idx: int):
        image_path = self.scene_dir / "images" / f"{idx:08d}.jpg"
        depth_path = self.scene_dir / "depths" / f"{idx:08d}.npy"
        mask_path = self.scene_dir / "binary_masks" / f"{idx:08d}.png"
        cam_path = self.scene_dir / "cams" / f"{idx:08d}_cam.txt"

        rgb_image = cv2.cvtColor(cv2.imread(image_path.as_posix()), cv2.COLOR_BGR2RGB)
        depthmap = np.load(depth_path)
        depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

        mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_UNCHANGED) / 255.0
        mask = mask.astype(np.float32)

        mask[mask > 0.5] = 1.0
        mask[mask < 0.5] = 0.0

        mask = cv2.resize(mask, (depthmap.shape[1], depthmap.shape[0]), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
        mask = cv2.erode(mask, kernel, iterations=1)
        depthmap = depthmap * mask

        with open(cam_path, "r") as f:
            cur_intrinsics, extrinsic = load_cam_mvsnet(f)

        intrinsics = cur_intrinsics[:3, :3]
        camera_pose = inverse_se3(extrinsic)

        return {
            "image": rgb_image,
            "depth": depthmap,
            "intrinsics": intrinsics,
            "pose": camera_pose,
        }

    def __len__(self):
        return self.length

    def __str__(self):
        return self.scene_id
