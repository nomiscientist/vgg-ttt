# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
from torch.utils.data.dataset import ConcatDataset

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

INFOS = {
    "train": {
        "chess": [1, 2, 4, 6],
        "fire": [1, 2],
        "heads": [2],
        "office": [1, 3, 4, 5, 8, 10],
        "pumpkin": [2, 3, 6, 8],
        "redkitchen": [1, 2, 5, 7, 8, 11, 13],
        "stairs": [2, 3, 5, 6],
    },
    "test": {
        "chess": [3, 5],
        "fire": [3, 4],
        "heads": [1],
        "office": [2, 6, 7, 9],
        "pumpkin": [1, 7],
        "redkitchen": [3, 4, 6, 12, 14],
        "stairs": [1, 4],
    },
}


class SevenScenesScene:
    """Interface for loading frames from 7scenes dataset sequences."""

    def __init__(self, root_dir: Path | str, seq_id: str):
        """Initialize the SevenScenesScene.

        Args:
            root_dir: The root directory of the 7scenes dataset.
            seq_id: The sequence ID, e.g., "chess/seq-01".
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Root directory {self.root_dir} not found")

        self.seq_dir = self.root_dir / seq_id

        self.length = len(list(self.seq_dir.glob("*.color.png")))

    def __getitem__(self, idx: int):
        """Load a single frame in the sequence."""

        im_idx = f"{idx:06d}"

        fx, fy, cx, cy = 525, 525, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        impath = str(self.seq_dir / f"frame-{im_idx}.color.png")
        depthpath = str(self.seq_dir / f"frame-{im_idx}.depth.proj.png")
        posepath = str(self.seq_dir / f"frame-{im_idx}.pose.txt")

        rgb_image = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB)
        depthmap = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

        is_valid = depthmap != 65535
        depthmap[~is_valid] = 0
        depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
        depthmap[depthmap > 10] = 0
        depthmap[depthmap < 1e-3] = 0

        camera_pose = np.loadtxt(posepath).astype(np.float32)
        return {
            "image": rgb_image,
            "depth": depthmap,
            "intrinsics": intrinsics_.copy(),
            "pose": camera_pose,
        }

    def __len__(self):
        return self.length

    def __str__(self):
        return self.seq_dir.relative_to(self.root_dir).as_posix()


class SevenScenes:
    """Interface for loading all frames across all sequences for a scene."""

    def __init__(
        self,
        root_dir: Path | str,
        seq_id: str,
        split: Literal["train", "test", "all"] = "test",
        preproc_fun: Callable | None = None,
    ):
        if split == "all":
            seq_ids = INFOS["test"][seq_id] + INFOS["train"][seq_id]
        else:
            seq_ids = INFOS[split][seq_id]

        self.preproc_fun = preproc_fun if preproc_fun is not None else lambda x: x

        self.seqs = []
        for id_ in seq_ids:
            self.seqs.append(SevenScenesScene(root_dir, f"{seq_id}/seq-{id_:02d}"))
        self.dataset = ConcatDataset(self.seqs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.preproc_fun(self.dataset[idx])


# def seven_scenes_few_view(root_dir: Path) -> Iterable[SevenScenesScene]:
#     # Few-view setting images: https://github.com/nianticlabs/simplerecon/blob/main/data_splits/7Scenes/dvmvs_split/test_eight_view_deepvmvs.txt
#     fewview_txt = root_dir / "test_eight_view_deepvmvs.txt"
#
#     with fewview_txt.open("r") as f:
#         for line in f.readlines():
#             scene, *idxs = line.strip().split(" ")
#             yield scene, (load_7scenes_frame(root_dir / scene, id_) for id_ in idxs)


def seven_scenes(root_dir: Path):
    for scene in SCENES:
        scene_dir = root_dir / scene
        test_seqs = [
            scene_dir / f"seq-{line[len('sequence') :].strip().zfill(2)}"
            for line in (scene_dir / "TestSplit.txt").open("r").readlines()
        ]

        for test_seq in test_seqs:
            yield SevenScenesScene(root_dir, test_seq.relative_to(root_dir).as_posix())
