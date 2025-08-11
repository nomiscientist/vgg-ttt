# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging

import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from vggttt.data.utils import asnumpy
from vggttt.evaluation.pointmaps.utils import align_to_gt
from vggttt.geometry import inverse_se3, rotmat_angle_diff
from vggttt.nets.vggt.metrics import build_pair_index, calculate_auc_np, compare_translation_by_angle

_logger = logging.getLogger(__name__)


def translation_angle_err(tvec_gt: torch.Tensor, tvec_pred: torch.Tensor, ambiguity: bool = True) -> torch.Tensor:
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_rad = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_rad * 180.0 / torch.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
    return rel_tangle_deg


def compute_pairwise_pose_errs(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise relative pose errors from global poses (translation angle error, orientation angle error).

    For this, generate all possible indices of pairs (basically the lower triangular part of the adjacency matrix of a
    fully connected graph).
    """
    src, tgt = build_pair_index(len(pred))

    # Pi3 and VGGT uses w2c translation vectors for evaluation
    # https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/test_co3d.py#L163
    rel_pose_pred = inverse_se3(pred[tgt]) @ pred[src]
    rel_pose_gt = inverse_se3(gt[tgt]) @ gt[src]

    return (
        translation_angle_err(rel_pose_pred[:, :3, 3], rel_pose_gt[:, :3, 3]),
        rotmat_angle_diff(rel_pose_pred[:, :3, :3], rel_pose_gt[:, :3, :3], unit="deg"),
    )


def compute_pose_acc_metrics(t_errs_deg, rot_errs_deg, thresholds=(5, 15, 30), max_threshold=30) -> dict[str, float]:
    metrics = {}

    for th in thresholds:
        metrics[f"RRA@{th}"] = np.mean(rot_errs_deg < th) * 100
        metrics[f"RTA@{th}"] = np.mean(t_errs_deg < th) * 100

    metrics[f"mAA@{max_threshold}"] = calculate_auc_np(rot_errs_deg, t_errs_deg, max_threshold=max_threshold) * 100
    return metrics


class PoseAcc(Metric):
    """Compute Relative rotation accuracy"""

    r_errs: list[torch.Tensor]
    t_errs: list[torch.Tensor]

    def __init__(self, prefix: str = "", thresholds: tuple[int, ...] = (5, 15, 30), max_threshold: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = thresholds
        self.max_threshold = max_threshold
        self.prefix = prefix

        self.add_state("t_errs", default=[], dist_reduce_fx="cat")
        self.add_state("r_errs", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs):
        for p, t in zip(preds, target):
            rel_t_errs, rel_rot_errs = compute_pairwise_pose_errs(p, t)
            self.t_errs.append(rel_t_errs)
            self.r_errs.append(rel_rot_errs)

    def compute(self):
        if (isinstance(self.r_errs, list) and not self.r_errs) or (isinstance(self.t_errs, list) and not self.t_errs):
            return {}

        return {
            f"{self.prefix}{k}": v
            for k, v in compute_pose_acc_metrics(
                asnumpy(dim_zero_cat(self.t_errs)),
                asnumpy(dim_zero_cat(self.r_errs)),
                thresholds=self.thresholds,
                max_threshold=self.max_threshold,
            ).items()
        }


class ScaleShiftInvMAE(Metric):
    """Compute scale- and shift-invariant mean absolute error for point clouds."""

    mae: list[torch.Tensor]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("mae", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, is_valids: torch.BoolTensor, *args, **kwargs):
        for pred, tgt, is_valid in zip(preds, targets, is_valids):
            if not is_valid.any():
                continue
            pred_points = torch.from_numpy(align_to_gt(asnumpy(pred), asnumpy(tgt), asnumpy(is_valid))).to(pred)
            self.mae.append(torch.mean(torch.abs(pred_points[is_valid] - tgt[is_valid])))

    def compute(self):
        metrics = {}
        metrics["mae"] = torch.mean(torch.stack(self.mae) if isinstance(self.mae, list) else self.mae)
        return metrics
