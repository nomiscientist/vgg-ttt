# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

_logger = logging.getLogger(__name__)

def compare_translation_by_angle(t_gt: torch.Tensor, t: torch.Tensor, eps: float = 1e-15, default_err: float = 180.0):
    """Normalize the translation vectors and compute the angle between them.

    From: https://github.com/facebookresearch/vggt/blob/4b8be14b574b58c91ecd699122daf3d8004901d4/evaluation/test_co3d.py#L91-L139
    """
    t_norm = torch.norm(t, dim=-1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=-1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=-1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    is_invalid = torch.isnan(err_t) | torch.isinf(err_t)
    if is_invalid.any():
        _logger.warning(f"Invalid translation vectors. Setting default error to {default_err}.")
    err_t[is_invalid] = default_err
    return err_t


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram))


def build_pair_index(N, B=1):
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2
