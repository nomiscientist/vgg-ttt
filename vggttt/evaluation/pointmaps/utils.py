# SPDX-FileCopyrightText: Copyright (c) 2025 CUT3R authors
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This file is adapted from CUT3R:
#   https://github.com/CUT3R/CUT3R/blob/main/eval/mv_recon/utils.py
# Original work licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0):
#   https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modifications by NVIDIA:
#   - Refactored `accuracy` and `completion` to return keyed dicts.
#   - Added `umeyama`, `align_to_gt`, and `icp` helpers.

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c.astype(np.float32), R.astype(np.float32), t.astype(np.float32)


def align_to_gt(pred_pts: np.ndarray, gt_pts: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Coarse align predicted points to ground truth points."""
    c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
    return c * np.einsum("nhwj, ij -> nhwi", pred_pts, R) + t.T


def accuracy(gt_pts, pred_pts, gt_normals=None, rec_normals=None):
    tree = KDTree(gt_pts, compact_nodes=False, balanced_tree=False)
    distances, idx = tree.query(pred_pts, k=1, workers=8)

    metrics = {"acc": float(np.mean(distances)), "acc_median": float(np.median(distances))}
    if gt_normals is None or rec_normals is None:
        return metrics

    normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
    normal_dot = np.abs(normal_dot)

    return {**metrics, "nc1": float(np.mean(normal_dot)), "nc1_med": float(np.median(normal_dot))}


def completion(gt_pts, pred_pts, gt_normals=None, rec_normals=None):
    tree = KDTree(pred_pts, compact_nodes=False, balanced_tree=False)
    distances, idx = tree.query(gt_pts, k=1, workers=8)

    metrics = {"comp": float(np.mean(distances)), "comp_median": float(np.median(distances))}
    if gt_normals is None or rec_normals is None:
        return metrics

    normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
    normal_dot = np.abs(normal_dot)
    return {**metrics, "nc2": float(np.mean(normal_dot)), "nc2_med": float(np.median(normal_dot))}


def icp(
    pred_pts: np.ndarray, gt_pts: np.ndarray, threshold: float | None = 0.1
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_pts)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)

    if threshold is None:
        return pcd, pcd_gt

    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, pcd_gt, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    pcd = pcd.transform(reg_p2p.transformation)
    return pcd, pcd_gt
