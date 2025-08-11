# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging
from math import ceil, floor
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from vggttt.geometry import inverse_se3, unproject_depthmap_to_point_map
from vggttt.nets.vggt.utils.geometry import closed_form_inverse_se3
from vggttt.nets.vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggttt.utils.dist import gather_varlen_tensor, get_sp_group

_logger = logging.getLogger(__name__)


def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.

    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points_of_n_imgs: int | None = None,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.

    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.

    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)

    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)

    new_world_points = None
    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = world_points @ R.mT[:, None, None] + t[:, None, None, None]

    new_cam_points = None
    new_depths = None
    if scale_by_points_of_n_imgs != 0:
        dist = new_world_points[:, :scale_by_points_of_n_imgs].norm(dim=-1)
        dist_sum = (dist * point_masks[:, :scale_by_points_of_n_imgs]).sum(dim=[1, 2, 3])
        valid_count = point_masks[:, :scale_by_points_of_n_imgs].sum(dim=[1, 2, 3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)

        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = depths.clone() / avg_scale.view(-1, 1, 1, 1)

        if cam_points is not None:
            new_cam_points = cam_points.clone() / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics, cam_points, new_world_points, depths

    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    return new_extrinsics, new_cam_points, new_world_points, new_depths


def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.

    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside
                                  [-hard_max, hard_max] will be clamped. If None,
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor

    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor), torch.zeros_like(input_tensor), input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        values_out_of_range = (input_tensor > hard_max).any() or (input_tensor < -hard_max).any()
        if values_out_of_range:
            logging.warning(f"Tensor {loss_name} contains values outside range [-{hard_max}, {hard_max}]. Clamping.")

        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor


class MultitaskLoss(torch.nn.Module):
    """Multi-task loss module that combines different loss types for VGGT.

    Supports:
    - Camera loss
    - Depth loss
    - Point loss
    - Tracking loss
    """

    def __init__(self, camera=None, depth=None, point=None, track=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track

    def forward(self, predictions, batch, return_viz_data: bool = False) -> torch.Tensor:
        """
        Compute the total multi-task loss.

        Args:
            predictions: Dict containing model predictions for different tasks
             - depth: (B, N, H, W)
             - depth_conf: (B, N, H, W)
             - pose_enc_list: list[(B, N, 10)]
             - world_points: (B, N, H, W, 3)
             - world_points_conf: (B, N, H, W)

            batch: Dict containing ground truth data and masks
             - point_masks: (B, N, H, W)
             - extrinsics: (B, N, 3, 4)
             - intrinsics: (B, N, 3, 3)
             - images: (B, N, 3, H, W)
             - depths: (B, N, H, W)
             - world_points: (B, N, H, W, 3)
             - world_points_conf: (B, N, H, W)

        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}

        B, N, _, H, W = batch["img"].shape

        sp_group = get_sp_group()
        # Gather predictions from all ranks along sequence dimension
        pred_points = pred_points_conf = None
        if "global" in predictions:
            pred_points = gather_varlen_tensor(predictions["global"]["pts3d"], group=sp_group, dim=1)
            pred_points_conf = gather_varlen_tensor(predictions["global"]["conf"], group=sp_group, dim=1)
        pred_pose_enc = gather_varlen_tensor(predictions["pose_enc"], group=sp_group, dim=1)
        pred_pose_enc_list = [gather_varlen_tensor(t, group=sp_group, dim=1) for t in predictions["pose_enc_list"]]
        pred_depths = gather_varlen_tensor(predictions["depth"], group=sp_group, dim=1)
        pred_depth_confs = gather_varlen_tensor(predictions["depth_conf"], group=sp_group, dim=1)

        # Gather ground truth from all ranks along sequence dimension
        gt_poses = gather_varlen_tensor(batch["gt_poses"], group=sp_group, dim=1)
        mask = gather_varlen_tensor(batch["mask"], group=sp_group, dim=1)
        gt_world_points = gather_varlen_tensor(batch["pts3d"], group=sp_group, dim=1)
        gt_depths = gather_varlen_tensor(batch["depth"], group=sp_group, dim=1)
        intrinsics = gather_varlen_tensor(batch["intrinsics"], group=sp_group, dim=1)

        if "track" in predictions:
            # Track predictions
            pred_track = [gather_varlen_tensor(t, group=sp_group, dim=1) for t in predictions["track"]]
            pred_track_conf = gather_varlen_tensor(predictions["track_conf"], group=sp_group, dim=1)
            pred_track_vis = gather_varlen_tensor(predictions["track_vis"], group=sp_group, dim=1)

            # Track ground truth
            track = gather_varlen_tensor(batch["track"], group=sp_group, dim=1)
            track_is_visible = gather_varlen_tensor(batch["track_is_visible"], group=sp_group, dim=1)
            # This is already a (B, #tracks) tensor and does not have a sequence dimension
            track_is_pos_sample = batch["track_is_pos_sample"]

        # Downstream code assumes world-to-camera / camera-from-world extrinsics.
        gt_extrinsics = inverse_se3(gt_poses)

        gt_extrinsics, _, gt_world_points, gt_depths = normalize_camera_extrinsics_and_points_batch(
            extrinsics=gt_extrinsics,
            world_points=gt_world_points,
            depths=gt_depths,
            point_masks=mask,
        )

        is_outlier_point = (gt_world_points.abs() > 100.0).any(-1)
        mask = mask & ~is_outlier_point

        # Setup for visualization.
        if return_viz_data:
            loss_dict["is_valid"] = mask
            loss_dict["gt_global_pts"] = gt_world_points
            loss_dict["gt_c2w_pose"] = inverse_se3(gt_extrinsics)

            loss_dict["pred_global_conf"] = pred_depth_confs.detach()
            pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                pred_pose_enc.detach(), image_size_hw=(H, W)
            )
            pred_c2w_pose = inverse_se3(pred_extrinsics)
            loss_dict["pred_global_pts"] = unproject_depthmap_to_point_map(
                pred_depths[..., 0].detach(), pred_intrinsics, pred_c2w_pose
            )[0]
            loss_dict["pred_c2w_pose"] = pred_c2w_pose
            loss_dict["pred_intrinsics"] = pred_intrinsics

        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(
                pred_pose_encodings=pred_pose_enc_list,
                valid_frame_mask=mask[:, 0].sum(dim=[-1, -2]) > 100,
                gt_extrinsics=gt_extrinsics,
                gt_intrinsics=intrinsics,
                image_hw=(H, W),
                **self.camera,
            )

            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)

        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(
                pred_depth=pred_depths,
                pred_depth_conf=pred_depth_confs,
                gt_depth=gt_depths,
                gt_depth_mask=mask.clone(),  # 3D points derived from depth map, so we use the same mask
                is_synthetic=batch["is_synthetic"],
                **self.depth,
            )
            depth_loss = (
                depth_loss_dict["loss_conf_depth"]
                + depth_loss_dict["loss_reg_depth"]
                + depth_loss_dict["loss_grad_depth"]
            )
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted
        if "global" in predictions:
            point_loss_dict = compute_point_loss(
                pred_points=pred_points,
                pred_points_conf=pred_points_conf,
                gt_points=gt_world_points,
                gt_points_mask=mask,
                is_synthetic=batch["is_synthetic"],
                **self.point,
            )
            point_loss = (
                point_loss_dict["loss_conf_point"]
                + point_loss_dict["loss_reg_point"]
                + point_loss_dict["loss_grad_point"]
            )
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # Tracking loss
        if "track" in predictions:
            track_loss, track_losses = compute_track_loss(
                tracks_pred=pred_track,
                confidence_pred=pred_track_conf,
                vis_pred=pred_track_vis,
                tracks_gt=track,
                tracks_valid_mask=track_is_pos_sample,
                tracks_vis_mask=track_is_visible,
            )
            total_loss = total_loss + self.track["weight"] * track_loss
            loss_dict.update(track_losses)

        return total_loss, loss_dict


def compute_camera_loss(
    pred_pose_encodings: list[torch.Tensor],
    valid_frame_mask: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    loss_type="l1",  # "l1" or "l2" loss
    gamma=0.6,  # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,  # weight for translation loss
    weight_rot=1.0,  # weight for rotation loss
    weight_focal=0.5,  # weight for focal length loss
    **kwargs,
):
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type,
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = avg_loss_T * weight_trans + avg_loss_R * weight_rot + avg_loss_FL * weight_focal

    # Return loss dictionary with individual components
    return {"loss_camera": total_camera_loss, "loss_T": avg_loss_T, "loss_R": avg_loss_R, "loss_FL": avg_loss_FL}


def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.

    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)

    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        # https://github.com/facebookresearch/vggt/issues/140
        loss_T = F.huber_loss(pred_pose_enc[..., :3], gt_pose_enc[..., :3], delta=0.1, reduction="none")
        loss_R = F.huber_loss(pred_pose_enc[..., 3:7], gt_pose_enc[..., 3:7], delta=0.1, reduction="none")
        loss_FL = F.huber_loss(pred_pose_enc[..., 7:], gt_pose_enc[..., 7:], delta=0.1, reduction="none")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_point_loss(
    pred_points,
    pred_points_conf,
    gt_points,
    gt_points_mask,
    gamma=1.0,
    alpha=0.2,
    gradient_loss_fn=None,
    valid_range=-1,
    per_point_weight: torch.Tensor | None = None,
    is_synthetic: torch.BoolTensor | None = None,
    **kwargs,
):
    """
    Compute point loss.

    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """

    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")

    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {
            "loss_conf_point": dummy_loss,
            "loss_reg_point": dummy_loss,
            "loss_grad_point": dummy_loss,
        }
        return loss_dict

    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_points,
        gt_points,
        gt_points_mask,
        conf=pred_points_conf,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range,
        loss_name="point_loss",
        per_point_weight=per_point_weight,
        is_synthetic=is_synthetic,
    )

    loss_dict = {
        "loss_conf_point": loss_conf,
        "loss_reg_point": loss_reg,
        "loss_grad_point": loss_grad,
    }

    return loss_dict


def compute_depth_loss(
    pred_depth,
    pred_depth_conf,
    gt_depth,
    gt_depth_mask,
    gamma=1.0,
    alpha=0.2,
    gradient_loss_fn=None,
    valid_range=-1,
    is_synthetic: torch.BoolTensor | None = None,
    **kwargs,
):
    """
    Compute depth loss.

    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]  # (B, H, W, 1)

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {
            "loss_conf_depth": dummy_loss,
            "loss_reg_depth": dummy_loss,
            "loss_grad_depth": dummy_loss,
        }
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_depth,
        gt_depth,
        gt_depth_mask,
        conf=pred_depth_conf,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range,
        loss_name="depth_loss",
        is_synthetic=is_synthetic,
    )

    loss_dict = {
        "loss_conf_depth": loss_conf,
        "loss_reg_depth": loss_reg,
        "loss_grad_depth": loss_grad,
    }

    return loss_dict


def regression_loss(
    pred,
    gt,
    mask,
    conf=None,
    gradient_loss_fn=None,
    gamma=1.0,
    alpha=0.2,
    valid_range=-1,
    loss_name="loss",
    per_point_weight: torch.Tensor | None = None,
    is_synthetic: torch.BoolTensor | None = None,
):
    """
    Core regression loss function with confidence weighting and optional gradient loss.

    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss

    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
        per_point_weight: (B, S, H, W) per-point weight.

    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, f"{loss_name}_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask].clamp(min=1e-6))
    loss_conf = check_and_fix_inf_nan(loss_conf, f"{loss_name}_conf")

    if per_point_weight is not None:
        loss_conf = loss_conf * per_point_weight[mask]

    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if is_synthetic.any():
        if "conf" in gradient_loss_fn:
            to_feed_conf = conf[is_synthetic].flatten(0, 1)
        else:
            to_feed_conf = None

        # Compute gradient loss if specified for spatial smoothness
        if "normal" in gradient_loss_fn:
            # Surface normal-based gradient loss
            loss_grad = gradient_loss_multi_scale_wrapper(
                pred[is_synthetic].flatten(0, 1),
                gt[is_synthetic].flatten(0, 1),
                mask[is_synthetic].flatten(0, 1),
                gradient_loss_fn=normal_loss,
                scales=3,
                conf=to_feed_conf,
            )
        elif "grad" in gradient_loss_fn:
            # Standard gradient-based loss
            loss_grad = gradient_loss_multi_scale_wrapper(
                pred[is_synthetic].flatten(0, 1),
                gt[is_synthetic].flatten(0, 1),
                mask[is_synthetic].flatten(0, 1),
                gradient_loss_fn=gradient_loss,
                conf=to_feed_conf,
            )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range > 0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"{loss_name}_conf")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range > 0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"{loss_name}_reg")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn=None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.

    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-6, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.

    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.

    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals, gt_valids = point_map_to_normal(target, mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.

    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.

    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.

    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization

    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.amp.autocast("cuda", enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]  # B,H,W,3
        up = pts[:, :-2, 1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:, 1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        # Compute direction vectors from center to neighbors
        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir, left_dir, dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir, dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir, up_dir, dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1]
        v3 = padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.

    This helps remove outliers that could destabilize training.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(input, q, dim=None, keepdim: bool = False, *, interpolation: str = "nearest") -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            f"Supported interpolations currently are {{'nearest', 'lower', 'higher'}} (got '{interpolation}')!"
        )

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()

    return out.squeeze(dim)


def compute_track_loss(
    tracks_pred: list[torch.Tensor],
    confidence_pred: torch.Tensor | None,
    vis_pred: torch.Tensor,
    tracks_gt: torch.Tensor,
    tracks_valid_mask: torch.Tensor,
    tracks_vis_mask: torch.Tensor,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute multi-scale tracking, visibility and confidence losses.

    Args:
        tracks_pred (list[Tensor]): List of *K* trajectory predictions with increasing refinement. Each tensor has
            shape ``(B, S, N, 2)``.
        confidence_pred (Tensor | None): Predicted inlier/confidence logits shaped ``(B, S, N)``.  Can be *None* if the
            model does not output confidences.
        vis_pred (Tensor): Visibility logits ``(B, S, N)`` predicting whether a point is visible in a given frame.
        tracks_gt (Tensor): Ground-truth trajectories ``(B, S, N, 2)``.
        tracks_valid_mask (Tensor): Static per-point validity ``(B, N)``.
        tracks_vis_mask (Tensor): Ground-truth visibility ``(B, S, N)``.

    Returns:
        total_loss (Tensor): Scalar loss = track + conf + vis.
        losses (dict): Individual components ``{"track_loss", "conf_loss", "vis_loss"}``.
    """
    # If there are no valid points, return dummy loss
    if tracks_valid_mask.sum() == 0:
        track_loss = sum([(p * 0.0).mean() for p in tracks_pred])
        conf_loss = (confidence_pred * 0.0).mean()
        vis_loss = (vis_pred * 0.0).mean()
        loss = track_loss + conf_loss + vis_loss
        return loss, {"track_loss": track_loss, "conf_loss": conf_loss, "vis_loss": vis_loss}

    # Compute tracking loss using sequence_loss
    tracks_valid_mask = tracks_valid_mask.unsqueeze(1).repeat(1, tracks_gt.shape[1], 1)
    track_loss = sequence_loss(
        flow_preds=tracks_pred,
        flow_gt=tracks_gt,
        vis=tracks_vis_mask,
        valids=tracks_valid_mask,
    )

    vis_loss = F.binary_cross_entropy_with_logits(
        vis_pred[tracks_valid_mask], tracks_vis_mask[tracks_valid_mask].float()
    )
    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)

    # within 3 pixels
    if confidence_pred is not None:
        gt_conf_mask = (tracks_gt - tracks_pred[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(
            confidence_pred[tracks_valid_mask], gt_conf_mask[tracks_valid_mask].float()
        )
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    losses = {
        "track_loss": track_loss,
        "conf_loss": conf_loss,
        "vis_loss": vis_loss,
    }
    total_loss = track_loss + conf_loss + vis_loss
    return total_loss, losses


def sequence_loss(
    flow_preds: list[torch.Tensor],
    flow_gt: torch.Tensor,
    vis: torch.Tensor,
    valids: torch.Tensor,
    gamma: float = 0.8,
    vis_aware: bool = False,
    vis_aware_w: float = 0.1,
) -> torch.Tensor:
    """Per-stage L1 loss on trajectories (RAFT-style deep supervision).

    Args:
        flow_preds (list[Tensor]): List of *K* predicted flows ``(B, S, N, 2)``.
        flow_gt (Tensor): Ground-truth flow ``(B, S, N, 2)``.
        vis (Tensor): Visibility mask ``(B, S, N)``.
        valids (Tensor): Static validity mask ``(B, S, N)``.
        gamma (float, *0.8*): Exponential stage weighting ``gamma^(K-i-1)``.
        vis_aware (bool, *False*): If *True* weight loss by visibility.
        vis_aware_w (float, *0.1*): Extra visibility weight.

    Returns:
        Scalar mean flow loss over all prediction stages.
    """
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3)  # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            # Add, don't add to the mask itself.
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        elif num_valid_points > 2:
            i_loss = i_loss[combined_mask]
            flow_loss += i_weight * i_loss.mean()
        else:
            # When there are too few valid points, add a dummy loss to maintain gradients
            i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
            flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss


def reduce_masked_mean(
    input: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Masked mean helper.

    This utility averages elements of *input* that are selected by *mask* in a
    numerically stable way and works as a drop-in replacement for
    ``torch.mean`` when per-element validity masks are involved.

    The implementation is differentiable w.r.t. both *input* and *mask* (the
    latter is often a float tensor coming from e.g. confidence estimation).

    Args:
        input: Tensor to average.
        mask:  Boolean or float tensor broadcastable to *input* (1 = keep).
        dim:   Dimension along which to reduce.  When *None* reduce all dims.
        keepdim: Keep the reduced dimension in the output.
        eps:  Small constant to avoid division by zero.

    Returns:
        The masked mean as a tensor with the requested shape.
    """
    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (eps + denom)
    return mean
