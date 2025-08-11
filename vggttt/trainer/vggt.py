# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging
from typing import Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from aim import Figure
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributed.nn import all_reduce

from vggttt.data.utils import asnumpy
from vggttt.metrics import PoseAcc, ScaleShiftInvMAE
from vggttt.utils.dist import gather_varlen_tensor, get_sp_group, init_sp_group
from vggttt.utils.optim import freeze_weights, make_parameter_groups
from vggttt.vis.plotly import PointcloudVisualizer

_logger = logging.getLogger(__name__)


class VGGT(L.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.strict_loading = False

        self.loss = instantiate(cfg.loss)
        self.model = None

        self.save_hyperparameters()

        # Measure median errors for entire graph of absolute poses
        self.pose_acc = PoseAcc()
        self.pc_metrics = ScaleShiftInvMAE()

    def configure_model(self):
        if self.model is not None:
            return
        self.model = instantiate(self.cfg.model)
        self.model.load_pretrained_weights()

    def setup(self, stage: str):
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        init_sp_group(global_rank, world_size, self.cfg.sequence_parallel_size)

    def forward(self, img: torch.Tensor, **kwargs):
        """Run model on graph(s) of images."""
        pred = self.model(img, **kwargs)
        return pred

    def _compute_loss(self, batch, split: Literal["train", "val", "test"], batch_idx: int):
        """Compute the loss for a batch of data."""
        pred = self(
            img=batch["img"],
            query_points=batch["track"][:, 0] if "track" in batch else None,  # First frame query points (B, N, 2)
        )

        if "pts3d" in batch:
            loss, details = self.loss(pred, batch, return_viz_data=split == "val")
        else:
            assert not self.training
            loss, details = 0.0, {}

        sp_group = get_sp_group()
        if sp_group is not None and sp_group.size() > 1 and isinstance(loss, torch.Tensor):
            loss = all_reduce(loss, op=dist.ReduceOp.AVG, group=sp_group)

        gt_world_pts = details.pop("gt_global_pts", None)
        pred_world_pts = details.pop("pred_global_pts", None)
        pred_world_pts_conf = details.pop("pred_global_conf", None)
        is_valid_pts = details.pop("is_valid", None)
        pred_c2w_pose = details.pop("pred_c2w_pose", None)
        gt_c2w_pose = details.pop("gt_c2w_pose", None)
        pred_intrinsics = details.pop("pred_intrinsics", None)

        if split == "train" or is_valid_pts is None:
            return loss, details

        ##############
        # Evaluation #
        ##############

        # For visualization but perform on all ranks to prevent hang due to distributed
        imgs = gather_varlen_tensor(batch["img"], group=sp_group, dim=1)
        gt_intrinsics = gather_varlen_tensor(batch["intrinsics"], group=sp_group, dim=1)

        # Compute metrics on all ranks
        if pred_c2w_pose is not None:
            self.pose_acc.update(pred_c2w_pose.to(torch.float32), gt_c2w_pose.to(torch.float32))
            self.pc_metrics.update(pred_world_pts, gt_world_pts, is_valid_pts)

        # Only rank 0 computes metrics and does visualization
        if not self.trainer.is_global_zero:
            return loss, details

        # Only visualize first batch
        if batch_idx != 0:
            return loss, details

        MAX_VIS_POINTS = 20_000
        downsample = max(1, is_valid_pts[0].sum().item() // MAX_VIS_POINTS)
        vis_valid = asnumpy(is_valid_pts[0])
        vis_pred = asnumpy(pred_world_pts[0])
        # Convert confidence values to RGB colormap
        vis_conf = asnumpy(pred_world_pts_conf[0])
        vis_gt = asnumpy(gt_world_pts[0])
        rgb = asnumpy(imgs[0].permute(0, 2, 3, 1))

        gt_poses = asnumpy(gt_c2w_pose[0])
        gt_intrs = asnumpy(gt_intrinsics[0])

        # Normalize confidence values to [0, 1] range if needed
        vis_conf_normalized = np.interp(vis_conf, (vis_conf.min(), vis_conf.max()), (-1, +1))
        # Apply colormap (using matplotlib's viridis colormap)
        vis_conf_rgb = plt.cm.viridis(vis_conf_normalized)[..., :3]  # Shape (n, 3)

        vis = (
            PointcloudVisualizer()
            .add_pointcloud(vis_gt[vis_valid][::downsample], color=rgb[vis_valid][::downsample], name="GT PC")
            .add_pointcloud(
                vis_pred[vis_valid][::downsample], color=vis_conf_rgb[vis_valid][::downsample], name="Pred PC"
            )
            .add_cameras(gt_poses, intrinsics=gt_intrs, color=(0, 1.0, 0), name="GT")
        )

        if pred_c2w_pose is not None:
            pred_poses = asnumpy(pred_c2w_pose[0])
            vis.add_cameras(pred_poses, intrinsics=asnumpy(pred_intrinsics[0]), color=(1.0, 0, 0), name="Pred. poses")

        self.logger.experiment.track(Figure(vis.show()), step=self.current_epoch, name="global_pts_pred")
        return loss, details

    def training_step(self, batch, batch_idx):
        if self.cfg.profile_memory:
            torch.cuda.memory._record_memory_history()
        loss, details = self._compute_loss(batch, split="train", batch_idx=batch_idx)
        if self.cfg.profile_memory:
            torch.cuda.memory._dump_snapshot("/tmp/mem_profile.pkl")
            1 / 0

        if torch.isnan(loss) or not torch.isfinite(loss):
            raise RuntimeError(f"Training diverged. Loss: {loss}")
        details = {f"train/{k}": v for k, v in details.items()}
        self.log_dict(
            {"train/loss": loss, **details},
            prog_bar=True,
            batch_size=batch["img"].size(0),
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            rank_zero_only=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.pose_acc.compute(), sync_dist=True)
        self.pose_acc.reset()

        self.log_dict(self.pc_metrics.compute(), sync_dist=True)
        self.pc_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.pose_acc.compute(), sync_dist=True)
        self.pose_acc.reset()

        self.log_dict(self.pc_metrics.compute(), sync_dist=True)
        self.pc_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, details = self._compute_loss(batch, split="val", batch_idx=batch_idx)
        self.log("val/loss", loss, batch_size=batch["img"].size(0), on_epoch=True, sync_dist=True, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        self._compute_loss(batch, split="test", batch_idx=batch_idx)

    def configure_optimizers(self):
        freeze_weights(self, self.cfg.get("freeze_weight_patterns", ()))
        params = make_parameter_groups(self, self.cfg.get("optimizer_groups", ()))

        optimizer = instantiate(self.cfg.optimizer, params)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(self.cfg.get("warmup_steps", 0), total_steps)
        warmup_ratio = self.cfg.get("warmup_ratio", 1.0)

        scheduler = [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_ratio, total_iters=warmup_steps)]
        if self.cfg.lr_cos_decay:
            scheduler.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.estimated_stepping_batches - warmup_steps,
                    eta_min=self.cfg.lr_final,
                )
            )
        return [optimizer], [{"scheduler": torch.optim.lr_scheduler.ChainedScheduler(scheduler), "interval": "step"}]

    def on_train_epoch_start(self) -> None:
        # WHYYYYYYYYYYY LIGHTNING????
        # It does not call set_epoch for some reason... despite it supposedly being fixed.
        # https://github.com/Lightning-AI/pytorch-lightning/issues/13316
        if self.trainer.train_dataloader is not None:
            if callable(getattr(self.trainer.train_dataloader.batch_sampler, "set_epoch", None)):
                self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch)
