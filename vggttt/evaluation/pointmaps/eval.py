# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Evaluation 3D reconstruction following Spann3R/CUT3R.

Supported datasets:
- 7scenes
- NRGBD
- DTU
- ETH3D
"""

import json
import logging
from collections import defaultdict
from functools import partial
from itertools import islice, pairwise
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
import torch
from einops import asnumpy, rearrange
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from vggttt.data.utils import move_to_device
from vggttt.evaluation.data import preproc_fun
from vggttt.evaluation.pointmaps.utils import accuracy, align_to_gt, completion, icp
from vggttt.geometry import inverse_se3, unproject_depthmap_to_point_map
from vggttt.utils.bench import PeakMemory, Timer
from vggttt.utils.dist import get_sp_group, init_sp_group

_logger = logging.getLogger(__name__)


def register_logging_hook(model: torch.nn.Module, hook_modules: list[str] = ["logging_hook"]):
    outputs = defaultdict(list)

    def get_hook(module_name: str):
        def hook_fn(module, input, output):
            outputs[module_name].append(move_to_device(output, torch.device("cpu")))

        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if name.split(".")[-1] not in hook_modules:
            continue
        hooks.append(module.register_forward_hook(get_hook(name)))
    return outputs, hooks


class Sequences:
    """Load sequences from sequence-to-frame IDs mapping."""

    def __init__(
        self,
        base_dataset: type[torch.utils.data.Dataset],
        seq_ids_map: dict[str, list[int]],
        preproc_fun: Callable | None = None,
    ):
        self._index = list(seq_ids_map)
        self.seq_ids_map = seq_ids_map
        self.base_dataset = base_dataset
        self.preproc_fun = preproc_fun if preproc_fun is not None else lambda x: x

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int):
        seq_id = self._index[idx]

        frame_ids = self.seq_ids_map[seq_id]

        if isinstance(frame_ids, tuple):
            ref_view_idxs, interp_view_idxs = frame_ids
        else:
            ref_view_idxs = frame_ids
            interp_view_idxs = []

        dataset = self.base_dataset(seq_id=seq_id)

        frames = []
        for frame_id in ref_view_idxs + interp_view_idxs:
            frames.append(self.preproc_fun(dataset[frame_id]))

        out = {}
        for k in frames[0].keys():
            out[k] = torch.stack([f[k] for f in frames], dim=0)

        pose = inverse_se3(out["pose"][[0]]) @ out["pose"]
        out["pts3d"], out["is_valid"] = unproject_depthmap_to_point_map(out["depth"], out["intrinsics"], pose)
        out["seq_id"] = seq_id
        out["num_ref_views"] = len(ref_view_idxs)
        out["num_interp_views"] = len(interp_view_idxs)
        return out


def evaluate(
    model: torch.nn.Module,
    sequences: torch.utils.data.DataLoader,
    output_dir: Path,
    icp_threshold: float | None,
    device: torch.device,
    dtype: torch.dtype,
    rank: int = 0,
    world_size: int = 1,
    create_vis: bool = False,
    fwd_kwargs: dict | None = None,
    store_log_outputs: bool = False,
    overwrite: bool = False,
):
    fwd_kwargs = fwd_kwargs or {}
    metrics = []

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "metrics.json"

    if not overwrite and output_json.is_file():
        with Path(output_json).open("r") as f:
            metrics = json.load(f)

        if isinstance(metrics, dict):
            _logger.info(f"Metrics already computed for {output_dir.name}. Skipping...")
            return

    sp_group = get_sp_group()
    num_done_seqs = len(metrics)
    for seq in tqdm(islice(sequences, num_done_seqs, None), total=len(sequences), initial=num_done_seqs):
        images = seq["image"].to(device)
        seq_name = seq["seq_id"]
        sanitized_seq_name = seq_name.replace("/", "-")
        num_ref_views = seq["num_ref_views"]
        num_interp_views = seq["num_interp_views"]

        _logger.debug(
            f"[RANK {rank}/{world_size}] Running model with num_ref_views: {num_ref_views}, num_interp_views: {num_interp_views}"
        )
        if store_log_outputs:
            log_outputs, hooks = register_logging_hook(model)

        if world_size > 1:
            # Wait until all ranks have loaded the images
            sp_group.barrier()

        with torch.autocast("cuda", dtype=dtype), Timer() as timer, PeakMemory() as peak_memory, torch.inference_mode():
            out = model.infer(images, **fwd_kwargs)

        if store_log_outputs:
            for hook in hooks:
                hook.remove()

        if world_size > 1:
            sp_group.barrier()

        # Only perform metric computatin on first rank
        if rank != 0:
            continue

        if store_log_outputs and log_outputs:
            torch.save(log_outputs, output_dir / f"{sanitized_seq_name}_log_outputs.pth")
            del log_outputs

        is_valid = asnumpy(seq["is_valid"][:num_ref_views])
        gt_pts = asnumpy(seq["pts3d"][:num_ref_views])
        pred_pts = asnumpy(out["pts3d"][:num_ref_views])

        _logger.info("Aligning to GT")
        pred_pts = align_to_gt(pred_pts, gt_pts, is_valid)

        _logger.info("Running ICP")
        pcd, pcd_gt = icp(pred_pts[is_valid], gt_pts[is_valid], threshold=icp_threshold)

        _logger.info("Estimating normals")
        pcd.estimate_normals()
        pcd_gt.estimate_normals()

        gt_normal = np.asarray(pcd_gt.normals)
        pred_normal = np.asarray(pcd.normals)
        gt_points = np.asarray(pcd_gt.points).astype(np.float32)
        pred_points = np.asarray(pcd.points).astype(np.float32)

        if create_vis:
            from vggttt.vis.trimesh import PointcloudVisualizer

            rgb = asnumpy((rearrange(images[:num_ref_views], "b c h w -> b h w c") * 255.0)).astype(np.uint8)
            (
                PointcloudVisualizer()
                .add_pointcloud(pred_points, color=(255, 0, 0), name="Pred PC")
                .add_pointcloud(gt_points, color=rgb[is_valid], name="GT PC")
                .show(output_dir / f"{sanitized_seq_name}_post_icp.glb")
            )

        _logger.info("Computing metrics")
        num_total_views = num_ref_views + num_interp_views * world_size
        metrics.append(
            {
                "seq_name": seq_name,
                **accuracy(gt_points, pred_points, gt_normal, pred_normal),
                **completion(gt_points, pred_points, gt_normal, pred_normal),
                "num_ref_views": num_ref_views,
                "num_interp_views": num_interp_views * world_size,  # Additional views are sharded across ranks
                "num_total_views": num_total_views,
                "forward_time_s": timer.get(),
                "fps": num_total_views / timer.get(),
                **peak_memory.get(),
            }
        )

        if output_json:
            with Path(output_json).open("w") as f:
                json.dump(metrics, f, indent=4)

        torch.cuda.empty_cache()

    if rank != 0:
        return

    agg_metrics = {}
    for k in metrics[0].keys():
        try:
            float(metrics[0][k])
        except ValueError:
            continue

        agg_metrics[k] = np.mean([seq_metrics[k] for seq_metrics in metrics])

    if "nc1" in agg_metrics and "nc2" in agg_metrics:
        agg_metrics["nc"] = (agg_metrics["nc1"] + agg_metrics["nc2"]) / 2.0
        agg_metrics["nc_med"] = (agg_metrics["nc1_med"] + agg_metrics["nc2_med"]) / 2.0

    agg_metrics["cd"] = (agg_metrics["acc"] + agg_metrics["comp"]) / 2.0
    agg_metrics["cd_median"] = (agg_metrics["acc_median"] + agg_metrics["comp_median"]) / 2.0

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with Path(output_json).open("w") as f:
            json.dump({"per_scene_results": metrics, **agg_metrics}, f, indent=4)


def get_view_idxs(n_images: int, n_extra_views: int, subsample: int | None = None, n_ref_views: int | None = None):
    """Get the view indices for the reference and supporting views."""
    assert (n_ref_views is not None) ^ (subsample is not None)

    ref_view_idxs = np.linspace(0, n_images, n_ref_views, dtype=int, endpoint=False).tolist()

    intervals = list(pairwise(ref_view_idxs + [n_images]))
    num_intervals = len(intervals)

    extra_views_per_interval = n_extra_views // num_intervals
    extra_views_per_interval_list = np.array([extra_views_per_interval] * num_intervals)
    extra_views_remainder = n_extra_views % num_intervals
    extra_views_per_interval_list[:extra_views_remainder] += 1

    support_view_idxs = []
    for interval, num_extra_views in zip(intervals, extra_views_per_interval_list):
        _, *views, _ = np.linspace(interval[0], interval[1], num_extra_views + 2, dtype=int, endpoint=True).tolist()
        support_view_idxs.extend(views)

    return ref_view_idxs, support_view_idxs


@hydra.main(config_path="../config", config_name="pointmap", version_base=None)
def main(cfg: DictConfig):
    _logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    assert cfg.output_dir is not None

    import os

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    # Set the device to the local rank to avoid multiple processes using the same GPU
    device = torch.device(f"cuda:{LOCAL_RANK}")
    torch.cuda.set_device(device)

    if WORLD_SIZE > 1:
        torch.distributed.init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE)
        init_sp_group(rank=RANK, world_size=WORLD_SIZE, sequence_parallel_size=WORLD_SIZE)

    dataset = hydra.utils.instantiate(cfg.data.dataset)

    if hasattr(cfg.data, "seq_id_map"):
        with open(cfg.data.seq_id_map, "r") as f:
            seq_ids_map = json.load(f)
    else:
        seq_ids_map = {}
        for eval_scene in cfg.data.eval_scenes:
            total_num_imgs = len(dataset(seq_id=eval_scene))
            ref_view_idxs, interp_view_idxs = get_view_idxs(
                total_num_imgs, n_extra_views=cfg.data.additional_views, n_ref_views=cfg.data.n_ref_views
            )
            interp_view_idxs = interp_view_idxs[RANK::WORLD_SIZE]
            ref_view_idxs = ref_view_idxs if RANK == 0 else []
            seq_ids_map[eval_scene] = (ref_view_idxs, interp_view_idxs)

    preprocessing_function = partial(preproc_fun, target_size=cfg.target_img_size, patch_size=cfg.patch_size)
    seqs = Sequences(dataset, seq_ids_map, preproc_fun=preprocessing_function)
    loader = DataLoader(
        seqs, batch_size=1, shuffle=False, num_workers=1, drop_last=False, collate_fn=lambda x: x[0], prefetch_factor=1
    )
    _logger.info("Loaded dataset")
    dtype = torch.bfloat16 if torch.cuda.get_device_properties(device).major >= 8 else torch.float16

    model = hydra.utils.instantiate(cfg.model).eval().to(device)
    _logger.info("Loaded model")

    with sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True
    ):
        evaluate(
            model=model,
            sequences=loader,
            icp_threshold=cfg.data.icp_threshold,
            output_dir=Path(cfg.output_dir) / cfg.data.name,
            device=device,
            dtype=dtype,
            create_vis=cfg.create_vis,
            fwd_kwargs=cfg.fwd_kwargs,
            overwrite=cfg.overwrite,
            rank=RANK,
            world_size=WORLD_SIZE,
        )


if __name__ == "__main__":
    main()
