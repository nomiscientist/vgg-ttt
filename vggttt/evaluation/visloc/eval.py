# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Evaluation for visual localization.

Supported datasets:
- 7scenes
- Wayspots
"""

import json
import logging
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import roma
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from vggttt.evaluation.data import Frame, preproc_fun
from vggttt.geometry import Rt_to_4x4, inverse_se3, rotmat_angle_diff, transform_pc
from vggttt.utils.bench import Timer

_logger = logging.getLogger(__name__)


def combine_batches(frames: list[Frame]):
    out = {}
    for k, v in frames[0].items():
        out[k] = torch.stack([b[k] for b in frames], dim=0)
    return out


def prepare_input(data: dict[str, Any], add_poses: bool, add_intrinsics: bool) -> list[dict[str, Any]]:
    views = []
    device = data["image"].device
    for view_idx in range(len(data["image"])):
        view = {"img": data["image"][[view_idx]]}
        if add_poses:
            view["camera_poses"] = data["pose"][[view_idx]]
            view["is_metric_scale"] = torch.tensor([True], device=device, dtype=torch.bool)
        if add_intrinsics:
            view["intrinsics"] = data["intrinsics"][[view_idx]]
        views.append(view)
    return views


def _apply_sim3_to_poses(
    poses: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply a Sim(3) transformation to camera-to-world poses."""

    rot_part = torch.einsum("i j,... j k->... i k", rotation, poses[..., :3, :3])
    trans_part = torch.einsum("i j,... j->... i", rotation, poses[..., :3, 3])
    aligned = poses.clone()
    aligned[..., :3, :3] = rot_part
    aligned[..., :3, 3] = scale * trans_part + translation
    return aligned


def get_sim3_align(gt_poses: torch.Tensor, pred_poses: torch.Tensor):
    R, t, s = roma.rigid_points_registration(gt_poses[:, :3, 3], pred_poses[:, :3, 3], compute_scaling=True)
    return s, R, t


@torch.inference_mode()
def evaluate(
    dataset,
    scene: str,
    model: torch.nn.Module,
    map_imgs_stride: int,
    query_imgs_stride: int,
    dtype: torch.dtype,
    output_dir: Path,
    align_sim3: bool,
    num_workers: int,
    use_map_cameras: bool = False,
    use_query_intrinsics: bool = False,
    map_kwargs: dict[str, Any] | None = None,
    patch_size: int = 14,
    target_img_size: int = 518,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    map_kwargs = map_kwargs or {}
    preproc = partial(preproc_fun, target_size=target_img_size, patch_size=patch_size)

    # Map data
    map_data = dataset(seq_id=scene, split="train", preproc_fun=preproc)
    if map_imgs_stride > 1:
        map_data = Subset(map_data, range(0, len(map_data), map_imgs_stride))
    map_data = DataLoader(
        map_data, batch_size=16, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=lambda x: x
    )
    map_frames: list[Frame] = list(chain(*map_data))
    map_data = combine_batches(map_frames)
    to_map_origin = inverse_se3(map_data["pose"][[0]])
    map_data["pose"] = to_map_origin @ map_data["pose"]

    # Query data
    query_dataset = dataset(seq_id=scene, split="test", preproc_fun=preproc)
    if query_imgs_stride > 1:
        query_dataset = Subset(query_dataset, range(0, len(query_dataset), query_imgs_stride))
    query_data = DataLoader(query_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)

    views = prepare_input(map_data, add_poses=use_map_cameras, add_intrinsics=use_map_cameras)
    with torch.autocast("cuda", dtype=dtype):
        map_out = model.map(views, **map_kwargs)

    for k, v in map_out.items():
        if isinstance(v, torch.Tensor):
            map_out[k] = v.cpu()

    if align_sim3:
        s, R, t = get_sim3_align(map_out["pose"], map_data["pose"])
    else:
        s, R, t = torch.tensor(1.0), torch.eye(3), torch.zeros(3)

    map_out["pose"] = _apply_sim3_to_poses(map_out["pose"], s, R, t)
    map_out["pts3d"] = transform_pc(map_out["pts3d"], Rt_to_4x4(R, t, s))

    # Localize query frames
    r_errors = []
    t_errors = []
    query_times = []
    for i, batch in enumerate(tqdm(query_data, desc="Localizing query frames")):
        batch["pose"] = to_map_origin @ batch["pose"]

        views = prepare_input(batch, add_poses=False, add_intrinsics=use_query_intrinsics)

        with torch.autocast("cuda", dtype=dtype), Timer() as timer:
            query_out = model.query(views, **map_kwargs)

        query_times.append(timer.get())

        for k, v in query_out.items():
            if isinstance(v, torch.Tensor):
                query_out[k] = v.cpu()

        query_out["pose"] = _apply_sim3_to_poses(query_out["pose"], s, R, t)
        query_out["pts3d"] = transform_pc(query_out["pts3d"], Rt_to_4x4(R, t, s))

        pred_pose = query_out["pose"]
        gt_pose = batch["pose"]

        r_errors.append(rotmat_angle_diff(pred_pose[:, :3, :3].cpu(), gt_pose[:, :3, :3], unit="deg").numpy())
        t_errors.append((pred_pose[:, :3, 3].cpu() - gt_pose[:, :3, 3]).norm(dim=-1).numpy())

    return {
        "r_error_deg": np.mean(r_errors).item(),
        "t_error_m": np.mean(t_errors).item(),
        "acc_10_deg_cm": np.mean((np.array(r_errors) < 10) & (np.array(t_errors) < 0.1)).item() * 100,
        "acc_20_deg_cm": np.mean((np.array(r_errors) < 20) & (np.array(t_errors) < 0.2)).item() * 100,
        "acc_30_deg_cm": np.mean((np.array(r_errors) < 30) & (np.array(t_errors) < 0.3)).item() * 100,
        "query_time_s": np.mean(query_times).item(),
        "fps": len(query_dataset) / (np.sum(query_times).item() + 1e-6),
    }


@hydra.main(config_path="../config", config_name="visloc", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    assert cfg.output_dir is not None

    dataset = hydra.utils.instantiate(cfg.data.dataset)
    _logger.info("Loaded dataset")

    device = torch.device(cfg.device)
    dtype = torch.bfloat16 if torch.cuda.get_device_properties(device).major >= 8 else torch.float16

    model = hydra.utils.instantiate(cfg.model).eval().to(cfg.device)
    _logger.info("Loaded model")

    output_dir = Path(cfg.output_dir) / cfg.data.name
    output_dir.mkdir(parents=True, exist_ok=True)

    per_scene_metrics = []
    for scene in cfg.data.eval_scenes:
        metrics = evaluate(
            dataset=dataset,
            scene=scene,
            model=model,
            map_imgs_stride=cfg.data.map_imgs_stride,
            query_imgs_stride=cfg.data.query_imgs_stride,
            use_map_cameras=cfg.use_map_cameras,
            use_query_intrinsics=cfg.use_query_intrinsics,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            align_sim3=cfg.align_sim3,
            num_workers=cfg.num_workers,
            patch_size=cfg.patch_size,
            target_img_size=cfg.target_img_size,
            map_kwargs=cfg.map_kwargs,
        )
        per_scene_metrics.append({"scene": scene, **metrics})

    avg_metrics = {}
    for metric_name in per_scene_metrics[0].keys():
        if metric_name == "scene":
            continue

        avg_metrics[metric_name] = np.mean([m[metric_name] for m in per_scene_metrics]).item()

    avg_metrics["per_scene"] = per_scene_metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)


if __name__ == "__main__":
    main()
