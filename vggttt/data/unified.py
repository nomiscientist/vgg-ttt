# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Unified dataloading for pre-processed data."""

import logging
import random
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from dataverse.datasets import CameraModel, DataField, DataSource
from dataverse.datasets.base import BaseDataset
from einops import asnumpy

from vggttt.data.view_sampling import ViewSamplerConf, sample_views
from vggttt.geometry import inverse_se3, transform_pc, unproject_depthmap
from vggttt.nets.vggt.img import get_target_shape, process_one_image
from vggttt.nets.vggt.training.track import build_tracks_by_depth

_logger = logging.getLogger(__name__)


DEFAULT_FIELDS_TO_LOAD = [
    DataField.IMAGE_RGB,
    DataField.CAMERA_C2W_TRANSFORM,
    DataField.CAMERA_INTRINSICS,
    DataField.DEPTH,
    DataField.CAMERA_MODEL,
    DataField.KEY,
    DataField.DISTORTION_COEFF,
    DataField.BACKGROUND_MASK,
]

MAX_RETRIES = 5


class DataverseAdapter(torch.utils.data.Dataset):
    """Dataset adapter for dataloading from dataverse-format datasets."""

    def __init__(
        self,
        dataverse_dataset: BaseDataset,
        augment_func: Callable,
        view_sampler: ViewSamplerConf = ViewSamplerConf(strategy="pose_similarity", settings={"expand_range": 10}),
        filter_depth_quantile: float | None = None,
        long_side_img_size: int = 256,
        patch_size: int = 14,
        aspect_ratio: float = 384 / 512,
        rescale_aug_pct: float = 0.0,
        max_depth_thresh: float | None = None,
        overlap_mats_path: str | None = None,
        overlap_version: Literal["min", "sym"] = "sym",
        add_track: bool = False,
        **kwargs,
    ):
        """Initialize dataset.

        Args:
            dataverse_dataset: Dataverse dataset to load from.
            augment_func: Augmentation function applied to each image independently.
            view_sampler: Configuration for how to sample other views from a scene.
            filter_depth_quantile: Depth values greater than this quantile are considered invalid.
            long_side_img_size: Longer side of the images to resize to.
            patch_size: Size of the patch to sample from the image.
            aspect_ratio: Default aspect ratio of the images output by the dataset. May be overriden through arguments
                to the `__getitem__` method.
            rescale_aug_pct: Maximum percentage to "zoom into" images as augmentation.
            max_depth_thresh: Maximum depth value beyond which depths are considered invalid.
        """
        super().__init__()

        self.dataverse_dataset = dataverse_dataset
        assert dataverse_dataset.num_videos() > 0, "DataverseAdapter requires at least one video"

        self.augment_func = augment_func
        self.view_sampler = view_sampler
        self.filter_depth_quantile = filter_depth_quantile
        self.long_side_img_size = (long_side_img_size // patch_size) * patch_size
        self.patch_size = patch_size
        self.aspect_ratio = aspect_ratio
        self.rescale_aug_pct = rescale_aug_pct
        self.max_depth_thresh = max_depth_thresh
        self.overlap_mats_path = overlap_mats_path
        self.add_track = add_track
        self.overlap_version = overlap_version

        self.dataset_id = self.dataverse_dataset.__class__.__name__
        meta = self.dataverse_dataset.get_dataset_metadata(0)
        self.is_metric = meta.is_metric_scale
        self.is_synthetic = meta.data_origin == DataSource.SYNTHETIC

    def read_from_dataverse(
        self, video_idx: int, frame_idxs: list[int], view_idxs: list[int]
    ) -> tuple[str, list[dict[str, np.ndarray]]]:
        available_fields = self.dataverse_dataset.available_data_fields()
        fields_to_read = [f for f in DEFAULT_FIELDS_TO_LOAD if f in available_fields]

        data = None
        try:
            data = self.dataverse_dataset.read(video_idx, frame_idxs, view_idxs, data_fields=fields_to_read)

            out = [
                {"key": f"{self.dataset_id}-{frame_idx}-{view_idx}"}
                for frame_idx, view_idx in zip(frame_idxs, view_idxs)
            ]
            for i, img in enumerate(data[DataField.IMAGE_RGB]):  # (b, 3, h, w) tensor or list
                # TODO: If already numpy arrays in correct format, pass data through
                out[i]["rgb"] = asnumpy(img.permute(1, 2, 0) * 255.0).astype(np.uint8)

            for i, depth in enumerate(data[DataField.DEPTH]):  # (b, h, w) tensor or list
                out[i]["depth"] = asnumpy(depth).astype(np.float32)

            for i, T_c2w in enumerate(data[DataField.CAMERA_C2W_TRANSFORM]):  # (b, 4, 4) tensor or list
                out[i]["T_c2w"] = asnumpy(T_c2w).astype(np.float32)

            for i, intrinsics in enumerate(data[DataField.CAMERA_INTRINSICS]):  # (b, 3, 3) tensor or list
                intr_mat = np.eye(3, dtype=np.float32)
                intr_mat[[0, 1, 0, 1], [0, 1, 2, 2]] = asnumpy(intrinsics)
                out[i]["intrinsics"] = intr_mat

            if DataField.BACKGROUND_MASK in data:
                for i, mask in enumerate(data[DataField.BACKGROUND_MASK]):  # (b, h, w) tensor or list
                    out[i]["mask"] = asnumpy(mask)

            # FIXME: Potentially want to do undistortion here to support more datasets
            if DataField.CAMERA_MODEL in data:
                assert all(CameraModel.PINHOLE.value == model for model in data[DataField.CAMERA_MODEL])
            else:
                assert DataField.DISTORTION_COEFF not in data

        except Exception as e:
            error_msg = f"video_idx: {video_idx}, frame_idxs: {frame_idxs}, view_idxs: {view_idxs}"
            key = error_msg + f", key: {data[DataField.KEY]}" if data else error_msg
            _logger.exception("Error reading %s", key)
            raise e

        return data[DataField.KEY], out

    def process_sample(
        self, sample: dict[str, Any], aspect_ratio: float, rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        orig_image = sample["rgb"]
        depth_map = sample["depth"]
        if "mask" in sample:
            is_background = sample["mask"]

            # Randomly mask the RGB-part of the background of the image
            if rng.random() < 0.5:
                orig_image[is_background] = 0
                depth_map[is_background] = 0.0

        image, depth_map, extri_opencv, intri_opencv, track = process_one_image(
            image=orig_image,
            depth_map=depth_map,
            extri_opencv=inverse_se3(sample["T_c2w"].astype(np.float32)),
            intri_opencv=sample["intrinsics"].astype(np.float32),
            target_image_shape=get_target_shape(aspect_ratio, self.long_side_img_size, self.patch_size),
            aug_scale=float(rng.uniform(1 - self.rescale_aug_pct, 1)),
            filepath=sample.get("key", None),
            rng=rng,
        )
        image = self.augment_func(image=image)["image"]

        view = {
            "true_shape": torch.tensor(image.shape[1:]),
            "img": image.to(torch.float32),
            "background_mask": torch.from_numpy(np.isclose(depth_map, 0) | np.isinf(depth_map) | np.isnan(depth_map)),
            "mask": torch.ones(*image.shape[1:], dtype=torch.bool),
            "intrinsics": torch.from_numpy(intri_opencv).to(torch.float32),
            "gt_poses": torch.from_numpy(inverse_se3(extri_opencv)).to(torch.float32),
        }

        depth = torch.from_numpy(depth_map).to(torch.float32)
        invalid_depth = depth.isnan() | (~depth.isfinite()) | (depth <= 0.0)
        if self.max_depth_thresh is not None:
            invalid_depth |= depth > self.max_depth_thresh

        if self.filter_depth_quantile is not None and not invalid_depth.all():
            depth_quantile = torch.quantile(depth[~invalid_depth], self.filter_depth_quantile)
            # Set to invalid if depth is greater than the quantile
            invalid_depth |= depth > depth_quantile

        view["mask"] &= ~invalid_depth
        depth[invalid_depth] = 0.0

        view["depth"] = depth

        # Store the points in local/global reference frame
        cam_pts3d, is_valid = unproject_depthmap(depth, K=view["intrinsics"].to(depth))
        view["cam_pts3d"] = cam_pts3d
        view["pts3d"] = transform_pc(cam_pts3d, view["gt_poses"])
        view["mask"] &= is_valid
        return view

    def get_view_overlap_mat(self, video_idx: int, num_frames: int) -> np.ndarray:
        """Load overlap matrices for the given video."""
        if self.overlap_mats_path is not None:
            video_id = self.dataverse_dataset.get_video_id(video_idx)
            covis_files_dir = Path(self.overlap_mats_path) / video_id / "covisibility" / "v0"

            # Assume only one covisibility file per video
            covis_file = list(covis_files_dir.glob(f"{self.overlap_version}_pairwise_covisibility*{num_frames}.npy"))[0]
            return np.load(covis_file, mmap_mode="r")

        if hasattr(self.dataverse_dataset, "get_view_overlap_mat"):
            return self.dataverse_dataset.get_view_overlap_mat(video_idx)

        raise ValueError(f"No view overlap matrix found {self.dataset_id}, video_idx: {video_idx}")

    def get_connected_components_per_view(self, video_idx: int, num_frames: int) -> np.ndarray:
        """Get the size of the connected component each view belongs to."""
        if self.overlap_mats_path is not None:
            video_id = self.dataverse_dataset.get_video_id(video_idx)
            assert self.overlap_mats_path is not None, "overlap_mats_path is not set"
            covis_files_dir = Path(self.overlap_mats_path) / video_id / "covisibility" / "v0"

            # Format of filename `connected_component_sizes_thresh=0.3_sym_pairwise_covisibility--384x384.npy`
            connected_components_sizes_file = list(
                covis_files_dir.glob(f"connected_component_sizes*_{self.overlap_version}_*{num_frames}.npy")
            )[0]
            return np.load(connected_components_sizes_file, mmap_mode="r")

        if hasattr(self.dataverse_dataset, "get_connected_components_per_view"):
            return self.dataverse_dataset.get_connected_components_per_view(video_idx)

        raise ValueError(f"No connected components per view found {self.dataset_id}, video_idx: {video_idx}")

    def sample_view_idxs(self, video_idx: int, num_images: int, seed: int) -> tuple[list[int], list[int]]:
        """Sample frame indices to load from the dataset."""

        num_frames = self.dataverse_dataset.num_frames(video_idx)
        num_views = self.dataverse_dataset.num_views(video_idx)
        num_total_images = num_frames * num_views

        if num_frames == 0:
            return [], []

        # Uniformly sample the start index
        rng = np.random.default_rng(seed)
        generator = torch.Generator().manual_seed(seed)
        start_idx = torch.randint(0, num_frames, size=(1,), generator=generator).item()
        current_view_idx = torch.randint(0, num_views, size=(1,), generator=generator).item()

        common_kwargs = {
            "start_idx": start_idx,
            "num_images": num_images,
            "max_idx": num_frames,
            "seed": seed,
            "current_view_idx": current_view_idx,
            "num_views": num_views,
        }

        view_sampler = self.view_sampler
        if self.view_sampler["strategy"] == "overlap":
            connected_components_sizes = self.get_connected_components_per_view(video_idx, num_total_images)

            # Start sampling from a node that belongs to a connnected component with at least `num_images``
            possible_start_idxs = np.nonzero(connected_components_sizes >= num_images)[0]
            if len(possible_start_idxs) > 0:
                start_img_idx = rng.choice(possible_start_idxs)
            else:
                # If no sufficiently big connected components, sample start proportional to connected component sizes
                weights = connected_components_sizes / connected_components_sizes.sum()
                start_img_idx = rng.choice(num_total_images, p=weights)

            common_kwargs["start_idx"] = start_img_idx // num_views
            common_kwargs["current_view_idx"] = start_img_idx % num_views

            common_kwargs["pairwise_overlap"] = self.get_view_overlap_mat(video_idx, num_total_images)
        elif self.view_sampler["strategy"] == "pose_similarity":
            # Load all poses for computing pose similarity
            frame_idxs = list(range(num_frames))
            data = self.dataverse_dataset.read(video_idx, frame_idxs, data_fields=[DataField.CAMERA_C2W_TRANSFORM])
            common_kwargs["poses"] = asnumpy(data[DataField.CAMERA_C2W_TRANSFORM])

        out = sample_views(view_sampler, **common_kwargs)
        if isinstance(out, tuple):
            return out
        else:
            return out, [current_view_idx] * len(out)

    def getitem_impl(
        self, video_idx: int, n_imgs: int, aspect_ratio: float, current_epoch: int, seed: int
    ) -> dict[str, torch.Tensor]:
        for _ in range(MAX_RETRIES):  # Max attempts to sample a video with frames
            frame_idxs, view_idxs = self.sample_view_idxs(video_idx, n_imgs, seed=seed)
            if frame_idxs:
                break

            _logger.warning("Dataset %s, video_idx %s has no frames, retrying.", self.dataset_id, video_idx)

            # We do not pass a explicit seed here since this might be called in a loop
            video_idx = random.randint(0, len(self) - 1)
        else:
            raise ValueError(
                f"Tried {MAX_RETRIES} times to sample a video with frames, but failed. "
                f"Dataset: {self.dataset_id}, last tried video_idx: {video_idx}"
            )

        maybe_worker_info = torch.utils.data.get_worker_info()
        worker_id = -1 if not maybe_worker_info else maybe_worker_info.id
        _logger.debug("[WORKER %s] video_idx: %s, frame_idxs: %s", worker_id, video_idx, frame_idxs)

        if len(frame_idxs) < n_imgs:
            _logger.debug(
                f"{self.dataset_id}, video_idx={video_idx} graph contains {len(frame_idxs)} < {n_imgs} images."
            )

        # Pad to the desired length with exisiting images
        num_pad = n_imgs - len(frame_idxs)
        frame_idxs = np.pad(frame_idxs, (0, num_pad), mode="wrap").tolist()
        view_idxs = np.pad(view_idxs, (0, num_pad), mode="wrap").tolist()

        key, data = self.read_from_dataverse(video_idx, frame_idxs, view_idxs)
        random_state = np.random.default_rng(seed)
        views = [self.process_sample(sample, aspect_ratio=aspect_ratio, rng=random_state) for sample in data]

        video_key = self.dataset_id + " " + key + f" video_idx={video_idx} frame_idxs={frame_idxs}"
        batch: dict[str, torch.Tensor] = {"key": video_key}
        for k in views[0].keys():
            batch[k] = torch.stack([v[k] for v in views])

        _logger.debug(
            "Aspect ratio %s, num images %s, current_epoch %s, Sampled idxs: %s",
            aspect_ratio,
            n_imgs,
            current_epoch,
            frame_idxs,
        )

        # Transform to first view to make sure that values are not crazy large which might cause problems if using
        # bfloat16 precision.
        to_first_cam = inverse_se3(batch["gt_poses"][[0]]).expand(batch["gt_poses"].shape[0], -1, -1)
        batch["pts3d"] = transform_pc(batch["pts3d"], to_first_cam)
        batch["gt_poses"] = to_first_cam @ batch["gt_poses"]

        if self.add_track:
            batch["track"], batch["track_is_visible"], batch["track_is_pos_sample"] = build_tracks_by_depth(
                extrinsics=inverse_se3(batch["gt_poses"]),
                intrinsics=batch["intrinsics"],
                world_points=batch["pts3d"],
                depths=batch["depth"],
                point_masks=batch["mask"],
                pos_rel_thres=0.05,
                target_track_num=512,
                seq_name=video_key,
            )

        batch["is_metric"] = torch.tensor(self.is_metric, dtype=torch.bool)
        batch["is_synthetic"] = torch.tensor(self.is_synthetic, dtype=torch.bool)
        return batch

    @torch.no_grad()
    def __getitem__(self, video_idx: int | tuple[int, dict[str, Any]]) -> dict[str, torch.Tensor]:
        aspect_ratio = self.aspect_ratio
        n_imgs = 2
        current_epoch = 0
        seed = None
        if isinstance(video_idx, tuple):
            video_idx, data = video_idx
            n_imgs = data.get("n_imgs", n_imgs)
            aspect_ratio = data.get("aspect_ratio", aspect_ratio)
            current_epoch = data.get("epoch", current_epoch)
            seed = data.get("seed", None)
        seed = seed or current_epoch | (video_idx << 32)  # Fallback if seed is not provided through sampler

        for _ in range(MAX_RETRIES):  # Max attempts to sample a video with frames
            batch = self.getitem_impl(video_idx, n_imgs, aspect_ratio, current_epoch, seed)

            tensor_names = []
            is_nan = []
            is_inf = []
            for k, v in batch.items():
                if not isinstance(v, torch.Tensor):
                    continue
                is_nan.append(torch.isnan(v).any().item())
                is_inf.append(torch.isinf(v).any().item())
                tensor_names.append(k)

            batch_is_clean = not any(is_nan) and not any(is_inf)
            if batch_is_clean and batch["mask"].any():  # batch is clean
                break

            video_key = batch["key"]
            if batch_is_clean:  # Batch does not have NaNs/Infs but no valid pixels.
                error_msg = f"Did not find any valid pixels in {video_key}. Retrying..."
            else:
                error_msg = f"Found NaNs/Infs in {video_key}."
                for name, nan, inf in zip(tensor_names, is_nan, is_inf):
                    error_msg += f"\n{name}: NaN={nan}, Inf={inf}"
                error_msg += "\nRetrying..."

            _logger.warning(error_msg)
            # We do not pass a explicit seed here since this might be called in a loop
            video_idx = random.randint(0, len(self) - 1)
        else:
            raise ValueError(f"Tried {MAX_RETRIES} times to sample a batch without NaNs or infs but failed.")

        return batch

    def __len__(self):
        return self.dataverse_dataset.num_videos()
