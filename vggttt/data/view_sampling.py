# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Functions for sampling views from a dataset."""

from collections import defaultdict
from typing import Any, Literal, TypedDict

import numpy as np

from vggttt.geometry import rotmat_angle_diff

Strategy = Literal["sequential", "overlap", "random", "pose_similarity"]


class ViewSamplerConf(TypedDict):
    strategy: Strategy
    settings: dict[str, Any]


def sample_views(view_sampler: ViewSamplerConf, **kwargs) -> list[int] | tuple[list[int], list[int]]:
    strategy = view_sampler["strategy"]
    settings = view_sampler["settings"]
    if strategy == "sequential":
        return sample_views_sequential(**settings, **kwargs)
    elif strategy == "overlap":
        return sample_views_overlap(**settings, **kwargs)
    elif strategy == "random":
        return sample_views_random(**settings, **kwargs)
    elif strategy == "pose_similarity":
        return sample_views_pose_similarity(**settings, **kwargs)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")


def sample_views_sequential(
    start_idx: int,
    num_images: int,
    max_idx: int,
    img_range: int = 25,
    seed: int = 0,
    allow_duplicates: bool = False,
    view_transition_prob: float = 0.0,
    valid_view_transitions: dict[int, list[int]] | None = None,
    sort_by_frames: bool = False,
    **kwargs,
) -> tuple[list[int], list[int]]:
    """Sample views such that each frame is at most `img_range` frames away from every other frame.

    With probability `view_transition_prob`, transitions to a new view instead of expanding the sampling range.
    """
    sampler = np.random.default_rng(seed)

    # Sample starting view
    current_view = 0
    if valid_view_transitions:
        current_view = int(sampler.choice(list(valid_view_transitions.keys())))

    current_frame = start_idx
    frames_to_views = defaultdict(list)
    frames_to_views[current_frame] = [current_view]

    for _ in range(num_images - 1):
        # Decide whether to transition to a new view
        if (
            valid_view_transitions
            and view_transition_prob > 0
            and current_view in valid_view_transitions
            and len(valid_view_transitions[current_view]) > 0
            and sampler.random() < view_transition_prob
        ):
            # Transition to a new view
            available_views = valid_view_transitions[current_view]
            available_views = [v for v in available_views if v not in frames_to_views[current_frame]]

            if available_views:
                current_view = int(sampler.choice(available_views))
                frames_to_views[current_frame].append(current_view)
                continue

        # Stay in current view and expand sampling range
        max_sampled_frame = max(frames_to_views.keys())
        min_sampled_frame = min(frames_to_views.keys())
        frontier_left = (max(0, min_sampled_frame - img_range), min_sampled_frame)
        frontier_right = (max_sampled_frame + 1, min(max_sampled_frame + img_range, max_idx))

        options = list(range(*frontier_left)) + list(range(*frontier_right))
        if not options:
            options = [
                i for i in range(min_sampled_frame, max_sampled_frame) if allow_duplicates or i not in frames_to_views
            ]

        if not options:
            break

        current_frame = int(sampler.choice(options))
        frames_to_views[current_frame].append(current_view)

    img_idxs = []
    view_idxs = []

    frame_idxs = list(frames_to_views.keys())
    if sort_by_frames:
        frame_idxs = sorted(frame_idxs)

    for f in frame_idxs:
        vs = frames_to_views[f]
        for v in vs:
            img_idxs.append(f)
            view_idxs.append(v)
    return img_idxs, view_idxs


def sample_views_overlap(
    start_idx: int,
    num_images: int,
    pairwise_overlap: np.ndarray,
    num_views: int,
    current_view_idx: int = 0,
    min_overlap: float = 0.2,
    seed: int = 0,
    allow_duplicates: bool = False,
    **kwargs,
) -> list[int] | tuple[list[int], list[int]]:
    """Sample views with overlap."""
    sampler = np.random.default_rng(seed)

    # Convert from frame and view indices to global index used to index the pairwise overlap matrix
    img_idxs = [start_idx * num_views + current_view_idx]
    for _ in range(num_images - 1):
        cur_frames_overlap = pairwise_overlap[img_idxs]

        # Sample any frame with overlap > min_overlap
        weights = (cur_frames_overlap > min_overlap).any(axis=0).astype(np.float32)

        # If duplicates are not allowed, exclude already sampled indices
        if not allow_duplicates:
            weights[img_idxs] = 0

        if weights.sum() == 0:
            break

        img_idxs.append(sampler.choice(len(weights), p=weights / weights.sum()))

    # Convert global index to frame and view indices
    if num_views > 1:
        img_idxs = np.array(img_idxs)
        frame_idxs = (img_idxs // num_views).tolist()
        view_idxs = (img_idxs % num_views).tolist()
        return frame_idxs, view_idxs
    return img_idxs


def sample_views_random(start_idx: int, num_images: int, max_idx: int, seed: int = 0, **kwargs) -> list[int]:
    """Sample views fully randomly with replacement."""
    sampler = np.random.default_rng(seed)
    return [start_idx] + sampler.integers(low=0, high=max_idx, size=num_images - 1).tolist()


def rank_views_by_pose_similarity(
    start_idx: int, poses: np.ndarray, lambda_t: float = 1.0, normalize: bool = True
) -> np.ndarray:
    """Rank views by pose similarity. Useful for unordered image collections."""
    pos = poses[:, :3, 3].copy()
    if normalize:
        pos -= pos.mean(axis=0)
        avg_scale = np.mean(np.linalg.norm(pos, axis=1))
        pos /= avg_scale

    t_dists = np.linalg.norm(pos - pos[start_idx], axis=1)
    r_dists = rotmat_angle_diff(poses[:, :3, :3], poses[[start_idx], :3, :3], unit="rad")
    r_dists = np.rad2deg(r_dists) / 180.0
    dists = r_dists + lambda_t * t_dists
    sorted_idxs = np.argsort(dists)
    return sorted_idxs


def sample_views_pose_similarity(
    start_idx: int, num_images: int, max_idx: int, poses: np.ndarray, expand_range: int = 25, seed: int = 0, **kwargs
) -> list[int]:
    """Sample views in a range ranked by pose similarity."""
    sampler = np.random.default_rng(seed)
    assert poses.shape[0] == max_idx, "poses must be of shape (max_idx, 4, 4)"

    idxs_ranked_by_similarity = rank_views_by_pose_similarity(start_idx, poses)

    # Exclude start_idx from the population (it's always at index 0 due to distance 0)
    available_idxs = idxs_ranked_by_similarity[1:expand_range]
    num_samples_to_draw = min(num_images - 1, len(available_idxs))

    idxs = sampler.choice(available_idxs, size=num_samples_to_draw, replace=False)
    return [start_idx] + idxs.tolist()
