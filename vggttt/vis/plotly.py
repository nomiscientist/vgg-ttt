# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import numpy as np
import plotly.graph_objects as go
from PIL import Image


class PointcloudVisualizer:
    def __init__(self):
        """Initialize empty plotly figure for visualizing 3D pointclouds and cameras"""
        self.fig = go.Figure()

    def add_pointcloud(
        self,
        points: np.ndarray,
        point_size: int = 2,
        color: tuple | np.ndarray = (100, 100, 100),
        name: str | None = None,
    ):
        """Add a pointcloud to the visualization

        Args:
            points: (N,3) array of 3D points
            point_size: Size of points in visualization
            color: RGB tuple for point color
        """
        self.fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=color),
                name=name or "Point Cloud",
            )
        )
        return self

    def add_camera(
        self,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        frustum_scale: float = 0.05,
        name: str = None,
        color: None | np.ndarray = None,
        legendgroup: str | None = None,
        image: np.ndarray | None = None,
        image_scale: float = 1.0,
        **kwargs,
    ):
        """Add a single camera frustum to the visualization

        Args:
            pose: (4,4) camera-to-world transformation matrix
            intrinsics: (3,3) camera intrinsics matrix
            frustum_scale: Scaling factor for frustum size
            name: Optional name for the camera trace
            color: Optional color for the camera trace
            legendgroup: Optional legend group for the camera trace
            image: Optional (H,W,3) RGB image array to display at the image plane
            image_scale: Scaling factor for the image plane size relative to frustum_scale
        """
        if color is None:
            color = np.array([255, 0, 0], dtype=np.uint8)

        # Compute frustum corners using the helper method
        corners_world = self._compute_frustum_corners(pose, intrinsics, frustum_scale)

        # Camera center
        t = pose[:3, 3]

        # Create frustum lines
        x, y, z = [], [], []
        for j in range(4):
            # Frustum edges
            x.extend([t[0], corners_world[0, j]])
            y.extend([t[1], corners_world[1, j]])
            z.extend([t[2], corners_world[2, j]])

            # Image plane connections
            x.extend([corners_world[0, j], corners_world[0, (j + 1) % 4]])
            y.extend([corners_world[1, j], corners_world[1, (j + 1) % 4]])
            z.extend([corners_world[2, j], corners_world[2, (j + 1) % 4]])

        self.fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    color=f"rgb({color[0] * 255},{color[1] * 255},{color[2] * 255})"
                    if isinstance(color, np.ndarray)
                    else color,
                    width=4,
                ),
                name=name or legendgroup or "Camera",
                legendgroup=legendgroup,
                showlegend=kwargs.pop("showlegend", True),
            )
        )

        # Add image thumbnail at the image plane if provided
        if image is not None:
            self._add_image_plane(
                pose, intrinsics, image, frustum_scale, image_scale, name, legendgroup, kwargs.get("showlegend", True)
            )

        return self

    def add_cameras(
        self,
        poses: np.ndarray,
        intrinsics: np.ndarray,
        frustum_scale: float = 0.05,
        color: np.ndarray | None = None,
        name: str | None = None,
        images: list[np.ndarray] | None = None,
        image_scale: float = 1.0,
    ):
        """Add multiple camera frustums to the visualization

        Args:
            poses: (N,4,4) array of camera-to-world transformation matrices
            intrinsics: (N,3,3) array of camera intrinsics matrices
            frustum_scale: Scaling factor for frustum size
            color: Array of RGB values for camera colors
            name: Name for the camera group
            images: Optional list of (H,W,3) RGB image arrays to display at image planes
            image_scale: Scaling factor for the image plane size relative to frustum_scale
        """

        if color is None:
            color = np.zeros((len(poses), 3), dtype=np.uint8)

        color = np.array(color)
        if color.ndim == 1:
            color = np.tile(color, (len(poses), 1))

        name = name or "Cameras"
        for i, (pose, K, c) in enumerate(zip(poses, intrinsics, color)):
            showlegend = i == 0
            image = images[i] if images is not None and i < len(images) else None
            self.add_camera(
                pose,
                K,
                frustum_scale=frustum_scale,
                name=None,
                color=c,
                legendgroup=name,
                showlegend=showlegend,
                image=image,
                image_scale=image_scale,
            )
        return self

    def _compute_frustum_corners(
        self,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Compute frustum corner positions in world coordinates

        Args:
            pose: (4,4) camera-to-world transformation matrix
            intrinsics: (3,3) camera intrinsics matrix
            scale: Scaling factor for frustum size

        Returns:
            (3,4) array of world coordinates for the 4 frustum corners
        """
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Frustum corners in image space
        W = 2 * intrinsics[0, 2]
        H = 2 * intrinsics[1, 2]
        corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]])

        # Convert to 3D rays
        Kinv = np.linalg.inv(intrinsics)
        rays = Kinv @ corners.T
        rays /= rays[2]  # Normalize z=1 plane
        rays *= scale

        # Transform to world coordinates
        corners_world = R @ rays + t.reshape(-1, 1)

        return corners_world

    def _add_image_plane(
        self,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        image: np.ndarray,
        frustum_scale: float,
        image_scale: float,
        name: str | None,
        legendgroup: str | None,
        showlegend: bool,
    ):
        """Add an image plane mesh with the camera image as texture

        The image plane corners will align exactly with the frustum corners when image_scale=1.0
        """
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Compute corners using the same method as the frustum, but with image scaling
        scaled_frustum_scale = frustum_scale * image_scale

        # Get image dimensions for coordinate mapping
        Kinv = np.linalg.inv(intrinsics)

        # Use a more reasonable resolution for better performance
        target_resolution = 64
        H, W = image.shape[:2]

        # Resize image to target resolution
        img_resized = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
        img_resized = img_resized.resize((target_resolution, target_resolution), Image.LANCZOS)
        img_array = np.array(img_resized) / 255.0  # Normalize to [0,1] for plotly

        # Note: We need to be careful about coordinate system mapping
        # Image coords: (0,0) at top-left, x goes right, y goes down
        # Camera coords: (0,0) at bottom-left, x goes right, y goes up
        # Flip image vertically to match camera coordinate system
        img_array = np.flipud(img_array)
        j_coords, i_coords = np.meshgrid(
            np.linspace(0, W, target_resolution), np.linspace(0, H, target_resolution), indexing="xy"
        )

        # Create homogeneous coordinates for all points at once
        ones = np.ones_like(i_coords)
        img_coords = np.stack([j_coords, i_coords, ones], axis=-1)  # Shape: (H, W, 3)

        # Vectorized transformation to camera coordinates
        rays = np.einsum("ij,hwj->hwi", Kinv, img_coords)  # Apply Kinv to each point
        rays = rays / rays[..., 2:3]  # Normalize z=1 plane
        rays *= scaled_frustum_scale

        # Transform to world coordinates (vectorized)
        world_coords = np.einsum("ij,hwj->hwi", R, rays) + t  # Shape: (H, W, 3)

        # Convert RGB to grayscale for surface coloring
        if img_array.ndim == 3:  # RGB image
            gray_array = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        else:
            gray_array = img_array

        self.fig.add_trace(
            go.Surface(
                x=world_coords[..., 0],
                y=world_coords[..., 1],
                z=world_coords[..., 2],
                surfacecolor=gray_array,
                colorscale="gray",
                opacity=0.9,
                name=f"{name or legendgroup or 'Camera'} Image" if showlegend else None,
                legendgroup=legendgroup,
                showlegend=False,
                hovertemplate="<b>Camera Image</b><extra></extra>",
                lighting=dict(ambient=0.8, diffuse=0.8, specular=0.1),
                showscale=False,
            )
        )

    def show(self):
        """Display the visualization"""
        self.fig.update_layout(
            scene=dict(
                aspectmode="data",
                camera=dict(eye=dict(x=0.0, y=-2.5, z=0.0), up=dict(x=0, y=-1.0, z=0)),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )
        return self.fig
