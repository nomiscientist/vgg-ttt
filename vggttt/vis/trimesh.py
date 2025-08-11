# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image

from vggttt.data.utils import asnumpy

_logger = logging.getLogger(__name__)


def trimesh_to_pyrender(tscene: trimesh.Scene, pscene: pyrender.Scene) -> pyrender.Scene:
    # Iterate through trimesh scene and convert geometries
    for node_name in tscene.graph.nodes_geometry:
        transform, geometry_name = tscene.graph[node_name]
        geometry = tscene.geometry[geometry_name]

        if isinstance(geometry, trimesh.points.PointCloud):
            # Convert point cloud to pyrender mesh (using small spheres or points)
            # Get point colors if available
            colors = geometry.colors[:, :3] if geometry.colors is not None else None

            # Create a pyrender Mesh from the point cloud
            # We'll use pyrender's point cloud rendering (creates small squares)
            mesh = pyrender.Mesh.from_points(geometry.vertices, colors=colors)
            pscene.add(mesh, pose=transform)
        elif isinstance(geometry, trimesh.Trimesh):
            # Convert trimesh mesh to pyrender mesh
            mesh = pyrender.Mesh.from_trimesh(geometry)
            pscene.add(mesh, pose=transform)
        elif isinstance(geometry, trimesh.path.Path3D):
            # Create tube meshes from line segments
            vertices = geometry.vertices
            meshes = []

            for entity in geometry.entities:
                if not hasattr(entity, "points"):
                    continue

                # Get the line segment points
                points = entity.points
                for i in range(len(points) - 1):
                    # Create a tube for each line segment
                    start = vertices[points[i]]
                    end = vertices[points[i + 1]]

                    # Create a cylinder between the two points
                    height = np.linalg.norm(end - start)
                    if height <= 0:
                        continue

                    cylinder = trimesh.creation.cylinder(radius=0.01, height=height, sections=8)

                    # Transform cylinder to align with the line segment
                    direction = (end - start) / height
                    z_axis = np.array([0, 0, 1])

                    # Compute rotation to align z-axis with direction
                    rotation = np.eye(4)
                    if not np.allclose(direction, z_axis):
                        rotation_axis = np.cross(z_axis, direction)
                        if np.linalg.norm(rotation_axis) > 1e-6:
                            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                            rotation = trimesh.transformations.rotation_matrix(angle, rotation_axis)

                    # Translate to midpoint
                    midpoint = (start + end) / 2
                    rotation[:3, 3] = midpoint

                    cylinder.apply_transform(rotation)

                    # Set color if available
                    if hasattr(entity, "color") and entity.color is not None:
                        color_array = np.array(entity.color)
                        if color_array.max() > 1.0:
                            color_array = color_array / 255.0
                        cylinder.visual.vertex_colors = color_array

                    meshes.append(cylinder)

            # Combine all tube meshes and add to scene
            if meshes:
                combined_mesh = trimesh.util.concatenate(meshes)
                mesh = pyrender.Mesh.from_trimesh(combined_mesh)
                pscene.add(mesh, pose=transform)
        return pscene


class PointcloudVisualizer:
    def __init__(self):
        """Initialize empty trimesh scene for visualizing 3D pointclouds and cameras."""
        self.scene = trimesh.Scene()
        self._mesh_counter = 0

    def add_pointcloud(
        self,
        points: np.ndarray,
        color: tuple | np.ndarray = (100, 100, 100),
        name: str | None = None,
        max_pts: int = 100_000,
    ):
        """Add a pointcloud to the visualization.

        Args:
            points: (N,3) array of 3D points
            point_size: Size of points in visualization (used as sphere radius * 0.001)
            color: RGB tuple for point color (0-255 range) or (0-1 range)
            name: Optional name for the pointcloud

        Returns:
            self for method chaining
        """
        points = asnumpy(points)
        if points is None or points.size == 0:
            return self

        color_array = asnumpy(color)
        if color_array is None:
            raise ValueError("color must not be None")
        if color_array.max() > 1.0:
            color_array = color_array / 255.0
        if color_array.ndim == 1:
            colors = np.tile(color_array, (len(points), 1))
        else:
            colors = color_array

        downsample = len(points) // max_pts + 1
        points = points[::downsample]
        colors = colors[::downsample]

        # Create point cloud using trimesh
        point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors)

        mesh_name = name or f"pointcloud_{self._mesh_counter}"
        self.scene.add_geometry(point_cloud, node_name=mesh_name)
        self._mesh_counter += 1

        return self

    def add_camera(
        self,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        frustum_scale: float = 0.1,
        name: str = None,
        color: None | np.ndarray = None,
        legendgroup: str | None = None,
        image: np.ndarray | None = None,
        image_scale: float = 1.0,
        **kwargs,
    ):
        """Add a single camera frustum to the visualization.

        Args:
            pose: (4,4) camera-to-world transformation matrix
            intrinsics: (3,3) camera intrinsics matrix
            frustum_scale: Scaling factor for frustum size
            name: Optional name for the camera trace
            color: Optional color for the camera trace (0-1 range)
            legendgroup: Optional legend group for the camera trace
            image: Optional (H,W,3) RGB image array to display at the image plane
            image_scale: Scaling factor for the image plane size relative to frustum_scale

        Returns:
            self for method chaining
        """
        pose_array = asnumpy(pose)
        intrinsics_array = asnumpy(intrinsics)
        if pose_array is None or intrinsics_array is None:
            raise ValueError("pose and intrinsics must not be None")

        if color is None:
            color_array = np.array([1.0, 0.0, 0.0])
        else:
            color_array = asnumpy(color)
            if color_array is None:
                raise ValueError("color must not be None")

        if color_array.max() > 1.0:
            color_array = color_array / 255.0

        # Create camera frustum mesh
        frustum_mesh = self._create_frustum_mesh(pose_array, intrinsics_array, frustum_scale, color_array)

        mesh_name = name or legendgroup or f"camera_{self._mesh_counter}"
        self.scene.add_geometry(frustum_mesh, node_name=mesh_name)
        self._mesh_counter += 1

        # Add image plane if provided
        if image is not None:
            image_array = asnumpy(image)
            image_mesh = self._create_image_plane_mesh(
                pose_array, intrinsics_array, image_array, frustum_scale, image_scale
            )
            if image_mesh is not None:
                image_name = f"{mesh_name}_image"
                self.scene.add_geometry(image_mesh, node_name=image_name)
                self._mesh_counter += 1

        return self

    def add_cameras(
        self,
        poses: np.ndarray,
        intrinsics: np.ndarray,
        frustum_scale: float = 0.1,
        color: np.ndarray | None = None,
        name: str | None = None,
        images: list[np.ndarray] | None = None,
        image_scale: float = 1.0,
    ):
        """Add multiple camera frustums to the visualization.

        Args:
            poses: (N,4,4) array of camera-to-world transformation matrices
            intrinsics: (N,3,3) array of camera intrinsics matrices
            frustum_scale: Scaling factor for frustum size
            color: Array of RGB values for camera colors
            name: Name for the camera group
            images: Optional list of (H,W,3) RGB image arrays to display at image planes
            image_scale: Scaling factor for the image plane size relative to frustum_scale

        Returns:
            self for method chaining
        """
        poses_array = asnumpy(poses)
        intrinsics_array = asnumpy(intrinsics)
        if poses_array is None or intrinsics_array is None:
            raise ValueError("poses and intrinsics must not be None")

        if color is None:
            color_array = np.zeros((len(poses_array), 3))
        else:
            color_array = asnumpy(color)
            if color_array is None:
                raise ValueError("color must not be None")
        if color_array.ndim == 1:
            color_array = np.tile(color_array, (len(poses_array), 1))
        if color_array.max() > 1.0:
            color_array = color_array / 255.0

        name_prefix = name or "camera"
        for i, (pose_item, K_item, c_item) in enumerate(zip(poses_array, intrinsics_array, color_array)):
            image = images[i] if images is not None and i < len(images) else None
            self.add_camera(
                pose_item,
                K_item,
                frustum_scale=frustum_scale,
                name=f"{name_prefix}_{i}",
                color=c_item,
                image=image,
                image_scale=image_scale,
            )
        return self

    def _create_frustum_mesh(
        self, pose: np.ndarray, intrinsics: np.ndarray, scale: float, color: np.ndarray
    ) -> trimesh.path.Path3D:
        """Create a camera frustum as a 3D path/wireframe.

        Args:
            pose: (4,4) camera-to-world transformation matrix
            intrinsics: (3,3) camera intrinsics matrix
            scale: Scaling factor for frustum size
            color: RGB color array

        Returns:
            trimesh.path.Path3D representing the camera frustum
        """
        pose_array = asnumpy(pose)
        intrinsics_array = asnumpy(intrinsics)
        color_array = asnumpy(color)

        # Compute frustum corners
        corners_world = self._compute_frustum_corners(pose_array, intrinsics_array, scale)
        t = pose_array[:3, 3]  # Camera center

        # Create line segments for the frustum
        vertices = []
        entities = []

        # Add camera center
        vertices.append(t)
        center_idx = 0

        # Add frustum corners
        for i in range(4):
            vertices.append(corners_world[:, i])

        vertices = np.array(vertices)

        # Create line entities connecting center to corners
        for i in range(4):
            corner_idx = i + 1
            entities.append(trimesh.path.entities.Line([center_idx, corner_idx], color=color_array))

        # Create line entities connecting corners to form image plane
        for i in range(4):
            corner_idx1 = i + 1
            corner_idx2 = (i + 1) % 4 + 1
            entities.append(trimesh.path.entities.Line([corner_idx1, corner_idx2], color=color_array))

        # Create the 3D path
        path = trimesh.path.Path3D(vertices=vertices, entities=entities)

        return path

    def _compute_frustum_corners(
        self,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        """Compute frustum corner positions in world coordinates.

        Args:
            pose: (4,4) camera-to-world transformation matrix
            intrinsics: (3,3) camera intrinsics matrix
            scale: Scaling factor for frustum size

        Returns:
            (3,4) array of world coordinates for the 4 frustum corners
        """
        pose_array = asnumpy(pose)
        intrinsics_array = asnumpy(intrinsics)

        R = pose_array[:3, :3]
        t = pose_array[:3, 3]

        # Frustum corners in image space
        W = 2 * intrinsics_array[0, 2]
        H = 2 * intrinsics_array[1, 2]
        corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]])

        # Convert to 3D rays
        Kinv = np.linalg.inv(intrinsics_array)
        rays = Kinv @ corners.T
        rays /= rays[2]  # Normalize z=1 plane

        # Normalize scale by focal length to maintain consistent image plane area
        # The area at z=1 is proportional to 1/(fx*fy), so we scale by sqrt(fx*fy)
        # to maintain constant area across different focal lengths
        fx, fy = intrinsics_array[0, 0], intrinsics_array[1, 1]
        focal_scale = np.sqrt(fx * fy) / 1000.0  # Divide by 1000 for reasonable scaling
        normalized_scale = scale * focal_scale

        rays *= normalized_scale

        # Transform to world coordinates
        corners_world = R @ rays + t.reshape(-1, 1)

        return corners_world

    def _create_image_plane_mesh(
        self,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        image: np.ndarray,
        frustum_scale: float,
        image_scale: float,
    ) -> trimesh.Trimesh | None:
        """Create an image plane mesh with the camera image as texture.

        Args:
            pose: (4,4) camera-to-world transformation matrix
            intrinsics: (3,3) camera intrinsics matrix
            image: (H,W,3) RGB image array
            frustum_scale: Scaling factor for frustum size
            image_scale: Scaling factor for the image plane size relative to frustum_scale

        Returns:
            trimesh.Trimesh representing the image plane.
        """
        pose_array = asnumpy(pose)
        intrinsics_array = asnumpy(intrinsics)
        image_array = asnumpy(image)

        # Compute the scaled frustum corners for the image plane
        scaled_frustum_scale = frustum_scale * image_scale
        corners_world = self._compute_frustum_corners(pose_array, intrinsics_array, scaled_frustum_scale)

        # Create a quad mesh for the image plane
        vertices = corners_world.T  # (4, 3)

        # Define faces for the quad (two triangles, double-sided for visibility from both directions)
        faces = np.array(
            [
                [0, 1, 2],  # First triangle (front-facing)
                [0, 2, 3],  # Second triangle (front-facing)
                [2, 1, 0],  # First triangle (back-facing)
                [3, 2, 0],  # Second triangle (back-facing)
            ]
        )

        # Define UV coordinates for texture mapping
        # Map corners to texture coordinates: bottom-left, bottom-right, top-right, top-left
        uv_coords = np.array(
            [
                [0.0, 1.0],  # bottom-left corner
                [1.0, 1.0],  # bottom-right corner
                [1.0, 0.0],  # top-right corner
                [0.0, 0.0],  # top-left corner
            ]
        )

        # Convert image to PIL Image
        if image_array.max() <= 1.0:
            # Image is in [0,1] range, convert to [0,255]
            pil_image = Image.fromarray((image_array * 255).astype(np.uint8))
        else:
            # Image is already in [0,255] range
            pil_image = Image.fromarray(image_array.astype(np.uint8))

        # Create mesh with texture
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Apply texture using TextureVisuals
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=pil_image)

        return mesh

    def show(self, output_path: str | Path):
        """Export the visualization as a GLB file.

        Args:
            output_path: Path where to save the GLB file

        Returns:
            The output path where the file was saved
        """
        # Export the scene as GLB
        transformed_scene = self.scene.apply_transform(np.diag([1, -1, -1, 1]))
        transformed_scene.export(output_path.as_posix() if isinstance(output_path, Path) else output_path)
