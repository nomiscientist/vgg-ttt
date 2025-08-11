# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import math

import cv2
import numpy as np
import PIL
import torch
import torchvision.transforms.functional as TF
from PIL import Image

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


def get_target_shape(aspect_ratio: float, img_size: int, patch_size: int = 1) -> np.ndarray:
    """Calculate the target shape based on the given aspect ratio.

    Args:
        aspect_ratio: Target aspect ratio (width/height)
        img_size: Original image size for the longer side
        patch_size: Patch size

    Returns:
        Target image shape [height, width]
    """
    if aspect_ratio >= 1.0:
        # Landscape or square: width >= height, so width is the longer side
        width = img_size
        height = img_size / aspect_ratio
    else:
        # Portrait: width < height, so height is the longer side
        height = img_size
        width = img_size * aspect_ratio

    height = round(height / patch_size) * patch_size
    width = round(width / patch_size) * patch_size
    return np.array([height, width])


def process_one_image(
    image: np.ndarray,
    target_image_shape: np.ndarray,
    depth_map: np.ndarray | None = None,
    extri_opencv: np.ndarray | None = None,
    intri_opencv: np.ndarray | None = None,
    track: np.ndarray | None = None,
    filepath: str | None = None,
    aug_scale: float | None = None,
    test_mode: bool = False,
    rotation_aug_prob: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """
    Process a single image and its associated data.

    This method handles image transformations, depth processing, and coordinate conversions.

    Args:
        image: Input image array
        depth_map: Depth map array
        extri_opencv: Extrinsic camera matrix (OpenCV convention)
        intri_opencv: Intrinsic camera matrix (OpenCV convention)
        original_size: Original image size [height, width]
        target_image_shape: Target image shape after processing
        track: Optional tracking information. Defaults to None.
        filepath: Optional file path for debugging. Defaults to None.
        safe_bound: Safety margin for cropping operations. Defaults to 4.
        test_mode: Crop along the long side of the image and do not rotate to match `target_image_shape` orientation.
        aug_scale: Scale augmentation to apply to the image. Defaults to no scale augmentation.

    Returns:
        tuple: (
            image (numpy.ndarray): Processed image,
            depth_map (numpy.ndarray): Processed depth map,
            extri_opencv (numpy.ndarray): Updated extrinsic matrix,
            intri_opencv (numpy.ndarray): Updated intrinsic matrix,
            world_coords_points (numpy.ndarray): 3D points in world coordinates,
            cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
            point_mask (numpy.ndarray): Boolean mask of valid points,
            track (numpy.ndarray, optional): Updated tracking information
        )
    """
    rng = np.random.default_rng() if rng is None else rng
    # Make copies to avoid in-place operations affecting original data
    image = np.copy(image)
    original_size = np.array(image.shape[:2])

    depth_map = np.copy(depth_map) if depth_map is not None else None
    extri_opencv = np.copy(extri_opencv) if extri_opencv is not None else None
    intri_opencv = np.copy(intri_opencv) if intri_opencv is not None else None
    track = np.copy(track) if track is not None else None

    # Apply random scale augmentation during training if enabled
    if aug_scale:
        # Avoid random padding by capping at 1.0
        random_scale = min(aug_scale, 1.0)
        aug_size = original_size * random_scale
        aug_size = aug_size.astype(np.int32)
    else:
        aug_size = original_size

    # Move principal point to the image center and crop if necessary
    image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
        image,
        depth_map,
        intri_opencv,
        aug_size,
        track=track,
        filepath=filepath,
    )

    original_size = np.array(image.shape[:2])  # update original_size
    target_shape = np.array(target_image_shape, dtype=int)

    def _orientation(shape: np.ndarray) -> int:
        height, width = int(shape[0]), int(shape[1])
        if height == width:
            return 0
        return 1 if height > width else -1

    desired_orientation = _orientation(target_shape)

    # Handle landscape vs. portrait orientation
    rotate_to_match_target = False
    orig_aspect = original_size[1] / original_size[0]
    original_is_approx_square = 0.5 < orig_aspect < 2.0

    # If we have close to square image, we can crop simply using the target shape.
    if original_is_approx_square and not test_mode:
        if rng.random() < rotation_aug_prob:
            target_shape = target_shape[::-1]
            rotate_to_match_target = True
    else:
        # If we have a non-square image, we want to crop along the long side of the image so we do not loose too
        # much context.
        if _orientation(original_size) != 0 and _orientation(original_size) != desired_orientation:
            target_shape = target_shape[::-1]
            rotate_to_match_target = not test_mode

    # Resize images and update intrinsics
    image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
        image,
        depth_map,
        intri_opencv,
        target_shape,
        original_size,
        track=track,
        pixel_center=False,
    )

    # Ensure final crop to target shape
    image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
        image,
        depth_map,
        intri_opencv,
        target_shape,
        track=track,
        filepath=filepath,
        center_crop=True,
        strict=True,
    )

    # Apply 90-degree rotation if needed
    if rotate_to_match_target and not test_mode:
        clockwise = rng.random() > 0.5
        image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            clockwise=clockwise,
            track=track,
        )

    return image, depth_map, extri_opencv, intri_opencv, track


#####################################################################################################################
def crop_image_depth_and_intrinsic_by_pp(
    image, depth_map, intrinsic, target_shape, track=None, filepath=None, strict=False, center_crop=False
):
    """Crops the given image and depth map around the camera's principal point, as defined by `intrinsic`.

    Specifically:
      - Ensures that the crop is centered on (cx, cy).
      - Optionally pads the image (and depth map) if `strict=True` and the result is smaller than `target_shape`.
      - Shifts the camera intrinsic matrix (and `track` if provided) accordingly.

    Args:
        image (np.ndarray):
            Input image array of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array of shape (H, W), or None if not available.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3). The principal point is assumed to be at (intrinsic[1,2], intrinsic[0,2]).
        target_shape (tuple[int, int]):
            Desired output shape.
        track (np.ndarray or None):
            Optional array of shape (N, 2). Interpreted as (x, y) pixel coordinates. Will be shifted after cropping.
        filepath (str or None):
            An optional file path for debug logging (only used if strict mode triggers warnings).
        strict (bool):
            If True, will zero-pad to ensure the exact target_shape even if the cropped region is smaller.

    Raises:
        AssertionError:
            If the input image is smaller than `target_shape`.
        ValueError:
            If the cropped image is larger than `target_shape` (in strict mode), which should not normally happen.

    Returns:
        tuple:
            (cropped_image, cropped_depth_map, updated_intrinsic, updated_track)

            - cropped_image (np.ndarray): Cropped (and optionally padded) image.
            - cropped_depth_map (np.ndarray or None): Cropped (and optionally padded) depth map.
            - updated_intrinsic (np.ndarray): Intrinsic matrix adjusted for the crop.
            - updated_track (np.ndarray or None): Track array adjusted for the crop, or None if track was not provided.
    """
    original_size = np.array(image.shape)
    intrinsic = np.copy(intrinsic) if intrinsic is not None else None

    if original_size[0] < target_shape[0]:
        error_message = (
            f"Width check failed: original width {original_size[0]} is less than target width {target_shape[0]}."
        )
        raise AssertionError(error_message)

    if original_size[1] < target_shape[1]:
        error_message = (
            f"Height check failed: original height {original_size[1]} is less than target height {target_shape[1]}."
        )
        raise AssertionError(error_message)

    if intrinsic is not None and not center_crop:
        # Identify principal point (cx, cy) from intrinsic
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
    else:
        # Center crop
        cx = original_size[1] / 2
        cy = original_size[0] / 2

    # Compute how far we can crop in each direction
    if strict:
        half_x = min((target_shape[1] / 2), cx)
        half_y = min((target_shape[0] / 2), cy)
    else:
        half_x = min((target_shape[1] / 2), cx, original_size[1] - cx)
        half_y = min((target_shape[0] / 2), cy, original_size[0] - cy)

    # Compute starting indices
    start_x = math.floor(cx - half_x)
    start_y = math.floor(cy - half_y)

    assert start_x >= 0
    assert start_y >= 0

    # Compute ending indices
    if strict:
        end_x = start_x + target_shape[1]
        end_y = start_y + target_shape[0]
    else:
        end_x = start_x + math.floor(2 * half_x)
        end_y = start_y + math.floor(2 * half_y)

    # Perform the crop
    image = image[start_y:end_y, start_x:end_x, :]
    if depth_map is not None:
        depth_map = depth_map[start_y:end_y, start_x:end_x]

    # Shift the principal point in the intrinsic
    if intrinsic is not None:
        intrinsic[0, 2] = intrinsic[0, 2] - start_x
        intrinsic[1, 2] = intrinsic[1, 2] - start_y

    # Adjust track if provided
    if track is not None:
        track[:, 0] = track[:, 0] - start_x
        track[:, 1] = track[:, 1] - start_y

    # If strict, zero-pad if the new shape is smaller than target_shape
    if strict and (image.shape[:2] != target_shape).any():
        print(f"{filepath} does not meet the target shape")
        current_h, current_w = image.shape[:2]
        target_h, target_w = target_shape[0], target_shape[1]
        pad_h = target_h - current_h
        pad_w = target_w - current_w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"The cropped image is bigger than the target shape: "
                f"cropped=({current_h},{current_w}), "
                f"target=({target_h},{target_w})."
            )
        image = np.pad(
            image,
            pad_width=((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        if depth_map is not None:
            depth_map = np.pad(
                depth_map,
                pad_width=((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

    return image, depth_map, intrinsic, track


def resize_image_depth_and_intrinsic(
    image: np.ndarray,
    depth_map: np.ndarray | None,
    intrinsic: np.ndarray | None,
    target_shape: np.ndarray,
    original_size: np.ndarray,
    track: np.ndarray | None,
    pixel_center: bool = True,
):
    """
    Resizes the given image and depth map (if provided) to slightly larger than `target_shape`,
    updating the intrinsic matrix (and track array if present). Optionally uses random rescaling
    to create some additional margin (based on `rescale_aug`).

    Steps:
      1. Compute a scaling factor so that the resized result is at least `target_shape + safe_bound`.
      2. Apply an optional triangular random factor if `rescale_aug=True`.
      3. Resize the image with LANCZOS if downscaling, BICUBIC if upscaling.
      4. Resize the depth map with nearest-neighbor.
      5. Update the camera intrinsic and track coordinates (if any).

    Args:
        image (np.ndarray):
            Input image array (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array (H, W), or None if unavailable.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3).
        target_shape (np.ndarray or tuple[int, int]):
            Desired final shape (height, width).
        original_size (np.ndarray or tuple[int, int]):
            Original size of the image in (height, width).
        track (np.ndarray or None):
            Optional (N, 2) array of pixel coordinates. Will be scaled.
        pixel_center (bool):
            If True, accounts for 0.5 pixel center shift during resizing.
        rescale_aug (bool):
            If True, randomly increase the `safe_bound` within a certain range to simulate augmentation.

    Returns:
        tuple:
            (resized_image, resized_depth_map, updated_intrinsic, updated_track)

            - resized_image (np.ndarray): The resized image.
            - resized_depth_map (np.ndarray or None): The resized depth map.
            - updated_intrinsic (np.ndarray): Camera intrinsic updated for new resolution.
            - updated_track (np.ndarray or None): Track array updated or None if not provided.

    Raises:
        AssertionError:
            If the shapes of the resized image and depth map do not match.
    """
    resize_scales = target_shape / original_size
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic) if intrinsic is not None else None

    # Convert image to PIL for resizing
    image = Image.fromarray(image)
    input_resolution = np.array(image.size)
    output_resolution = np.ceil(input_resolution * max_resize_scale).astype(int)
    image = image.resize(tuple(output_resolution), resample=lanczos if max_resize_scale < 1 else bicubic)
    image = np.array(image)

    if depth_map is not None:
        depth_map = cv2.resize(depth_map, output_resolution, interpolation=cv2.INTER_NEAREST)

    actual_size = np.array(image.shape[:2])
    actual_resize_scale = np.max(actual_size / original_size)

    if intrinsic is not None:
        if pixel_center:
            intrinsic[0, 2] = intrinsic[0, 2] + 0.5
            intrinsic[1, 2] = intrinsic[1, 2] + 0.5

        intrinsic[:2, :] = intrinsic[:2, :] * actual_resize_scale

        if pixel_center:
            intrinsic[0, 2] = intrinsic[0, 2] - 0.5
            intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    if track is not None:
        track = track * actual_resize_scale

    if depth_map is not None:
        assert image.shape[:2] == depth_map.shape[:2]
    return image, depth_map, intrinsic, track


def rotate_90_degrees(image, depth_map, extri_opencv, intri_opencv, clockwise=True, track=None):
    """
    Rotates the input image, depth map, and camera parameters by 90 degrees.

    Applies one of two 90-degree rotations:
    - Clockwise
    - Counterclockwise (if clockwise=False)

    The extrinsic and intrinsic matrices are adjusted accordingly to maintain
    correct camera geometry. Track coordinates are also updated if provided.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        extri_opencv (np.ndarray):
            Extrinsic matrix (3x4) in OpenCV convention.
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        clockwise (bool):
            If True, rotates the image 90 degrees clockwise; else 90 degrees counterclockwise.
        track (np.ndarray or None):
            Optional (N, 2) track array. Will be rotated accordingly.

    Returns:
        tuple:
            (
                rotated_image,
                rotated_depth_map,
                new_extri_opencv,
                new_intri_opencv,
                new_track
            )

            Where each is the updated version after the rotation.
    """
    image_height, image_width = image.shape[:2]

    # Rotate the image and depth map
    rotated_image, rotated_depth_map = rotate_image_and_depth_rot90(image, depth_map, clockwise)
    # Adjust the intrinsic matrix
    if intri_opencv is not None:
        new_intri_opencv = adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise)
    else:
        new_intri_opencv = None

    if track is not None:
        new_track = adjust_track_rot90(track, image_width, image_height, clockwise)
    else:
        new_track = None

    # Adjust the extrinsic matrix
    if extri_opencv is not None:
        new_extri_opencv = adjust_extrinsic_matrix_rot90(extri_opencv, clockwise)
    else:
        new_extri_opencv = None

    return (
        rotated_image,
        rotated_depth_map,
        new_extri_opencv,
        new_intri_opencv,
        new_track,
    )


def rotate_image_and_depth_rot90(image, depth_map, clockwise):
    """
    Rotates the given image and depth map by 90 degrees (clockwise or counterclockwise),
    using a transpose+flip pattern.

    Args:
        image (np.ndarray):
            Input image of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map of shape (H, W), or None if not available.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        tuple:
            (rotated_image, rotated_depth_map)
    """
    rotated_depth_map = None
    if clockwise:
        rotated_image = np.transpose(image, (1, 0, 2))  # Transpose height and width
        rotated_image = np.flip(rotated_image, axis=1)  # Flip horizontally
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=1)
    else:
        rotated_image = np.transpose(image, (1, 0, 2))  # Transpose height and width
        rotated_image = np.flip(rotated_image, axis=0)  # Flip vertically
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=0)
    return np.copy(rotated_image), np.copy(rotated_depth_map)


def adjust_extrinsic_matrix_rot90(extri_opencv, clockwise):
    """
    Adjusts the extrinsic matrix (3x4) for a 90-degree rotation of the image.

    The rotation is in the image plane. This modifies the camera orientation
    accordingly. The function applies either a clockwise or counterclockwise
    90-degree rotation.

    Args:
        extri_opencv (np.ndarray):
            Extrinsic matrix (3x4) in OpenCV convention.
        clockwise (bool):
            If True, rotate extrinsic for a 90-degree clockwise image rotation;
            otherwise, counterclockwise.

    Returns:
        np.ndarray:
            A new 3x4 extrinsic matrix after the rotation.
    """
    R = extri_opencv[:3, :3]
    t = extri_opencv[:3, 3]

    if clockwise:
        R_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    else:
        R_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    new_R = np.dot(R_rotation, R)
    new_t = np.dot(R_rotation, t)
    new_extri_opencv = np.hstack((new_R, new_t.reshape(-1, 1)))
    return new_extri_opencv


def adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise):
    """
    Adjusts the intrinsic matrix (3x3) for a 90-degree rotation of the image in the image plane.

    Args:
        intri_opencv (np.ndarray):
            Intrinsic matrix (3x3).
        image_width (int):
            Original width of the image.
        image_height (int):
            Original height of the image.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        np.ndarray:
            A new 3x3 intrinsic matrix after the rotation.
    """
    fx, fy, cx, cy = (
        intri_opencv[0, 0],
        intri_opencv[1, 1],
        intri_opencv[0, 2],
        intri_opencv[1, 2],
    )

    new_intri_opencv = np.eye(3)
    if clockwise:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx

    return new_intri_opencv


def adjust_track_rot90(track, image_width, image_height, clockwise):
    """
    Adjusts a track (N, 2) for a 90-degree rotation of the image in the image plane.

    Args:
        track (np.ndarray):
            (N, 2) array of pixel coordinates, each row is (x, y).
        image_width (int):
            Original image width.
        image_height (int):
            Original image height.
        clockwise (bool):
            Whether the rotation is 90 degrees clockwise or counterclockwise.

    Returns:
        np.ndarray:
            A new track of shape (N, 2) after rotation.
    """
    if clockwise:
        # (x, y) -> (y, image_width - 1 - x)
        new_track = np.stack((track[:, 1], image_width - 1 - track[:, 0]), axis=-1)
    else:
        # (x, y) -> (image_height - 1 - y, x)
        new_track = np.stack((image_height - 1 - track[:, 1], track[:, 0]), axis=-1)

    return new_track


def load_and_preprocess_images(
    image_path_list, mode="crop", target_size: int = 518, patch_size: int = 14, pad_value: float = 1.0
):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        if isinstance(image_path, str):
            img = Image.open(image_path)
        elif isinstance(image_path, np.ndarray):
            img = Image.fromarray(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background_value = int(255 * pad_value)
            background = Image.new("RGBA", img.size, (background_value, background_value, background_value, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / patch_size) * patch_size  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / patch_size) * patch_size  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by patch_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = TF.to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
