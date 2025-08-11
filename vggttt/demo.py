# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

"""Gradio demo for VGG-T³.

Input:
- a set of images of a scene.
- or a video of a scene (.mp4, .avi, etc.)

Output:
- Camera parameters (intrinsics and extrinsics)
- Point maps
- Depth maps
"""

import logging
import os
import socket
import sys
import time

import cv2
import gradio as gr
import numpy as np
import viser
from scipy.spatial.transform import Rotation

from vggttt.nets.vggt.img import load_and_preprocess_images
from vggttt.nets.vggt.models.vggt import VGGT

_logger = logging.getLogger(__name__)

# Global model instance
_model: VGGT | None = None

# Global viser server instance
_viser_server: viser.ViserServer | None = None
_viser_port: int = 8890
_gradio_port: int = 7860


def _print_port_forwarding_info() -> None:
    """Print SSH port forwarding instructions if running on a remote machine."""
    if not (os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION")):
        return

    hostname = socket.gethostname()
    ssh_cmd = (
        f"ssh -L {_gradio_port}:localhost:{_gradio_port} "
        f"-L {_viser_port}:localhost:{_viser_port} {hostname}"
    )
    browser_url = f"http://localhost:{_gradio_port}"

    # ANSI colors only when stderr is a TTY and NO_COLOR is not set.
    if sys.stderr.isatty() and not os.environ.get("NO_COLOR"):
        bold, cyan, yellow, reset = "\033[1m", "\033[96m", "\033[93m", "\033[0m"
    else:
        bold = cyan = yellow = reset = ""

    sep = "=" * 78
    _logger.info(
        f"\n{cyan}{sep}{reset}\n"
        f"{bold}{cyan}Remote machine detected (SSH session).{reset} "
        f"Forward both ports via SSH:\n\n"
        f"    {bold}{yellow}{ssh_cmd}{reset}\n\n"
        f"Then open {bold}{cyan}{browser_url}{reset} in your browser.\n"
        f"{cyan}{sep}{reset}"
    )


def get_model(model_path: str | None = None) -> VGGT:
    """Load model once and cache it."""
    global _model
    if _model is None:
        _logger.info("Loading VGG-T³ model...")
        _model = VGGT.from_pretrained(model_path)
        _model = _model.cuda().eval()
        _logger.info("Model loaded successfully")
    return _model


def get_viser_server() -> viser.ViserServer:
    """Get or create the viser server instance."""
    global _viser_server
    if _viser_server is None:
        _logger.info(f"Starting viser server on port {_viser_port}...")
        _viser_server = viser.ViserServer(host="0.0.0.0", port=_viser_port)
        _logger.info(f"Viser server started at http://localhost:{_viser_port}")

    return _viser_server


def extract_frames_from_video(video_path: str, fps: float = 1.0) -> list[np.ndarray]:
    """Extract frames from a video file at the given FPS.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to extract.

    Returns:
        List of frames as numpy arrays in RGB format.
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("Could not read video metadata")

    # Compute frame indices to sample at the target FPS.
    frame_step = max(1.0, video_fps / fps)
    frame_indices = np.arange(0, total_frames, frame_step).astype(int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    if not frames:
        raise ValueError("Could not extract any frames from video")

    return frames


def _compute_initial_viewpoint(
    poses: np.ndarray,
    points: np.ndarray,
    confidence: np.ndarray,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an initial viewer pose that frames the reconstructed scene.

    Places the viewer slightly above and behind the camera cluster, looking at
    the scene's robust center, with up aligned to the cameras' shared up axis.

    Args:
        poses: Camera-to-world poses, shape (S, 4, 4), OpenCV convention
            (+X right, +Y down, +Z forward in the camera frame).
        points: Flat point cloud, shape (N, 3).
        confidence: Per-point confidence, shape (N,).
        conf_threshold: Cutoff used to discard noisy points when estimating
            scene center and extent.

    Returns:
        position: (3,) viewer position in world coordinates.
        look_at: (3,) world point the viewer is looking at.
        up: (3,) world up direction.
    """
    cam_positions = poses[:, :3, 3]
    cam_centroid = cam_positions.mean(axis=0)

    # World up: average of cameras' local "up" (-Y in OpenCV) directions.
    up = -poses[:, :3, 1].mean(axis=0)
    up_norm = np.linalg.norm(up)
    up = up / up_norm if up_norm > 1e-6 else np.array([0.0, -1.0, 0.0])

    # Robust scene center & extent from confident points; fall back to cameras.
    mask = confidence > conf_threshold
    scene_points = points[mask] if mask.sum() >= 100 else cam_positions
    scene_center = np.median(scene_points, axis=0)

    pt_extent = float(np.percentile(np.linalg.norm(scene_points - scene_center, axis=1), 90))
    cam_extent = float(np.linalg.norm(cam_positions - scene_center, axis=1).max())
    extent = max(pt_extent, cam_extent, 0.1)

    # Viewing direction (from viewer toward scene). Prefer cameras' average
    # forward axis for typical front-facing captures; fall back to the line
    # from camera centroid to scene center for ambiguous cases (e.g., orbits).
    avg_forward = poses[:, :3, 2].mean(axis=0)
    fwd_norm = np.linalg.norm(avg_forward)
    if fwd_norm > 0.5:
        view_dir = avg_forward / fwd_norm
    else:
        to_scene = scene_center - cam_centroid
        to_scene_norm = np.linalg.norm(to_scene)
        view_dir = to_scene / to_scene_norm if to_scene_norm > 1e-6 else -up

    distance = extent * 1.5
    position = cam_centroid - view_dir * distance + up * distance * 0.3
    return position, scene_center, up


def create_visualization(
    pts3d: np.ndarray,
    conf: np.ndarray,
    images: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    conf_threshold: float | None = None,
    max_points: int = 1000000,
    client_host: str = "localhost",
) -> str:
    """Create a 3D visualization with point cloud and camera frustums using viser.

    Args:
        pts3d: Point cloud of shape (S, H, W, 3).
        conf: Confidence of shape (S, H, W).
        images: Images of shape (S, H, W, 3) in [0, 1].
        poses: Camera poses of shape (S, 4, 4).
        intrinsics: Camera intrinsics of shape (S, 3, 3).
        conf_threshold: Initial confidence threshold. If None, uses 7% quantile.
        max_points: Maximum number of points to display.
        client_host: Hostname of the client to construct the viser URL.

    Returns:
        HTML string with iframe embedding the viser visualization.
    """
    server = get_viser_server()

    # Clear previous scene and GUI elements
    server.scene.reset()
    server.gui.reset()

    # Flatten point cloud and get colors from images
    points = pts3d.reshape(-1, 3)
    confidence = conf.reshape(-1)
    colors = images.reshape(-1, 3)

    # Initial threshold: confidence is in [1, inf)
    min_conf = 1.0
    max_conf = round(np.max(confidence), ndigits=0)
    if conf_threshold is None:
        conf_threshold = min_conf + (max_conf - min_conf) * 0.2

    # Add interactive sliders
    gui_threshold = server.gui.add_slider(
        "Confidence Threshold",
        min=min_conf,
        max=max_conf,
        step=0.1,
        initial_value=conf_threshold,
    )
    gui_point_size = server.gui.add_slider(
        "Point Size",
        min=0.001,
        max=0.02,
        step=0.001,
        initial_value=0.002,
    )

    def update_point_cloud():
        current_threshold = gui_threshold.value
        current_point_size = gui_point_size.value

        # Filter by confidence
        mask = confidence > current_threshold
        p = points[mask]
        c = colors[mask]

        # Subsample if too many points
        if len(p) > max_points:
            # Use fixed seed to avoid flickering when sliding
            rng = np.random.default_rng(42)
            indices = rng.choice(len(p), max_points, replace=False)
            p = p[indices]
            c = c[indices]

        # Convert colors to uint8 for viser
        c_uint8 = (c * 255).astype(np.uint8)

        # Add point cloud to viser scene
        server.scene.add_point_cloud(
            name="/scene/pointcloud",
            points=p.astype(np.float32),
            colors=c_uint8,
            point_size=current_point_size,
            point_shape="rounded",
        )

    # Register callbacks and perform initial update
    gui_threshold.on_update(lambda _: update_point_cloud())
    gui_point_size.on_update(lambda _: update_point_cloud())
    update_point_cloud()

    # Add camera frustums with images
    frustum_scale = 0.05
    for i, (pose, K, image) in enumerate(zip(poses, intrinsics, images)):
        # Extract camera parameters
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        W, H = int(2 * cx), int(2 * cy)

        # Compute FOV from focal length
        fov = 2 * np.arctan(H / (2 * fy))

        # Convert rotation matrix to quaternion (w, x, y, z) for viser
        quat_wxyz = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]

        # Resize image for frustum visualization (e.g., to 128px height)
        vis_h = 128
        vis_w = int(vis_h * (W / H))
        vis_image = cv2.resize(image, (vis_w, vis_h))
        vis_image_uint8 = (vis_image * 255).astype(np.uint8)

        # Add camera frustum
        server.scene.add_camera_frustum(
            name=f"/scene/cameras/camera_{i}",
            fov=fov,
            aspect=W / H,
            scale=frustum_scale,
            wxyz=quat_wxyz,
            position=pose[:3, 3],
            color=(0, 0, 0),
            image=vis_image_uint8,
        )

    init_position, init_look_at, init_up = _compute_initial_viewpoint(
        poses, points, confidence, conf_threshold
    )
    server.scene.set_up_direction(tuple(init_up.tolist()))

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        _logger.info(
            f"Setting initial camera position={init_position}, look_at={init_look_at}, up={init_up}"
        )
        with server.atomic():
            client.camera.position = tuple(init_position.tolist())
            client.camera.look_at = tuple(init_look_at.tolist())

    # Log point count for debugging
    _logger.info(f"Visualization: {len(points)} points, {len(poses)} cameras")

    # Return HTML with iframe embedding viser
    # Add timestamp to force iframe reload and show updated scene
    timestamp = int(time.time() * 1000)

    # Embed viser via the same host the user is browsing from. With SSH port
    # forwarding (or local access) this resolves correctly to the user's
    # machine on the viser port.
    src = f"http://{client_host}:{_viser_port}?t={timestamp}"
    _logger.info(f"Embedding viser iframe at {src}")

    return f"""
    <div style="border-radius: 18px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.08); border: 2.5px solid #e0e4ef;">
        <iframe
            src="{src}"
            style="width: 100%; height: 700px; border: none; border-radius: 0;"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope"
        ></iframe>
    </div>
    """


def run_inference(
    image_files: list | None,
    video_file: str | None,
    use_global_pred: bool,
    video_fps: float,
    memory_efficient: bool,
    progress: gr.Progress = gr.Progress(),
    request: gr.Request = None,
) -> str | None:
    """Run VGGT inference on input images or video.

    Returns:
        HTML string with embedded viser visualization, or None on failure.
    """
    try:
        progress(0.1, desc="Loading inputs...")

        # Load images from either source
        if video_file is not None:
            frames = extract_frames_from_video(video_file, fps=video_fps)
            images = load_and_preprocess_images(frames)
            _logger.info(f"Extracted and preprocessed {len(frames)} frames from video")
        elif image_files is not None and len(image_files) > 0:
            image_paths = [f.name for f in image_files]
            images = load_and_preprocess_images(image_paths)
            _logger.info(f"Loaded and preprocessed {len(image_paths)} images")
        else:
            return None

        progress(0.2, desc="Loading model...")
        model = get_model()

        # Move images to GPU and run inference
        progress(0.3, desc="Running inference...")
        images_gpu = images.cuda()

        results = model.infer(
            images_gpu,
            memory_efficient_inference=memory_efficient,
            use_global_pred=use_global_pred,
        )

        progress(0.7, desc="Creating visualizations...")

        # Convert results to numpy
        pts3d = results["pts3d"].numpy()
        conf = results["conf"].numpy()
        poses = results["pose"].numpy()
        intrinsics = results["intrinsics"].numpy()

        # Original images for coloring point cloud
        images_np = images.permute(0, 2, 3, 1).numpy()

        # Construct client host for viser iframe
        client_host = "localhost"
        if request is not None:
            host_header = request.headers.get("host", "localhost")
            client_host = host_header.split(":")[0]

        # Create visualization with point cloud and cameras
        vis_html = create_visualization(pts3d, conf, images_np, poses, intrinsics, client_host=client_host)

        progress(1.0, desc="Done!")

        return vis_html

    except Exception as e:
        _logger.exception("Inference failed")
        return None


def create_demo():
    """Create the Gradio interface."""
    # Initialize viser server at startup
    get_viser_server()

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <div align="center">
            <h1>VGG-T³: Offline Feed-Forward 3D Reconstruction at Scale</h1>
            </div>
            """,
            sanitize_html=False,
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")

                with gr.Tab("🖼️ Images"):
                    image_input = gr.File(
                        label="Upload Images",
                        file_count="multiple",
                        file_types=["image"],
                        height=300,
                    )

                with gr.Tab("🎬 Video"):
                    video_input = gr.Video(label="Upload Video")
                    video_fps = gr.Slider(
                        minimum=0.5,
                        maximum=30,
                        value=1,
                        step=0.5,
                        label="Extraction FPS",
                        info="Frames per second to extract from the video",
                    )

                gr.Markdown("### ⚙️ Settings")

                use_global_pred = gr.Checkbox(
                    label="Use Global Prediction",
                    value=True,
                    info="Global point map prediction vs. depth + camera pose",
                )

                memory_efficient = gr.Checkbox(
                    label="Memory Efficient Mode",
                    value=False,
                    info="Use for many images or limited GPU memory",
                )

                run_btn = gr.Button("🚀 Run Reconstruction", variant="primary", size="lg")

            with gr.Column(scale=3):
                gr.Markdown("### 📊 Results")

                with gr.Tab("☁️ Point Cloud"):
                    pointcloud_html = gr.HTML(
                        value="""
                        <div style="
                            width: 100%;
                            height: 700px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            background: white;
                            border-radius: 8px;
                            color: #8892b0;
                            font-family: system-ui, -apple-system, sans-serif;
                        ">
                            <p>Run reconstruction to view 3D point cloud</p>
                        </div>
                        """,
                        label="3D Point Cloud",
                    )

        # Connect the button
        run_btn.click(
            fn=run_inference,
            inputs=[image_input, video_input, use_global_pred, video_fps, memory_efficient],
            outputs=[pointcloud_html],
        )

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="VGG-T³ Gradio Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/vgg-ttt",
        help="Model to use for inference",
    )
    args = parser.parse_args()

    get_model(args.model)

    _print_port_forwarding_info()

    demo = create_demo()
    demo.launch(share=False, server_port=_gradio_port)
