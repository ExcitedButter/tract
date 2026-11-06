"""
Video dataset loader for real-world Franka data with hybrid track conditioning.

Hybrid format (matching the LIBERO hybrid model):
- Agentview: 7 mesh vertices only (gripper keypoints)
- Eye-in-hand: 25 grid + 7 mesh = 32 points

Tracks are rendered as colored dot heatmaps with Gaussian blur.
"""

import glob
import os
import random

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Import Gaussian blur kernel from This_and_That_VDM
try:
    from utils.optical_flow_utils import bivariate_Gaussian
except ImportError:
    def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
        """Fallback bivariate Gaussian kernel."""
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sig_x**2))
        return kernel / kernel.sum()


def generate_hybrid_trajectory_vis(tracked_points, config, view_name):
    """Render trajectory heatmap for hybrid conditioning.

    Args:
        tracked_points: (T, N, 2) normalized [0,1] coords. N=7 for agentview, N=32 for wrist.
        config: dict with height, width, video_seq_length, etc.
        view_name: 'agentview' or 'eye_in_hand'

    Returns:
        (T, 3, H, W) float32 in [0, 1]
    """
    H = config.get("height", 320)
    W = config.get("width", 320)
    seq_len = config.get("video_seq_length", 16)
    num_points = tracked_points.shape[1]

    # Scale blur parameters relative to 128px base
    scale_factor = H / 128.0
    if config.get("dilate", True):
        kernel_size = int(49 * scale_factor) if scale_factor > 1 else 49
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = 7 * scale_factor if scale_factor > 1 else 7
        blur_kernel = bivariate_Gaussian(kernel_size, sigma, sigma, 0, grid=None, isotropic=True)
    else:
        blur_kernel = None

    # Dot sizes
    if num_points == 7:
        dot_range = config.get("dot_range_agentview", 12)
    else:
        dot_range = config.get("dot_range_wrist", 8)

    use_view_colors = config.get("use_view_specific_colors", True)
    result = np.zeros((seq_len, 3, H, W), dtype=np.float32)

    for t in range(min(seq_len, len(tracked_points))):
        base_img = np.ones((H, W, 3), dtype=np.float32) * 255

        for p_idx in range(num_points):
            x_norm, y_norm = tracked_points[t, p_idx]
            if x_norm <= 0 and y_norm <= 0:
                continue  # skip zero/hidden points

            x = int(np.clip(x_norm * W, 0, W - 1))
            y = int(np.clip(y_norm * H, 0, H - 1))

            # Color selection (BGR format for cv2)
            if num_points == 7:
                # Agentview: all mesh, blue
                if use_view_colors and view_name == "agentview":
                    color = [255, 0, 0]  # BGR blue
                else:
                    color = [0, 0, 255]  # BGR red
            else:
                # Wrist: first 25 are grid (light), last 7 are mesh (bold)
                if p_idx < 25:
                    color = [150, 150, 255]  # BGR light red
                else:
                    color = [0, 0, 255]  # BGR bold red

            # Draw square dot
            for i in range(-dot_range, dot_range + 1):
                for j in range(-dot_range, dot_range + 1):
                    px, py = x + j, y + i
                    if 0 <= px < W and 0 <= py < H:
                        base_img[py, px] = color

        if blur_kernel is not None:
            base_img = cv2.filter2D(base_img, -1, blur_kernel)

        # BGR → RGB, normalize to [0, 1], HWC → CHW
        rgb = cv2.cvtColor(base_img.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        result[t] = rgb.transpose(2, 0, 1)

    return result


TASK_DESCRIPTIONS = {
    "task_1": "put the pink noodle in the bowl",
    "task_2": "put the bread in the bowl",
    "task_3": "pour the pasta into the pan",
    "task_4": "close the right cabinet door",
}


class Video_Realworld_Hybrid_Dataset(Dataset):
    """Real-world Franka dataset with hybrid track conditioning."""

    def __init__(self, config, device, tokenizer=None, normalize=True):
        self.config = config
        self.device = device
        self.tokenizer = tokenizer

        self.video_seq_length = config.get("video_seq_length", 16)
        self.height = config.get("height", 320)
        self.width = config.get("width", 320)
        self.mode = config.get("split", "train")
        self.target_view = config.get("training_view", "both")

        dataset_path = config["dataset_path"]
        tasks = sorted([d for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith("task_")])

        self.samples = []
        for task in tasks:
            split_dir = os.path.join(dataset_path, task, self.mode)
            if not os.path.exists(split_dir):
                continue
            for f in sorted(glob.glob(os.path.join(split_dir, "*.hdf5"))):
                self.samples.append((f, task))

        # Tokenize per-task prompts (track model uses text conditioning)
        self.task_prompt_ids = {}
        if tokenizer is not None:
            for task_key, desc in TASK_DESCRIPTIONS.items():
                tokens = tokenizer(desc, max_length=77, padding="max_length",
                                   truncation=True, return_tensors="pt")
                self.task_prompt_ids[task_key] = tokens.input_ids.squeeze(0)
            # Fallback for unknown tasks
            fallback = tokenizer("a robot arm performing a manipulation task",
                                 max_length=77, padding="max_length",
                                 truncation=True, return_tensors="pt")
            self.task_prompt_ids["_fallback"] = fallback.input_ids.squeeze(0)

        print(f"[{self.mode.upper()}] Loaded {len(self.samples)} demos, "
              f"view={self.target_view}, hybrid conditioning")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5_path, task_name = self.samples[idx]

        try:
            with h5py.File(hdf5_path, "r") as f:
                # View selection
                if self.target_view == "both":
                    view_idx = idx % 2
                elif self.target_view == "eye_in_hand":
                    view_idx = 1
                else:
                    view_idx = 0

                view_name = "agentview" if view_idx == 0 else "eye_in_hand"

                # Load video
                video = np.array(f[f"root/{view_name}/video"][0])  # (T, 3, H, W)
                total_frames = video.shape[0]
                seq_len = self.video_seq_length

                # Sample window
                if total_frames < seq_len:
                    start = 0
                    pad_len = seq_len - total_frames
                else:
                    if self.mode == "train":
                        start = random.randint(0, total_frames - seq_len)
                    else:
                        start = (total_frames - seq_len) // 2
                    pad_len = 0

                end = min(start + seq_len, total_frames)
                frames = video[start:end]

                if pad_len > 0:
                    frames = np.concatenate([frames, np.repeat(frames[-1:], pad_len, axis=0)], axis=0)

                # Normalize video to [-1, 1]
                video_tensor = torch.from_numpy(frames.astype(np.float32) / 127.5 - 1.0)

                # Load tracks based on view
                mesh_key = "agentview_vertices_2d" if view_idx == 0 else "eyeinhand_vertices_2d"
                mesh_tracks = np.array(f[f"root/mesh_data/vertex_tracking/{mesh_key}"])  # (T, 7, 2)
                mesh_tracks = mesh_tracks[start:end]

                if pad_len > 0:
                    mesh_tracks = np.concatenate(
                        [mesh_tracks, np.repeat(mesh_tracks[-1:], pad_len, axis=0)], axis=0)

                if view_idx == 0:
                    # Agentview: 7 mesh points only
                    conditioning_points = mesh_tracks  # (seq_len, 7, 2)
                else:
                    # Eye-in-hand: 25 grid + 7 mesh = 32 points
                    # Prefer per-window tracks (fresh 5x5 grid per window)
                    perwindow_key = "root/eye_in_hand/grid_tracks_perwindow"
                    old_key = "root/eye_in_hand/grid_tracks"
                    if perwindow_key in f:
                        # Shape: (num_windows, 16, 25, 2) — index by start frame
                        pw_data = f[perwindow_key]
                        if start < pw_data.shape[0]:
                            grid_tracks = np.array(pw_data[start])  # (16, 25, 2)
                        else:
                            # Past last window, use last available
                            grid_tracks = np.array(pw_data[-1])
                        # Pad if needed
                        if pad_len > 0:
                            grid_tracks = np.concatenate(
                                [grid_tracks, np.repeat(grid_tracks[-1:], pad_len, axis=0)], axis=0)
                    elif old_key in f:
                        # Fallback to old format
                        grid_tracks = np.array(f[old_key][0])  # (T, 25, 2)
                        grid_tracks = grid_tracks[start:end]
                        if pad_len > 0:
                            grid_tracks = np.concatenate(
                                [grid_tracks, np.repeat(grid_tracks[-1:], pad_len, axis=0)], axis=0)
                    else:
                        grid_tracks = np.zeros((seq_len, 25, 2), dtype=np.float32)

                    conditioning_points = np.concatenate([grid_tracks, mesh_tracks], axis=1)  # (seq_len, 32, 2)

                # Render trajectory heatmap
                controlnet_cond = generate_hybrid_trajectory_vis(
                    conditioning_points, self.config, view_name)
                controlnet_cond = torch.from_numpy(controlnet_cond)

                # Coordinate values (mean of tracked points per frame)
                coord_values = torch.from_numpy(
                    conditioning_points.mean(axis=1).astype(np.float32))  # (seq_len, 2)

                # Motion bucket
                motion_bucket = torch.tensor(
                    self.config.get("motion_bucket_id", 180), dtype=torch.float32)

                # Prompt tokens (per-task description)
                prompt = self.task_prompt_ids.get(
                    task_name, self.task_prompt_ids.get("_fallback", torch.zeros(77, dtype=torch.long)))

                return {
                    "video_frames": video_tensor,
                    "controlnet_condition": controlnet_cond,
                    "reflected_motion_bucket_id": motion_bucket,
                    "controlnet_image_index": torch.arange(seq_len, dtype=torch.int32),
                    "prompt": prompt,
                    "coordinate_values": coord_values,
                }

        except Exception as e:
            print(f"Error loading {hdf5_path}: {e}")
            import traceback
            traceback.print_exc()
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        n = self.video_seq_length
        return {
            "video_frames": torch.zeros((n, 3, self.height, self.width), dtype=torch.float32),
            "controlnet_condition": torch.zeros((n, 3, self.height, self.width), dtype=torch.float32),
            "reflected_motion_bucket_id": torch.tensor(180.0, dtype=torch.float32),
            "controlnet_image_index": torch.arange(n, dtype=torch.int32),
            "prompt": torch.zeros(77, dtype=torch.long),
            "coordinate_values": torch.zeros((n, 2), dtype=torch.float32),
        }
