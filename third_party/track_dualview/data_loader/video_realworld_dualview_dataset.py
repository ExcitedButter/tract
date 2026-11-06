"""Dual-view dataset loader for track-conditioned SVD+ControlNet WM.

Extension of `video_realworld_hybrid_dataset.py`. Always loads BOTH views
(agentview + eye_in_hand) at `per_view_size × per_view_size` and stacks them
vertically to produce:
    video             (T, 3, 2S, S)
    controlnet_cond   (T, 3, 2S, S)

Agentview is the top S rows, eye_in_hand the bottom S rows. The hybrid
heatmap renderer is reused per view — this file adds no new rendering logic.

If the hdf5 has precomputed VAE latents (`root/video_latents_stacked` and
`root/controlnet_latents_perwindow`), the fast path returns those latents
directly and skips heatmap rendering / VAE encode at train time. The first
frame is still read as raw pixels (cheap — single frame) because training
needs it for CLIP input and the conditional-frame noise-aug which must
happen in pixel space.
"""

import glob
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader.video_realworld_hybrid_dataset import (
    TASK_DESCRIPTIONS,
    generate_hybrid_trajectory_vis,
)


VIDEO_LAT_KEY = "root/video_latents_stacked"
CTRL_LAT_KEY = "root/controlnet_latents_perwindow"


class Video_Realworld_Dualview_Dataset(Dataset):
    """Real-world Franka dataset with dual-view vertical stacking."""

    def __init__(self, config, device, tokenizer=None, normalize=True, force_legacy=False):
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.force_legacy = force_legacy

        self.video_seq_length = config.get("video_seq_length", 16)
        self.per_view_size = config.get("per_view_size", 224)
        self.mode = config.get("split", "train")

        cfg_h = config.get("height", 2 * self.per_view_size)
        cfg_w = config.get("width", self.per_view_size)
        assert cfg_h == 2 * self.per_view_size and cfg_w == self.per_view_size, (
            f"dualview expects height=2*per_view_size and width=per_view_size, "
            f"got height={cfg_h} width={cfg_w} per_view_size={self.per_view_size}")

        # Per-view config for generate_hybrid_trajectory_vis (H=W=per_view_size).
        self._view_config = dict(config)
        self._view_config["height"] = self.per_view_size
        self._view_config["width"] = self.per_view_size

        dataset_path = config["dataset_path"]
        tasks = sorted([d for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d))
                        and d.startswith("task_")])

        self.samples = []
        for task in tasks:
            split_dir = os.path.join(dataset_path, task, self.mode)
            if not os.path.exists(split_dir):
                continue
            for f in sorted(glob.glob(os.path.join(split_dir, "*.hdf5"))):
                self.samples.append((f, task))

        self.task_prompt_ids = {}
        if tokenizer is not None:
            for task_key, desc in TASK_DESCRIPTIONS.items():
                tokens = tokenizer(desc, max_length=77, padding="max_length",
                                   truncation=True, return_tensors="pt")
                self.task_prompt_ids[task_key] = tokens.input_ids.squeeze(0)
            fallback = tokenizer("a robot arm performing a manipulation task",
                                 max_length=77, padding="max_length",
                                 truncation=True, return_tensors="pt")
            self.task_prompt_ids["_fallback"] = fallback.input_ids.squeeze(0)

        # Quick probe on first demo to decide fast/slow path advertised to log.
        precomputed = False
        if self.samples:
            try:
                with h5py.File(self.samples[0][0], "r") as f:
                    precomputed = VIDEO_LAT_KEY in f and CTRL_LAT_KEY in f
            except Exception:
                precomputed = False

        effective_fast = precomputed and not self.force_legacy
        print(f"[{self.mode.upper()}] Loaded {len(self.samples)} demos, "
              f"dualview per_view={self.per_view_size}, "
              f"stacked={2 * self.per_view_size}x{self.per_view_size}, "
              f"precomputed_latents={precomputed} fast_path={effective_fast} "
              f"force_legacy={self.force_legacy}")

    def __len__(self):
        return len(self.samples)

    def _load_view_window(self, f, view_name: str, start: int, seq_len: int, pad_len: int):
        video = np.array(f[f"root/{view_name}/video"][0])  # (T, 3, S, S)
        end = min(start + seq_len, video.shape[0])
        frames = video[start:end]
        if pad_len > 0:
            frames = np.concatenate(
                [frames, np.repeat(frames[-1:], pad_len, axis=0)], axis=0)
        return frames  # (seq_len, 3, S, S)

    def _load_first_frame_stacked(self, f, start: int) -> np.ndarray:
        """Read ONE frame from each view at index `start`, stack → (3, 2S, S) uint8."""
        agent = np.array(f["root/agentview/video"][0, start])     # (3, S, S)
        wrist = np.array(f["root/eye_in_hand/video"][0, start])   # (3, S, S)
        return np.concatenate([agent, wrist], axis=1)             # (3, 2S, S)

    def __getitem__(self, idx):
        hdf5_path, task_name = self.samples[idx]

        try:
            with h5py.File(hdf5_path, "r") as f:
                S = self.per_view_size
                seq_len = self.video_seq_length

                total_frames = f["root/agentview/video"].shape[1]
                if total_frames < seq_len:
                    start = 0
                    pad_len = seq_len - total_frames
                else:
                    if self.mode == "train":
                        start = random.randint(0, total_frames - seq_len)
                    else:
                        start = (total_frames - seq_len) // 2
                    pad_len = 0

                motion_bucket = torch.tensor(
                    self.config.get("motion_bucket_id", 180), dtype=torch.float32)
                prompt = self.task_prompt_ids.get(
                    task_name,
                    self.task_prompt_ids.get("_fallback", torch.zeros(77, dtype=torch.long)))

                # ---------- Precomputed fast path ----------
                if (not self.force_legacy) and VIDEO_LAT_KEY in f and CTRL_LAT_KEY in f:
                    vid_lat_full = f[VIDEO_LAT_KEY]  # (T, 4, 2S/8, S/8) fp16
                    end = min(start + seq_len, vid_lat_full.shape[0])
                    vid_lat = np.asarray(vid_lat_full[start:end])
                    if pad_len > 0:
                        vid_lat = np.concatenate(
                            [vid_lat, np.repeat(vid_lat[-1:], pad_len, axis=0)], axis=0)

                    ctrl_lat_full = f[CTRL_LAT_KEY]  # (num_windows, seq_len, 4, 2S/8, S/8) fp16
                    idx_ctrl = min(start, ctrl_lat_full.shape[0] - 1)
                    ctrl_lat = np.asarray(ctrl_lat_full[idx_ctrl])  # (seq_len, 4, 2S/8, S/8)

                    # First frame pixels (for CLIP + cond-frame noise aug) — 1 frame read.
                    first_frame_uint8 = self._load_first_frame_stacked(f, start)            # (3, 2S, S) uint8
                    first_frame = first_frame_uint8.astype(np.float32) / 127.5 - 1.0        # [-1, 1]

                    # Coord summary for telemetry — zeros are fine (not used in loss).
                    coord_values = torch.zeros((seq_len, 2), dtype=torch.float32)

                    return {
                        "video_latents": torch.from_numpy(vid_lat),                         # fp16 scaled
                        "controlnet_latents": torch.from_numpy(ctrl_lat),                   # fp16 unscaled
                        "first_frame": torch.from_numpy(first_frame),                       # (3, 2S, S) [-1,1]
                        "reflected_motion_bucket_id": motion_bucket,
                        "controlnet_image_index": torch.arange(seq_len, dtype=torch.int32),
                        "prompt": prompt,
                        "coordinate_values": coord_values,
                    }

                # ---------- Legacy path: render on the fly ----------
                agent_frames = self._load_view_window(f, "agentview", start, seq_len, pad_len)
                wrist_frames = self._load_view_window(f, "eye_in_hand", start, seq_len, pad_len)
                video = np.concatenate([agent_frames, wrist_frames], axis=2)  # (T, 3, 2S, S)
                video_tensor = torch.from_numpy(video.astype(np.float32) / 127.5 - 1.0)

                agent_mesh = np.array(
                    f["root/mesh_data/vertex_tracking/agentview_vertices_2d"]
                )[start:start + seq_len]
                wrist_mesh = np.array(
                    f["root/mesh_data/vertex_tracking/eyeinhand_vertices_2d"]
                )[start:start + seq_len]
                if pad_len > 0:
                    agent_mesh = np.concatenate(
                        [agent_mesh, np.repeat(agent_mesh[-1:], pad_len, axis=0)], axis=0)
                    wrist_mesh = np.concatenate(
                        [wrist_mesh, np.repeat(wrist_mesh[-1:], pad_len, axis=0)], axis=0)

                perwindow_key = "root/eye_in_hand/grid_tracks_perwindow"
                old_key = "root/eye_in_hand/grid_tracks"
                if perwindow_key in f:
                    pw_data = f[perwindow_key]
                    if start < pw_data.shape[0]:
                        grid_tracks = np.array(pw_data[start])
                    else:
                        grid_tracks = np.array(pw_data[-1])
                    if pad_len > 0:
                        grid_tracks = np.concatenate(
                            [grid_tracks, np.repeat(grid_tracks[-1:], pad_len, axis=0)], axis=0)
                elif old_key in f:
                    grid_tracks = np.array(f[old_key][0])[start:start + seq_len]
                    if pad_len > 0:
                        grid_tracks = np.concatenate(
                            [grid_tracks, np.repeat(grid_tracks[-1:], pad_len, axis=0)], axis=0)
                else:
                    grid_tracks = np.zeros((seq_len, 25, 2), dtype=np.float32)

                wrist_points = np.concatenate([grid_tracks, wrist_mesh], axis=1)  # (T, 32, 2)

                agent_hm = generate_hybrid_trajectory_vis(
                    agent_mesh, self._view_config, "agentview")
                wrist_hm = generate_hybrid_trajectory_vis(
                    wrist_points, self._view_config, "eye_in_hand")
                controlnet_cond = np.concatenate([agent_hm, wrist_hm], axis=2)  # (T, 3, 2S, S)
                controlnet_cond = torch.from_numpy(controlnet_cond)

                coord_points = np.concatenate([agent_mesh, wrist_mesh], axis=1)
                coord_values = torch.from_numpy(
                    coord_points.mean(axis=1).astype(np.float32))

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
        H = 2 * self.per_view_size
        W = self.per_view_size
        return {
            "video_frames": torch.zeros((n, 3, H, W), dtype=torch.float32),
            "controlnet_condition": torch.zeros((n, 3, H, W), dtype=torch.float32),
            "reflected_motion_bucket_id": torch.tensor(180.0, dtype=torch.float32),
            "controlnet_image_index": torch.arange(n, dtype=torch.int32),
            "prompt": torch.zeros(77, dtype=torch.long),
            "coordinate_values": torch.zeros((n, 2), dtype=torch.float32),
        }
