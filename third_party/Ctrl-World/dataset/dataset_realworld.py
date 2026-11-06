"""
Dataset loader for real-world Franka data in Ctrl-World format.
Based on dataset_droid_exp33.py but adapted for:
  - 2 real views + 1 zero-padded (stacked to latent height 72)
  - 10Hz data with down_sample=1
  - num_history=1, num_frames=16
  - 7D actions [pos(3), axis_angle(3), gripper(1)]
"""

import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset_mix(Dataset):
    def __init__(self, args, mode="val"):
        super().__init__()
        self.args = args
        self.mode = mode

        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []

        dataset_root_path = args.dataset_root_path
        dataset_names = args.dataset_names.split("+")
        dataset_meta_info_path = args.dataset_meta_info_path
        dataset_cfgs = args.dataset_cfgs.split("+")
        self.prob = args.prob

        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            data_json_path = f"{dataset_meta_info_path}/{dataset_cfg}/{mode}_sample.json"

            with open(data_json_path, "r") as f:
                samples = json.load(f)
            dataset_path = [os.path.join(dataset_root_path, dataset_name) for _ in samples]
            print(f"Dataset '{dataset_name}' [{mode}]: {len(samples)} samples")
            self.dataset_path_all.append(dataset_path)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))

            # Normalization bounds
            with open(f"{dataset_meta_info_path}/{dataset_name}/stat.json", "r") as f:
                data_stat = json.load(f)
                state_p01 = np.array(data_stat["state_01"])[None, :]
                state_p99 = np.array(data_stat["state_99"])[None, :]
                self.norm_all.append((state_p01, state_p99))

        self.max_id = max(self.samples_len)
        print(f"samples_len: {self.samples_len}, max_id: {self.max_id}")

    def __len__(self):
        return self.max_id

    def _load_latent_video(self, video_path, frame_ids):
        with open(video_path, "rb") as file:
            video_tensor = torch.load(file, weights_only=False)
            video_tensor.requires_grad = False
        max_frames = video_tensor.size()[0]
        frame_ids = [int(fid) if fid < max_frames else max_frames - 1 for fid in frame_ids]
        return video_tensor[frame_ids]

    def _get_frames(self, label, frame_ids, cam_id, video_dir):
        video_path = label["latent_videos"][cam_id]["latent_video_path"]
        video_path = os.path.join(video_dir, video_path)
        try:
            frames = self._load_latent_video(video_path, frame_ids)
        except Exception:
            video_path = video_path.replace("latent_videos", "latent_videos_svd")
            frames = self._load_latent_video(video_path, frame_ids)
        return frames

    def normalize_bound(self, data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def __getitem__(self, index):
        # Sample dataset
        dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]
        dataset_dir = dataset_path[index]

        # Load annotation
        ann_file = f'{dataset_dir}/{self.args.annotation_name}/{self.mode}/{sample["episode_id"]}.json'
        with open(ann_file, "r") as f:
            label = json.load(f)

        # Frame indices
        # down_sample=1 for our 10Hz data (no further downsampling)
        num_history = self.args.num_history  # 1
        num_frames = self.args.num_frames    # 16
        total_frames = num_history + num_frames  # 17

        frame_len = label["video_length"] - 1
        frame_now = sample["frame_ids"][0]

        # Random temporal skip for data augmentation
        skip = random.randint(1, 2)

        # Build frame indices: [history..., current, future...]
        rgb_id = []
        # History frames (just 1)
        for i in range(num_history, 0, -1):
            rgb_id.append(int(frame_now - i * skip))
        # Current + future
        rgb_id.append(frame_now)
        for i in range(1, num_frames):
            rgb_id.append(int(frame_now + i * skip))

        rgb_id = np.array(rgb_id)
        rgb_id = np.clip(rgb_id, 0, frame_len).tolist()
        rgb_id = [int(fid) for fid in rgb_id]

        # State indices: same as rgb_id since down_sample=1
        state_id = np.array(rgb_id) * self.args.down_sample

        # Load latents from 3 views (view 2 is zeros)
        data = dict()
        data["text"] = label["texts"][0]

        latent_h = 72  # 3 views * 24
        latent_w = 40
        latent = torch.zeros((total_frames, 4, latent_h, latent_w), dtype=torch.float32)

        # DROID view order: 0=exterior_1, 1=exterior_2, 2=wrist
        # Our data: slot 0=agentview, slot 1=zeros, slot 2=eye_in_hand
        view0 = self._get_frames(label, rgb_id, cam_id=0, video_dir=dataset_dir)  # agentview
        view2 = self._get_frames(label, rgb_id, cam_id=2, video_dir=dataset_dir)  # eye_in_hand (wrist)

        latent[:, :, 0:24] = view0     # slot 0: agentview (= DROID exterior_1)
        # latent[:, :, 24:48] stays zeros  # slot 1: no second exterior
        latent[:, :, 48:72] = view2    # slot 2: eye_in_hand (= DROID wrist)

        data["latent"] = latent.float()

        # Actions: 7D from annotation (matches DROID loader pattern)
        # cartesian_position: (T, 6) [pos, axis_angle]
        # gripper_position: (T,) scalar → add newaxis for concat
        cartesian_pose = np.array(label["observation.state.cartesian_position"])[state_id]
        gripper_pose = np.array(label["observation.state.gripper_position"])[state_id][..., np.newaxis]
        action = np.concatenate((cartesian_pose, gripper_pose), axis=-1)  # (num_frames, 7)
        action = self.normalize_bound(action, state_p01, state_p99)
        data["action"] = torch.tensor(action).float()

        return data


if __name__ == "__main__":
    from config_realworld import wm_args
    args = wm_args()
    train_dataset = Dataset_mix(args, mode="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    for data in train_loader:
        print(f"latent: {data['latent'].shape}")   # (B, 17, 4, 72, 40)
        print(f"action: {data['action'].shape}")    # (B, 17, 7)
        print(f"text: {data['text']}")
        break
