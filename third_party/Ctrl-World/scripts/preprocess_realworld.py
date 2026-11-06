"""
Preprocess real-world Franka data for Ctrl-World training.

- Videos: from data/task_data/ (raw 672x376 MP4)
  → center-crop to 5:3 aspect (627x376) → resize to 320x192 → SVD VAE encode
- Actions: from data/task_data_320_square_tracks/ (10D, gripper binarized)
  → convert rot6d to axis-angle → save 7D + 10D

Output: Ctrl-World dataset format under dataset_example/realworld/

Usage:
    cd third_party/Ctrl-World
    source ../../.venv/bin/activate
    # Single GPU:
    python scripts/preprocess_realworld.py \
        --svd_path ../../checkpoints/svd
    # Multi-GPU:
    accelerate launch scripts/preprocess_realworld.py \
        --svd_path ../../checkpoints/svd
"""

import argparse
import glob
import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.action_conversion import actions_10d_to_7d

# Camera serial → view name
SERIAL_TO_VIEW = {
    "14203948": "agentview",
    "18204585": "eye_in_hand",
}

TASKS = ["task_1", "task_2", "task_3", "task_4"]

TASK_INSTRUCTIONS = {
    "task_1": "put the pink noodle in the bowl",
    "task_2": "put the bread in the bowl",
    "task_3": "pour the pasta into the pan",
    "task_4": "close the right cabinet door",
}

# Raw: 672x376. Center-crop width to 627 (5:3 ratio), then resize to 320x192.
RAW_W, RAW_H = 672, 376
TARGET_W, TARGET_H = 320, 192
# 5:3 crop: height stays 376, width = 376 * 5/3 = 626.67 → 627 (round up, will be resized anyway)
CROP_W = round(RAW_H * TARGET_W / TARGET_H)  # 627
CROP_LEFT = (RAW_W - CROP_W) // 2  # 22
CROP_RIGHT = CROP_LEFT + CROP_W    # 649


def read_mp4_frames(mp4_path):
    """Read all frames from MP4. Returns (T, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)


def crop_and_resize(frames):
    """672x376 → center-crop to 627x376 → resize to 320x192.
    Input/Output: (T, H, W, 3) uint8 RGB.
    """
    cropped = frames[:, :, CROP_LEFT:CROP_RIGHT, :]  # (T, 376, 627, 3)
    resized = np.zeros((len(cropped), TARGET_H, TARGET_W, 3), dtype=np.uint8)
    for i in range(len(cropped)):
        resized[i] = cv2.resize(cropped[i], (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)
    return resized


def convert_actions_7d(actions_10d):
    """Convert 10D → 7D using shared utility (models/action_conversion.py)."""
    return actions_10d_to_7d(actions_10d)


def get_split_demos(tracks_dir):
    """Read train/val split from 320_square_tracks directory.
    Returns {task: {demo_name: split}}.
    """
    splits = {}
    for task in TASKS:
        splits[task] = {}
        for split in ["train", "val"]:
            split_dir = os.path.join(tracks_dir, task, split)
            if not os.path.exists(split_dir):
                continue
            for f in glob.glob(os.path.join(split_dir, "*.hdf5")):
                demo_name = os.path.splitext(os.path.basename(f))[0]
                splits[task][demo_name] = split
    return splits


def process_demo(
    raw_dir, tracks_dir, output_dir, task, demo_name, split, vae, device
):
    """Process a single demo: videos from raw MP4, actions from tracks HDF5."""
    demo_dir = os.path.join(raw_dir, task, demo_name)
    tracks_path = os.path.join(tracks_dir, task, split, f"{demo_name}.hdf5")
    episode_id = f"{task}_{demo_name}"

    # Skip if already done
    latent_check = os.path.join(output_dir, "latent_videos", split, episode_id, "0.pt")
    if os.path.exists(latent_check):
        return "skip"

    # Load actions from tracks HDF5
    with h5py.File(tracks_path, "r") as f:
        actions_10d = np.array(f["root/actions"])  # (T, 10)
    T = actions_10d.shape[0]

    # Convert actions
    actions_7d = convert_actions_7d(actions_10d)

    # Separate fields for annotation (DROID-compatible format)
    # cartesian_position: (T, 6) list of lists — [pos(3), axis_angle(3)]
    # gripper_position: (T,) flat list of scalars — matches DROID format
    cartesian_position = actions_7d[:, 0:6].tolist()
    gripper_position = actions_7d[:, 6].tolist()

    # Process videos from raw MP4
    # DROID view order: 0=exterior_1 (third-person), 1=exterior_2, 2=wrist
    # Our mapping: agentview→slot 0, zeros→slot 1, eye_in_hand→slot 2
    VIEW_SLOT = {
        "agentview": 0,      # exterior_1 equivalent
        "eye_in_hand": 2,    # wrist equivalent
    }

    video_dir = os.path.join(output_dir, "videos", split, episode_id)
    latent_dir = os.path.join(output_dir, "latent_videos", split, episode_id)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)

    view_latents = {}
    for serial, view_name in SERIAL_TO_VIEW.items():
        slot = VIEW_SLOT[view_name]
        mp4_path = os.path.join(demo_dir, f"serial_{serial}_left.mp4")
        if not os.path.exists(mp4_path):
            return f"missing mp4: {mp4_path}"

        frames = read_mp4_frames(mp4_path)  # (T_raw, 376, 672, 3)
        if len(frames) > T:
            frames = frames[:T]
        elif len(frames) < T:
            return f"frame mismatch: raw={len(frames)}, actions={T}"

        # Crop and resize to 320x192
        frames_resized = crop_and_resize(frames)  # (T, 192, 320, 3) uint8

        # Save MP4
        save_mp4(frames_resized, os.path.join(video_dir, f"{slot}.mp4"), fps=10)

        # VAE encode
        x = torch.from_numpy(frames_resized).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
        x = x.to(device)
        with torch.no_grad():
            latents = []
            for i in range(0, len(x), 32):
                batch = x[i:i+32]
                latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                latents.append(latent)
            latent_tensor = torch.cat(latents, dim=0)  # (T, 4, 24, 40)
        view_latents[slot] = latent_tensor
        torch.save(latent_tensor, os.path.join(latent_dir, f"{slot}.pt"))

    # Slot 1: zeros (no second exterior camera)
    zero_latent = torch.zeros_like(list(view_latents.values())[0])
    torch.save(zero_latent, os.path.join(latent_dir, "1.pt"))
    black_frames = np.zeros((T, TARGET_H, TARGET_W, 3), dtype=np.uint8)
    save_mp4(black_frames, os.path.join(video_dir, "1.mp4"), fps=10)

    # Write annotation JSON
    instruction = TASK_INSTRUCTIONS[task]
    annotation = {
        "texts": [instruction],
        "episode_id": episode_id,
        "video_length": T,
        "videos": [
            {"video_path": f"videos/{split}/{episode_id}/0.mp4"},
            {"video_path": f"videos/{split}/{episode_id}/1.mp4"},
            {"video_path": f"videos/{split}/{episode_id}/2.mp4"},
        ],
        "latent_videos": [
            {"latent_video_path": f"latent_videos/{split}/{episode_id}/0.pt"},
            {"latent_video_path": f"latent_videos/{split}/{episode_id}/1.pt"},
            {"latent_video_path": f"latent_videos/{split}/{episode_id}/2.pt"},
        ],
        "states": actions_7d.tolist(),
        "states_10d": actions_10d.tolist(),
        "observation.state.cartesian_position": cartesian_position,
        "observation.state.gripper_position": gripper_position,
    }

    ann_dir = os.path.join(output_dir, "annotation", split)
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, f"{episode_id}.json"), "w") as f:
        json.dump(annotation, f, indent=2)

    return "ok"


def save_mp4(frames_rgb, path, fps=10):
    """Save (T, H, W, 3) uint8 RGB array as MP4."""
    T, H, W, _ = frames_rgb.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for i in range(T):
        writer.write(cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/task_data",
                        help="Raw data with MP4s")
    parser.add_argument("--tracks_dir", default="data/task_data_320_square_tracks",
                        help="320 square tracks HDF5 (for actions, split)")
    parser.add_argument("--output_dir", default="dataset_example/realworld",
                        help="Output in Ctrl-World dataset format")
    parser.add_argument("--svd_path", default="checkpoints/svd",
                        help="Path to SVD model (for VAE)")
    args = parser.parse_args()

    print(f"Raw videos:  {args.raw_dir}")
    print(f"Actions:     {args.tracks_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"SVD VAE:     {args.svd_path}")

    # Load VAE
    from diffusers.models import AutoencoderKLTemporalDecoder
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        is_main = accelerator.is_main_process
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_path, subfolder="vae").to(device)
    vae.eval()
    vae.requires_grad_(False)
    print(f"VAE loaded on {device}")

    # Get splits
    splits = get_split_demos(args.tracks_dir)

    # Build work list
    work = []
    for task in TASKS:
        for demo_name, split in sorted(splits[task].items()):
            work.append((task, demo_name, split))

    print(f"Total demos: {len(work)}")

    # Process
    ok, skip, err = 0, 0, 0
    for task, demo_name, split in tqdm(work, desc="Processing", disable=not is_main):
        result = process_demo(
            args.raw_dir, args.tracks_dir, args.output_dir,
            task, demo_name, split, vae, device
        )
        if result == "ok":
            ok += 1
        elif result == "skip":
            skip += 1
        else:
            err += 1
            if is_main:
                print(f"  ERROR {task}/{demo_name}: {result}")

    if is_main:
        print(f"\nDone: {ok} processed, {skip} skipped, {err} errors")

        # Quick verification
        sample_latent = glob.glob(os.path.join(args.output_dir, "latent_videos", "train", "*", "0.pt"))
        if sample_latent:
            t = torch.load(sorted(sample_latent)[0], map_location="cpu")
            print(f"Sample latent shape: {t.shape}")  # expect (T, 4, 24, 40)

        sample_ann = glob.glob(os.path.join(args.output_dir, "annotation", "train", "*.json"))
        if sample_ann:
            with open(sorted(sample_ann)[0]) as f:
                ann = json.load(f)
            print(f"Sample annotation: episode={ann['episode_id']}, T={ann['video_length']}, "
                  f"7D action shape=({len(ann['states'])}, {len(ann['states'][0])})")


if __name__ == "__main__":
    main()
