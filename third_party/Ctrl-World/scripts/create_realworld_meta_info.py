"""
Create meta info (normalization stats + train/val samples) for real-world Ctrl-World dataset.

Usage:
    cd third_party/Ctrl-World
    python scripts/create_realworld_meta_info.py
"""

import glob
import json
import os

import numpy as np


def compute_stats(annotations_dir, action_key="states"):
    """Compute 1st and 99th percentile bounds from all training annotations."""
    all_states = []
    ann_files = sorted(glob.glob(os.path.join(annotations_dir, "train", "*.json")))
    for ann_file in ann_files:
        with open(ann_file) as f:
            ann = json.load(f)
        all_states.extend(ann[action_key])
    all_states = np.array(all_states)
    print(f"  Collected {all_states.shape[0]} frames, dim={all_states.shape[1]}")

    state_01 = np.percentile(all_states, 1, axis=0).tolist()
    state_99 = np.percentile(all_states, 99, axis=0).tolist()
    return {"state_01": state_01, "state_99": state_99}


def create_samples(annotations_dir, split, sample_stride=5):
    """Create sample list with multiple starting frames per trajectory.

    Each sample is a starting frame from which the dataloader will extract
    a window of [num_history + num_frames] frames.
    """
    ann_files = sorted(glob.glob(os.path.join(annotations_dir, split, "*.json")))
    samples = []
    for ann_file in ann_files:
        with open(ann_file) as f:
            ann = json.load(f)
        episode_id = ann["episode_id"]
        T = ann["video_length"]

        # Sample starting frames every sample_stride frames
        # Leave room for the prediction window (num_history=1 + num_frames=16 = 17)
        max_start = T - 17
        if max_start < 0:
            print(f"  WARNING: {episode_id} too short ({T} frames), skipping")
            continue

        for frame_idx in range(0, max_start, sample_stride):
            samples.append({
                "episode_id": episode_id,
                "frame_ids": [frame_idx],
            })

    print(f"  {split}: {len(ann_files)} trajectories → {len(samples)} samples (stride={sample_stride})")
    return samples


def main():
    dataset_dir = "dataset_example/realworld"
    output_dir = "dataset_meta_info/realworld"
    os.makedirs(output_dir, exist_ok=True)

    annotations_dir = os.path.join(dataset_dir, "annotation")

    # 7D stats
    print("Computing 7D normalization stats...")
    stats_7d = compute_stats(annotations_dir, action_key="states")
    with open(os.path.join(output_dir, "stat.json"), "w") as f:
        json.dump(stats_7d, f, indent=2)
    print(f"  state_01: {[f'{v:.4f}' for v in stats_7d['state_01']]}")
    print(f"  state_99: {[f'{v:.4f}' for v in stats_7d['state_99']]}")

    # 10D stats (backup)
    print("Computing 10D normalization stats...")
    stats_10d = compute_stats(annotations_dir, action_key="states_10d")
    with open(os.path.join(output_dir, "stat_10d.json"), "w") as f:
        json.dump(stats_10d, f, indent=2)

    # Train/val samples
    print("Creating train samples...")
    train_samples = create_samples(annotations_dir, "train", sample_stride=5)
    with open(os.path.join(output_dir, "train_sample.json"), "w") as f:
        json.dump(train_samples, f, indent=2)

    print("Creating val samples...")
    val_samples = create_samples(annotations_dir, "val", sample_stride=10)
    with open(os.path.join(output_dir, "val_sample.json"), "w") as f:
        json.dump(val_samples, f, indent=2)

    print(f"\nDone. Output: {output_dir}/")
    print(f"  stat.json (7D), stat_10d.json (10D)")
    print(f"  train_sample.json ({len(train_samples)} samples)")
    print(f"  val_sample.json ({len(val_samples)} samples)")


if __name__ == "__main__":
    main()
