"""
Create meta info (normalization stats + train/val samples) for LIBERO Ctrl-World dataset.

Usage:
    cd third_party/Ctrl-World
    python scripts/create_libero_meta_info.py
"""

import glob
import json
import os

import numpy as np


def compute_stats(annotations_dir, action_key="states"):
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
    ann_files = sorted(glob.glob(os.path.join(annotations_dir, split, "*.json")))
    samples = []
    for ann_file in ann_files:
        with open(ann_file) as f:
            ann = json.load(f)
        episode_id = ann["episode_id"]
        T = ann["video_length"]
        max_start = T - 17
        if max_start < 0:
            print(f"  WARNING: {episode_id} too short ({T} frames), skipping")
            continue
        for frame_idx in range(0, max_start, sample_stride):
            samples.append({"episode_id": episode_id, "frame_ids": [frame_idx]})
    print(f"  {split}: {len(ann_files)} trajectories -> {len(samples)} samples (stride={sample_stride})")
    return samples


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="dataset_example/libero")
    # NOTE: dataset_meta_info/libero/stat.json is also the normalization used at inference
    # (scripts/vlac_pipeline/ctrl_world_wm.py); write elsewhere when experimenting on a subset.
    parser.add_argument("--output_dir", default="dataset_meta_info/libero")
    cli = parser.parse_args()
    dataset_dir = cli.dataset_dir
    output_dir = cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    annotations_dir = os.path.join(dataset_dir, "annotation")

    print("Computing 7D normalization stats...")
    stats = compute_stats(annotations_dir, action_key="states")
    with open(os.path.join(output_dir, "stat.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  state_01: {[f'{v:.4f}' for v in stats['state_01']]}")
    print(f"  state_99: {[f'{v:.4f}' for v in stats['state_99']]}")

    print("Creating train samples...")
    train_samples = create_samples(annotations_dir, "train", sample_stride=5)
    with open(os.path.join(output_dir, "train_sample.json"), "w") as f:
        json.dump(train_samples, f, indent=2)

    print("Creating val samples...")
    val_samples = create_samples(annotations_dir, "val", sample_stride=10)
    with open(os.path.join(output_dir, "val_sample.json"), "w") as f:
        json.dump(val_samples, f, indent=2)

    print(f"\nDone. Output: {output_dir}/")
    print(f"  stat.json, train_sample.json ({len(train_samples)}), val_sample.json ({len(val_samples)})")


if __name__ == "__main__":
    main()
