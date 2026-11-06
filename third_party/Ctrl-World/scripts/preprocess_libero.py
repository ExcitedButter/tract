"""
Preprocess LIBERO RLDS data for Ctrl-World training.

Reads openvla/modified_libero_rlds TFRecords and converts to Ctrl-World format:
  - Images: 256x256 JPEG → center-crop to 5:3 → resize to 320x192 → SVD VAE encode
  - States: 8D RLDS observation → 7D absolute [pos(3), axis_angle(3), gripper_width(1)]
  - Text: language_instruction from RLDS

Output: Ctrl-World dataset under dataset_example/libero/

Usage:
    cd third_party/Ctrl-World
    # needs TensorFlow to read the RLDS TFRecords (separate env, e.g. tensorflow-cpu==2.15.1 + torch + diffusers)

    # Single GPU:
    python scripts/preprocess_libero.py --svd_path ../../checkpoints/svd

    # Multi-GPU:
    accelerate launch --num_processes 8 scripts/preprocess_libero.py --svd_path ../../checkpoints/svd
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUITES = [
    "libero_spatial_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_10_no_noops",
]

TARGET_W, TARGET_H = 320, 192
SOURCE_SIZE = 256
CROP_H = round(SOURCE_SIZE * TARGET_H / TARGET_W)  # 154
CROP_TOP = (SOURCE_SIZE - CROP_H) // 2  # 51


def parse_rlds_episodes(tfrecord_path):
    """Parse a TFRecord file into a list of episode dicts."""
    import tensorflow as tf

    raw_ds = tf.data.TFRecordDataset(tfrecord_path)
    episodes = []
    for raw_record in raw_ds:
        ex = tf.train.SequenceExample()
        ex.ParseFromString(raw_record.numpy())
        ctx = ex.context.feature

        states = np.array(ctx["steps/observation/state"].float_list.value, dtype=np.float32)
        actions = np.array(ctx["steps/action"].float_list.value, dtype=np.float32)
        n_steps = len(ctx["steps/is_first"].int64_list.value)
        states = states.reshape(n_steps, 8)
        actions = actions.reshape(n_steps, 7)

        img_bytes_list = list(ctx["steps/observation/image"].bytes_list.value)
        wrist_bytes_list = list(ctx["steps/observation/wrist_image"].bytes_list.value)
        lang = ctx["steps/language_instruction"].bytes_list.value[0].decode("utf-8")
        file_path = ctx["episode_metadata/file_path"].bytes_list.value[0].decode("utf-8")

        episodes.append({
            "states_8d": states,
            "actions": actions,
            "image_bytes": img_bytes_list,
            "wrist_bytes": wrist_bytes_list,
            "text": lang,
            "file_path": file_path,
            "n_steps": n_steps,
        })
    return episodes


def decode_and_resize(jpeg_bytes, tf_module):
    """Decode JPEG bytes, center-crop to 5:3 aspect, resize to TARGET_W x TARGET_H."""
    img = tf_module.io.decode_jpeg(jpeg_bytes).numpy()  # (256, 256, 3)
    cropped = img[CROP_TOP : CROP_TOP + CROP_H, :, :]  # (154, 256, 3)
    resized = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)
    return resized


def state_8d_to_7d(states_8d):
    """Convert RLDS 8D state to Ctrl-World 7D.

    8D: [ee_pos(3), ee_ori_axisangle(3), gripper_L, gripper_R]
    7D: [ee_pos(3), ee_ori_axisangle(3), gripper_width]
    """
    gripper_width = states_8d[:, 6:7] - states_8d[:, 7:8]
    return np.concatenate([states_8d[:, :6], gripper_width], axis=1)


def vae_encode_frames(frames, vae, device, batch_size=32):
    """Encode (T, H, W, 3) uint8 frames to VAE latents (T, 4, 24, 40)."""
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
    latents = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size].to(device)
            z = vae.encode(batch).latent_dist.sample() * vae.config.scaling_factor
            latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def process_episode(ep, episode_id, split, output_dir, vae, device, tf_module, vae_batch_size=32):
    """Process a single RLDS episode into Ctrl-World format."""
    latent_check = os.path.join(output_dir, "latent_videos", split, episode_id, "0.pt")
    if os.path.exists(latent_check):
        return "skip"

    T = ep["n_steps"]
    states_7d = state_8d_to_7d(ep["states_8d"])

    agentview_frames = np.zeros((T, TARGET_H, TARGET_W, 3), dtype=np.uint8)
    wrist_frames = np.zeros((T, TARGET_H, TARGET_W, 3), dtype=np.uint8)
    for t in range(T):
        agentview_frames[t] = decode_and_resize(ep["image_bytes"][t], tf_module)
        wrist_frames[t] = decode_and_resize(ep["wrist_bytes"][t], tf_module)

    agent_latent = vae_encode_frames(agentview_frames, vae, device, vae_batch_size)
    wrist_latent = vae_encode_frames(wrist_frames, vae, device, vae_batch_size)
    zero_latent = torch.zeros_like(agent_latent)

    latent_dir = os.path.join(output_dir, "latent_videos", split, episode_id)
    os.makedirs(latent_dir, exist_ok=True)
    torch.save(agent_latent, os.path.join(latent_dir, "0.pt"))
    torch.save(zero_latent, os.path.join(latent_dir, "1.pt"))
    torch.save(wrist_latent, os.path.join(latent_dir, "2.pt"))

    ann = {
        "texts": [ep["text"]],
        "episode_id": episode_id,
        "video_length": T,
        "videos": [],
        "latent_videos": [
            {"latent_video_path": f"latent_videos/{split}/{episode_id}/{s}.pt"}
            for s in [0, 1, 2]
        ],
        "states": states_7d.tolist(),
        "observation.state.cartesian_position": states_7d[:, :6].tolist(),
        "observation.state.gripper_position": states_7d[:, 6].tolist(),
    }
    ann_dir = os.path.join(output_dir, "annotation", split)
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, f"{episode_id}.json"), "w") as f:
        json.dump(ann, f, indent=2)

    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds_dir", default="raw_data/modified_libero_rlds")
    parser.add_argument("--output_dir", default="dataset_example/libero")
    parser.add_argument("--svd_path", default="checkpoints/svd")
    parser.add_argument("--suites", nargs="+", default=None)
    parser.add_argument("--vae_batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    suites = args.suites or SUITES

    import tensorflow as tf

    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        device = accelerator.device
        local_rank = accelerator.process_index
        world_size = accelerator.num_processes
        is_main = accelerator.is_main_process
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        world_size = 1
        is_main = True

    from diffusers.models import AutoencoderKLTemporalDecoder
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_path, subfolder="vae").to(device)
    vae.eval()
    vae.requires_grad_(False)
    if is_main:
        print(f"VAE loaded on {device}")

    work = []
    for suite in suites:
        suite_dir = os.path.join(args.rlds_dir, suite)
        tfrecords = sorted(glob.glob(os.path.join(suite_dir, "1.0.0", "*.tfrecord*")))
        if is_main:
            print(f"Scanning {suite}: {len(tfrecords)} shards")
        suite_short = suite.replace("_no_noops", "")

        global_ep_idx = 0
        for tfr in tfrecords:
            episodes = parse_rlds_episodes(tfr)
            for ep in episodes:
                work.append((ep, suite_short, global_ep_idx))
                global_ep_idx += 1
        if is_main:
            print(f"  {suite}: {global_ep_idx} episodes total")

    n_total = len(work)
    if is_main:
        print(f"Total episodes across all suites: {n_total}")

    ok, skip, err = 0, 0, 0
    for idx in tqdm(range(n_total), desc="Processing", disable=not is_main):
        if idx % world_size != local_rank:
            continue

        ep, suite_short, ep_idx = work[idx]
        n_suite = sum(1 for w in work if w[1] == suite_short)
        n_val = max(1, int(n_suite * args.val_ratio))
        split = "val" if ep_idx >= n_suite - n_val else "train"
        episode_id = f"{suite_short}_{ep_idx:04d}"

        try:
            result = process_episode(
                ep, episode_id, split, args.output_dir,
                vae, device, tf, args.vae_batch_size,
            )
            if result == "ok":
                ok += 1
            elif result == "skip":
                skip += 1
        except Exception as e:
            err += 1
            if is_main:
                print(f"  ERROR {episode_id}: {e}")

    if is_main:
        print(f"\nDone: {ok} processed, {skip} skipped, {err} errors")
        sample_latent = glob.glob(os.path.join(args.output_dir, "latent_videos", "train", "*", "0.pt"))
        if sample_latent:
            t = torch.load(sorted(sample_latent)[0], map_location="cpu")
            print(f"Sample latent shape: {t.shape}")
        sample_ann = glob.glob(os.path.join(args.output_dir, "annotation", "train", "*.json"))
        if sample_ann:
            with open(sorted(sample_ann)[0]) as f:
                ann = json.load(f)
            print(f"Sample: ep={ann['episode_id']}, T={ann['video_length']}, text='{ann['texts'][0][:50]}'")


if __name__ == "__main__":
    main()
