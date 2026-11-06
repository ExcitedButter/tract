"""Standalone track dualview WM inference.

No dependency on scripts/vlac_pipeline/. All paths relative to this directory.

Usage:
    cd third_party/track_dualview
    python inference.py --help
    python inference.py --test
"""

import logging
import os
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# tract repo root (third_party/track_dualview -> repo); checkpoints/ lives there.
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))

# Add this repo to path so svd/ imports work
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

logger = logging.getLogger(__name__)

# Default paths — override via constructor args or env vars
DEFAULT_CHECKPOINT = os.environ.get(
    "TRACK_DUALVIEW_CHECKPOINT",
    next(
        (p for p in (os.path.join(REPO_ROOT, "checkpoints", "track_dualview", "hybrid-checkpoint-128000"),
                     os.path.join(THIS_DIR, "checkpoints", "hybrid-checkpoint-128000")) if os.path.isdir(p)),
        os.path.join(REPO_ROOT, "checkpoints", "track_dualview", "hybrid-checkpoint-128000"),
    ),
)
DEFAULT_SVD_PATH = os.environ.get(
    "SVD_BASE_PATH",
    os.path.join(REPO_ROOT, "checkpoints", "svd")
    if os.path.isfile(os.path.join(REPO_ROOT, "checkpoints", "svd", "model_index.json"))
    else "stabilityai/stable-video-diffusion-img2vid",
)
DEFAULT_TEXT_ENCODER_PATH = os.environ.get(
    "TEXT_ENCODER_PATH",
    "stabilityai/stable-diffusion-2-1-base",
)
# Training parquet / Ctrl-World use 16-step windows; hybrid ckpt config may still say 14.
DEFAULT_NUM_FRAMES = 16
TRACK_TARGETS_RAW_DIM = 78  # 7 agent mesh + 25 wrist grid + 7 wrist mesh
DEFAULT_MOTION_BUCKET_ID = 180  # dualview training default
# 7 gripper points × (x,y) × 2 cameras (agentview + wrist); no wrist grid.
GRIPPER_TRACK_DIM = 28


def _bivariate_gaussian(kernel_size, sigma):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()


def split_tracks_78d(tracks_78d):
    T = tracks_78d.shape[0]
    flat = tracks_78d.reshape(T, 39, 2)
    return flat[:, 0:7, :], flat[:, 7:32, :], flat[:, 32:39, :]


def amplify_view_track_displacement(
    tracks: np.ndarray,
    *,
    view: str = "wrist",
    gain: float = 1.0,
) -> np.ndarray:
    """Scale per-frame displacement from t=0 (stronger ControlNet wrist drive)."""
    if abs(gain - 1.0) < 1e-6:
        return np.asarray(tracks, dtype=np.float32)
    out = np.asarray(tracks, dtype=np.float32).copy()
    t_len = out.shape[0]
    if out.shape[-1] == TRACK_TARGETS_RAW_DIM:
        flat = out.reshape(t_len, 39, 2)
        if view == "wrist":
            idx = slice(7, 39)
        elif view == "agent":
            idx = slice(0, 7)
        else:
            idx = slice(0, 39)
    elif out.shape[-1] == GRIPPER_TRACK_DIM:
        flat = out.reshape(t_len, 14, 2)
        idx = slice(7, 14) if view == "wrist" else slice(0, 7)
    else:
        return out
    base = flat[0, idx].copy()
    for t in range(t_len):
        flat[t, idx] = np.clip(base + gain * (flat[t, idx] - base), 0.0, 1.0)
    return flat.reshape(t_len, -1)


def render_view_heatmap(tracks, view, size, dilate=True, *, tight_gripper=False):
    T, N, _ = tracks.shape
    scale_factor = size / 128.0
    if dilate and not tight_gripper:
        kernel_size = int(49 * scale_factor) if scale_factor > 1 else 49
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = 7 * scale_factor if scale_factor > 1 else 7
        blur_kernel = _bivariate_gaussian(kernel_size, sigma)
    else:
        blur_kernel = None

    if tight_gripper:
        dot_range = 4 if view == "agentview" else 3
    else:
        dot_range = 12 if view == "agentview" else 8
    result = np.zeros((T, 3, size, size), dtype=np.float32)

    for t in range(T):
        base = np.ones((size, size, 3), dtype=np.float32) * 255
        for p_idx in range(N):
            x_norm, y_norm = tracks[t, p_idx]
            if x_norm <= 0 and y_norm <= 0:
                continue
            x = int(np.clip(x_norm * size, 0, size - 1))
            y = int(np.clip(y_norm * size, 0, size - 1))
            if view == "agentview":
                color = [255, 0, 0]
            else:
                color = [150, 150, 255] if p_idx < 25 else [0, 0, 255]
            for i in range(-dot_range, dot_range + 1):
                for j in range(-dot_range, dot_range + 1):
                    px, py = x + j, y + i
                    if 0 <= px < size and 0 <= py < size:
                        base[py, px] = color
        if blur_kernel is not None:
            base = cv2.filter2D(base, -1, blur_kernel)
        rgb = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        result[t] = rgb.transpose(2, 0, 1)
    return result


class TrackDualViewWM:
    def __init__(
        self,
        checkpoint_path=DEFAULT_CHECKPOINT,
        svd_path=DEFAULT_SVD_PATH,
        text_encoder_path=DEFAULT_TEXT_ENCODER_PATH,
        per_view_size=192,
    ):
        self.checkpoint_path = checkpoint_path
        self.svd_path = svd_path
        self.text_encoder_path = text_encoder_path
        self.per_view_size = per_view_size
        self.gripper_tight_heatmap = False
        self._pipeline = None
        self._controlnet = None
        self._text_encoder = None
        self._tokenizer = None

    def load(self, device="cuda", dtype=torch.float16, num_frames: int | None = None):
        from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        from svd.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionControlNetPipeline
        from svd.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
        from svd.temporal_controlnet import ControlNetModel

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            os.path.join(self.checkpoint_path, "unet"), torch_dtype=dtype)
        ckpt_frames = int(getattr(unet.config, "num_frames", DEFAULT_NUM_FRAMES))
        if num_frames is not None and num_frames != ckpt_frames:
            logger.warning(
                "Overriding UNet num_frames %d -> %d for inference",
                ckpt_frames,
                num_frames,
            )
            unet.config.num_frames = int(num_frames)
        controlnet = ControlNetModel.from_pretrained(
            os.path.join(self.checkpoint_path, "controlnet"), torch_dtype=dtype)
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.svd_path, subfolder="vae", variant="fp16", torch_dtype=dtype)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.svd_path, subfolder="image_encoder", variant="fp16", torch_dtype=dtype)
        feature_extractor = CLIPImageProcessor.from_pretrained(
            self.svd_path, subfolder="feature_extractor")
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.svd_path, subfolder="scheduler")

        pipeline = StableVideoDiffusionControlNetPipeline(
            vae=vae, image_encoder=image_encoder, unet=unet,
            scheduler=scheduler, feature_extractor=feature_extractor,
        ).to(device)
        # Progress bars dominate eval logs and add sync overhead; enable via env for debug.
        quiet = os.environ.get("TRACK_DUALVIEW_PIPELINE_DEBUG", "").strip().lower() not in {
            "1", "true", "yes", "on",
        }
        if quiet:
            try:
                pipeline.set_progress_bar_config(disable=True)
            except Exception:  # noqa: BLE001
                pass

        self._pipeline = pipeline
        self._controlnet = controlnet.to(device)
        self.num_frames = int(getattr(unet.config, "num_frames", DEFAULT_NUM_FRAMES))

        try:
            from transformers import AutoTokenizer, CLIPTextModel
            self._text_encoder = CLIPTextModel.from_pretrained(
                self.text_encoder_path, subfolder="text_encoder").to(device, dtype=dtype)
            self._text_encoder.requires_grad_(False)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.text_encoder_path, subfolder="tokenizer", use_fast=False)
        except Exception as e:
            logger.warning("Text encoder not loaded: %s", e)

        self._device = device
        self._dtype = dtype
        logger.info(
            "TrackDualViewWM loaded from %s (per_view=%d, num_frames=%d)",
            self.checkpoint_path,
            self.per_view_size,
            self.num_frames,
        )

    def _stack_views(self, agentview_pil, wrist_pil):
        S = self.per_view_size
        a = agentview_pil.convert("RGB").resize((S, S), Image.BICUBIC)
        w = wrist_pil.convert("RGB").resize((S, S), Image.BICUBIC)
        canvas = Image.new("RGB", (S, 2 * S))
        canvas.paste(a, (0, 0))
        canvas.paste(w, (0, S))
        return canvas

    def render_track_heatmaps(self, tracks):
        """Render conditioning heatmaps (dualview: stack agent top, wrist bottom on H)."""
        from data_loader.video_realworld_hybrid_dataset import generate_hybrid_trajectory_vis

        S = self.per_view_size
        arr = np.asarray(tracks, dtype=np.float32)
        view_config = {
            "height": S,
            "width": S,
            "video_seq_length": arr.shape[0],
            "dilate": True,
            "use_view_specific_colors": True,
            "dot_range_agentview": 12,
            "dot_range_wrist": 8,
        }
        if arr.ndim == 2 and arr.shape[-1] == GRIPPER_TRACK_DIM:
            flat = arr.reshape(arr.shape[0], 14, 2)
            agent_hm = generate_hybrid_trajectory_vis(flat[:, :7], view_config, "agentview")
            wrist_hm = generate_hybrid_trajectory_vis(flat[:, 7:], view_config, "eye_in_hand")
            return np.concatenate([agent_hm, wrist_hm], axis=2)
        agent_mesh, wrist_grid, wrist_mesh = split_tracks_78d(arr)
        wrist_cond = np.concatenate([wrist_grid, wrist_mesh], axis=1)
        agent_hm = generate_hybrid_trajectory_vis(agent_mesh, view_config, "agentview")
        wrist_hm = generate_hybrid_trajectory_vis(wrist_cond, view_config, "eye_in_hand")
        return np.concatenate([agent_hm, wrist_hm], axis=2)

    def _tokenize_prompt(self, prompt):
        if not prompt or self._tokenizer is None or self._text_encoder is None:
            return None
        inputs = self._tokenizer(
            [prompt], max_length=self._tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return inputs.input_ids.to(self._device)

    def generate_video(
        self,
        agentview_pil,
        wrist_pil,
        track_predictions_78d,
        prompt=None,
        num_inference_steps=25,
        num_frames: int | None = None,
        *,
        wrist_track_gain: float = 1.0,
        motion_bucket_id: int = DEFAULT_MOTION_BUCKET_ID,
        controlnet_conditioning_scale: float = 1.0,
        inner_conditioning_scale: float = 1.0,
        noise_aug_strength: float = 0.1,
    ):
        S = self.per_view_size
        num_frames = num_frames or getattr(self, "num_frames", DEFAULT_NUM_FRAMES)
        tracks = np.asarray(track_predictions_78d, dtype=np.float32)
        if tracks.shape[0] != num_frames:
            raise ValueError(f"tracks must have T={num_frames}, got {tracks.shape}")
        if tracks.ndim != 2 or tracks.shape[-1] not in (GRIPPER_TRACK_DIM, TRACK_TARGETS_RAW_DIM):
            raise ValueError(
                f"tracks last dim must be {GRIPPER_TRACK_DIM} (gripper7×2 views) or "
                f"{TRACK_TARGETS_RAW_DIM} (track_targets_raw), got {tracks.shape}"
            )
        tracks = amplify_view_track_displacement(tracks, view="wrist", gain=wrist_track_gain)

        stacked_first = self._stack_views(agentview_pil, wrist_pil)
        cond = self.render_track_heatmaps(tracks)

        prompt_tokens = self._tokenize_prompt(prompt) if prompt else None

        with torch.no_grad():
            output = self._pipeline(
                image=stacked_first,
                controlnet=self._controlnet,
                condition_img=cond,
                prompt=prompt_tokens,
                use_text=prompt_tokens is not None,
                text_encoder=self._text_encoder,
                height=2 * S, width=S,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                max_guidance_scale=3.0,
                fps=7,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                inner_conditioning_scale=inner_conditioning_scale,
                use_instructpix2pix=False,
                guess_mode=False,
                image_guidance_scale=2.5,
            )

        stacked_frames = output.frames[0]
        agentview_frames = [f.crop((0, 0, S, S)) for f in stacked_frames]
        wrist_frames = [f.crop((0, S, S, 2 * S)) for f in stacked_frames]
        return agentview_frames, wrist_frames

    def batch_generate_video(
        self,
        agentview_pil,
        wrist_pil,
        tracks_batch,
        prompt=None,
        num_inference_steps=25,
        num_frames: int | None = None,
        vae_encode_chunk: int = 16,
        *,
        wrist_track_gain: float = 1.0,
        motion_bucket_id: int = DEFAULT_MOTION_BUCKET_ID,
        controlnet_conditioning_scale: float = 1.0,
        inner_conditioning_scale: float = 1.0,
        noise_aug_strength: float = 0.1,
    ):
        """Generate B dual-view videos in one diffusion pass.

        Args:
            tracks_batch: (B, T, 78|28) float32 tracks in [0, 1].

        Returns:
            (agent_list, wrist_list) each length B; every entry is a list of T PILs.
        """
        S = self.per_view_size
        num_frames = num_frames or getattr(self, "num_frames", DEFAULT_NUM_FRAMES)
        tracks_batch = np.asarray(tracks_batch, dtype=np.float32)
        if tracks_batch.ndim != 3:
            raise ValueError(f"tracks_batch must be (B,T,D), got {tracks_batch.shape}")
        B, T, D = tracks_batch.shape
        if T != num_frames:
            raise ValueError(f"tracks_batch T must be {num_frames}, got {T}")
        if D not in (GRIPPER_TRACK_DIM, TRACK_TARGETS_RAW_DIM):
            raise ValueError(f"tracks last dim must be {GRIPPER_TRACK_DIM} or {TRACK_TARGETS_RAW_DIM}, got {D}")

        amplified = np.stack(
            [
                amplify_view_track_displacement(tracks_batch[i], view="wrist", gain=wrist_track_gain)
                for i in range(B)
            ],
            axis=0,
        )
        stacked_first = self._stack_views(agentview_pil, wrist_pil)
        heatmaps = np.stack([self.render_track_heatmaps(amplified[i]) for i in range(B)], axis=0)
        h_flat = torch.from_numpy(heatmaps.reshape(B * T, *heatmaps.shape[2:])).to(
            self._device, dtype=self._dtype
        )
        prompt_tokens = self._tokenize_prompt(prompt) if prompt else None
        if prompt_tokens is not None and prompt_tokens.shape[0] == 1 and B > 1:
            prompt_tokens = prompt_tokens.repeat(B, 1)

        lat_h = 2 * S // 8
        lat_w = S // 8
        with torch.no_grad():
            cond_parts = []
            for i in range(0, B * T, vae_encode_chunk):
                chunk = h_flat[i : i + vae_encode_chunk]
                cond_parts.append(self._pipeline.vae.encode(chunk).latent_dist.mode())
            cond_latent = torch.cat(cond_parts, dim=0).reshape(B, T, 4, lat_h, lat_w)

            img_t = self._pipeline.image_processor.preprocess(stacked_first, height=2 * S, width=S)
            img_t = img_t.to(self._device, dtype=self._dtype)
            img_t = img_t + torch.randn_like(img_t) * noise_aug_strength
            image_latent = self._pipeline.vae.encode(img_t).latent_dist.mode().repeat(B, 1, 1, 1)

            output = self._pipeline(
                image=[stacked_first] * B,
                controlnet=self._controlnet,
                condition_img=None,
                image_latent=image_latent,
                controlnet_cond_latent=cond_latent,
                prompt=prompt_tokens,
                use_text=prompt_tokens is not None,
                text_encoder=self._text_encoder,
                height=2 * S,
                width=S,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                max_guidance_scale=3.0,
                fps=7,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                inner_conditioning_scale=inner_conditioning_scale,
                use_instructpix2pix=False,
                guess_mode=False,
                image_guidance_scale=2.5,
            )

        agent_list, wrist_list = [], []
        for b_frames in output.frames:
            agent_list.append([f.crop((0, 0, S, S)) for f in b_frames])
            wrist_list.append([f.crop((0, S, S, 2 * S)) for f in b_frames])
        return agent_list, wrist_list


def test():
    import h5py
    import imageio

    print("=== Track Dualview WM Standalone Test ===")
    print(f"Checkpoint: {DEFAULT_CHECKPOINT}")
    print(f"SVD: {DEFAULT_SVD_PATH}")

    # Check paths
    for name, path in [("svd", os.path.join(THIS_DIR, "svd")),
                        ("checkpoint", DEFAULT_CHECKPOINT),
                        ("svd_weights", DEFAULT_SVD_PATH),
                        ("text_encoder", DEFAULT_TEXT_ENCODER_PATH)]:
        exists = os.path.exists(path)
        print(f"  {name}: {'✓' if exists else '✗ (optional)' if name == 'text_encoder' else '✗'} {path}")
        if not exists and name != "text_encoder":
            print(f"  Required path missing!")
            return False

    # Load model
    print("\nLoading model...")
    wm = TrackDualViewWM()
    wm.load(device="cuda")
    print("  Model loaded.")

    # Load test data
    data_path = "data/task_data_320_square_tracks/task_1/val/session_demo_15.hdf5"
    if not os.path.exists(data_path):
        print(f"  Test data not found: {data_path}")
        return False

    with h5py.File(data_path, "r") as f:
        av_video = f["root/agentview/video"][0]
        ei_video = f["root/eye_in_hand/video"][0]
        av_mesh = f["root/mesh_data/vertex_tracking/agentview_vertices_2d"][:]
        ei_mesh = f["root/mesh_data/vertex_tracking/eyeinhand_vertices_2d"][:]
        wrist_grid = f["root/eye_in_hand/grid_tracks_perwindow"][20]

    frame_idx = 40
    av_pil = Image.fromarray(av_video[frame_idx].transpose(1, 2, 0))
    ei_pil = Image.fromarray(ei_video[frame_idx].transpose(1, 2, 0))

    nf = wm.num_frames
    sl = slice(frame_idx, frame_idx + nf)
    tracks_78d = np.concatenate([
        av_mesh[sl].reshape(nf, 14),
        wrist_grid[:nf].reshape(nf, 50),
        ei_mesh[sl].reshape(nf, 14),
    ], axis=1).astype(np.float32)

    print(f"\nGenerating video (frame_idx={frame_idx})...")
    agent_frames, wrist_frames = wm.generate_video(
        av_pil, ei_pil, tracks_78d, prompt="put the pink noodle in the bowl"
    )
    print(f"  Generated {len(agent_frames)} frames, size={agent_frames[0].size}")

    # Save output
    out_dir = os.path.join(THIS_DIR, "test_output")
    os.makedirs(out_dir, exist_ok=True)

    av_pil.save(os.path.join(out_dir, "input_agentview.png"))
    ei_pil.save(os.path.join(out_dir, "input_wrist.png"))

    writer = imageio.get_writer(os.path.join(out_dir, "generated.mp4"), fps=10)
    for i, (a, w) in enumerate(zip(agent_frames, wrist_frames)):
        a.save(os.path.join(out_dir, f"frame_{i:02d}_agent.png"))
        w.save(os.path.join(out_dir, f"frame_{i:02d}_wrist.png"))
        stacked = Image.new("RGB", (a.width, a.height + w.height))
        stacked.paste(a, (0, 0))
        stacked.paste(w, (0, a.height))
        writer.append_data(np.array(stacked))
    writer.close()

    print(f"\n  Output saved to: {out_dir}/")
    print("  PASS")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.test:
        test()
