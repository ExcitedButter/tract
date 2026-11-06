"""Ctrl-World world model wrapper.

SVD UNet with cross-attention action conditioning.
Resolution: 320×192 per view, 3 views stacked → 320×576.
Actions: 7D axis-angle (converted from 10D internally).

Output: center-cropped agentview 192×192 square frames for VLAC.
"""

import json
import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

from .utils import center_crop_square

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CTRL_WORLD_DIR = os.path.join(PROJECT_ROOT, "third_party", "Ctrl-World")

logger = logging.getLogger(__name__)

CTRL_WORLD_LIBERO_CKPT = os.path.join(
    PROJECT_ROOT, "checkpoints", "ctrl_world", "libero_ctrlworld", "checkpoint-20000.pt"
)
CTRL_WORLD_REALWORLD_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "ctrl_world", "realworld_ctrlworld_v3")
CTRL_WORLD_DROID_BASE = os.path.join(PROJECT_ROOT, "checkpoints", "ctrl_world", "droid", "checkpoint-10000.pt")


def _find_latest_realworld_ctrlworld_checkpoint() -> str:
    """Return the latest real-world Ctrl-World checkpoint, or the DROID base as fallback."""
    import glob
    import re

    pattern = os.path.join(CTRL_WORLD_REALWORLD_DIR, "checkpoint-*.pt")
    matches = glob.glob(pattern)
    if matches:

        def step(p):
            m = re.search(r"checkpoint-(\d+)\.pt$", p)
            return int(m.group(1)) if m else 0

        latest = max(matches, key=step)
        logger.info("Auto-selected latest Ctrl-World real-world checkpoint: %s", latest)
        return latest
    logger.warning(
        "No real-world Ctrl-World checkpoint found in %s — falling back to DROID base",
        CTRL_WORLD_REALWORLD_DIR,
    )
    return CTRL_WORLD_DROID_BASE


def _default_ctrlworld_checkpoint() -> str:
    if os.path.isfile(CTRL_WORLD_LIBERO_CKPT):
        return CTRL_WORLD_LIBERO_CKPT
    return _find_latest_realworld_ctrlworld_checkpoint()


DEFAULT_CHECKPOINT = _default_ctrlworld_checkpoint()
DEFAULT_SVD_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "svd")
DEFAULT_CLIP_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "clip-vit-base-patch32")
DEFAULT_STAT_PATH = os.path.join(CTRL_WORLD_DIR, "dataset_meta_info", "realworld", "stat.json")
LIBERO_STAT_PATH = os.path.join(CTRL_WORLD_DIR, "dataset_meta_info", "libero", "stat.json")


def _resolve_stat_path(checkpoint_path: str, stat_path: str) -> str:
    """Pick LIBERO vs realworld normalization stats from checkpoint path."""
    if stat_path != DEFAULT_STAT_PATH:
        return stat_path
    ckpt = (checkpoint_path or "").lower()
    if "libero" in ckpt and os.path.isfile(LIBERO_STAT_PATH):
        return LIBERO_STAT_PATH
    return stat_path


def _normalize_bound(data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
    """Normalize data to [clip_min, clip_max] using min/max bounds."""
    ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, clip_min, clip_max)


class CtrlWorldModel:
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        svd_path: str = DEFAULT_SVD_PATH,
        clip_path: str = DEFAULT_CLIP_PATH,
        stat_path: str = DEFAULT_STAT_PATH,
    ):
        self.checkpoint_path = checkpoint_path
        self.svd_path = svd_path
        self.clip_path = clip_path
        self.stat_path = _resolve_stat_path(checkpoint_path, stat_path)
        self._model = None
        self._state_p01 = None
        self._state_p99 = None

    def load(self, device: str = "cuda", dtype=torch.bfloat16):
        if CTRL_WORLD_DIR not in sys.path:
            sys.path.insert(0, CTRL_WORLD_DIR)

        from models.ctrl_world import CrtlWorld

        # Reuse Ctrl-World inference path resolution (local checkpoints or HF hub).
        scripts_dir = os.path.join(CTRL_WORLD_DIR, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from inference_world_model import resolve_clip_path, resolve_svd_path

        svd_user = None if self.svd_path == DEFAULT_SVD_PATH else self.svd_path
        clip_user = None if self.clip_path == DEFAULT_CLIP_PATH else self.clip_path
        svd_resolved = resolve_svd_path(svd_user, auto_download=True)
        clip_resolved = resolve_clip_path(clip_user)

        # Build args namespace matching config_realworld.py
        class Args:
            pass

        args = Args()
        args.svd_model_path = svd_resolved
        args.clip_model_path = clip_resolved
        args.width = 320
        args.height = 192
        args.num_frames = 16
        args.num_history = 1
        args.action_dim = 7
        args.text_cond = True
        args.frame_level_cond = True
        args.dtype = dtype

        model = CrtlWorld(args)
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.to(device).to(dtype)
        model.eval()

        # Load normalization stats
        with open(self.stat_path) as f:
            stat = json.load(f)
        self._state_p01 = np.array(stat["state_01"])[None, :]  # (1, 7)
        self._state_p99 = np.array(stat["state_99"])[None, :]  # (1, 7)

        self._model = model
        self._device = device
        self._dtype = dtype
        self._args = args
        logger.info("Ctrl-World WM loaded from %s", self.checkpoint_path)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _convert_10d_to_7d(self, actions_10d: np.ndarray) -> np.ndarray:
        """Convert 10D absolute EE actions to 7D axis-angle format.

        10D: [pos(3), rot6d(6), gripper_width(1)]
        7D:  [pos(3), axis_angle(3), gripper_01(1)]
        """
        sys.path.insert(0, CTRL_WORLD_DIR) if CTRL_WORLD_DIR not in sys.path else None
        from models.action_conversion import actions_10d_to_7d
        return actions_10d_to_7d(actions_10d, gripper_max=0.08)

    def _convert_policy_actions_to_7d(
        self,
        actions: np.ndarray,
        *,
        initial_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map policy actions to 16 rows of 7D absolute states for Ctrl-World.

        - Real-world policies: (16, 10) absolute EE → 7D via ``actions_10d_to_7d``.
        - LIBERO policies: (16, 32) OSC delta → roll out with ``LiberoActionConverter``.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[-1] == 10:
            return self._convert_10d_to_7d(actions)

        if actions.shape[-1] < 7:
            raise ValueError(f"Expected action dim >= 7, got shape {actions.shape}")

        from models.libero_action_converter import LiberoActionConverter

        conv = LiberoActionConverter(device=str(self._device))
        adapter = os.path.join(
            PROJECT_ROOT, "checkpoints", "ctrl_world", "libero_action_adapter.pt"
        )
        if os.path.isfile(adapter):
            conv.load_adapter(adapter, device=str(self._device))
        elif not getattr(self, "_warned_no_adapter", False):
            logger.warning(
                "LIBERO action adapter not found at %s; using the approximate analytical "
                "delta->state conversion (results will differ from the learned adapter).",
                adapter,
            )
            self._warned_no_adapter = True

        deltas = actions[..., :7]
        if initial_state is not None and len(initial_state) >= 8:
            init_7d = conv.initial_state_from_obs(np.asarray(initial_state, dtype=np.float64)[:8])
        else:
            init_7d = np.zeros(7, dtype=np.float64)

        states = conv.trajectory(init_7d, deltas)  # (17, 7)
        return states[1 : 1 + len(deltas)].astype(np.float32)

    def _encode_frame_to_latent(self, frame_pil: Image.Image) -> torch.Tensor:
        """Encode a single PIL frame to VAE latent space."""
        frame_np = np.array(frame_pil).astype(np.float32)
        frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        frame_t = frame_t / 255.0 * 2 - 1  # normalize to [-1, 1]
        frame_t = frame_t.to(self._device, dtype=self._dtype)
        with torch.no_grad():
            latent = self._model.vae.encode(frame_t).latent_dist.sample()
            latent = latent * self._model.vae.config.scaling_factor
        return latent  # (1, 4, H/8, W/8)

    def generate_video(
        self,
        agentview_pil: Image.Image,
        wrist_pil: Image.Image,
        actions_10d: np.ndarray,
        prompt: str = "",
        num_inference_steps: int = 50,
        return_all_views: bool = False,
        initial_state: np.ndarray | None = None,
    ) -> list:
        """Generate 16-frame video and return center-cropped agentview frames.

        Args:
            agentview_pil: PIL 320×192 agentview frame.
            wrist_pil: PIL 320×192 wrist frame.
            actions_10d: (16, 10) or (16, 32) policy actions — converted to 7D internally.
            initial_state: optional robot state for LIBERO delta rollout (8D RLDS or 10D).
            prompt: task description for text conditioning.
            num_inference_steps: diffusion steps.

        Returns:
            List of 16 PIL Images at 192×192 (center-cropped agentview).
        """
        import einops

        # Convert actions: 10D → 7D → normalize to [-1,1]
        actions_7d = self._convert_policy_actions_to_7d(
            actions_10d, initial_state=initial_state
        )  # (16, 7)
        # Prepend a dummy history action (repeat first action)
        actions_7d_full = np.concatenate([actions_7d[:1], actions_7d], axis=0)  # (17, 7)
        actions_norm = _normalize_bound(actions_7d_full, self._state_p01, self._state_p99)
        actions_t = torch.from_numpy(actions_norm).unsqueeze(0).to(self._device, dtype=self._dtype)

        # Encode action embeddings with optional text
        with torch.no_grad():
            action_hidden = self._model.action_encoder(
                actions_t,
                texts=[prompt] if prompt else None,
                text_tokinizer=self._model.tokenizer,
                text_encoder=self._model.text_encoder,
                frame_level_cond=True,
            )  # (1, 17, 1024)

        # Encode frames: stack 3 views [agentview, zeros, wrist] → (1, 4, 72, 40)
        lat_agent = self._encode_frame_to_latent(agentview_pil)   # (1, 4, 24, 40)
        lat_zeros = torch.zeros_like(lat_agent)                    # (1, 4, 24, 40)
        lat_wrist = self._encode_frame_to_latent(wrist_pil)       # (1, 4, 24, 40)
        cond_latent = torch.cat([lat_agent, lat_zeros, lat_wrist], dim=2)  # (1, 4, 72, 40)

        # History latents: model expects num_history (1) frames of history.
        # We use the current frame latent as the single history frame, matching
        # upstream Ctrl-World rollout_replay_traj.py pattern where his_cond has shape (1, num_history, 4, H, W).
        his_cond = cond_latent.unsqueeze(1)  # (1, 1, 4, 72, 40)

        # Run diffusion pipeline
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        with torch.no_grad():
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                self._model.pipeline,
                image=cond_latent,
                text=action_hidden,
                width=self._args.width,
                height=self._args.height * 3,
                num_frames=self._args.num_frames,
                history=his_cond,
                num_inference_steps=num_inference_steps,
                decode_chunk_size=8,
                max_guidance_scale=1.0,
                fps=7,
                motion_bucket_id=127,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
                his_cond_zero=False,
            )

        # latents: (1, 16, 4, 72, 40)
        # Rearrange: split 3 views
        latents = einops.rearrange(latents, 'b f c (m h) w -> (b m) f c h w', m=3)
        # latents[0] = agentview, latents[1] = zeros, latents[2] = wrist

        def _decode_view(view_latents):
            frames = []
            for t in range(view_latents.shape[1]):
                lat = view_latents[:, t] / self._model.vae.config.scaling_factor
                decoded = self._model.vae.decode(lat, num_frames=1).sample
                frame = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255)
                frame = frame[0].permute(1, 2, 0).to(torch.float32).cpu().numpy().astype(np.uint8)
                frames.append(Image.fromarray(frame))
            return frames

        with torch.no_grad():
            agentview_frames = [center_crop_square(f) for f in _decode_view(latents[0:1])]
            if return_all_views:
                wrist_frames = [center_crop_square(f) for f in _decode_view(latents[2:3])]
                return agentview_frames, wrist_frames

        return agentview_frames

    def batch_generate_video(
        self,
        agentview_pil: Image.Image,
        wrist_pil: Image.Image,
        actions_batch: np.ndarray,
        prompt: str = "",
        num_inference_steps: int = 50,
        return_all_views: bool = False,
    ) -> list:
        """Generate B videos in one forward pass.

        Args:
            agentview_pil: PIL 320×192 agentview frame (same for all).
            wrist_pil: PIL 320×192 wrist frame (same for all).
            actions_batch: (B, 16, 10) numpy.
            prompt: task description.
            num_inference_steps: diffusion steps.

        Returns:
            List of B lists, each containing 16 PIL Images at 192×192.
        """
        import einops
        B = actions_batch.shape[0]

        # Convert all B action sequences: 10D → 7D → normalize
        actions_7d_list = []
        for i in range(B):
            a7 = self._convert_10d_to_7d(actions_batch[i])  # (16, 7)
            a7_full = np.concatenate([a7[:1], a7], axis=0)   # (17, 7)
            a7_norm = _normalize_bound(a7_full, self._state_p01, self._state_p99)
            actions_7d_list.append(a7_norm)
        actions_t = torch.from_numpy(
            np.stack(actions_7d_list)
        ).to(self._device, dtype=self._dtype)  # (B, 17, 7)

        with torch.no_grad():
            # Batched action encoding
            action_hidden = self._model.action_encoder(
                actions_t,
                texts=[prompt] * B if prompt else None,
                text_tokinizer=self._model.tokenizer,
                text_encoder=self._model.text_encoder,
                frame_level_cond=True,
            )  # (B, 17, 1024)

            # Encode frame once, repeat B times
            lat_agent = self._encode_frame_to_latent(agentview_pil)  # (1, 4, 24, 40)
            lat_zeros = torch.zeros_like(lat_agent)
            lat_wrist = self._encode_frame_to_latent(wrist_pil)
            cond_latent = torch.cat([lat_agent, lat_zeros, lat_wrist], dim=2)  # (1, 4, 72, 40)
            cond_latent = cond_latent.repeat(B, 1, 1, 1)  # (B, 4, 72, 40)

            his_cond = cond_latent.unsqueeze(1)  # (B, 1, 4, 72, 40)

            from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                self._model.pipeline,
                image=cond_latent,
                text=action_hidden,
                width=self._args.width,
                height=self._args.height * 3,
                num_frames=self._args.num_frames,
                history=his_cond,
                num_inference_steps=num_inference_steps,
                decode_chunk_size=8,
                max_guidance_scale=1.0,
                fps=7,
                motion_bucket_id=127,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
                his_cond_zero=False,
            )

        # latents: (B, 16, 4, 72, 40) → split 3 views
        latents = einops.rearrange(latents, 'b f c (m h) w -> (b m) f c h w', m=3)
        # latents[0::3] = agentview for each batch item

        def _decode_view_latents(view_lat):
            frames = []
            for t in range(view_lat.shape[1]):
                lat = view_lat[:, t] / self._model.vae.config.scaling_factor
                decoded = self._model.vae.decode(lat, num_frames=1).sample
                frame = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255)
                frame = frame[0].permute(1, 2, 0).to(torch.float32).cpu().numpy().astype(np.uint8)
                frames.append(center_crop_square(Image.fromarray(frame)))
            return frames

        results = []
        wrist_results = [] if return_all_views else None
        with torch.no_grad():
            for b in range(B):
                results.append(_decode_view_latents(latents[b * 3: b * 3 + 1]))
                if return_all_views:
                    wrist_results.append(_decode_view_latents(latents[b * 3 + 2: b * 3 + 3]))

        if return_all_views:
            return results, wrist_results
        return results
