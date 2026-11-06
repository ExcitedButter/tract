"""LIBERO closed-loop action selection: OpenPI → Ctrl-World (action WM) → VLAC.

Used by ``evaluate_full_action.py``. Mirrors ``libero_action_selection.py`` but
conditions the world model on policy **actions** (Ctrl-World) instead of tracks.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from .utils import center_crop_to_aspect

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CTRLWORLD_CHECKPOINT = str(
    REPO_ROOT / "checkpoints" / "ctrl_world" / "libero_ctrlworld" / "checkpoint-20000.pt"
)
DEFAULT_SVD_PATH = str(REPO_ROOT / "checkpoints" / "svd")
_LOCAL_CLIP = REPO_ROOT / "checkpoints" / "clip-vit-base-patch32"
DEFAULT_CLIP_PATH = (
    str(_LOCAL_CLIP) if (_LOCAL_CLIP / "config.json").is_file() else "openai/clip-vit-base-patch32"
)
DEFAULT_VLAC_MODEL = str(REPO_ROOT / "checkpoints" / "VLAC-8b")
DEFAULT_OPENPI_CHECKPOINT = str(
    REPO_ROOT / "checkpoints" / "pi05_libero_stage2p5_joint" / "pi05_final" / "29999"
)

DEFAULT_DIVERSITY_KWARGS: dict[str, Any] = {
    "sampling_mode": "tsr",
    "tsr_k": 5.0,
    "noise_scale": 3.0,
    "noise_type": "sobol",
}


@dataclass
class SelectionResult:
    actions: np.ndarray
    track_predictions: np.ndarray | None
    best_idx: int
    scores: list[float]
    n_candidates: int
    policy_ms: float
    wm_ms: float
    vlac_ms: float


@dataclass
class LiberoCtrlWorldSelector:
    """Sample N policy candidates, imagine with Ctrl-World, pick via dual-view VLAC."""

    config_name: str = "pi05_libero_stage2p5_joint"
    checkpoint_dir: str = DEFAULT_OPENPI_CHECKPOINT
    ctrlworld_checkpoint: str = DEFAULT_CTRLWORLD_CHECKPOINT
    svd_path: str = DEFAULT_SVD_PATH
    clip_path: str = DEFAULT_CLIP_PATH
    vlac_model: str = DEFAULT_VLAC_MODEL
    n_candidates: int = 20
    wm_num_inference_steps: int = 25
    policy_mem_fraction: float = 0.35
    diversity_kwargs: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_DIVERSITY_KWARGS))
    smooth_sigma: float = 2.0
    agent_weight: float = 0.6
    wrist_weight: float = 0.4

    def __post_init__(self) -> None:
        self._policy = None
        self._world_model = None
        self._critic = None

    def load(
        self,
        *,
        policy_device: str = "cuda",
        wm_device: str = "cuda",
        vlac_device: str = "cuda",
    ) -> None:
        from scripts.vlac_pipeline.ctrl_world_wm import CtrlWorldModel
        from scripts.vlac_pipeline.openpi_policy import OpenPIPolicy
        from scripts.vlac_pipeline.vlac_dualview_critic import VLACDualViewCritic

        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(self.policy_mem_fraction))

        logger.info("Loading OpenPI policy for candidate generation...")
        self._policy = OpenPIPolicy(
            config_name=self.config_name,
            checkpoint_dir=self.checkpoint_dir,
            mem_fraction=self.policy_mem_fraction,
        )
        self._policy.load(device=policy_device)

        logger.info("Loading Ctrl-World (action WM) from %s...", self.ctrlworld_checkpoint)
        wm_kwargs: dict[str, Any] = {
            "checkpoint_path": self.ctrlworld_checkpoint,
            "svd_path": self.svd_path,
        }
        # Only pass a local CLIP dir; HF repo ids are resolved inside CtrlWorldModel
        # when clip_path is left at its default / unset.
        if Path(self.clip_path).is_dir() and (Path(self.clip_path) / "config.json").is_file():
            wm_kwargs["clip_path"] = self.clip_path
        self._world_model = CtrlWorldModel(**wm_kwargs)
        self._world_model.load(device=wm_device)

        logger.info("Loading VLAC dual-view critic from %s...", self.vlac_model)
        self._critic = VLACDualViewCritic(
            model_path=self.vlac_model,
            agent_weight=self.agent_weight,
            wrist_weight=self.wrist_weight,
        )
        self._critic.load(device=vlac_device)
        logger.info(
            "LiberoCtrlWorldSelector ready (n_candidates=%d, diversity=%s).",
            self.n_candidates,
            self.diversity_kwargs,
        )

    @property
    def ready(self) -> bool:
        return self._policy is not None and self._world_model is not None and self._critic is not None

    def infer_direct(self, raw_item: dict[str, Any]) -> dict[str, Any]:
        assert self._policy is not None
        return self._policy.infer(raw_item, return_tracks=True)

    @staticmethod
    def _wm_frames(agent_img: np.ndarray, wrist_img: np.ndarray) -> tuple[Image.Image, Image.Image]:
        """Ctrl-World expects 320×192 per view after center-crop."""
        av = center_crop_to_aspect(np.asarray(agent_img), 320, 192)
        ei = center_crop_to_aspect(np.asarray(wrist_img), 320, 192)
        return Image.fromarray(av.astype(np.uint8)), Image.fromarray(ei.astype(np.uint8))

    def select_best_candidate(
        self,
        raw_item: dict[str, Any],
        *,
        agent_img: np.ndarray,
        wrist_img: np.ndarray,
        prompt: str | None = None,
        n_candidates: int | None = None,
    ) -> SelectionResult:
        if not self.ready:
            raise RuntimeError("LiberoCtrlWorldSelector.load() must be called first")

        assert self._policy is not None
        assert self._world_model is not None
        assert self._critic is not None

        n = int(n_candidates if n_candidates is not None else self.n_candidates)
        task = prompt if prompt is not None else str(raw_item.get("prompt", ""))
        initial_state = np.asarray(raw_item.get("state"), dtype=np.float32)

        t0 = time.time()
        candidates = self._policy.generate_candidates(
            raw_item, n_candidates=n, **self.diversity_kwargs
        )
        if self.smooth_sigma > 0:
            for c in candidates:
                actions = np.asarray(c["actions"], dtype=np.float32)
                for d in range(actions.shape[1]):
                    actions[:, d] = gaussian_filter1d(actions[:, d], sigma=self.smooth_sigma)
                c["actions"] = actions
        policy_ms = (time.time() - t0) * 1000.0
        logger.info(
            "Generated %d candidates in %.0fms (diversity=%s, smooth=%.1f)",
            len(candidates),
            policy_ms,
            self.diversity_kwargs,
            self.smooth_sigma,
        )

        wm_av, wm_ei = self._wm_frames(agent_img, wrist_img)
        scores: list[float] = []
        wm_ms_total = 0.0
        vlac_ms_total = 0.0
        for i, cand in enumerate(candidates):
            t_wm = time.time()
            try:
                agent_frames, wrist_frames = self._world_model.generate_video(
                    wm_av,
                    wm_ei,
                    cand["actions"],
                    prompt=task,
                    num_inference_steps=self.wm_num_inference_steps,
                    return_all_views=True,
                    initial_state=initial_state,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ctrl-World failed for candidate %d: %s", i, exc, exc_info=True)
                scores.append(float("-inf"))
                continue
            wm_ms_total += (time.time() - t_wm) * 1000.0

            t_vlac = time.time()
            try:
                score = float(self._critic.score_dualview(agent_frames, wrist_frames, task))
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLAC failed for candidate %d: %s", i, exc, exc_info=True)
                score = float("-inf")
            vlac_ms_total += (time.time() - t_vlac) * 1000.0
            scores.append(score)
            logger.info(
                "  Candidate %d: score=%.2f (WM cumulative %.0fms, VLAC cumulative %.0fms)",
                i,
                score,
                wm_ms_total,
                vlac_ms_total,
            )

        if not scores or not np.isfinite(scores).any():
            logger.warning("All candidates failed scoring; falling back to candidate 0")
            best_idx = 0
        else:
            best_idx = int(np.nanargmax(np.asarray(scores, dtype=np.float64)))

        best = candidates[best_idx]
        logger.info(
            "Best candidate: %d (score=%.2f) policy=%.0fms wm=%.0fms vlac=%.0fms",
            best_idx,
            scores[best_idx] if scores else float("nan"),
            policy_ms,
            wm_ms_total,
            vlac_ms_total,
        )
        tracks = best.get("track_predictions")
        return SelectionResult(
            actions=np.asarray(best["actions"], dtype=np.float32),
            track_predictions=np.asarray(tracks, dtype=np.float32) if tracks is not None else None,
            best_idx=best_idx,
            scores=scores,
            n_candidates=n,
            policy_ms=policy_ms,
            wm_ms=wm_ms_total,
            vlac_ms=vlac_ms_total,
        )
