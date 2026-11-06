"""LIBERO closed-loop action selection: OpenPI candidates → Track WM → VLAC.

Used by ``evaluate_full_track.py``. Does not use real-world StateToTracks;
mesh / wrist track inputs already come from the LIBERO mesh env via
``_prepare_raw_item``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACK_WM_CHECKPOINT = str(
    REPO_ROOT / "checkpoints" / "track_dualview" / "hybrid-checkpoint-128000"
)
DEFAULT_SVD_PATH = str(REPO_ROOT / "checkpoints" / "svd")
DEFAULT_VLAC_MODEL = str(REPO_ROOT / "checkpoints" / "VLAC-8b")
DEFAULT_OPENPI_CHECKPOINT = str(
    REPO_ROOT / "checkpoints" / "pi05_libero_stage2p5_joint" / "pi05_final" / "29999"
)

# Empirically validated candidate-diversity defaults.
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
class LiberoActionSelector:
    """Sample N policy candidates, imagine with Track DualView WM, pick via VLAC."""

    config_name: str = "pi05_libero_stage2p5_joint"
    checkpoint_dir: str = DEFAULT_OPENPI_CHECKPOINT
    track_wm_checkpoint: str = DEFAULT_TRACK_WM_CHECKPOINT
    svd_path: str = DEFAULT_SVD_PATH
    vlac_model: str = DEFAULT_VLAC_MODEL
    n_candidates: int = 20
    wm_per_view_size: int = 192
    wm_num_frames: int = 16
    wm_num_inference_steps: int = 25
    # Batch size for Track WM diffusion (+ VLAC scoring). 1 = sequential (slow).
    wm_batch_size: int = 4
    policy_mem_fraction: float = 0.35
    diversity_kwargs: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_DIVERSITY_KWARGS))
    smooth_sigma: float = 2.0
    agent_weight: float = 0.6
    wrist_weight: float = 0.4

    def __post_init__(self) -> None:
        self._policy = None
        self._world_model = None
        self._critic = None
        self._policy_device = "cuda"
        self._wm_device = "cuda"
        self._vlac_device = "cuda"

    def load(
        self,
        *,
        policy_device: str = "cuda",
        wm_device: str = "cuda",
        vlac_device: str = "cuda",
    ) -> None:
        """Load OpenPI + Track DualView WM + VLAC dual-view critic."""
        self._policy_device = policy_device
        self._wm_device = wm_device
        self._vlac_device = vlac_device

        from scripts.vlac_pipeline.openpi_policy import OpenPIPolicy
        from scripts.vlac_pipeline.vlac_dualview_critic import VLACDualViewCritic

        track_dir = str(REPO_ROOT / "third_party" / "track_dualview")
        if track_dir not in sys.path:
            sys.path.insert(0, track_dir)
        from inference import TrackDualViewWM  # noqa: PLC0415

        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(self.policy_mem_fraction))

        logger.info("Loading OpenPI policy for candidate generation...")
        self._policy = OpenPIPolicy(
            config_name=self.config_name,
            checkpoint_dir=self.checkpoint_dir,
            mem_fraction=self.policy_mem_fraction,
        )
        self._policy.load(device=policy_device)

        svd = self.svd_path if Path(self.svd_path).exists() else "stabilityai/stable-video-diffusion-img2vid"
        logger.info(
            "Loading Track DualView WM from %s (per_view=%d, num_frames=%d, svd=%s)...",
            self.track_wm_checkpoint,
            self.wm_per_view_size,
            self.wm_num_frames,
            svd,
        )
        self._world_model = TrackDualViewWM(
            checkpoint_path=self.track_wm_checkpoint,
            svd_path=svd,
            per_view_size=self.wm_per_view_size,
        )
        self._world_model.load(device=wm_device, num_frames=self.wm_num_frames)

        logger.info("Loading VLAC dual-view critic from %s...", self.vlac_model)
        self._critic = VLACDualViewCritic(
            model_path=self.vlac_model,
            agent_weight=self.agent_weight,
            wrist_weight=self.wrist_weight,
        )
        self._critic.load(device=vlac_device)
        logger.info(
            "LiberoActionSelector ready (n_candidates=%d, wm_batch_size=%d, diversity=%s).",
            self.n_candidates,
            self.wm_batch_size,
            self.diversity_kwargs,
        )

    @property
    def ready(self) -> bool:
        return self._policy is not None and self._world_model is not None and self._critic is not None

    def infer_direct(self, raw_item: dict[str, Any]) -> dict[str, Any]:
        """Single policy.infer (no selection) — same semantics as evaluate.py."""
        assert self._policy is not None
        return self._policy.infer(raw_item, return_tracks=True)

    @staticmethod
    def _to_pil(img: np.ndarray) -> Image.Image:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _match_track_horizon(tracks: np.ndarray, num_frames: int) -> np.ndarray:
        """Pad / truncate track predictions to WM horizon."""
        t = np.asarray(tracks, dtype=np.float32)
        if t.ndim == 1:
            t = t.reshape(1, -1)
        if t.shape[0] == num_frames:
            return t
        if t.shape[0] > num_frames:
            return t[:num_frames]
        pad = np.repeat(t[-1:], num_frames - t.shape[0], axis=0)
        return np.concatenate([t, pad], axis=0)

    def select_best_candidate(
        self,
        raw_item: dict[str, Any],
        *,
        agent_img: np.ndarray,
        wrist_img: np.ndarray,
        prompt: str | None = None,
        n_candidates: int | None = None,
    ) -> SelectionResult:
        """Generate candidates, WM-rollout each, VLAC-score, return best actions."""
        if not self.ready:
            raise RuntimeError("LiberoActionSelector.load() must be called first")

        assert self._policy is not None
        assert self._world_model is not None
        assert self._critic is not None

        n = int(n_candidates if n_candidates is not None else self.n_candidates)
        task = prompt if prompt is not None else str(raw_item.get("prompt", ""))

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

        agent_pil = self._to_pil(agent_img)
        wrist_pil = self._to_pil(wrist_img)
        wm_horizon = int(getattr(self._world_model, "num_frames", self.wm_num_frames))
        batch_size = max(1, int(self.wm_batch_size))
        use_batch = batch_size > 1 and hasattr(self._world_model, "batch_generate_video")

        scores: list[float] = [float("-inf")] * len(candidates)
        wm_ms_total = 0.0
        vlac_ms_total = 0.0

        def _score_one(i: int, tracks: np.ndarray) -> None:
            nonlocal wm_ms_total, vlac_ms_total
            t_wm = time.time()
            try:
                agent_frames, wrist_frames = self._world_model.generate_video(
                    agent_pil,
                    wrist_pil,
                    tracks,
                    prompt=task,
                    num_inference_steps=self.wm_num_inference_steps,
                    num_frames=wm_horizon,
                )
            except Exception as exc:  # noqa: BLE001 — keep eval loop alive
                logger.warning("Track WM failed for candidate %d: %s", i, exc)
                return
            wm_ms_total += (time.time() - t_wm) * 1000.0
            t_vlac = time.time()
            try:
                scores[i] = float(self._critic.score_dualview(agent_frames, wrist_frames, task))
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLAC failed for candidate %d: %s", i, exc)
            vlac_ms_total += (time.time() - t_vlac) * 1000.0

        for start in range(0, len(candidates), batch_size if use_batch else 1):
            end = min(start + (batch_size if use_batch else 1), len(candidates))
            idxs = list(range(start, end))
            tracks_list = [
                self._match_track_horizon(candidates[i]["track_predictions"], wm_horizon) for i in idxs
            ]

            if use_batch and len(idxs) > 1:
                tracks_batch = np.stack(tracks_list, axis=0)
                t_wm = time.time()
                try:
                    agent_list, wrist_list = self._world_model.batch_generate_video(
                        agent_pil,
                        wrist_pil,
                        tracks_batch,
                        prompt=task,
                        num_inference_steps=self.wm_num_inference_steps,
                        num_frames=wm_horizon,
                    )
                    wm_ms_total += (time.time() - t_wm) * 1000.0
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Track WM batch failed for candidates %s (%s); falling back to sequential",
                        idxs,
                        exc,
                    )
                    for i, tracks in zip(idxs, tracks_list):
                        _score_one(i, tracks)
                        logger.info(
                            "  Candidate %d: score=%.2f (WM cumulative %.0fms, VLAC cumulative %.0fms)",
                            i,
                            scores[i],
                            wm_ms_total,
                            vlac_ms_total,
                        )
                    continue

                t_vlac = time.time()
                try:
                    batch_scores = self._critic.score_dualview_batch(agent_list, wrist_list, task)
                    for i, sc in zip(idxs, batch_scores):
                        scores[i] = float(sc)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("VLAC batch failed for candidates %s: %s", idxs, exc)
                    for j, i in enumerate(idxs):
                        try:
                            scores[i] = float(
                                self._critic.score_dualview(agent_list[j], wrist_list[j], task)
                            )
                        except Exception as exc_one:  # noqa: BLE001
                            logger.warning("VLAC failed for candidate %d: %s", i, exc_one)
                vlac_ms_total += (time.time() - t_vlac) * 1000.0
                for i in idxs:
                    logger.info(
                        "  Candidate %d: score=%.2f (WM cumulative %.0fms, VLAC cumulative %.0fms)",
                        i,
                        scores[i],
                        wm_ms_total,
                        vlac_ms_total,
                    )
            else:
                i = idxs[0]
                _score_one(i, tracks_list[0])
                logger.info(
                    "  Candidate %d: score=%.2f (WM cumulative %.0fms, VLAC cumulative %.0fms)",
                    i,
                    scores[i],
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
        return SelectionResult(
            actions=np.asarray(best["actions"], dtype=np.float32),
            track_predictions=np.asarray(best.get("track_predictions"), dtype=np.float32)
            if best.get("track_predictions") is not None
            else None,
            best_idx=best_idx,
            scores=scores,
            n_candidates=n,
            policy_ms=policy_ms,
            wm_ms=wm_ms_total,
            vlac_ms=vlac_ms_total,
        )
