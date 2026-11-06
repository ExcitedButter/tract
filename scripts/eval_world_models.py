#!/usr/bin/env python3
"""Evaluate Ctrl-World vs Track-DualView on 16-step horizons across replan windows.

Default: one task from ``libero_goal_task`` (integral slot 05: report Task 2, ``task_index=1``,
language *put the plate on the stove* / stem ``put_the_bowl_on_the_stove``). Override with
``--suites`` / ``--suite`` / ``--task-index``, or restore the four base suites via
``--suites libero_10_task libero_goal libero_object libero_spatial``.

Each suite prepares ``--num-horizon-windows`` (default 5) windows per episode: 16 predicted frames
per window, aligned with evaluate2 replans at ``0, replan_step_size, 2*replan_step_size, ...``.

Prepare data (pick one):
  1) evaluate2 export under ``1/simulated/episodes/<suite>/`` with
     ``actions.npy``, ``tracks.npy``, ``images_agentview.npy``, ``images_wrist.npy``
  2) LeRobot parquet (``libero_cotracker_densegrid_merged``, has ``track_targets_raw``), e.g.
     ``.../episode_000401.parquet`` for libero_goal_task task 1 (put the bowl on the stove demo)

Per window at env step ``t`` (replan index ``r``):
  - init image/state from frame ``t``
  - Ctrl-World: actions[t:t+16]
  - Track-DV: tracks[r] replan plan (16 x 39 x 2)
  - GT: frames [t+1:t+17]

Canonical 192x192 videos under ``video/simulated/<suite>/*_h00..h04/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

OPENPI_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = OPENPI_ROOT / "1"
INPUT_SIM_ROOT = INPUT_ROOT / "simulated" / "episodes"
VIDEO_ROOT = OPENPI_ROOT / "video"
SIM_ROOT = VIDEO_ROOT / "simulated"
RESULTS_PATH = VIDEO_ROOT / "results.txt"

DEFAULT_SUITES = ("libero_goal_task",)
DEFAULT_SUITE = "libero_goal_task"
# pi05_final report Task 2; same stem as libero_goal task_index 1 (put_the_bowl_on_the_stove).
DEFAULT_TASK_INDEX = 1
DEFAULT_ROLLOUT_INDEX = 0
DEFAULT_NUM_HORIZON_WINDOWS = 5
DEFAULT_REPLAN_STEP_SIZE = 5
LEROBOT_TRACK_DATASET = "libero_cotracker_densegrid_merged"

# (suite, task_index) -> LeRobot episode_index in libero_cotracker_densegrid_merged
SUITE_PARQUET_EPISODE: dict[tuple[str, int], int] = {
    ("libero_10_task", 0): 0,
    ("libero_10_task", 4): 7,
    ("libero_10", 0): 0,
    ("libero_10", 4): 7,
    ("libero_goal", 0): 399,
    ("libero_goal", 1): 401,
    ("libero_goal_task", 0): 399,
    ("libero_goal_task", 1): 401,
    ("libero_object", 0): 813,
    ("libero_object_task", 0): 813,
    ("libero_spatial", 0): 0,
    ("libero_spatial_task", 0): 0,
}

# suite alias -> canonical key in SUITE_PARQUET_EPISODE
SUITE_ALIASES: dict[str, str] = {
    "libero_goal_task": "libero_goal",
    "libero_object_task": "libero_object",
    "libero_spatial_task": "libero_spatial",
    "libero_10": "libero_10_task",
}


def _canonical_suite_name(suite: str) -> str:
    return SUITE_ALIASES.get(suite, suite)


def resolve_parquet_path(*, suite: str, task_index: int) -> Path:
    """Parquet with CoTracker track_targets_raw for the given suite task index."""
    suite_key = _canonical_suite_name(suite)
    episode_index = SUITE_PARQUET_EPISODE.get(
        (suite, task_index),
        SUITE_PARQUET_EPISODE.get((suite_key, task_index), task_index),
    )
    return (
        OPENPI_ROOT
        / "lerobot_cache"
        / "physical-intelligence"
        / LEROBOT_TRACK_DATASET
        / "data"
        / "chunk-000"
        / f"episode_{episode_index:06d}.parquet"
    )


DEFAULT_PARQUET_PATH = resolve_parquet_path(
    suite=DEFAULT_SUITE, task_index=DEFAULT_TASK_INDEX
)

MODELS = (
    ("ctrlworld", "ctrlworld_pred_{view}.mp4"),
    ("trackdv", "trackdv_pred_{view}.mp4"),
)
VIEWS = ("agentview", "wrist")
GT_PATTERN = "gt_{view}.mp4"


@dataclass
class ViewMetrics:
    psnr: float
    ssim: float
    lpips: float
    fid: float
    fvd: float


def load_video(path: Path, max_frames: int | None = None) -> np.ndarray:
    """Load video as (T, H, W, 3) RGB uint8."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"No frames in {path}")
    return np.stack(frames, axis=0)


def save_video(path: Path, frames: np.ndarray, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(frames.shape[2]), int(frames.shape[1])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {path}")
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


def center_crop_to_aspect(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop to target aspect ratio, then resize."""
    h, w = frame.shape[:2]
    target_aspect = float(target_w) / float(target_h)
    current_aspect = float(w) / float(h)
    if current_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = (w - new_w) // 2
        cropped = frame[:, x0 : x0 + new_w]
    else:
        new_h = int(round(w / target_aspect))
        y0 = (h - new_h) // 2
        cropped = frame[y0 : y0 + new_h, :]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def resize_with_pad(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize preserving all content; pad with black bars if needed."""
    img = np.asarray(frame, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def canonicalize_video_192(frames: np.ndarray, size: int = 192) -> np.ndarray:
    """Convert frames to canonical 192x192 with no content cropping."""
    out = []
    for f in frames:
        out.append(resize_with_pad(np.asarray(f, dtype=np.uint8), size, size))
    return np.stack(out, axis=0)


def _copy_as_canonical(src: Path, dst: Path, eval_size: int, fps: int) -> None:
    video = load_video(src)
    can = canonicalize_video_192(video, size=eval_size)
    save_video(dst, can, fps=fps)


def _save_canonical_from_npy(npy_path: Path, out_path: Path, eval_size: int, fps: int) -> None:
    arr = np.load(npy_path)
    if arr.ndim != 4:
        raise ValueError(f"Expected (T,H,W,3) in {npy_path}, got {arr.shape}")
    can = canonicalize_video_192(arr.astype(np.uint8), size=eval_size)
    save_video(out_path, can, fps=fps)


def align_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def compute_psnr_ssim(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    psnrs, ssims = [], []
    for p, g in zip(pred, gt):
        psnrs.append(peak_signal_noise_ratio(g, p, data_range=255))
        ssims.append(structural_similarity(g, p, channel_axis=2, data_range=255))
    return float(np.mean(psnrs)), float(np.mean(ssims))


class LPIPSMetric:
    def __init__(self, device: str):
        import lpips

        self.device = device
        self.model = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def mean_distance(self, pred: np.ndarray, gt: np.ndarray) -> float:
        dists = []
        for p, g in zip(pred, gt):
            pt = torch.from_numpy(p).permute(2, 0, 1).float() / 127.5 - 1.0
            gt_t = torch.from_numpy(g).permute(2, 0, 1).float() / 127.5 - 1.0
            pt = pt.unsqueeze(0).to(self.device)
            gt_t = gt_t.unsqueeze(0).to(self.device)
            dists.append(float(self.model(pt, gt_t).item()))
        return float(np.mean(dists))


def _frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    diff = mu1 - mu2
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


class InceptionFID:
    """FID via Inception-v3 features (pytorch_fid)."""

    def __init__(self, device: str):
        from pytorch_fid.inception import InceptionV3

        self.device = device
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(device).eval()

    @torch.no_grad()
    def _features(self, frames: np.ndarray) -> np.ndarray:
        feats = []
        for img in frames:
            x = torch.from_numpy(img).permute(2, 0, 1).float()
            x = x.unsqueeze(0).to(self.device)
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = (x - 127.5) / 128.0
            out = self.model(x)[0]
            if out.size(2) != 1 or out.size(3) != 1:
                out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
            feats.append(out.squeeze().cpu().numpy())
        return np.stack(feats, axis=0)

    def fid(self, pred: np.ndarray, gt: np.ndarray) -> float:
        f1 = self._features(pred)
        f2 = self._features(gt)
        mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0)
        sigma1 = np.cov(f1, rowvar=False)
        sigma2 = np.cov(f2, rowvar=False)
        return _frechet_distance(mu1, sigma1, mu2, sigma2)


class I3DFVD:
    """FVD using I3D (Kinetics) features via torch hub."""

    def __init__(self, device: str):
        self.device = device
        self.model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True).to(device).eval()

    @torch.no_grad()
    def _video_features(self, frames: np.ndarray) -> np.ndarray:
        clip = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        clip = clip.unsqueeze(0).to(self.device)
        clip = torch.nn.functional.interpolate(
            clip, size=(frames.shape[0], 224, 224), mode="trilinear", align_corners=False
        )
        out = self.model(clip)
        if isinstance(out, dict):
            out = out.get("logits", next(iter(out.values())))
        return out.reshape(-1).cpu().numpy()

    def fvd(self, pred: np.ndarray, gt: np.ndarray) -> float:
        augments = (pred, pred[::-1].copy(), pred[::2], pred[1::2])
        pred_feats = np.stack([self._video_features(c) for c in augments if len(c) >= 8], axis=0)
        gt_feats = np.stack([self._video_features(c) for c in (gt, gt[::-1].copy(), gt[::2], gt[1::2]) if len(c) >= 8], axis=0)
        mu1, mu2 = pred_feats.mean(0), gt_feats.mean(0)
        sigma1 = np.cov(pred_feats, rowvar=False)
        sigma2 = np.cov(gt_feats, rowvar=False)
        if sigma1.ndim == 0:
            sigma1 = np.array([[float(sigma1)]])
        if sigma2.ndim == 0:
            sigma2 = np.array([[float(sigma2)]])
        return _frechet_distance(mu1, sigma1, mu2, sigma2)


def resolve_pred_path(folder: Path, model: str, view: str) -> Path | None:
    path = folder / dict(MODELS)[model].format(view=view)
    return path if path.is_file() else None


def discover_eval_folders(sim_root: Path, *, suite: str | None = None) -> list[Path]:
    """Find folders containing GT + Ctrl-World + Track evaluation videos."""
    suite_key = suite
    folders: list[Path] = []
    for p in sorted(sim_root.rglob("*")):
        if not p.is_dir():
            continue
        if suite_key is not None:
            try:
                rel = p.relative_to(sim_root)
            except ValueError:
                continue
            if not rel.parts or rel.parts[0] != suite_key:
                continue
        has_eval_videos = all((p / GT_PATTERN.format(view=v)).is_file() for v in VIEWS) and all(
            (p / dict(MODELS)[m].format(view=v)).is_file() for m, _ in MODELS for v in VIEWS
        )
        if has_eval_videos:
            folders.append(p)
    return folders


def _load_episode_inputs(epi: Path) -> dict[str, np.ndarray]:
    actions = np.load(epi / "actions.npy")
    tracks = np.load(epi / "tracks.npy")
    av = np.load(epi / "images_agentview.npy")
    wi = np.load(epi / "images_wrist.npy")
    state_path = epi / "states.npy"
    if state_path.is_file():
        states = np.load(state_path)
    else:
        states = np.zeros((len(actions) + 1, 8), dtype=np.float32)
    return {
        "images_agentview": av,
        "images_wrist": wi,
        "actions": actions,
        "states": states,
        "tracks": tracks,
    }


def discover_evaluate2_bundle_dir(
    suite: str,
    task_index: int,
    rollout_index: int,
) -> Path | None:
    """Find evaluate2 metrics export: .../1/simulated/episodes/<suite>/*_task_rollout_*/."""
    suite_dir = INPUT_SIM_ROOT / suite
    if not suite_dir.is_dir():
        return None
    needle = f"_{task_index:02d}_{rollout_index:02d}_"
    candidates: list[Path] = []
    for d in suite_dir.iterdir():
        if not d.is_dir() or needle not in d.name:
            continue
        if (d / "actions.npy").is_file() and (d / "tracks.npy").is_file():
            candidates.append(d)
    if not candidates:
        return None
    return sorted(candidates)[0]


def _save_metrics_videos(
    out_dir: Path,
    *,
    gt_av: np.ndarray,
    gt_wi: np.ndarray,
    cw_av: np.ndarray,
    cw_wi: np.ndarray,
    td_av: np.ndarray,
    td_wi: np.ndarray,
    fps: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_video(out_dir / "gt_agentview.mp4", gt_av, fps=fps)
    save_video(out_dir / "gt_wrist.mp4", gt_wi, fps=fps)
    save_video(out_dir / "ctrlworld_pred_agentview.mp4", cw_av, fps=fps)
    save_video(out_dir / "ctrlworld_pred_wrist.mp4", cw_wi, fps=fps)
    save_video(out_dir / "trackdv_pred_agentview.mp4", td_av, fps=fps)
    save_video(out_dir / "trackdv_pred_wrist.mp4", td_wi, fps=fps)


def _to_pil_192(frame: np.ndarray, size: int) -> Image.Image:
    can = canonicalize_video_192(np.asarray([frame], dtype=np.uint8), size=size)[0]
    return Image.fromarray(can)


def _to_pil_ctrlworld(frame: np.ndarray) -> Image.Image:
    """Ctrl-World preferred view size: 320x192 (no crop, keep full content)."""
    # Preserve full 256x256 FoV: resize + pad to 320x192 instead of aspect-cropping.
    img = resize_with_pad(np.asarray(frame, dtype=np.uint8), 320, 192)
    return Image.fromarray(img)


def _decode_parquet_image(cell) -> np.ndarray:
    if isinstance(cell, dict) and "bytes" in cell:
        raw = cell["bytes"]
    elif isinstance(cell, (bytes, bytearray, memoryview)):
        raw = bytes(cell)
    else:
        arr = np.asarray(cell)
        if arr.ndim == 3 and arr.dtype == np.uint8:
            return arr
        raise TypeError(f"Unsupported image cell type: {type(cell)}")
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode parquet image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _track_targets_raw_to_2d(cell) -> np.ndarray:
    """Convert parquet track_targets_raw cell into (H,78) float32."""
    if hasattr(cell, "as_py"):
        cell = cell.as_py()

    # First try direct conversion (fast path for already numeric arrays).
    try:
        arr = np.asarray(cell, dtype=np.float32)
    except Exception:
        arr = None

    if arr is not None and arr.dtype != object:
        if arr.ndim == 3:  # (H,39,2)
            return arr.reshape(arr.shape[0], -1)
        if arr.ndim == 2 and arr.shape[-1] == 2:  # (39,2)
            return arr.reshape(1, -1)
        if arr.ndim == 2:  # (H,78)
            return arr
        if arr.ndim == 1 and arr.size % 78 == 0:
            return arr.reshape(arr.size // 78, 78)

    # Slow path for nested / ragged structures.
    def _is_seq(x) -> bool:
        return isinstance(x, (list, tuple, np.ndarray))

    def _flatten_numeric(x) -> np.ndarray:
        """Recursively flatten nested numeric leaves into 1D float32."""
        if _is_seq(x):
            parts = [_flatten_numeric(v) for v in x]
            if not parts:
                return np.zeros((0,), dtype=np.float32)
            return np.concatenate(parts, axis=0)
        # Scalar leaf
        return np.array([x], dtype=np.float32)

    def _to_rows(x) -> list[np.ndarray]:
        """
        Recursively split into rows of length 78.
        If x is a sequence whose each element can be a 78-row, treat as multiple rows.
        Otherwise flatten x as a single row.
        """
        if _is_seq(x):
            # Try interpreting children as separate rows.
            child_rows: list[np.ndarray] = []
            ok_children = True
            for c in x:
                flat_c = _flatten_numeric(c)
                if flat_c.size == 78:
                    child_rows.append(flat_c)
                elif flat_c.size == 0:
                    continue
                else:
                    ok_children = False
                    break
            if ok_children and child_rows:
                return child_rows

        flat = _flatten_numeric(x)
        if flat.size == 0:
            return []
        if flat.size % 78 != 0:
            raise ValueError(f"Unexpected flattened track_targets_raw size: {flat.size}")
        return [flat[i : i + 78] for i in range(0, flat.size, 78)]

    rows = _to_rows(cell)
    if not rows:
        raise ValueError("Empty track_targets_raw cell")
    return np.stack(rows, axis=0).astype(np.float32)


def _horizon_window_specs(
    num_windows: int,
    replan_step_size: int,
) -> list[tuple[int, int]]:
    """Return (env_start_step, replan_idx) for each 16-step prediction window."""
    return [(i * replan_step_size, i) for i in range(num_windows)]


def _horizon_output_dir(base_name: str, replan_idx: int) -> str:
    return f"{base_name}_h{replan_idx:02d}"


def _load_models_for_generation(args: argparse.Namespace):
    # Lazy imports keep base metrics-only mode lightweight.
    sys.path.insert(0, str(OPENPI_ROOT))
    sys.path.insert(0, str(OPENPI_ROOT / "third_party"))
    from scripts.vlac_pipeline.ctrl_world_wm import CtrlWorldModel
    from track_dualview.inference import TrackDualViewWM

    ctrl = CtrlWorldModel()
    ctrl.load(device=args.device)

    track = TrackDualViewWM(per_view_size=args.eval_size)
    track.load(device=args.device, num_frames=args.horizon)
    return ctrl, track


def _generate_predictions_from_inputs(
    ctrl_model,
    track_model,
    inputs: dict[str, np.ndarray],
    *,
    eval_size: int,
    horizon: int,
    start_step: int = 0,
    replan_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    av = np.asarray(inputs["images_agentview"], dtype=np.uint8)
    wi = np.asarray(inputs["images_wrist"], dtype=np.uint8)
    actions = np.asarray(inputs["actions"], dtype=np.float32)
    states = np.asarray(inputs["states"], dtype=np.float32)
    tracks = np.asarray(inputs["tracks"], dtype=np.float32)

    end_step = start_step + horizon
    if len(av) <= end_step or len(wi) <= end_step:
        raise ValueError(
            f"Need >= {end_step + 1} image frames for start_step={start_step}, "
            f"horizon={horizon}; got av={len(av)}, wi={len(wi)}"
        )
    if len(actions) < end_step:
        raise ValueError(
            f"Need >= {end_step} actions for start_step={start_step}, "
            f"horizon={horizon}; got {len(actions)}"
        )
    if len(states) <= start_step:
        raise ValueError(
            f"Need state at start_step={start_step}; got {len(states)} states"
        )
    if tracks.ndim < 2 or replan_idx >= tracks.shape[0]:
        raise ValueError(
            f"Need tracks.npy replan index {replan_idx}; got shape {tracks.shape}"
        )

    init_av = _to_pil_192(av[start_step], eval_size)
    init_wi = _to_pil_192(wi[start_step], eval_size)

    # GT next-horizon frames (compare prediction at t+1..t+horizon).
    gt_av = canonicalize_video_192(av[start_step + 1 : end_step + 1], size=eval_size)
    gt_wi = canonicalize_video_192(wi[start_step + 1 : end_step + 1], size=eval_size)

    act_h = actions[start_step:end_step]
    st0 = states[start_step]
    cw_av_pil, cw_wi_pil = ctrl_model.generate_video(
        init_av,
        init_wi,
        act_h,
        prompt="",
        num_inference_steps=25,
        return_all_views=True,
        initial_state=st0,
    )
    cw_av = canonicalize_video_192(np.stack([np.asarray(f.convert("RGB")) for f in cw_av_pil], axis=0), size=eval_size)
    cw_wi = canonicalize_video_192(np.stack([np.asarray(f.convert("RGB")) for f in cw_wi_pil], axis=0), size=eval_size)

    # evaluate2 stores per-replan tracks as (R,16,39,2)/(R,16,78).
    tr0 = tracks[replan_idx]
    if tr0.ndim == 3:
        tr0 = tr0.reshape(tr0.shape[0], -1)
    tr0 = np.asarray(tr0[:horizon], dtype=np.float32)
    td_av_pil, td_wi_pil = track_model.generate_video(
        init_av,
        init_wi,
        tr0,
        prompt="",
        num_inference_steps=8,
        num_frames=horizon,
    )
    td_av = canonicalize_video_192(np.stack([np.asarray(f.convert("RGB")) for f in td_av_pil], axis=0), size=eval_size)
    td_wi = canonicalize_video_192(np.stack([np.asarray(f.convert("RGB")) for f in td_wi_pil], axis=0), size=eval_size)
    return gt_av, gt_wi, cw_av, cw_wi, td_av, td_wi


def build_video_outputs_from_metrics_bundle(
    args: argparse.Namespace,
    episode_dir: Path,
) -> list[Path]:
    """Build canonical WM eval videos for each replan window (default 5 x 16-step)."""
    inputs = _load_episode_inputs(episode_dir)
    ctrl_model, track_model = _load_models_for_generation(args)
    out_dirs: list[Path] = []
    for start_step, replan_idx in _horizon_window_specs(
        args.num_horizon_windows, args.replan_step_size
    ):
        try:
            gt_av, gt_wi, cw_av, cw_wi, td_av, td_wi = _generate_predictions_from_inputs(
                ctrl_model,
                track_model,
                inputs,
                eval_size=args.eval_size,
                horizon=args.horizon,
                start_step=start_step,
                replan_idx=replan_idx,
            )
        except ValueError as exc:
            print(
                f"  skip horizon h{replan_idx:02d} "
                f"(start_step={start_step}): {exc}"
            )
            continue
        out_dir = SIM_ROOT / args.suite / _horizon_output_dir(episode_dir.name, replan_idx)
        _save_metrics_videos(
            out_dir,
            gt_av=gt_av,
            gt_wi=gt_wi,
            cw_av=cw_av,
            cw_wi=cw_wi,
            td_av=td_av,
            td_wi=td_wi,
            fps=args.fps,
        )
        out_dirs.append(out_dir)
        print(
            f"  saved horizon window h{replan_idx:02d} "
            f"(start_step={start_step}) -> {out_dir}"
        )
    if not out_dirs:
        raise ValueError(f"No horizon windows generated for {episode_dir}")
    return out_dirs


def build_video_outputs_from_parquet(args: argparse.Namespace) -> list[Path]:
    """Generate ``num_horizon_windows`` 16-step windows from parquet replan offsets."""
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("pandas is required to read parquet; run via `uv run python metrics.py`") from exc

    parquet_path = Path(args.parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    df = pd.read_parquet(parquet_path)
    if len(df) < 2:
        raise ValueError(f"Parquet has too few rows: {len(df)}")

    horizon = int(args.horizon)
    ctrl_model, track_model = _load_models_for_generation(args)
    base_name = f"parquet_{parquet_path.stem}"
    n_rows = len(df)
    max_start = (args.num_horizon_windows - 1) * args.replan_step_size
    min_rows = max_start + horizon + 1
    if n_rows < min_rows:
        raise ValueError(
            f"Need at least {min_rows} parquet rows for "
            f"{args.num_horizon_windows} windows (horizon={horizon}, "
            f"replan_step_size={args.replan_step_size}), got {n_rows}"
        )

    out_dirs: list[Path] = []
    for start_step, replan_idx in _horizon_window_specs(
        args.num_horizon_windows, args.replan_step_size
    ):
        end_step = start_step + horizon
        if end_step >= n_rows:
            print(
                f"  skip horizon h{replan_idx:02d} "
                f"(start_step={start_step}): need row {end_step}, got {n_rows}"
            )
            continue
        av0 = _decode_parquet_image(df["image"].iloc[start_step])
        wi0 = _decode_parquet_image(df["wrist_image"].iloc[start_step])
        init_av_cw = _to_pil_ctrlworld(av0)
        init_wi_cw = _to_pil_ctrlworld(wi0)
        init_av_td = _to_pil_192(av0, args.eval_size)
        init_wi_td = _to_pil_192(wi0, args.eval_size)

        gt_av = np.stack(
            [_decode_parquet_image(df["image"].iloc[t]) for t in range(start_step + 1, end_step + 1)],
            axis=0,
        )
        gt_wi = np.stack(
            [_decode_parquet_image(df["wrist_image"].iloc[t]) for t in range(start_step + 1, end_step + 1)],
            axis=0,
        )
        gt_av = canonicalize_video_192(gt_av, size=args.eval_size)
        gt_wi = canonicalize_video_192(gt_wi, size=args.eval_size)

        states = np.asarray(df["state"].iloc[start_step], dtype=np.float32)
        act = np.stack(
            [np.asarray(df["actions"].iloc[i], dtype=np.float32) for i in range(start_step, end_step)],
            axis=0,
        )
        cw_av_pil, cw_wi_pil = ctrl_model.generate_video(
            init_av_cw,
            init_wi_cw,
            act,
            prompt="",
            num_inference_steps=25,
            return_all_views=True,
            initial_state=states,
        )
        cw_av = canonicalize_video_192(
            np.stack([np.asarray(f.convert("RGB")) for f in cw_av_pil[:horizon]], axis=0),
            size=args.eval_size,
        )
        cw_wi = canonicalize_video_192(
            np.stack([np.asarray(f.convert("RGB")) for f in cw_wi_pil[:horizon]], axis=0),
            size=args.eval_size,
        )

        tr = _track_targets_raw_to_2d(df["track_targets_raw"].iloc[start_step])
        if tr.shape[0] < horizon:
            tr = np.concatenate([tr, np.repeat(tr[-1:], horizon - tr.shape[0], axis=0)], axis=0)
        td_av_pil, td_wi_pil = track_model.generate_video(
            init_av_td,
            init_wi_td,
            tr[:horizon],
            prompt="",
            num_inference_steps=8,
            num_frames=horizon,
        )
        td_av = canonicalize_video_192(
            np.stack([np.asarray(f.convert("RGB")) for f in td_av_pil[:horizon]], axis=0),
            size=args.eval_size,
        )
        td_wi = canonicalize_video_192(
            np.stack([np.asarray(f.convert("RGB")) for f in td_wi_pil[:horizon]], axis=0),
            size=args.eval_size,
        )

        out_dir = SIM_ROOT / args.suite / _horizon_output_dir(base_name, replan_idx)
        _save_metrics_videos(
            out_dir,
            gt_av=gt_av,
            gt_wi=gt_wi,
            cw_av=cw_av,
            cw_wi=cw_wi,
            td_av=td_av,
            td_wi=td_wi,
            fps=args.fps,
        )
        out_dirs.append(out_dir)
        print(
            f"  saved horizon window h{replan_idx:02d} "
            f"(start_step={start_step}) -> {out_dir}"
        )
    if not out_dirs:
        raise ValueError(f"No horizon windows generated for {parquet_path}")
    return out_dirs


def evaluate_folder(
    folder: Path,
    lpips_m: LPIPSMetric,
    fid_m: InceptionFID,
    fvd_m: I3DFVD,
) -> dict:
    out: dict = {
        "folder": str(folder),
        "preprocess": "canonical center-crop-square + resize to 192 before metrics",
        "models": {},
    }

    gt_cache: dict[str, np.ndarray] = {}
    for view in VIEWS:
        gt_path = folder / GT_PATTERN.format(view=view)
        if gt_path.is_file():
            gt_cache[view] = load_video(gt_path)

    for model, _ in MODELS:
        out["models"][model] = {}
        for view in VIEWS:
            if view not in gt_cache:
                continue
            pred_path = resolve_pred_path(folder, model, view)
            if pred_path is None:
                continue
            pred = load_video(pred_path)
            gt = gt_cache[view]
            pred, gt = align_length(pred, gt)
            psnr, ssim = compute_psnr_ssim(pred, gt)
            lp = lpips_m.mean_distance(pred, gt)
            fid = fid_m.fid(pred, gt)
            fvd = fvd_m.fvd(pred, gt)
            out["models"][model][view] = {
                **asdict(ViewMetrics(psnr=psnr, ssim=ssim, lpips=lp, fid=fid, fvd=fvd)),
                "pred_size": {"height": int(pred.shape[1]), "width": int(pred.shape[2])},
                "gt_size": {"height": int(gt.shape[1]), "width": int(gt.shape[2])},
                "num_frames": int(len(pred)),
            }
    return out


METRIC_HEADERS = (
    ("PSNR↑", 8),
    ("SSIM↑", 8),
    ("LPIPS↓", 8),
    ("FID↓", 10),
    ("FVD↓", 10),
)


def summarize_overall(all_results: list[dict]) -> dict[str, dict[str, float]]:
    acc: dict[tuple[str, str, str], list[float]] = {}
    for block in all_results:
        for model, _ in MODELS:
            for view in VIEWS:
                m = block["models"].get(model, {}).get(view)
                if not m:
                    continue
                for key in ("psnr", "ssim", "lpips", "fid", "fvd"):
                    acc.setdefault((model, view, key), []).append(float(m[key]))
    out: dict[str, dict[str, float]] = {}
    for model, _ in MODELS:
        out[model] = {}
        for view in VIEWS:
            for key in ("psnr", "ssim", "lpips", "fid", "fvd"):
                vals = acc.get((model, view, key), [])
                if vals:
                    out[model][f"{view}_{key}"] = float(np.mean(vals))
    return out


def format_results(all_results: list[dict]) -> str:
    lines = [
        "Video evaluation (pred vs GT)",
        "Canonical preprocess: center-crop-square + resize 192x192",
        "=" * 72,
        "",
    ]
    for block in all_results:
        folder = str(Path(block["folder"]).relative_to(VIDEO_ROOT))
        lines.append(f"[{folder}]")
        lines.append("-" * 72)
        hdr_metrics = "".join(f"{name:>{w}}" for name, w in METRIC_HEADERS)
        header = f"{'model':<12} {'view':<12}{hdr_metrics}"
        lines.append(header)
        lines.append("-" * 72)
        for model in ("ctrlworld", "trackdv"):
            for view in VIEWS:
                m = block["models"].get(model, {}).get(view)
                if m is None:
                    continue
                lines.append(
                    f"{model:<12} {view:<12}"
                    f"{m['psnr']:8.4f}{m['ssim']:8.4f}{m['lpips']:8.4f}"
                    f"{m['fid']:10.4f}{m['fvd']:10.4f}"
                )
        lines.append("")
    overall = summarize_overall(all_results)
    lines.append("[overall_mean]")
    lines.append("-" * 72)
    lines.append(f"{'model':<12} {'view':<12}{''.join(f'{n:>{w}}' for n, w in METRIC_HEADERS)}")
    lines.append("-" * 72)
    for model in ("ctrlworld", "trackdv"):
        row = overall.get(model, {})
        for view in VIEWS:
            keys = [f"{view}_{k}" for k in ("psnr", "ssim", "lpips", "fid", "fvd")]
            if not all(k in row for k in keys):
                continue
            lines.append(
                f"{model:<12} {view:<12}"
                f"{row[keys[0]]:8.4f}{row[keys[1]]:8.4f}{row[keys[2]]:8.4f}"
                f"{row[keys[3]]:10.4f}{row[keys[4]]:10.4f}"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_root", type=Path, default=VIDEO_ROOT, help="Output root, default=video")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="Default=video/results.txt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_prepare", action="store_true", help="Skip rebuilding canonical videos from /1 evaluate2 outputs")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--eval_size", type=int, default=192)
    parser.add_argument("--horizon", type=int, default=16, help="Prediction length per window (default 16)")
    parser.add_argument(
        "--num-horizon-windows",
        type=int,
        default=DEFAULT_NUM_HORIZON_WINDOWS,
        help="Number of replan windows per episode (default 5 -> 16*5 frames total)",
    )
    parser.add_argument(
        "--replan-step-size",
        type=int,
        default=DEFAULT_REPLAN_STEP_SIZE,
        help="Env-step offset between windows; matches evaluate2 replan_steps (default 5)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Run only one suite (overrides default four suites)",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=None,
        help=(
            "Run these suites sequentially "
            f"(default: {' '.join(DEFAULT_SUITES)})"
        ),
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=DEFAULT_TASK_INDEX,
        help=(
            "Task index within the suite (default 1 = libero_goal_task report Task 2, "
            "put the plate on the stove)"
        ),
    )
    parser.add_argument(
        "--rollout-index",
        type=int,
        default=DEFAULT_ROLLOUT_INDEX,
        help="Rollout index when loading evaluate2 exports from /1",
    )
    parser.add_argument(
        "--metrics-episode-dir",
        type=Path,
        default=None,
        help="Explicit evaluate2 export dir with actions/tracks/images npy files",
    )
    parser.add_argument(
        "--parquet_path",
        type=Path,
        default=None,
        help="LeRobot parquet fallback (default: libero_cotracker_densegrid_merged for --suite/--task-index)",
    )
    return parser.parse_args()


def _resolve_suites(args: argparse.Namespace) -> list[str]:
    if args.suites is not None:
        return list(args.suites)
    if args.suite is not None:
        return [args.suite]
    return list(DEFAULT_SUITES)


def _output_path_for_suite(base: Path, suite: str) -> Path:
    if base.suffix:
        return base.with_name(f"{base.stem}_{suite}{base.suffix}")
    return base.parent / f"{base.name}_{suite}.txt"


def _prepare_suite(args: argparse.Namespace, suite: str) -> None:
    args.suite = suite
    args.parquet_path = resolve_parquet_path(suite=suite, task_index=args.task_index)
    bundle_dir = args.metrics_episode_dir
    if bundle_dir is None:
        bundle_dir = discover_evaluate2_bundle_dir(
            suite, args.task_index, args.rollout_index
        )
    if bundle_dir is not None and bundle_dir.is_dir():
        print(f"[{suite}] Prepare from evaluate2 bundle: {bundle_dir}")
        build_video_outputs_from_metrics_bundle(args, bundle_dir)
    else:
        print(f"[{suite}] Prepare from parquet: {args.parquet_path}")
        build_video_outputs_from_parquet(args)


def _evaluate_suite(
    args: argparse.Namespace,
    suite: str,
    lpips_m: LPIPSMetric,
    fid_m: InceptionFID,
    fvd_m: I3DFVD,
) -> list[dict]:
    folders = discover_eval_folders(SIM_ROOT, suite=suite)
    if not folders:
        raise FileNotFoundError(
            f"No evaluable episode folders for suite={suite} under: {SIM_ROOT}"
        )
    print(f"[{suite}] Found {len(folders)} eval folder(s) (expect up to 5 horizon windows each)")
    if folders:
        print(f"[{suite}] Example: {folders[0]}")
    results: list[dict] = []
    for folder in folders:
        print(f"[{suite}] Evaluating {folder} ...")
        results.append(evaluate_folder(folder, lpips_m, fid_m, fvd_m))
    return results


def main() -> None:
    args = parse_args()
    global VIDEO_ROOT, SIM_ROOT
    VIDEO_ROOT = args.video_root
    SIM_ROOT = VIDEO_ROOT / "simulated"

    suites = _resolve_suites(args)
    print(f"Suites: {', '.join(suites)}")

    if not args.no_prepare:
        for suite in suites:
            print(f"\n--- prepare {suite} ---")
            try:
                _prepare_suite(args, suite)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[{suite}] SKIP prepare: {exc}")
        print(f"\nPrepared canonical videos under {SIM_ROOT}")

    lpips_m = LPIPSMetric(args.device)
    fid_m = InceptionFID(args.device)
    fvd_m = I3DFVD(args.device)

    if not SIM_ROOT.is_dir():
        raise FileNotFoundError(f"Missing folder: {SIM_ROOT}")

    for suite in suites:
        print("\n" + "=" * 72)
        print(f"SUITE: {suite}")
        print("=" * 72)
        try:
            all_results = _evaluate_suite(args, suite, lpips_m, fid_m, fvd_m)
        except FileNotFoundError as exc:
            print(f"[{suite}] SKIP evaluate: {exc}")
            continue
        text = format_results(all_results)
        out_path = _output_path_for_suite(args.output, suite)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        out_path.with_suffix(".json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        print(text)
        print(f"Saved: {out_path}")
        print(f"Saved: {out_path.with_suffix('.json')}")

    print(f"\nCanonical videos under: {SIM_ROOT}")


if __name__ == "__main__":
    main()
