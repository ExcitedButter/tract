#!/usr/bin/env python3
"""Single-frame dual-view track WM inference (This-and-That SVD + ControlNet).

Uses trained UNet+ControlNet (hybrid-checkpoint-128000) + frozen SVD VAE/encoder.
Track **eval gripper keypoints** (7 agent + 7 wrist); default loads tracks + RGB from LeRobot parquet.

Example (parquet, default):
    cd third_party/track_dualview
    CUDA_VISIBLE_DEVICES=0 python scripts/infer_single_frame.py \
        --parquet_path lerobot_cache/physical-intelligence/libero_cotracker_densegrid_merged/data/chunk-000/episode_000000.parquet \
        --checkpoint checkpoints/track_dualview/hybrid-checkpoint-128000 \
        --output_dir third_party/track_dualview/results

Example (LIBERO sim nudge):
    CUDA_VISIBLE_DEVICES=0 python scripts/infer_single_frame.py \
        --data_source sim \
        --openpi_root . \
        --libero_viewpoint_task 6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from inference import (  # noqa: E402
    GRIPPER_TRACK_DIM,
    TRACK_TARGETS_RAW_DIM,
    TrackDualViewWM,
)

LOGGER = logging.getLogger(__name__)
_HF_SVD = "stabilityai/stable-video-diffusion-img2vid"
_HF_TEXT = "stabilityai/stable-diffusion-2-1-base"
# Trained track WM (hybrid LIBERO+DROID @ step 128k); not raw SVD-only weights.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "checkpoints/track_dualview/hybrid-checkpoint-128000"
_DEFAULT_PARQUET = str(_REPO_ROOT) + "/" + (
    "lerobot_cache/physical-intelligence/"
    "libero_cotracker_densegrid_merged/data/chunk-000/episode_000000.parquet"
)
_DEFAULT_LEROBOT_ROOT = str(_REPO_ROOT) + "/" + (
    "lerobot_cache/physical-intelligence/"
    "libero_cotracker_densegrid_merged"
)
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "third_party/track_dualview/results"
# LeRobot parquet / Ctrl-World action horizon.
DEFAULT_HORIZON = 16
# Model train/infer resolution: square per view (center-crop input, same-size output).
DEFAULT_VIEW_SIZE = 192

# Match Ctrl-World single_frame left nudge (OSC axis 1, per-step delta).
_CTRLWORLD_LEFT_AXIS = 1
# Sign flipped vs earlier defaults: WM motion was inverted (arm moved, gripper static).
_CTRLWORLD_LEFT_PER_STEP = 0.36
GRIPPER_NUM = 7  # same as hybrid agentview conditioning

# LIBERO mesh tracker ids used in eval / training (gripper keypoints, not full arm).
AGENTVIEW_GRIPPER_VERTEX_IDS = [3001, 3004, 3014, 3067, 3040, 3081, 2898]
WRIST_GRIPPER_VERTEX_IDS = [2955, 3055, 3040, 2981, 2966, 3052, 3022]
# Only fingertip slots move; indices 0-4 stay fixed (wrist/base of gripper in image).
FINGERTIP_PT_IDX = (5, 6)


def _is_svd_repo(path: Path) -> bool:
    return path.is_dir() and (path / "model_index.json").is_file()


def resolve_svd_path(user: str | None) -> str:
    if user:
        p = Path(user).expanduser()
        if _is_svd_repo(p):
            return str(p)
        if "/" not in user and "\\" not in user:
            return user
        raise FileNotFoundError(f"Invalid SVD repo: {p}")
    for local in (_REPO_ROOT / "checkpoints" / "svd", REPO_DIR / "weights" / "stable-video-diffusion-img2vid"):
        if _is_svd_repo(local):
            return str(local)
    return _HF_SVD


def resolve_text_encoder_path(user: str | None) -> str:
    if user:
        p = Path(user).expanduser()
        if p.is_dir() and (p / "text_encoder" / "config.json").is_file():
            return str(p)
        if "/" not in user and "\\" not in user:
            return user
        raise FileNotFoundError(f"Invalid text encoder: {p}")
    local = REPO_DIR / "weights" / "stable-diffusion-2-1-base"
    if local.is_dir():
        return str(local)
    return _HF_TEXT


def _unwrap_robosuite_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def _norm_tracked_2d(pts_px: np.ndarray, h: int, w: int, *, flip_x: bool, flip_y: bool) -> np.ndarray:
    pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    if flip_x:
        pts[:, 0] = 1.0 - pts[:, 0]
    if flip_y:
        pts[:, 1] = 1.0 - pts[:, 1]
    return np.clip(pts, 0.0, 1.0)


def _setup_libero(openpi_root: Path):
    openpi_root = openpi_root.resolve()
    for p in (openpi_root / "third_party", openpi_root / "third_party" / "libero"):
        ps = str(p)
        if p.exists() and ps not in sys.path:
            sys.path.insert(0, ps)
    cfg_root = Path(os.environ.get("LIBERO_CONFIG_PATH", str(openpi_root / ".libero_local")))
    cfg_root.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_root)


def _build_viewpoint_mesh_env(openpi_root: Path, task_id: int, seed: int):
    _setup_libero(openpi_root)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.mesh_vertex_wrapper import MeshVertexWrapper

    bench = benchmark.get_benchmark("libero_viewpoint")()
    task = bench.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    env = MeshVertexWrapper(env, max_points=10000, include_non_mesh_geoms=True)
    return env, task, bench.get_task_init_states(task_id)


def _attach_mesh_tracker(env, agent_ids: list[int], wrist_ids: list[int]):
    from libero.libero.envs.mesh_vertex_tracker import MeshVertexTracker

    return MeshVertexTracker(
        env,
        agentview_vertex_ids=agent_ids,
        eye_in_hand_vertex_ids=wrist_ids,
        img_height=256,
        img_width=256,
    )


def _settle_env(env, inits, rollout_idx: int, num_wait: int = 10):
    dummy = [0.0] * 6 + [-1.0]
    env.reset()
    obs = env.set_init_state(inits[rollout_idx % len(inits)])
    for _ in range(num_wait):
        obs, _, done, _ = env.step(dummy)
        if done:
            break
    return obs


def open_viewpoint_env_with_gripper_tracks(
    openpi_root: Path, task_id: int, seed: int, rollout_idx: int
) -> tuple[object, str, dict]:
    """Reset sim with eval gripper vertex ids (7 agent + 7 wrist keypoints)."""
    env, task, inits = _build_viewpoint_mesh_env(openpi_root, task_id, seed)
    env = _attach_mesh_tracker(
        env, AGENTVIEW_GRIPPER_VERTEX_IDS, WRIST_GRIPPER_VERTEX_IDS
    )
    obs = _settle_env(env, inits, rollout_idx)
    LOGGER.info(
        "Tracking agent %s | wrist %s; only fingertip idx %s move",
        AGENTVIEW_GRIPPER_VERTEX_IDS,
        WRIST_GRIPPER_VERTEX_IDS,
        FINGERTIP_PT_IDX,
    )
    return env, task.language, obs


def get_initial_gripper_tracks_libero(
    openpi_root: Path,
    task_id: int,
    rollout_idx: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    env, lang, obs = open_viewpoint_env_with_gripper_tracks(
        openpi_root, task_id, seed, rollout_idx
    )
    t0 = tracks_gripper28_from_obs(obs)
    env.close()
    return t0, lang


def mask_fingertip_only_tracks(tracks: np.ndarray, base_t0: np.ndarray) -> np.ndarray:
    """Freeze gripper base points (idx 0-4); only fingertip slots 5-6 move per view."""
    base = np.asarray(base_t0, dtype=np.float32).reshape(14, 2)
    out = np.asarray(tracks, dtype=np.float32).copy().reshape(-1, 14, 2)
    for t in range(out.shape[0]):
        frozen = base.copy()
        for i in FINGERTIP_PT_IDX:
            frozen[i] = out[t, i]
            frozen[7 + i] = out[t, 7 + i]
        out[t] = frozen
    return out.reshape(-1, GRIPPER_TRACK_DIM)


def tracks_gripper28_from_obs(obs: dict, *, flip_x: bool = True, flip_y: bool = True) -> np.ndarray:
    """7 agent + 7 wrist gripper keypoints → 28 floats."""
    h, w = int(obs["agentview_image"].shape[0]), int(obs["agentview_image"].shape[1])
    agent_raw = np.asarray(obs["agentview_tracked_vertices_2d"], dtype=np.float32)
    wrist_raw = np.asarray(obs["eye_in_hand_tracked_vertices_2d"], dtype=np.float32)
    if agent_raw.size != GRIPPER_NUM * 2 or wrist_raw.size != GRIPPER_NUM * 2:
        raise ValueError(
            f"Expected {GRIPPER_NUM} tracked points per view, got "
            f"agentview {agent_raw.shape}, wrist {wrist_raw.shape}"
        )
    agent = _norm_tracked_2d(agent_raw, h, w, flip_x=flip_x, flip_y=flip_y)
    wrist = _norm_tracked_2d(wrist_raw, h, w, flip_x=flip_x, flip_y=flip_y)
    return np.concatenate([agent.reshape(-1), wrist.reshape(-1)]).astype(np.float32)


def collect_tracks_nudge_sim_gripper7(
    openpi_root: Path,
    task_id: int,
    rollout_idx: int,
    seed: int,
    *,
    num_frames: int = DEFAULT_HORIZON,
    axis: int = _CTRLWORLD_LEFT_AXIS,
    per_step_delta: float = _CTRLWORLD_LEFT_PER_STEP,
) -> tuple[np.ndarray, str]:
    """OSC nudge in sim; track only 7 gripper-sampled points × 2 cameras."""
    env, lang, obs = open_viewpoint_env_with_gripper_tracks(
        openpi_root, task_id, seed, rollout_idx
    )
    tracks = [tracks_gripper28_from_obs(obs)]
    action = np.zeros(7, dtype=np.float32)
    action[axis] = per_step_delta
    action[6] = -1.0
    for _ in range(num_frames - 1):
        obs, _, done, _ = env.step(action.tolist())
        tracks.append(tracks_gripper28_from_obs(obs))
        if done:
            while len(tracks) < num_frames:
                tracks.append(tracks[-1].copy())
            break
    env.close()
    return np.stack(tracks[:num_frames], axis=0).astype(np.float32), lang


def make_nudge_gripper_tracks(
    tracks_t0: np.ndarray,
    num_steps: int = DEFAULT_HORIZON,
    *,
    agent_dx: float = 0.12,
    agent_dy: float = 0.0,
    wrist_dx: float = 0.12,
    wrist_dy: float = 0.0,
    fingertip_only: bool = False,
) -> np.ndarray:
    """Shift gripper keypoints; default all 7 per view move together (rigid)."""
    base = np.asarray(tracks_t0, dtype=np.float32).reshape(14, 2).copy()
    move_idx = list(FINGERTIP_PT_IDX) if fingertip_only else list(range(GRIPPER_NUM))
    out = np.zeros((num_steps, GRIPPER_TRACK_DIM), dtype=np.float32)
    delta = np.array([agent_dx, agent_dy], dtype=np.float32)
    delta_w = np.array([wrist_dx, wrist_dy], dtype=np.float32)
    for t in range(num_steps):
        pts = base.copy()
        d = float(t)
        for i in move_idx:
            pts[i] = np.clip(pts[i] + d * delta, 0.0, 1.0)
            pts[7 + i] = np.clip(pts[7 + i] + d * delta_w, 0.0, 1.0)
        out[t] = pts.reshape(-1)
    return out


def load_pil(path: Path, size: int = DEFAULT_VIEW_SIZE) -> Image.Image:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(prepare_rgb(rgb, size))


def center_crop_square(rgb: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    """Return square crop and (orig_h, orig_w, y0, x0)."""
    h, w = rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return rgb[y0 : y0 + side, x0 : x0 + side], h, w, y0, x0


def remap_norm_points(
    pts: np.ndarray,
    orig_h: int,
    orig_w: int,
    y0: int,
    x0: int,
    side: int,
) -> np.ndarray:
    """Remap normalized xy from full frame to center-cropped square [0,1]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    px = pts[:, 0] * float(orig_w)
    py = pts[:, 1] * float(orig_h)
    out = np.stack([(px - x0) / float(side), (py - y0) / float(side)], axis=1)
    return np.clip(out, 0.0, 1.0)


def prepare_rgb(rgb: np.ndarray, size: int = DEFAULT_VIEW_SIZE) -> np.ndarray:
    """Symmetric center-crop to square, then resize to ``size x size``."""
    square, _, _, _, _ = center_crop_square(np.asarray(rgb, dtype=np.uint8))
    if square.shape[0] == size and square.shape[1] == size:
        return square
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def remap_tracks78_for_view(
    tracks_78d: np.ndarray,
    orig_h: int,
    orig_w: int,
    y0: int,
    x0: int,
    side: int,
) -> np.ndarray:
    """Apply the same center-crop square as ``prepare_rgb`` to normalized tracks."""
    flat = tracks_78d.reshape(tracks_78d.shape[0], 39, 2)
    remapped = np.stack(
        [remap_norm_points(flat[t], orig_h, orig_w, y0, x0, side) for t in range(len(flat))],
        axis=0,
    )
    return remapped.reshape(tracks_78d.shape).astype(np.float32)


def _decode_parquet_raw(cell) -> np.ndarray:
    """Decode parquet image cell to RGB uint8 without resize."""
    if isinstance(cell, dict) and "bytes" in cell:
        raw = cell["bytes"]
    elif isinstance(cell, (bytes, bytearray, memoryview)):
        raw = bytes(cell)
    else:
        arr = np.asarray(cell)
        if arr.dtype == np.uint8 and arr.ndim == 3:
            return arr
        raise TypeError(f"Unsupported parquet image cell: {type(cell)}")
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed on parquet image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_parquet_image(cell, size: int = DEFAULT_VIEW_SIZE) -> np.ndarray:
    return prepare_rgb(_decode_parquet_raw(cell), size)


def _stack_norm_points(cell) -> np.ndarray:
    pts = np.stack([np.asarray(p, dtype=np.float32) for p in np.asarray(cell)], axis=0)
    return np.clip(pts.reshape(-1, 2), 0.0, 1.0)


def _track_targets_raw_to_array(cell) -> np.ndarray:
    """Parse parquet track_targets_raw cell → (H, 39, 2)."""
    if hasattr(cell, "as_py"):
        cell = cell.as_py()
    arr = np.asarray(cell, dtype=np.float32)
    if arr.ndim == 3:
        return np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return np.clip(arr.reshape(-1, 39, 2), 0.0, 1.0)
    raise ValueError(f"Unexpected track_targets_raw shape: {arr.shape}")


def _tracks78_from_targets(raw: np.ndarray, horizon: int) -> np.ndarray:
    """(H,39,2) future window → (horizon, 78) for TrackDualViewWM."""
    if raw.shape[0] < horizon:
        raise ValueError(f"track_targets_raw has {raw.shape[0]} steps, need {horizon}")
    return raw[:horizon].reshape(horizon, TRACK_TARGETS_RAW_DIM).astype(np.float32)


def _log_track_drift(tracks_78d: np.ndarray, *, label: str) -> None:
    """Log per-view drift so static conditioning windows are obvious."""
    flat = tracks_78d.reshape(tracks_78d.shape[0], 39, 2)
    agent = flat[:, :7]
    grid = flat[:, 7:32]
    wrist_mesh = flat[:, 32:39]
    agent_d = float(np.max(np.abs(agent[-1] - agent[0])))
    grid_d = float(np.max(np.abs(grid[-1] - grid[0])))
    mesh_d = float(np.max(np.abs(wrist_mesh[-1] - wrist_mesh[0])))
    grid_step = float(np.max(np.abs(grid[1:] - grid[:-1]))) if len(grid) > 1 else 0.0
    LOGGER.info(
        "%s drift over %d steps: agent_mesh=%.4f wrist_grid=%.4f wrist_mesh=%.4f "
        "(max grid step=%.4f)",
        label,
        tracks_78d.shape[0],
        agent_d,
        grid_d,
        mesh_d,
        grid_step,
    )
    if grid_d < 0.02 and mesh_d < 0.02:
        LOGGER.warning(
            "%s: wrist tracks barely move — try --start_frame later in the episode",
            label,
        )


def _load_lerobot_task_text(meta_root: Path, task_index: int, episode_index: int | None) -> str:
    tasks_file = meta_root / "meta" / "tasks.jsonl"
    if tasks_file.is_file():
        with tasks_file.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if int(row.get("task_index", -1)) == int(task_index):
                    return str(row["task"])
    episodes_file = meta_root / "meta" / "episodes.jsonl"
    if episode_index is not None and episodes_file.is_file():
        with episodes_file.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if int(row.get("episode_index", -1)) == int(episode_index):
                    tasks = row.get("tasks") or []
                    if tasks:
                        return str(tasks[0])
    return ""


def load_parquet_episode_window(
    parquet_path: Path,
    *,
    meta_root: Path,
    start_frame: int,
    horizon: int,
    view_size: int = DEFAULT_VIEW_SIZE,
) -> tuple[Image.Image, Image.Image, np.ndarray, str, list[np.ndarray], list[np.ndarray]]:
    """Load init RGB + track_targets_raw (H,78) + GT RGB from LeRobot parquet.

    All RGB is center-cropped to square then resized to ``view_size`` (default 192).
    Tracks are remapped to match the same center crop.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    parquet_path = parquet_path.expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    if start_frame + horizon >= len(df):
        raise ValueError(
            f"Need rows [{start_frame}, {start_frame + horizon}] but episode has "
            f"{len(df)} rows in {parquet_path}"
        )

    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
    episode_index = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else None
    lang = _load_lerobot_task_text(meta_root, task_index, episode_index)
    if not lang:
        lang = parquet_path.stem

    ref_rgb = _decode_parquet_raw(df["image"].iloc[start_frame])
    _, oh, ow, y0, x0 = center_crop_square(ref_rgb)
    side = min(oh, ow)

    agent_pil = Image.fromarray(prepare_rgb(ref_rgb, view_size))
    wrist_pil = Image.fromarray(
        prepare_rgb(_decode_parquet_raw(df["wrist_image"].iloc[start_frame]), view_size)
    )

    raw_table = pq.read_table(
        parquet_path,
        columns=["track_targets_raw"],
    )
    raw = _track_targets_raw_to_array(raw_table["track_targets_raw"][start_frame])
    tracks_78d = _tracks78_from_targets(raw, horizon)
    tracks_78d = remap_tracks78_for_view(tracks_78d, oh, ow, y0, x0, side)

    gt_agent = []
    gt_wrist = []
    for t in range(start_frame + 1, start_frame + horizon + 1):
        gt_agent.append(_decode_parquet_image(df["image"].iloc[t], size=view_size))
        gt_wrist.append(_decode_parquet_image(df["wrist_image"].iloc[t], size=view_size))

    return agent_pil, wrist_pil, tracks_78d, lang, gt_agent, gt_wrist


def _ensure_rgb_size(rgb: np.ndarray, size: int) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def save_dual_view_outputs(
    out_dir: Path,
    run_tag: str,
    mode: str,
    horizon: int,
    *,
    pred_agent,
    pred_wrist,
    gt_agent_rgb: list[np.ndarray],
    gt_wrist_rgb: list[np.ndarray],
    per_view_size: int,
    fps: int,
) -> dict[str, Path]:
    """Write agent / wrist / stacked comparison MP4s (GT left, pred right per row).

    Matches Ctrl-World ``save_lerobot_dual_view_videos`` layout and naming.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(
        len(pred_agent),
        len(pred_wrist),
        len(gt_agent_rgb),
        len(gt_wrist_rgb),
        horizon,
    )
    pr_a = np.stack(
        [_ensure_rgb_size(np.array(f.convert("RGB")), per_view_size) for f in pred_agent[:n]]
    )
    pr_w = np.stack(
        [_ensure_rgb_size(np.array(f.convert("RGB")), per_view_size) for f in pred_wrist[:n]]
    )
    gt_a = np.stack([_ensure_rgb_size(g, per_view_size) for g in gt_agent_rgb[:n]])
    gt_w = np.stack([_ensure_rgb_size(g, per_view_size) for g in gt_wrist_rgb[:n]])

    agent_cmp = np.concatenate([gt_a, pr_a], axis=2)
    wrist_cmp = np.concatenate([gt_w, pr_w], axis=2)
    both_cmp = np.concatenate([agent_cmp, wrist_cmp], axis=0)

    paths = {
        "agent": out_dir / f"wm_{run_tag}_{mode}_h{horizon}_agent_gt_left_pred_right.mp4",
        "wrist": out_dir / f"wm_{run_tag}_{mode}_h{horizon}_wrist_gt_left_pred_right.mp4",
        "both": out_dir / f"wm_{run_tag}_{mode}_h{horizon}_agent_wrist_gt_left_pred_right.mp4",
        "pred_agent": out_dir / f"wm_{run_tag}_pred_agent_h{horizon}.mp4",
        "pred_wrist": out_dir / f"wm_{run_tag}_pred_wrist_h{horizon}.mp4",
        "gt_agent": out_dir / f"wm_{run_tag}_gt_agent_h{horizon}.mp4",
        "gt_wrist": out_dir / f"wm_{run_tag}_gt_wrist_h{horizon}.mp4",
    }
    imageio.mimwrite(str(paths["agent"]), agent_cmp, fps=fps)
    imageio.mimwrite(str(paths["wrist"]), wrist_cmp, fps=fps)
    imageio.mimwrite(str(paths["both"]), both_cmp, fps=fps)
    imageio.mimwrite(str(paths["pred_agent"]), pr_a, fps=fps)
    imageio.mimwrite(str(paths["pred_wrist"]), pr_w, fps=fps)
    imageio.mimwrite(str(paths["gt_agent"]), gt_a, fps=fps)
    imageio.mimwrite(str(paths["gt_wrist"]), gt_w, fps=fps)
    LOGGER.info("Saved agent  (GT|Pred): %s", paths["agent"])
    LOGGER.info("Saved wrist  (GT|Pred): %s", paths["wrist"])
    LOGGER.info("Saved stacked agent+wrist: %s", paths["both"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_source",
        choices=("parquet", "sim", "offset"),
        default="parquet",
        help="parquet: real gripper tracks from LeRobot (default); sim/offset: LIBERO nudge",
    )
    parser.add_argument(
        "--parquet_path",
        default=_DEFAULT_PARQUET,
        help="LeRobot episode parquet (data_source=parquet)",
    )
    parser.add_argument(
        "--lerobot_meta_root",
        default=_DEFAULT_LEROBOT_ROOT,
        help="LeRobot dataset root for meta/tasks.jsonl",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="First parquet row for init RGB + tracks (data_source=parquet)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Output directory for parquet dual-view videos",
    )
    parser.add_argument(
        "--agent_frame",
        default=str(_REPO_ROOT / "frames/agentview/frame_000000.png"),
        help="Init agent RGB (sim/offset only)",
    )
    parser.add_argument(
        "--wrist_frame",
        default=str(_REPO_ROOT / "frames/wrist/frame_000000.png"),
        help="Init wrist RGB (sim/offset only)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(_DEFAULT_CHECKPOINT),
        help="Trained hybrid UNet+ControlNet dir (default hybrid-checkpoint-128000)",
    )
    parser.add_argument("--svd_path", default=None)
    parser.add_argument("--text_encoder_path", default=None)
    parser.add_argument("--per_view_size", type=int, default=DEFAULT_VIEW_SIZE,
                        help="Per-view resolution after center-crop (default 192)")
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Track / GT / output length (default 16, same as Ctrl-World & parquet track_targets_raw)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="UNet temporal length (default: same as --horizon; ckpt config may say 14)",
    )
    parser.add_argument("--openpi_root", default=str(_REPO_ROOT))
    parser.add_argument("--libero_viewpoint_task", type=int, default=6)
    parser.add_argument("--rollout_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--nudge_axis",
        type=int,
        default=_CTRLWORLD_LEFT_AXIS,
        help="OSC axis for sim mode (1=y, same as Ctrl-World left)",
    )
    parser.add_argument(
        "--nudge_delta",
        type=float,
        default=_CTRLWORLD_LEFT_PER_STEP,
        help="OSC per-step delta for sim mode (default -0.36, larger left nudge)",
    )
    parser.add_argument(
        "--invert_track",
        action="store_true",
        help="Multiply all track nudge deltas by -1 (if gripper/arm motion looks reversed)",
    )
    parser.add_argument(
        "--track_delta_x",
        type=float,
        default=0.12,
        help="offset: per-frame normalized dx (14 frames ≈ 1.7 image width)",
    )
    parser.add_argument(
        "--track_delta_y",
        type=float,
        default=0.0,
        help="offset: per-frame normalized dy on agent gripper points",
    )
    parser.add_argument(
        "--wrist_track_delta_x",
        type=float,
        default=None,
        help="wrist mesh dx (default: same as --track_delta_x)",
    )
    parser.add_argument(
        "--wrist_track_delta_y",
        type=float,
        default=0.0,
        help="wrist mesh dy for gripper-local left in eye-in-hand view",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=8,
        help="Denoising steps (default 8: wrist moves; ≥12 collapses toward static init wrist)",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--wrist_track_gain",
        type=float,
        default=3.0,
        help="Amplify wrist grid+mesh displacement in ControlNet cond (default 3.0)",
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=180,
        help="SVD motion bucket (dualview training default 180)",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="ControlNet outer scale (dualview training default 1.0)",
    )
    parser.add_argument(
        "--output_video",
        default=str(REPO_DIR / "outputs" / "libero_viewpoint_task06_nudge_left.mp4"),
        help="Single agentview MP4 (sim/offset only)",
    )
    parser.add_argument(
        "--fingertip_only",
        action="store_true",
        help="Only move track points 5-6; default moves all 7 gripper keypoints",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.num_frames is None:
        args.num_frames = args.horizon
    track_sign = -1.0 if args.invert_track else 1.0
    args.track_delta_x *= track_sign
    args.track_delta_y *= track_sign
    args.nudge_delta *= track_sign
    if args.wrist_track_delta_x is not None:
        args.wrist_track_delta_x *= track_sign
    args.wrist_track_delta_y *= track_sign

    logging.basicConfig(level=logging.INFO)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    openpi_root = Path(args.openpi_root)
    gt_agent_rgb: list[np.ndarray] | None = None
    gt_wrist_rgb: list[np.ndarray] | None = None
    run_tag = "episode"

    if args.data_source == "parquet":
        parquet_path = Path(args.parquet_path)
        meta_root = Path(args.lerobot_meta_root)
        av_pil, wrist_pil, tracks_78d, lang, gt_agent_rgb, gt_wrist_rgb = load_parquet_episode_window(
            parquet_path,
            meta_root=meta_root,
            start_frame=args.start_frame,
            horizon=args.horizon,
            view_size=args.per_view_size,
        )
        run_tag = parquet_path.stem
        _log_track_drift(tracks_78d, label="track_targets_raw")
        LOGGER.info(
            "Tracks: track_targets_raw row %d (%d×78), GT rows [%d, %d] from %s",
            args.start_frame,
            args.horizon,
            args.start_frame + 1,
            args.start_frame + args.horizon,
            parquet_path,
        )
        tracks_for_wm = tracks_78d
    else:
        agent_path = Path(args.agent_frame)
        wrist_path = Path(args.wrist_frame)
        if not wrist_path.is_file():
            wrist_path = agent_path
        av_pil = load_pil(agent_path, size=args.per_view_size)
        wrist_pil = load_pil(wrist_path, size=args.per_view_size)

        LOGGER.info(
            "Track data_source=%s task=%d sign=%+.0f (invert_track=%s)",
            args.data_source,
            args.libero_viewpoint_task,
            track_sign,
            args.invert_track,
        )
        if args.data_source == "sim":
            tracks_raw, lang = collect_tracks_nudge_sim_gripper7(
                openpi_root,
                args.libero_viewpoint_task,
                args.rollout_idx,
                args.seed,
                num_frames=args.num_frames,
                axis=args.nudge_axis,
                per_step_delta=args.nudge_delta,
            )
            tracks_for_wm = (
                mask_fingertip_only_tracks(tracks_raw, tracks_raw[0])
                if args.fingertip_only
                else tracks_raw
            )
            LOGGER.info(
                "Tracks: sim OSC axis=%d step=%.3f (%s)",
                args.nudge_axis,
                args.nudge_delta,
                "fingertip-only" if args.fingertip_only else "all-7-gripper",
            )
        else:
            tracks_t0, lang = get_initial_gripper_tracks_libero(
                openpi_root,
                args.libero_viewpoint_task,
                args.rollout_idx,
                args.seed,
            )
            wdx = args.wrist_track_delta_x if args.wrist_track_delta_x is not None else args.track_delta_x
            tracks_raw = make_nudge_gripper_tracks(
                tracks_t0,
                num_steps=args.num_frames,
                agent_dx=args.track_delta_x,
                agent_dy=args.track_delta_y,
                wrist_dx=wdx,
                wrist_dy=args.wrist_track_delta_y,
                fingertip_only=args.fingertip_only,
            )
            tracks_for_wm = tracks_raw
            LOGGER.info(
                "Tracks: offset agent(%.4f,%.4f) wrist(%.4f,%.4f) (%s)",
                args.track_delta_x,
                args.track_delta_y,
                wdx,
                args.wrist_track_delta_y,
                "fingertip-only" if args.fingertip_only else "all-7-gripper",
            )
        run_tag = f"libero_task{args.libero_viewpoint_task}_{args.data_source}"

    prompt = args.prompt or lang
    pts = tracks_for_wm.reshape(tracks_for_wm.shape[0], -1, 2)
    tip = list(FINGERTIP_PT_IDX)
    if pts.shape[1] == 39:
        agent_pts = pts[:, :7]
        wrist_pts = pts[:, 32:39]
    else:
        agent_pts = pts[:, :7]
        wrist_pts = pts[:, 7:14]
    base_drift = float(
        np.max(np.abs(agent_pts[:, :5] - agent_pts[0:1, :5]))
        + np.max(np.abs(wrist_pts[:, :5] - wrist_pts[0:1, :5]))
    )
    tip_drift = float(
        np.max(np.abs(agent_pts[:, tip] - agent_pts[0:1, tip]))
        + np.max(np.abs(wrist_pts[:, tip] - wrist_pts[0:1, tip]))
    )
    LOGGER.info("Prompt: %s", prompt)
    LOGGER.info(
        "Drift base_pts=%.5f (should≈0) | fingertip_pts=%.5f (should>0)",
        base_drift,
        tip_drift,
    )

    wm = TrackDualViewWM(
        checkpoint_path=str(ckpt),
        svd_path=resolve_svd_path(args.svd_path),
        text_encoder_path=resolve_text_encoder_path(args.text_encoder_path),
        per_view_size=args.per_view_size,
    )
    LOGGER.info(
        "Loading trained TrackDualViewWM: UNet+ControlNet from %s | SVD base %s",
        ckpt,
        resolve_svd_path(args.svd_path),
    )
    wm.load(device=args.device, num_frames=args.num_frames)

    LOGGER.info("Generating %d-frame dual-view video...", wm.num_frames)
    pred_agent, pred_wrist = wm.generate_video(
        av_pil,
        wrist_pil,
        tracks_for_wm,
        prompt=prompt,
        num_inference_steps=args.num_inference_steps,
        num_frames=wm.num_frames,
        wrist_track_gain=args.wrist_track_gain,
        motion_bucket_id=args.motion_bucket_id,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )

    if args.data_source == "parquet":
        assert gt_agent_rgb is not None and gt_wrist_rgb is not None
        save_dual_view_outputs(
            Path(args.output_dir),
            run_tag,
            "track",
            args.horizon,
            pred_agent=pred_agent,
            pred_wrist=pred_wrist,
            gt_agent_rgb=gt_agent_rgb,
            gt_wrist_rgb=gt_wrist_rgb,
            per_view_size=args.per_view_size,
            fps=args.fps,
        )
    else:
        out_path = Path(args.output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        init_rgb = np.array(av_pil.convert("RGB").resize(
            (args.per_view_size, args.per_view_size), Image.BICUBIC
        ))
        writer = imageio.get_writer(str(out_path), fps=args.fps)
        writer.append_data(init_rgb)
        for f in pred_agent:
            writer.append_data(np.array(f))
        writer.close()
        LOGGER.info("Saved %d predicted agent frames + init → %s", len(pred_agent), out_path)


if __name__ == "__main__":
    main()
