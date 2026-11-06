#!/usr/bin/env python3
from __future__ import annotations

import tract.shared.local_cache_bootstrap
import collections
import dataclasses
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import h5py
from moviepy import ImageSequenceClip
import numpy as np
from openpi_client import image_tools

# Set temp/cache before importing torch so JIT/extensions do not fill root /tmp.
_pi_base = Path(os.environ.get("OPENPI_CACHE_BASE", str(Path(__file__).resolve().parents[1])))
_pi_tmp = _pi_base / "tmp"
_pi_tmp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_pi_tmp))
os.environ.setdefault("TEMP", str(_pi_tmp))
os.environ.setdefault("TMP", str(_pi_tmp))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_pi_base / "torch_extensions"))

OPENPI_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
_LIBERO_PATH_ORDER = (
    OPENPI_ROOT / "third_party",
    OPENPI_ROOT / "third_party" / "libero",
)
_BOOTSTRAP_PATHS = (
    OPENPI_ROOT / "src",
    OPENPI_ROOT / "packages" / "openpi-client" / "src",
    OPENPI_ROOT,
    *_LIBERO_PATH_ORDER,
    SCRIPTS_DIR,
)
for _p in _BOOTSTRAP_PATHS:
    p_str = str(_p)
    if _p.exists() and p_str not in sys.path:
        sys.path.insert(0, p_str)

libero_cfg_root = Path(os.environ.get("LIBERO_CONFIG_PATH", str(OPENPI_ROOT / ".libero_local")))
os.environ["LIBERO_CONFIG_PATH"] = str(libero_cfg_root)
libero_cfg_root.mkdir(parents=True, exist_ok=True)


def _parse_cli_kv_flag(flag: str) -> str | None:
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1].strip()
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1].strip()
    return None


def _configure_mujoco_gl_backend() -> None:
    """Prefer GLX/GLFW (xvfb) over EGL when render nodes are unavailable."""
    requested = (os.environ.get("OPENPI_MUJOCO_GL") or os.environ.get("MUJOCO_GL") or "glx").strip().lower()
    if requested in ("", "auto"):
        requested = "glx"
    os.environ["MUJOCO_GL"] = requested
    if requested in ("glx", "glfw", "osmesa"):
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)


def _install_dual_gpu_env(policy_gpu: str, render_gpu: str) -> int:
    """Put policy on cuda:0 (first visible GPU); MuJoCo EGL on render_gpu when applicable."""
    return _install_multi_gpu_env(
        policy_gpu=policy_gpu,
        render_gpu=render_gpu,
        wm_gpu=policy_gpu,
        vlac_gpu=policy_gpu,
    )[0]


def _install_multi_gpu_env(
    *,
    policy_gpu: str,
    render_gpu: str,
    wm_gpu: str,
    vlac_gpu: str,
) -> tuple[int, dict[str, str]]:
    """Expose policy/wm/vlac/render GPUs and return (render_id, cuda device map).

    ``device_map`` keys are ``policy`` / ``wm`` / ``vlac`` with values like ``cuda:0``.
    Under GLX/GLFW/OSMesa, ``render_id`` is ``-1`` (GLFW uses the X display, not EGL).
    """
    ids = {
        "policy": policy_gpu.strip(),
        "wm": wm_gpu.strip(),
        "vlac": vlac_gpu.strip(),
        "render": render_gpu.strip(),
    }
    for name, val in ids.items():
        if not val.isdigit():
            raise ValueError(f"--{name}-gpu must be a numeric GPU index, got {val!r}")

    # Preserve first-seen order so policy stays on cuda:0 when listed first.
    visible: list[str] = []
    for val in (ids["policy"], ids["wm"], ids["vlac"], ids["render"]):
        if val not in visible:
            visible.append(val)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible)

    gl = (os.environ.get("MUJOCO_GL") or "").strip().lower()
    render = ids["render"]
    if gl == "egl":
        if render == ids["policy"] and len(visible) == 1:
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        else:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = render
        os.environ["OPENPI_RENDER_GPU_DEVICE_ID"] = render
        render_id = int(render)
    else:
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        os.environ["OPENPI_RENDER_GPU_DEVICE_ID"] = ids["policy"]
        render_id = -1

    index = {phys: i for i, phys in enumerate(visible)}
    device_map = {
        "policy": f"cuda:{index[ids['policy']]}",
        "wm": f"cuda:{index[ids['wm']]}",
        "vlac": f"cuda:{index[ids['vlac']]}",
    }
    return render_id, device_map


def _apply_gpu_split_from_argv() -> None:
    cli_policy = _parse_cli_kv_flag("--policy-gpu")
    cli_render = _parse_cli_kv_flag("--render-gpu")
    cli_wm = _parse_cli_kv_flag("--wm-gpu")
    cli_vlac = _parse_cli_kv_flag("--vlac-gpu")
    policy = (cli_policy or os.environ.get("OPENPI_POLICY_GPU", "")).strip()
    render = (cli_render or os.environ.get("OPENPI_RENDER_GPU", "")).strip()
    wm = (cli_wm or os.environ.get("OPENPI_WM_GPU", "")).strip() or policy
    vlac = (cli_vlac or os.environ.get("OPENPI_VLAC_GPU", "")).strip() or policy
    if policy and render:
        _install_multi_gpu_env(
            policy_gpu=policy,
            render_gpu=render,
            wm_gpu=wm or policy,
            vlac_gpu=vlac or policy,
        )


def _warmup_glx_context(resolution: int = 128) -> None:
    """Create/destroy a tiny offscreen env before JAX/CUDA touches the process."""
    gl = (os.environ.get("MUJOCO_GL") or "").strip().lower()
    if gl not in ("glx", "glfw"):
        return
    from libero.libero import get_libero_path as _glp
    from libero.libero.envs import OffScreenRenderEnv as _OffEnv

    bddl_root = Path(_glp("bddl_files"))
    sample = next(bddl_root.joinpath("libero_object").glob("*.bddl"), None)
    if sample is None:
        sample = next(bddl_root.rglob("*.bddl"))
    warm = _OffEnv(
        bddl_file_name=str(sample),
        camera_heights=resolution,
        camera_widths=resolution,
        render_gpu_device_id=-1,
    )
    warm.reset()
    warm.close()


_configure_mujoco_gl_backend()
_apply_gpu_split_from_argv()
try:
    _warmup_glx_context(resolution=128)
except Exception as _warmup_exc:  # noqa: BLE001
    print(f"[evaluate_full_action] GL warmup failed before torch import: {_warmup_exc}", file=sys.stderr)

import torch
import tyro
from tract.policies import policy_config as _policy_config
from tract.training import config as training_config

libero_cfg_root = Path(os.environ.get("LIBERO_CONFIG_PATH", str(OPENPI_ROOT / ".libero_local")))
os.environ["LIBERO_CONFIG_PATH"] = str(libero_cfg_root)
libero_cfg_root.mkdir(parents=True, exist_ok=True)

_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.benchmark import libero_suites as REGISTERED_LIBERO_SUITES
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.mesh_vertex_tracker import MeshVertexTracker
from libero.libero.envs.mesh_vertex_wrapper import MeshVertexWrapper

from libero_gripper_vertex_ids import (
    apply_dynamic_gripper_vertex_ids,
    find_mesh_vertex_tracker,
    refresh_tracked_vertices_obs,
)
from libero_inits_from_panda import ik_arm_to_eef, target_init_from_source_vec

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

DYNAMIC_GRIPPER_VERTEX_SUITES = frozenset(
    {
        "libero_10_cross",
        "libero_10_sawyer",
        "libero_object_cross",
        "libero_goal_cross",
        "libero_spatial_cross",
        "libero_object_sawyer",
        "libero_goal_sawyer",
        "libero_spatial_sawyer",
    }
)


def _suite_uses_dynamic_gripper(suite_name: str) -> bool:
    name = suite_name.strip()
    if name in DYNAMIC_GRIPPER_VERTEX_SUITES:
        return True
    return name.endswith("_cross") or name.endswith("_sawyer")


COMMON_WRIST_VERTEX_IDS = [2955, 3055, 3040, 2981, 2966, 3052, 3022]
DEFAULT_VERTEX_IDS = {
    "agentview": [3001, 3004, 3014, 3067, 3040, 3081, 2898],
    "eye_in_hand": COMMON_WRIST_VERTEX_IDS,
}
KITCHEN_VERTEX_IDS = {
    "agentview": [3249, 3252, 3200, 3315, 3272, 3329, 3134],
    "eye_in_hand": COMMON_WRIST_VERTEX_IDS,
}
LIVING_ROOM_VERTEX_IDS = {
    "agentview": [3135, 3140, 3141, 3202, 3205, 3215, 3034],
    "eye_in_hand": COMMON_WRIST_VERTEX_IDS,
}
STUDY_VERTEX_IDS = {
    "agentview": [3352, 3355, 3365, 3418, 3391, 3432, 3249],
    "eye_in_hand": COMMON_WRIST_VERTEX_IDS,
}
SCENE_VERTEX_IDS = {
    "default": DEFAULT_VERTEX_IDS,
    "kitchen": KITCHEN_VERTEX_IDS,
    "living_room": LIVING_ROOM_VERTEX_IDS,
    "living": LIVING_ROOM_VERTEX_IDS,
    "study": STUDY_VERTEX_IDS,
}
SUITE_NAME_ALIASES: dict[str, str] = {
    "libero_10_viewpoint": "libero_viewpoint",
}


def _default_max_steps_for_registered_suite(suite_name: str) -> int:
    if suite_name == "libero_90":
        return 400
    if suite_name.startswith("libero_spatial"):
        return 220
    if suite_name.startswith("libero_object"):
        return 280
    if suite_name.startswith("libero_goal"):
        return 300
    return 520


SUITE_MAX_STEPS = {
    name: _default_max_steps_for_registered_suite(name) for name in REGISTERED_LIBERO_SUITES
}


def _normalize_suite_name(name: str) -> str:
    key = name.strip()
    if key in SUITE_MAX_STEPS:
        return key
    resolved = SUITE_NAME_ALIASES.get(key)
    if resolved is not None:
        logging.warning("Suite %r is not registered; using %r instead.", key, resolved)
        return resolved
    raise ValueError(
        f"Unknown suite: {name!r}. Registered suites: {sorted(SUITE_MAX_STEPS.keys())}"
    )


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    name: str
    config_name: str
    checkpoint_dir: str


@dataclasses.dataclass
class Args:
    """LIBERO eval with optional Ctrl-World (action WM) + VLAC selection.

    Default: policy samples N candidates → Ctrl-World imagines each from actions →
    VLAC dual-view scores → execute best ``replan_steps`` actions.

    For Track-WM selection use ``evaluate_full_track.py``. Use
    ``--no-use-action-selection`` for the same single-infer baseline as
    ``evaluate.py``. Use ``--replay-on-sawyer`` for HDF5 source replay.
    """

    config_name: str = "pi05_libero_stage2p5_joint"
    checkpoint_dir: str = str(OPENPI_ROOT / "checkpoints/pi05_libero_stage2p5_joint/pi05_final/29999")
    suites: tuple[str, ...] = ("libero_object_sawyer",)
    all_tasks: bool = False
    task_index: int = 0
    rollouts_per_task: int = 10
    resize_size: int = 224
    env_resolution: int = 256
    replan_steps: int = 5
    num_steps_wait: int = 10
    fps: int = 10
    seed: int = 7
    out_dir: str = str(OPENPI_ROOT / "results")
    run_name: str | None = None
    flip_image_x: bool = True
    flip_image_y: bool = True
    align_point_flips_with_image: bool = True
    xflip_points: bool = False
    yflip_points: bool = False
    fail_on_missing_checkpoint: bool = False
    save_original_rollout_video: bool = True
    save_rollout_predictions: bool = True
    libero_10_cross_robot: str = "ur5e"
    policy_gpu: str = "0"
    render_gpu: str = "0"
    # Physical GPU ids for Ctrl-World / VLAC (mapped into CUDA_VISIBLE_DEVICES).
    wm_gpu: str = ""  # default: same as policy_gpu
    vlac_gpu: str = ""  # default: same as policy_gpu
    # Dual Dynamics action selection (Ctrl-World / action WM + VLAC).
    use_action_selection: bool = True
    n_candidates: int = 20
    ctrlworld_checkpoint: str = str(OPENPI_ROOT / "checkpoints/ctrl_world/libero_ctrlworld/checkpoint-20000.pt")
    svd_path: str = str(OPENPI_ROOT / "checkpoints/svd")
    # Local folder if present; otherwise Ctrl-World resolves openai/clip-vit-base-patch32.
    clip_path: str = "openai/clip-vit-base-patch32"
    vlac_model: str = str(OPENPI_ROOT / "checkpoints/VLAC-8b")
    wm_num_inference_steps: int = 25
    policy_mem_fraction: float = 0.35
    sampling_mode: str = "tsr"
    tsr_k: float = 5.0
    noise_scale: float = 3.0
    noise_type: str = "sobol"
    smooth_sigma: float = 2.0
    vlac_agent_weight: float = 0.6
    vlac_wrist_weight: float = 0.4
    # Replay Franka/cross HDF5 executed_actions + tracks on *_sawyer (no policy).
    replay_on_sawyer: bool = False
    replay_source_suite: str = "libero_object"
    replay_sawyer_suite: str = "libero_object_sawyer"
    replay_hdf5_dir: str = str(OPENPI_ROOT / "results/pi05_final/libero_object/hdf5")
    replay_hdf5_policy_prefix: str = "stage2"
    # Map libero_object pruned_init → sawyer sim (same objects/base layout, IK-matched EEF).
    replay_use_panda_init: bool = True
    # Overlay source-suite HDF5 tracks on replay video (base frames always from env render).
    replay_overlay_source_tracks: bool = True
    # "delta": replay source OSC deltas; "eef_ik": track source EEF poses via IK on target arm.
    replay_mode: str = "eef_ik"
    # When set, the first replay rollout video is written to this path (overrides default naming).
    replay_output_video: str | None = None


def _scene_from_task(task: str) -> str:
    t = task.lower()
    if "kitchen" in t:
        return "kitchen"
    if "living" in t or "livingroom" in t:
        return "living_room"
    if "study" in t:
        return "study"
    return "default"


def _mesh_seeds_for_scene(scene: str) -> tuple[list[int], list[int]]:
    ids = SCENE_VERTEX_IDS.get(scene, DEFAULT_VERTEX_IDS)
    return ids["agentview"], ids["eye_in_hand"]


def _max_steps_for_suite(name: str) -> int:
    return SUITE_MAX_STEPS[_normalize_suite_name(name)]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float32, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _make_grid_25() -> np.ndarray:
    u = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    v = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)


def _ensure_n_points_xy(
    points_2d: np.ndarray,
    n: int = 7,
    *,
    label: str = "mesh",
) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2) if pts.size else np.zeros((0, 2), dtype=np.float32)
    elif pts.ndim > 2:
        pts = pts.reshape(-1, 2)
    if pts.shape[0] >= n:
        return pts[:n].astype(np.float32, copy=False)
    if pts.shape[0] == 0:
        fallback = np.linspace(0.35, 0.65, n, dtype=np.float32)
        out = np.stack([fallback, np.full(n, 0.55, dtype=np.float32)], axis=-1)
        logging.debug("%s mesh: 0 valid vertices; using placeholder points", label)
        return out
    pad = np.repeat(pts[-1:], n - pts.shape[0], axis=0)
    return np.concatenate([pts, pad], axis=0).astype(np.float32)


def _to_norm_xy(points_2d: np.ndarray, h: int, w: int) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=np.float32).copy()
    pts[..., 0] /= float(w)
    pts[..., 1] /= float(h)
    return np.clip(pts, 0.0, 1.0)


def _flip_xy(points: np.ndarray, *, xflip: bool, yflip: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()
    if xflip:
        pts[..., 0] = 1.0 - pts[..., 0]
    if yflip:
        pts[..., 1] = 1.0 - pts[..., 1]
    return pts


def _align_pred_tracks_for_display(
    pred: np.ndarray,
    *,
    flip_image_x: bool,
    flip_image_y: bool,
) -> np.ndarray:
    """Map policy track outputs onto flipped display images.

    - Agent mesh (0:7): y-flip only (overlay vs policy vertical axis).
    - Wrist grid (7:32): mirror fixed lattice with image x/y flips.
    - Wrist mesh (32:39): unchanged (already aligned in ``_prepare_raw_item``).
    """
    out = np.asarray(pred, dtype=np.float32).reshape(-1, 39, 2).copy()
    if flip_image_y:
        agent = out[:, :7, :].copy()
        agent[..., 1] = 1.0 - agent[..., 1]
        out[:, :7, :] = agent
    if flip_image_x or flip_image_y:
        grid = out[:, 7:32, :].copy()
        if flip_image_x:
            grid[..., 0] = 1.0 - grid[..., 0]
        if flip_image_y:
            grid[..., 1] = 1.0 - grid[..., 1]
        out[:, 7:32, :] = grid
    return out


def _pt_to_px(pt: np.ndarray, h: int, w: int) -> tuple[int, int]:
    x = float(pt[0])
    y = float(pt[1])
    if -3.0 <= x <= 3.0 and -3.0 <= y <= 3.0:
        x *= w
        y *= h
    return int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))


_TRACK_COLOR_RGB_START = (255, 72, 8)
_TRACK_COLOR_RGB_MID = (255, 248, 200)
_TRACK_COLOR_RGB_END = (8, 210, 255)


def _pred_tracks_horizon_window(
    pred_tracks: np.ndarray | None,
    *,
    horizon_offset: int,
    track_horizon: int = 16,
) -> np.ndarray | None:
    if pred_tracks is None or np.asarray(pred_tracks).size == 0:
        return None
    pred = np.asarray(pred_tracks, dtype=np.float32).reshape(-1, 39, 2)
    start = int(horizon_offset)
    if start >= pred.shape[0]:
        return None
    chunk = pred[start : min(start + track_horizon, pred.shape[0])]
    return chunk if chunk.shape[0] >= 1 else None


def _track_horizon_color_bgr(t: int, horizon: int) -> tuple[int, int, int]:
    u = float(t) / float(max(int(horizon) - 1, 1))
    if u <= 0.5:
        v = u * 2.0
        rgb = tuple(
            int(round(a + (b - a) * v))
            for a, b in zip(_TRACK_COLOR_RGB_START, _TRACK_COLOR_RGB_MID, strict=True)
        )
    else:
        v = (u - 0.5) * 2.0
        rgb = tuple(
            int(round(a + (b - a) * v))
            for a, b in zip(_TRACK_COLOR_RGB_MID, _TRACK_COLOR_RGB_END, strict=True)
        )
    return (rgb[2], rgb[1], rgb[0])


def _draw_future_tracks_colormap(
    img: np.ndarray,
    future: np.ndarray,
    *,
    line_width: int = 3,
    marker_radius: int = 3,
) -> None:
    h, w = img.shape[:2]
    horizon = int(future.shape[0])
    npts = int(future.shape[1])
    for i in range(npts):
        for t in range(horizon):
            color = _track_horizon_color_bgr(t, horizon)
            x, y = _pt_to_px(future[t, i], h, w)
            cv2.circle(img, (x, y), marker_radius, color, -1, cv2.LINE_AA)
        for t in range(horizon - 1):
            color = _track_horizon_color_bgr(t, horizon)
            p0 = _pt_to_px(future[t, i], h, w)
            p1 = _pt_to_px(future[t + 1, i], h, w)
            cv2.line(img, p0, p1, color, line_width, cv2.LINE_AA)


def _render_frame(
    base_img: np.ndarray,
    wrist_img: np.ndarray,
    *,
    pred_tracks: np.ndarray | None = None,
) -> np.ndarray:
    """Raw cameras + colored future tracks (pred_tracks already display-aligned)."""
    left = np.asarray(base_img, dtype=np.uint8).copy()
    right = np.asarray(wrist_img, dtype=np.uint8).copy()
    if pred_tracks is not None and pred_tracks.size > 0:
        pred = np.asarray(pred_tracks, dtype=np.float32).reshape(-1, 39, 2)
        _draw_future_tracks_colormap(left, pred[:, :7])
        _draw_future_tracks_colormap(right, pred[:, 7:32])
        _draw_future_tracks_colormap(right, pred[:, 32:39])
    return np.concatenate([left, right], axis=1).copy()


def _render_original_frame(base_img: np.ndarray, wrist_img: np.ndarray) -> np.ndarray:
    left = np.asarray(base_img, dtype=np.uint8).copy()
    right = np.asarray(wrist_img, dtype=np.uint8).copy()
    return np.concatenate([left, right], axis=1).copy()


def _copy_obs_camera_rgb(obs: dict[str, Any], key: str) -> np.ndarray:
    """Snapshot one camera image from obs (robosuite reuses render buffers)."""
    arr = np.asarray(obs[key])
    if np.issubdtype(arr.dtype, np.floating):
        return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.array(arr, dtype=np.uint8, copy=True)


def _flip_camera_pair(
    agent_img: np.ndarray,
    wrist_img: np.ndarray,
    *,
    flip_image_x: bool,
    flip_image_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if flip_image_x and flip_image_y:
        agent_out = np.ascontiguousarray(agent_img[::-1, ::-1, :])
        wrist_out = np.ascontiguousarray(wrist_img[::-1, ::-1, :])
    else:
        agent_out = agent_img
        wrist_out = wrist_img
        if flip_image_y:
            agent_out = agent_out[::-1, :, :]
            wrist_out = wrist_out[::-1, :, :]
        if flip_image_x:
            agent_out = agent_out[:, ::-1, :]
            wrist_out = wrist_out[:, ::-1, :]
        agent_out = np.ascontiguousarray(agent_out)
        wrist_out = np.ascontiguousarray(wrist_out)
    return agent_out, wrist_out


def _render_env_camera_rgb(inner: Any, camera_name: str, *, resolution: int) -> np.ndarray:
    """Fresh offscreen render for one camera (avoids robosuite obs buffer aliasing)."""
    img = inner.sim.render(
        camera_name=camera_name,
        width=int(resolution),
        height=int(resolution),
        depth=False,
    )
    return np.asarray(img, dtype=np.uint8).copy()


def _obs_to_display_images(
    obs: dict[str, Any],
    *,
    env: Any | None = None,
    env_resolution: int = 256,
    resize_size: int,
    flip_image_x: bool,
    flip_image_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Sim -> copy cameras -> flip -> resize uint8 (video-safe; prefers direct sim.render)."""
    if env is not None:
        inner = _unwrap_inner_env(env)
        inner.sim.forward()
        agent_raw = _render_env_camera_rgb(inner, "agentview", resolution=env_resolution)
        wrist_raw = _render_env_camera_rgb(inner, "robot0_eye_in_hand", resolution=env_resolution)
    else:
        agent_raw = _copy_obs_camera_rgb(obs, "agentview_image")
        wrist_raw = _copy_obs_camera_rgb(obs, "robot0_eye_in_hand_image")
    agent_flip, wrist_flip = _flip_camera_pair(
        agent_raw,
        wrist_raw,
        flip_image_x=flip_image_x,
        flip_image_y=flip_image_y,
    )
    agent_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agent_flip, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_flip, resize_size, resize_size)
    )
    return np.asarray(agent_img, dtype=np.uint8).copy(), np.asarray(wrist_img, dtype=np.uint8).copy()


def _append_sim_rollout_video_frames(
    video_frames: list[np.ndarray],
    original_video_frames: list[np.ndarray] | None,
    obs: dict[str, Any],
    env: Any,
    *,
    env_resolution: int,
    resize_size: int,
    flip_image_x: bool,
    flip_image_y: bool,
    pred_tracks: np.ndarray | None,
    save_original_video: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Record one frame: env.step state -> sim.render cameras -> evaluate-style concat."""
    agent_img, wrist_img = _obs_to_display_images(
        obs,
        env=env,
        env_resolution=env_resolution,
        resize_size=resize_size,
        flip_image_x=flip_image_x,
        flip_image_y=flip_image_y,
    )
    video_frames.append(_render_frame(agent_img, wrist_img, pred_tracks=pred_tracks))
    if save_original_video and original_video_frames is not None:
        original_video_frames.append(_render_original_frame(agent_img, wrist_img))
    return agent_img, wrist_img


def _unwrap_inner_env(env: Any):
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    return cur


def _assert_flat_init_matches_sim(env: Any, init_vec: Any, *, label: str) -> None:
    inner = _unwrap_inner_env(env)
    need = 1 + int(inner.sim.model.nq) + int(inner.sim.model.nv)
    got = int(np.asarray(init_vec, dtype=np.float64).reshape(-1).size)
    if got == need:
        return
    try:
        roots = f"bddl_files={get_libero_path('bddl_files')!s}, init_states={get_libero_path('init_states')!s}"
    except Exception:
        roots = "(could not resolve get_libero_path)"
    raise ValueError(
        f"{label}: flattened init length {got} != simulation requirement {need} "
        f"(1+time+qpos+qvel with nq={inner.sim.model.nq}, nv={inner.sim.model.nv}). "
        f"Loaded assets: {roots}."
    )


def _offscreen_env_kwargs(resolution: int, *, render_gpu_device_id: int = -1) -> dict[str, Any]:
    rid = int(render_gpu_device_id)
    if rid < 0:
        env_rid = os.environ.get("OPENPI_RENDER_GPU_DEVICE_ID", "").strip()
        if env_rid.isdigit():
            rid = int(env_rid)
    return {
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": rid,
    }


def _build_mesh_env(
    task,
    resolution: int,
    seed: int,
    *,
    suite_name: str = "",
    render_gpu_device_id: int = -1,
):
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        **_offscreen_env_kwargs(resolution, render_gpu_device_id=render_gpu_device_id),
    )
    env.seed(seed)
    scene = _scene_from_task(task_description)
    agent_ids, wrist_ids = _mesh_seeds_for_scene(scene)
    env = MeshVertexWrapper(env, max_points=10000, include_non_mesh_geoms=True)
    env = MeshVertexTracker(
        env,
        agentview_vertex_ids=agent_ids,
        eye_in_hand_vertex_ids=wrist_ids,
        img_height=resolution,
        img_width=resolution,
    )
    env._eval_scene = scene  # type: ignore[attr-defined]
    env._eval_fallback_agent_ids = agent_ids  # type: ignore[attr-defined]
    env._eval_fallback_wrist_ids = wrist_ids  # type: ignore[attr-defined]
    env._eval_use_dynamic_gripper = _suite_uses_dynamic_gripper(suite_name)  # type: ignore[attr-defined]
    return env, task_description


def _maybe_refresh_gripper_query_points(
    env: Any,
    obs: dict[str, Any],
    *,
    resolution: int,
) -> dict[str, Any]:
    if not getattr(env, "_eval_use_dynamic_gripper", False):
        return obs
    agent_fb = getattr(env, "_eval_fallback_agent_ids", [])
    wrist_fb = getattr(env, "_eval_fallback_wrist_ids", [])
    apply_dynamic_gripper_vertex_ids(
        env,
        fallback_agent_ids=agent_fb,
        fallback_wrist_ids=wrist_fb,
        img_hw=(resolution, resolution),
    )
    obs = refresh_tracked_vertices_obs(env, obs)
    tracker = find_mesh_vertex_tracker(env)
    if tracker is not None:
        agent_ids = tracker.agentview_vertex_ids
        wrist_ids = tracker.eye_in_hand_vertex_ids
        if list(agent_ids) == list(agent_fb) and list(wrist_ids) == list(wrist_fb):
            logging.warning(
                "Dynamic gripper vertex selection fell back to Panda scene defaults; "
                "check robot gripper geoms."
            )
    return obs


def _load_policy(spec: PolicySpec):
    cfg = training_config.get_config(spec.config_name)
    eval_cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, include_track_targets=False))
    eval_data_cfg = eval_cfg.data.create(eval_cfg.assets_dirs, eval_cfg.model)
    return _policy_config.create_trained_policy(
        eval_cfg,
        spec.checkpoint_dir,
        repack_transforms=eval_data_cfg.repack_transforms,
    )


def _checkpoint_ready(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    return (p / "params").exists() or (p / "model.safetensors").exists()


def _resolve_run_name(args: Args, policies: list[PolicySpec]) -> str:
    if args.run_name:
        return args.run_name
    if not policies:
        raise ValueError("Cannot resolve run_name without at least one policy.")
    p = Path(policies[-1].checkpoint_dir).expanduser()
    name = p.name
    if name.isdigit():
        name = p.parent.name
    return name or "run"


def _prepare_raw_item(
    obs: dict[str, Any],
    task_description: str,
    *,
    resize_size: int,
    flip_image_x: bool,
    flip_image_y: bool,
    xflip_points: bool,
    yflip_points: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    agent_img, wrist_img = _obs_to_display_images(
        obs,
        resize_size=resize_size,
        flip_image_x=flip_image_x,
        flip_image_y=flip_image_y,
    )

    h, w = int(obs["agentview_image"].shape[0]), int(obs["agentview_image"].shape[1])
    agent_mesh = _ensure_n_points_xy(
        _to_norm_xy(obs["agentview_tracked_vertices_2d"], h=h, w=w), 7, label="agentview"
    )
    wrist_mesh = _ensure_n_points_xy(
        _to_norm_xy(obs["eye_in_hand_tracked_vertices_2d"], h=h, w=w), 7, label="wrist_mesh"
    )
    agent_mesh = _flip_xy(agent_mesh, xflip=xflip_points, yflip=yflip_points)
    wrist_mesh = _flip_xy(wrist_mesh, xflip=xflip_points, yflip=yflip_points)

    wrist_grid = _make_grid_25().astype(np.float32)
    wrist_tracks_32 = np.concatenate([wrist_grid, wrist_mesh], axis=0).astype(np.float32)
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)

    raw_item = {
        "image": agent_img,
        "wrist_image": wrist_img,
        "state": state,
        "actions": np.zeros((16, 7), dtype=np.float32),
        "prompt": str(task_description),
        "agentview_mesh_vertices_2d": agent_mesh.astype(np.float32),
        "wrist_mesh_vertices_2d": wrist_mesh.astype(np.float32),
        "wrist_tracks": wrist_tracks_32,
        "wrist_grid_mesh_32": wrist_tracks_32,
    }
    return raw_item, agent_img, wrist_img


def _task_ids_for_suite(suite, *, all_tasks: bool, task_index: int) -> list[int]:
    if all_tasks:
        return list(range(suite.n_tasks))
    return [int(np.clip(task_index, 0, suite.n_tasks - 1))]


@dataclasses.dataclass(frozen=True)
class SourceRolloutRecording:
    executed_actions: np.ndarray
    tracks: np.ndarray | None
    replan_steps: int
    source_hdf5_path: str
    source_success: bool | None


def _find_source_rollout_hdf5(
    hdf5_dir: Path,
    *,
    policy_prefix: str,
    source_suite: str,
    task_id: int,
    rollout_idx: int,
) -> Path:
    pattern = f"{policy_prefix}_{source_suite}_{task_id:02d}_{rollout_idx:02d}_*.hdf5"
    matches = sorted(hdf5_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No source HDF5 for task={task_id} rollout={rollout_idx} under {hdf5_dir} (pattern {pattern})"
        )
    return matches[0]


def _sawyer_init_from_panda_source(
    *,
    source_suite: str,
    sawyer_suite: str,
    task_id: int,
    rollout_idx: int,
    resolution: int,
    render_gpu_device_id: int = -1,
) -> np.ndarray:
    """Transfer libero_object (Franka) pruned_init to libero_object_sawyer sim state."""
    bench_dict = benchmark.get_benchmark_dict()
    task_src = bench_dict[source_suite]().get_task(task_id)
    task_sawyer = bench_dict[sawyer_suite]().get_task(task_id)
    panda_inits = bench_dict[source_suite]().get_task_init_states(task_id)
    panda_vec = panda_inits[rollout_idx % len(panda_inits)]
    if hasattr(panda_vec, "numpy"):
        panda_vec = panda_vec.numpy()
    panda_vec = np.asarray(panda_vec, dtype=np.float64).reshape(-1)

    bddl_root = Path(get_libero_path("bddl_files"))
    bddl_src = bddl_root / task_src.problem_folder / task_src.bddl_file
    bddl_sawyer = bddl_root / task_sawyer.problem_folder / task_sawyer.bddl_file
    return target_init_from_source_vec(
        panda_vec,
        bddl_source=bddl_src,
        bddl_target=bddl_sawyer,
        resolution=resolution,
        render_gpu_device_id=render_gpu_device_id,
    )


@dataclasses.dataclass(frozen=True)
class SourceEefKeyframe:
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_qpos: np.ndarray
    gripper_action: float


def _build_plain_libero_env(
    task,
    resolution: int,
    seed: int,
    *,
    render_gpu_device_id: int = -1,
):
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        **_offscreen_env_kwargs(resolution, render_gpu_device_id=render_gpu_device_id),
    )
    env.seed(seed)
    return env, task.language


def _panda_gripper_qpos_to_rethink(gripper_qpos: np.ndarray) -> np.ndarray:
    gq = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
    if gq.size >= 2:
        spread = 0.5 * (abs(float(gq[0])) + abs(float(gq[1])))
    else:
        spread = abs(float(gq[0])) if gq.size else 0.020833
    spread = float(np.clip(spread, 0.0, 0.04))
    return np.array([spread, -spread], dtype=np.float64)


def _set_robot_gripper_qpos_flat(sim: Any, gripper_qpos: np.ndarray) -> None:
    vals = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
    specs: list[tuple[int, int]] = []
    for j in range(sim.model.njnt):
        name = sim.model.joint(j).name
        if "gripper" not in name:
            continue
        adr = int(sim.model.jnt_qposadr[j])
        jtype = int(sim.model.jnt_type[j])
        if jtype == 3:
            sz = 1
        elif jtype == 2:
            sz = 4
        elif jtype == 0:
            sz = 7
        else:
            sz = int(sim.model.jnt_dof[j])
        specs.append((adr, sz))
    specs.sort(key=lambda x: x[0])
    for i, (adr, sz) in enumerate(specs):
        if i >= vals.size:
            break
        sim.data.qpos[adr : adr + min(sz, 1)] = vals[i : i + 1]


def _sync_osc_controller(env: Any) -> None:
    inner = _unwrap_inner_env(env)
    robot = inner.robots[0]
    controller = getattr(robot, "controller", None)
    if controller is None:
        return
    controller.update_initial_joints(robot._joint_positions)
    controller.reset_goal()


def _eef_pose_from_env(env: Any, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inner = _unwrap_inner_env(env)
    robot = inner.robots[0]
    eef_name = robot.robot_model.eef_name
    pos = inner.sim.data.get_body_xpos(eef_name).copy()
    quat = inner.sim.data.get_body_xquat(eef_name).copy()
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).copy()
    return pos, quat, gripper_qpos


def _record_source_eef_trajectory(
    *,
    source_suite: str,
    task_id: int,
    rollout_idx: int,
    actions: np.ndarray,
    seed: int,
    resolution: int,
    num_steps_wait: int,
    render_gpu_device_id: int,
) -> list[SourceEefKeyframe]:
    """Replay source actions on the source robot and record post-step EEF keyframes."""
    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[source_suite]()
    task = suite.get_task(task_id)
    env, _ = _build_plain_libero_env(
        task, resolution, seed, render_gpu_device_id=render_gpu_device_id
    )
    env.reset()
    init_vec = suite.get_task_init_states(task_id)[rollout_idx % len(suite.get_task_init_states(task_id))]
    if hasattr(init_vec, "numpy"):
        init_vec = init_vec.numpy()
    obs = env.set_init_state(np.asarray(init_vec, dtype=np.float64).reshape(-1))

    keyframes: list[SourceEefKeyframe] = []
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    for step_idx in range(actions.shape[0]):
        obs, _, done, _ = env.step(actions[step_idx].tolist())
        pos, quat, gripper_qpos = _eef_pose_from_env(env, obs)
        keyframes.append(
            SourceEefKeyframe(
                eef_pos=pos,
                eef_quat=quat,
                gripper_qpos=gripper_qpos,
                gripper_action=float(actions[step_idx, 6]),
            )
        )
        if done:
            break
    env.close()
    logging.info(
        "Recorded %d source EEF keyframes for %s task=%d rollout=%d",
        len(keyframes),
        source_suite,
        task_id,
        rollout_idx,
    )
    return keyframes


def _sawyer_step_eef_ik(
    env: Any,
    keyframe: SourceEefKeyframe,
    *,
    arm_warmstart: np.ndarray,
) -> tuple[Any, float, bool, dict[str, Any], np.ndarray]:
    """IK-match target EEF on Sawyer, then advance one env step (arm hold + source gripper)."""
    inner = _unwrap_inner_env(env)
    sim = inner.sim
    robot = inner.robots[0]
    q_arm = ik_arm_to_eef(
        sim,
        robot,
        keyframe.eef_pos,
        keyframe.eef_quat,
        arm_warmstart,
    )
    sim.data.qpos[robot._ref_joint_pos_indexes] = q_arm
    _set_robot_gripper_qpos_flat(sim, _panda_gripper_qpos_to_rethink(keyframe.gripper_qpos))
    sim.forward()
    _sync_osc_controller(env)
    hold_action = [0.0] * 6 + [float(keyframe.gripper_action)]
    obs, reward, done, info = env.step(hold_action)
    return obs, float(reward), bool(done), info, q_arm


def _load_source_rollout_hdf5(path: Path, *, default_replan_steps: int) -> SourceRolloutRecording:
    with h5py.File(path, "r") as f:
        if "executed_actions" in f:
            actions = np.asarray(f["executed_actions"], dtype=np.float32)
        elif "actions" in f:
            actions = np.asarray(f["actions"], dtype=np.float32)
        else:
            raise KeyError(f"{path}: missing executed_actions/actions dataset")
        tracks = None
        if "tracks" in f:
            tracks = np.asarray(f["tracks"], dtype=np.float32)
        replan = int(f.attrs.get("replan_steps", default_replan_steps))
        source_success = bool(f.attrs["success"]) if "success" in f.attrs else None
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{path}: expected actions (T, 7), got {actions.shape}")
    return SourceRolloutRecording(
        executed_actions=actions,
        tracks=tracks,
        replan_steps=max(1, replan),
        source_hdf5_path=str(path),
        source_success=source_success,
    )


def _source_tracks_window(
    tracks: np.ndarray | None,
    *,
    control_step: int,
    replan_steps: int,
    track_horizon: int = 16,
) -> np.ndarray | None:
    if tracks is None or tracks.size == 0:
        return None
    replan_idx = min(control_step // replan_steps, tracks.shape[0] - 1)
    horizon_offset = control_step % replan_steps
    return _pred_tracks_horizon_window(
        tracks[replan_idx],
        horizon_offset=horizon_offset,
        track_horizon=track_horizon,
    )


def _save_video(video_frames: list[np.ndarray], video_path: Path, fps: int) -> None:
    if not video_frames:
        return
    frames = [np.asarray(frame, dtype=np.uint8).copy() for frame in video_frames]
    write_kwargs: dict[str, Any] = {
        "codec": "libx264",
        "audio": False,
        "logger": None,
    }
    # ffmpeg infers muxer from extension; non-standard suffixes (e.g. .pm4) need explicit format.
    if video_path.suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        write_kwargs["ffmpeg_params"] = ["-f", "mp4"]
    ImageSequenceClip(frames, fps=fps).write_videofile(str(video_path), **write_kwargs)


def _save_rollout_predictions_hdf5(
    hdf5_path: Path,
    *,
    spec: PolicySpec,
    suite_name: str,
    task_id: int,
    task_description: str,
    rollout_idx: int,
    success: bool,
    reward_sum: float,
    replan_steps: int,
    executed_actions: list[np.ndarray],
    pred_action_chunks: list[np.ndarray],
    pred_track_chunks: list[np.ndarray | None],
    pred_env_steps: list[int],
    selection_records: list[dict[str, Any]] | None = None,
    action_selection_mode: str = "none",
) -> None:
    """Save executed actions and per-replan predictions (display-aligned tracks)."""
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        f.attrs["policy_name"] = spec.name
        f.attrs["config_name"] = spec.config_name
        f.attrs["checkpoint_dir"] = spec.checkpoint_dir
        f.attrs["suite"] = suite_name
        f.attrs["task_id"] = int(task_id)
        f.attrs["task_description"] = task_description
        f.attrs["rollout_idx"] = int(rollout_idx)
        f.attrs["success"] = bool(success)
        f.attrs["reward_sum"] = float(reward_sum)
        f.attrs["replan_steps"] = int(replan_steps)
        f.attrs["num_replans"] = int(len(pred_action_chunks))
        f.attrs["tracks_layout"] = "replan,horizon,39,2"
        f.attrs["tracks_coords"] = "display_aligned"
        f.attrs["action_selection"] = action_selection_mode

        if executed_actions:
            actions = np.stack(executed_actions, axis=0).astype(np.float32)
            f.create_dataset("actions", data=actions, compression="gzip")
            f.create_dataset("executed_actions", data=actions, compression="gzip")
        if pred_action_chunks:
            pred_actions = np.stack(pred_action_chunks, axis=0).astype(np.float32)
            f.create_dataset("pred_actions", data=pred_actions, compression="gzip")
        if pred_track_chunks:
            template = next((t for t in pred_track_chunks if t is not None), None)
            if template is not None:
                track_stack = []
                for tracks in pred_track_chunks:
                    if tracks is None:
                        track_stack.append(np.full(template.shape, np.nan, dtype=np.float32))
                    else:
                        track_stack.append(np.asarray(tracks, dtype=np.float32))
                track_arr = np.stack(track_stack, axis=0).astype(np.float32)
                f.create_dataset("tracks", data=track_arr, compression="gzip")
                if track_arr.ndim == 3:
                    f.create_dataset(
                        "tracks_flat",
                        data=track_arr.reshape(track_arr.shape[0], track_arr.shape[1], -1),
                        compression="gzip",
                    )
        if pred_env_steps:
            f.create_dataset(
                "pred_env_steps",
                data=np.asarray(pred_env_steps, dtype=np.int32),
                compression="gzip",
            )
        if selection_records:
            best_idxs = np.asarray(
                [int(r["best_idx"]) for r in selection_records], dtype=np.int32
            )
            f.create_dataset("best_candidate_idx", data=best_idxs, compression="gzip")
            score_rows = []
            for r in selection_records:
                scores = np.asarray(r["scores"], dtype=np.float32)
                score_rows.append(scores)
            # Ragged lengths should not happen (fixed n_candidates); pad just in case.
            max_n = max(len(s) for s in score_rows)
            score_mat = np.full((len(score_rows), max_n), np.nan, dtype=np.float32)
            for i, s in enumerate(score_rows):
                score_mat[i, : len(s)] = s
            f.create_dataset("candidate_scores", data=score_mat, compression="gzip")
            f.attrs["n_candidates"] = int(selection_records[0].get("n_candidates", max_n))


def _format_suite_summary_text(
    *,
    policy_name: str,
    suite_name: str,
    seed: int,
    rows: list[dict[str, Any]],
    per_task: list[dict[str, Any]],
) -> str:
    suite_rows = [r for r in rows if str(r["policy_name"]) == policy_name and str(r["suite"]) == suite_name]
    suite_task_rows = [
        r for r in per_task if str(r["policy_name"]) == policy_name and str(r["suite"]) == suite_name
    ]
    suite_task_rows = sorted(suite_task_rows, key=lambda x: int(x["task_id"]))

    total_rollouts = len(suite_rows)
    total_successes = int(sum(1 for r in suite_rows if r["success"]))
    total_rate = 100.0 * float(total_successes / max(1, total_rollouts))

    lines: list[str] = []
    lines.append("======================================================================")
    lines.append(f"PER-TASK SUCCESS RATE SUMMARY ({suite_name})")
    lines.append("======================================================================")
    for idx, task in enumerate(suite_task_rows, start=1):
        successes = int(task["successes"])
        rollouts = int(task["rollouts"])
        accuracy = 100.0 * float(task["accuracy"])
        task_desc = str(task["task_description"])
        lines.append(
            f"Task {idx:>2}: {successes:>2}/{rollouts:<2} = {accuracy:>6.1f}% | {task_desc}"
        )
    lines.append("======================================================================")
    lines.append(f"TOTAL: {total_successes}/{total_rollouts} = {total_rate:.1f}%")
    lines.append(f"Total episodes: {total_rollouts}")
    lines.append("")
    lines.append(f"task_suite_name: {suite_name}")
    lines.append(f"policy_name: {policy_name}")
    sel_modes = sorted(
        {
            str(r.get("action_selection", "unknown"))
            for r in suite_rows
            if "action_selection" in r
        }
    )
    if sel_modes:
        lines.append(f"action_selection: {','.join(sel_modes)}")
    lines.append(f"seed: {seed}")
    lines.append("")
    return "\n".join(lines)


def _write_run_text_report(
    *,
    out_root: Path,
    run_name: str,
    suite_names: tuple[str, ...],
    rows: list[dict[str, Any]],
    per_task: list[dict[str, Any]],
    seed: int,
) -> str:
    policy_names = sorted({str(r["policy_name"]) for r in rows})
    parts: list[str] = []
    for policy_name in policy_names:
        suites = sorted({str(r["suite"]) for r in rows if str(r["policy_name"]) == policy_name})
        parts.extend(
            _format_suite_summary_text(
                policy_name=policy_name,
                suite_name=suite_name,
                seed=seed,
                rows=rows,
                per_task=per_task,
            )
            for suite_name in suites
        )
    suite_slug = "_".join(suite_names)
    report_path = out_root / f"{run_name}_{suite_slug}.txt"
    report_path.write_text("\n".join(parts), encoding="utf-8")
    return str(report_path)


def _run_policy_on_suite(
    spec: PolicySpec,
    args: Args,
    suite_name: str,
    tracks_video_root: Path,
) -> list[dict[str, Any]]:
    use_selection = bool(args.use_action_selection)
    action_selection_mode = (
        f"ctrlworld_vlac,n={args.n_candidates}" if use_selection else "none (direct_policy)"
    )
    selector = None
    policy = None
    if use_selection:
        device_map = getattr(args, "device_map", None) or {
            "policy": "cuda:0",
            "wm": "cuda:0",
            "vlac": "cuda:0",
        }
        selector = _build_action_selector(spec, args)
        logging.info(
            "Loading action selector (Ctrl-World / action WM + VLAC), devices=%s",
            device_map,
        )
        selector.load(
            policy_device=device_map["policy"],
            wm_device=device_map["wm"],
            vlac_device=device_map["vlac"],
        )
    else:
        policy = _load_policy(spec)

    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[suite_name]()
    task_ids = _task_ids_for_suite(suite, all_tasks=args.all_tasks, task_index=args.task_index)

    rows: list[dict[str, Any]] = []
    point_xflip = args.flip_image_x if args.align_point_flips_with_image else args.xflip_points
    point_yflip = args.flip_image_y if args.align_point_flips_with_image else args.yflip_points
    max_steps = _max_steps_for_suite(suite_name)
    save_hdf5 = bool(args.save_rollout_predictions)
    save_original_video = bool(args.save_original_rollout_video)
    infer_return_tracks = True

    logging.info("Suite %s action_selection=%s", suite_name, action_selection_mode)
    if save_hdf5:
        logging.info(
            "HDF5 per rollout: %s/<suite>/hdf5/*.hdf5 (actions, pred_actions, display-aligned tracks)",
            tracks_video_root,
        )

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, task_description = _build_mesh_env(
            task,
            args.env_resolution,
            args.seed,
            suite_name=suite_name,
            render_gpu_device_id=args.render_gpu_device_id,
        )

        for rollout_idx in range(args.rollouts_per_task):
            env.reset()
            _raw_init = initial_states[rollout_idx % len(initial_states)]
            _assert_flat_init_matches_sim(
                env,
                _raw_init,
                label=f"{suite_name} task_id={task_id} rollout={rollout_idx}",
            )
            obs = env.set_init_state(_raw_init)
            obs = _maybe_refresh_gripper_query_points(env, obs, resolution=args.env_resolution)

            action_plan = collections.deque()
            video_frames: list[np.ndarray] = []
            original_video_frames: list[np.ndarray] = []
            executed_actions: list[np.ndarray] = []
            pred_action_chunks: list[np.ndarray] = []
            pred_track_chunks: list[np.ndarray | None] = []
            pred_env_steps: list[int] = []
            selection_records: list[dict[str, Any]] = []
            total_reward = 0.0
            done = False
            last_pred_tracks: np.ndarray | None = None
            control_step = 0

            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    total_reward += float(reward)
                    continue

                raw_item, agent_img, wrist_img = _prepare_raw_item(
                    obs,
                    task_description,
                    resize_size=args.resize_size,
                    flip_image_x=args.flip_image_x,
                    flip_image_y=args.flip_image_y,
                    xflip_points=point_xflip,
                    yflip_points=point_yflip,
                )

                if not action_plan:
                    if selector is not None:
                        sel = selector.select_best_candidate(
                            raw_item,
                            agent_img=agent_img,
                            wrist_img=wrist_img,
                            prompt=task_description,
                        )
                        action_chunk = np.asarray(sel.actions, dtype=np.float32)
                        tracks_raw = sel.track_predictions
                        selection_records.append(
                            {
                                "best_idx": sel.best_idx,
                                "scores": list(sel.scores),
                                "n_candidates": sel.n_candidates,
                                "policy_ms": sel.policy_ms,
                                "wm_ms": sel.wm_ms,
                                "vlac_ms": sel.vlac_ms,
                                "env_step": control_step,
                            }
                        )
                    else:
                        assert policy is not None
                        infer_out = policy.infer(raw_item, return_tracks=infer_return_tracks)
                        action_chunk = np.asarray(infer_out["actions"], dtype=np.float32)
                        tracks_raw = infer_out.get("track_predictions")

                    pred_action_chunks.append(action_chunk.copy())
                    pred_env_steps.append(control_step)
                    if tracks_raw is not None:
                        tracks_raw = np.asarray(tracks_raw, dtype=np.float32)
                        last_pred_tracks = _align_pred_tracks_for_display(
                            tracks_raw,
                            flip_image_x=args.flip_image_x,
                            flip_image_y=args.flip_image_y,
                        )
                        pred_track_chunks.append(last_pred_tracks.copy())
                    else:
                        pred_track_chunks.append(None)
                        last_pred_tracks = None
                    if len(action_chunk) < args.replan_steps:
                        raise ValueError(
                            f"Policy returned {len(action_chunk)} actions, "
                            f"expected at least replan_steps={args.replan_steps}"
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                horizon_offset = args.replan_steps - len(action_plan)
                pred_window = _pred_tracks_horizon_window(
                    last_pred_tracks,
                    horizon_offset=horizon_offset,
                )
                video_frames.append(
                    _render_frame(agent_img, wrist_img, pred_tracks=pred_window)
                )
                if save_original_video:
                    original_video_frames.append(
                        _render_original_frame(agent_img, wrist_img)
                    )

                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                executed_actions.append(action.copy())
                obs, reward, done, _ = env.step(action.tolist())
                total_reward += float(reward)
                control_step += 1
                if done:
                    break

            status = "success" if done else "failure"
            task_slug = task_description.replace(" ", "_")[:120]
            video_name = f"{spec.name}_{suite_name}_{task_id:02d}_{rollout_idx:02d}_{status}_{task_slug}.mp4"

            tracks_video_dir = tracks_video_root / suite_name
            tracks_video_dir.mkdir(parents=True, exist_ok=True)
            video_path = tracks_video_dir / video_name
            _save_video(video_frames, video_path, args.fps)

            hdf5_path: str | None = None
            if save_hdf5:
                hdf5_dir = tracks_video_dir / "hdf5"
                hdf5_dir.mkdir(parents=True, exist_ok=True)
                hdf5_file = hdf5_dir / video_name.replace(".mp4", ".hdf5")
                _save_rollout_predictions_hdf5(
                    hdf5_file,
                    spec=spec,
                    suite_name=suite_name,
                    task_id=task_id,
                    task_description=task_description,
                    rollout_idx=rollout_idx,
                    success=done,
                    reward_sum=total_reward,
                    replan_steps=args.replan_steps,
                    executed_actions=executed_actions,
                    pred_action_chunks=pred_action_chunks,
                    pred_track_chunks=pred_track_chunks,
                    pred_env_steps=pred_env_steps,
                    selection_records=selection_records or None,
                    action_selection_mode=action_selection_mode,
                )
                hdf5_path = str(hdf5_file)
                logging.info("Saved HDF5: %s", hdf5_path)

            original_video_path: str | None = None
            if save_original_video:
                original_video_dir = tracks_video_root / suite_name / "original_rollout"
                original_video_dir.mkdir(parents=True, exist_ok=True)
                original_path = original_video_dir / video_name
                _save_video(original_video_frames, original_path, args.fps)
                original_video_path = str(original_path)

            row = {
                "policy_name": spec.name,
                "config_name": spec.config_name,
                "checkpoint_dir": spec.checkpoint_dir,
                "suite": suite_name,
                "task_id": task_id,
                "task_description": task_description,
                "rollout_idx": rollout_idx,
                "success": bool(done),
                "reward_sum": float(total_reward),
                "steps": int(len(executed_actions)),
                "control_steps": control_step,
                "video_path": str(video_path),
                "predictions_hdf5_path": hdf5_path,
                "original_video_path": original_video_path,
                "has_predicted_tracks_overlay": bool(last_pred_tracks is not None),
                "action_selection": action_selection_mode,
                "num_selection_replans": len(selection_records),
            }
            rows.append(row)
            logging.info(
                "%s %s task=%d rollout=%d success=%s reward_sum=%.3f steps=%d",
                spec.name,
                suite_name,
                task_id,
                rollout_idx,
                done,
                total_reward,
                control_step,
            )

        env.close()

    return rows


def _run_source_replay_on_sawyer(
    args: Args,
    tracks_video_root: Path,
) -> list[dict[str, Any]]:
    """Execute source-suite HDF5 actions on *_sawyer; overlay source tracks on video."""
    sawyer_suite = _normalize_suite_name(args.replay_sawyer_suite)
    source_suite = _normalize_suite_name(args.replay_source_suite)
    hdf5_dir = Path(args.replay_hdf5_dir).expanduser()
    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"replay_hdf5_dir not found: {hdf5_dir}")

    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[sawyer_suite]()
    task_ids = _task_ids_for_suite(suite, all_tasks=args.all_tasks, task_index=args.task_index)

    rows: list[dict[str, Any]] = []
    point_xflip = args.flip_image_x if args.align_point_flips_with_image else args.xflip_points
    point_yflip = args.flip_image_y if args.align_point_flips_with_image else args.yflip_points
    max_steps = _max_steps_for_suite(sawyer_suite)
    save_hdf5 = bool(args.save_rollout_predictions)
    save_original_video = bool(args.save_original_rollout_video)
    replay_mode = str(args.replay_mode).strip().lower()
    if replay_mode not in {"delta", "eef_ik"}:
        raise ValueError(f"replay_mode must be 'delta' or 'eef_ik', got {args.replay_mode!r}")
    policy_label = "panda_replay" if source_suite == "libero_object" else "source_replay"
    if replay_mode == "eef_ik":
        policy_label = "panda_eef_ik_replay" if source_suite == "libero_object" else "eef_ik_replay"

    logging.info(
        "Source-action replay: %s -> %s, HDF5 from %s (prefix=%s, panda_init=%s, mode=%s)",
        source_suite,
        sawyer_suite,
        hdf5_dir,
        args.replay_hdf5_policy_prefix,
        args.replay_use_panda_init,
        replay_mode,
    )

    for task_id in task_ids:
        task = suite.get_task(task_id)
        sawyer_inits = suite.get_task_init_states(task_id)
        env: Any | None = None
        task_description = task.language

        for rollout_idx in range(args.rollouts_per_task):
            source_path = _find_source_rollout_hdf5(
                hdf5_dir,
                policy_prefix=args.replay_hdf5_policy_prefix.strip(),
                source_suite=source_suite,
                task_id=task_id,
                rollout_idx=rollout_idx,
            )
            recording = _load_source_rollout_hdf5(
                source_path,
                default_replan_steps=args.replan_steps,
            )
            replan_steps = recording.replan_steps
            eef_keyframes: list[SourceEefKeyframe] | None = None
            if replay_mode == "eef_ik":
                eef_keyframes = _record_source_eef_trajectory(
                    source_suite=source_suite,
                    task_id=task_id,
                    rollout_idx=rollout_idx,
                    actions=recording.executed_actions,
                    seed=args.seed,
                    resolution=args.env_resolution,
                    num_steps_wait=args.num_steps_wait,
                    render_gpu_device_id=args.render_gpu_device_id,
                )
                if not eef_keyframes:
                    raise RuntimeError(
                        f"No EEF keyframes recorded for {source_suite} task={task_id} rollout={rollout_idx}"
                    )

            if args.replay_use_panda_init:
                # IK init spins up temporary OffScreenRenderEnv; close main env first so EGL
                # on render_gpu is not left in a bad state (green/flickering videos).
                if env is not None:
                    env.close()
                    env = None
                _raw_init = _sawyer_init_from_panda_source(
                    source_suite=source_suite,
                    sawyer_suite=sawyer_suite,
                    task_id=task_id,
                    rollout_idx=rollout_idx,
                    resolution=args.env_resolution,
                    render_gpu_device_id=args.render_gpu_device_id,
                )
                logging.info(
                    "%s task=%d rollout=%d: init from %s pruned_init (IK → sawyer)",
                    sawyer_suite,
                    task_id,
                    rollout_idx,
                    source_suite,
                )
                env, task_description = _build_mesh_env(
                    task,
                    args.env_resolution,
                    args.seed,
                    suite_name=sawyer_suite,
                    render_gpu_device_id=args.render_gpu_device_id,
                )
                env.reset()
            else:
                _raw_init = sawyer_inits[rollout_idx % len(sawyer_inits)]
                if env is None:
                    env, task_description = _build_mesh_env(
                        task,
                        args.env_resolution,
                        args.seed,
                        suite_name=sawyer_suite,
                        render_gpu_device_id=args.render_gpu_device_id,
                    )
                env.reset()
            _assert_flat_init_matches_sim(
                env,
                _raw_init,
                label=f"{sawyer_suite} task_id={task_id} rollout={rollout_idx}",
            )
            obs = env.set_init_state(_raw_init)
            obs = _maybe_refresh_gripper_query_points(env, obs, resolution=args.env_resolution)

            video_frames: list[np.ndarray] = []
            original_video_frames: list[np.ndarray] = []
            executed_actions: list[np.ndarray] = []
            total_reward = 0.0
            done = False
            control_step = 0
            source_actions = recording.executed_actions
            arm_warmstart = np.asarray(
                _unwrap_inner_env(env).robots[0].robot_model.init_qpos, dtype=np.float64
            )

            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    total_reward += float(reward)
                    continue

                pred_window = None
                if args.replay_overlay_source_tracks:
                    pred_window = _source_tracks_window(
                        recording.tracks,
                        control_step=control_step,
                        replan_steps=replan_steps,
                    )
                _append_sim_rollout_video_frames(
                    video_frames,
                    original_video_frames if save_original_video else None,
                    obs,
                    env,
                    env_resolution=args.env_resolution,
                    resize_size=args.resize_size,
                    flip_image_x=args.flip_image_x,
                    flip_image_y=args.flip_image_y,
                    pred_tracks=pred_window,
                    save_original_video=save_original_video,
                )

                if control_step >= source_actions.shape[0]:
                    logging.warning(
                        "%s task=%d rollout=%d: exhausted source actions at step %d",
                        sawyer_suite,
                        task_id,
                        rollout_idx,
                        control_step,
                    )
                    break

                if replay_mode == "eef_ik":
                    assert eef_keyframes is not None
                    if control_step >= len(eef_keyframes):
                        logging.warning(
                            "%s task=%d rollout=%d: exhausted EEF keyframes at step %d",
                            sawyer_suite,
                            task_id,
                            rollout_idx,
                            control_step,
                        )
                        break
                    keyframe = eef_keyframes[control_step]
                    executed_actions.append(
                        np.asarray([0.0] * 6 + [keyframe.gripper_action], dtype=np.float32)
                    )
                    obs, reward, done, _, arm_warmstart = _sawyer_step_eef_ik(
                        env,
                        keyframe,
                        arm_warmstart=arm_warmstart,
                    )
                else:
                    action = source_actions[control_step]
                    executed_actions.append(np.asarray(action, dtype=np.float32))
                    obs, reward, done, _ = env.step(action.tolist())
                total_reward += float(reward)
                control_step += 1
                if done:
                    break

            status = "success" if done else "failure"
            task_slug = task_description.replace(" ", "_")[:120]
            video_name = (
                f"{policy_label}_{sawyer_suite}_{task_id:02d}_{rollout_idx:02d}_"
                f"{status}_{task_slug}.mp4"
            )

            tracks_video_dir = tracks_video_root / sawyer_suite / f"{source_suite}_replay"
            tracks_video_dir.mkdir(parents=True, exist_ok=True)
            if args.replay_output_video and rollout_idx == 0 and task_id == task_ids[0]:
                video_path = Path(args.replay_output_video).expanduser()
                video_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                video_path = tracks_video_dir / video_name
            _save_video(video_frames, video_path, args.fps)
            if args.replay_output_video and rollout_idx == 0 and task_id == task_ids[0]:
                logging.info("Replay video saved to %s (success=%s)", video_path, done)

            hdf5_path: str | None = None
            if save_hdf5:
                hdf5_dir_out = tracks_video_dir / "hdf5"
                hdf5_dir_out.mkdir(parents=True, exist_ok=True)
                hdf5_file = hdf5_dir_out / video_name.replace(".mp4", ".hdf5")
                with h5py.File(hdf5_file, "w") as f:
                    f.attrs["mode"] = "source_replay_on_sawyer"
                    f.attrs["source_suite"] = source_suite
                    f.attrs["source_hdf5"] = recording.source_hdf5_path
                    f.attrs["source_success"] = recording.source_success
                    f.attrs["replay_use_panda_init"] = bool(args.replay_use_panda_init)
                    f.attrs["replay_mode"] = replay_mode
                    f.attrs["suite"] = sawyer_suite
                    f.attrs["task_id"] = int(task_id)
                    f.attrs["rollout_idx"] = int(rollout_idx)
                    f.attrs["task_description"] = task_description
                    f.attrs["success"] = bool(done)
                    f.attrs["reward_sum"] = float(total_reward)
                    if executed_actions:
                        arr = np.stack(executed_actions, axis=0).astype(np.float32)
                        f.create_dataset("executed_actions", data=arr, compression="gzip")
                    if recording.tracks is not None:
                        f.create_dataset("source_tracks", data=recording.tracks, compression="gzip")
                hdf5_path = str(hdf5_file)

            original_video_path: str | None = None
            if save_original_video:
                original_video_dir = tracks_video_dir / "original_rollout"
                original_video_dir.mkdir(parents=True, exist_ok=True)
                original_path = original_video_dir / video_name
                _save_video(original_video_frames, original_path, args.fps)
                original_video_path = str(original_path)

            row = {
                "policy_name": policy_label,
                "config_name": "source_replay",
                "checkpoint_dir": recording.source_hdf5_path,
                "suite": sawyer_suite,
                "task_id": task_id,
                "task_description": task_description,
                "rollout_idx": rollout_idx,
                "success": bool(done),
                "reward_sum": float(total_reward),
                "steps": int(len(executed_actions)),
                "control_steps": control_step,
                "video_path": str(video_path),
                "predictions_hdf5_path": hdf5_path,
                "original_video_path": original_video_path,
                "has_predicted_tracks_overlay": recording.tracks is not None,
                "source_hdf5": recording.source_hdf5_path,
                "source_success": recording.source_success,
                "source_suite": source_suite,
            }
            rows.append(row)
            logging.info(
                "source_replay %s task=%d rollout=%d sawyer_success=%s source_success=%s "
                "reward_sum=%.3f steps=%d source=%s",
                sawyer_suite,
                task_id,
                rollout_idx,
                done,
                recording.source_success,
                total_reward,
                control_step,
                source_path.name,
            )

        if env is not None:
            env.close()

    return rows


def _per_task_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in rows:
        key = (str(r["policy_name"]), str(r["suite"]), int(r["task_id"]))
        if key not in buckets:
            buckets[key] = {
                "policy_name": key[0],
                "suite": key[1],
                "task_id": key[2],
                "task_description": str(r["task_description"]),
                "rollouts": 0,
                "successes": 0,
            }
        b = buckets[key]
        b["rollouts"] += 1
        if r["success"]:
            b["successes"] += 1
    out: list[dict[str, Any]] = []
    for _, b in sorted(buckets.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        n = max(1, int(b["rollouts"]))
        b = dict(b)
        b["accuracy"] = float(b["successes"]) / float(n)
        out.append(b)
    return out


def _finalize_gpu_config(args: Args) -> None:
    """Apply multi-GPU env vars before LIBERO envs / models are constructed."""
    policy = (args.policy_gpu or os.environ.get("OPENPI_POLICY_GPU", "")).strip()
    render = (args.render_gpu or os.environ.get("OPENPI_RENDER_GPU", "")).strip()
    wm = (args.wm_gpu or os.environ.get("OPENPI_WM_GPU", "")).strip() or policy
    vlac = (args.vlac_gpu or os.environ.get("OPENPI_VLAC_GPU", "")).strip() or policy
    if not policy or not render:
        raise ValueError(
            "Set both --policy-gpu and --render-gpu (or OPENPI_POLICY_GPU and OPENPI_RENDER_GPU)."
        )
    render_id, device_map = _install_multi_gpu_env(
        policy_gpu=policy,
        render_gpu=render,
        wm_gpu=wm,
        vlac_gpu=vlac,
    )
    args.render_gpu_device_id = render_id
    args.device_map = device_map


def _build_action_selector(spec: PolicySpec, args: Args):
    if str(OPENPI_ROOT) not in sys.path:
        sys.path.insert(0, str(OPENPI_ROOT))
    from scripts.vlac_pipeline.libero_ctrlworld_selection import LiberoCtrlWorldSelector

    diversity = {
        "sampling_mode": args.sampling_mode,
        "tsr_k": args.tsr_k,
        "noise_scale": args.noise_scale,
        "noise_type": args.noise_type,
    }
    return LiberoCtrlWorldSelector(
        config_name=spec.config_name,
        checkpoint_dir=spec.checkpoint_dir,
        ctrlworld_checkpoint=args.ctrlworld_checkpoint,
        svd_path=args.svd_path,
        clip_path=args.clip_path,
        vlac_model=args.vlac_model,
        n_candidates=args.n_candidates,
        wm_num_inference_steps=args.wm_num_inference_steps,
        policy_mem_fraction=args.policy_mem_fraction,
        diversity_kwargs=diversity,
        smooth_sigma=args.smooth_sigma,
        agent_weight=args.vlac_agent_weight,
        wrist_weight=args.vlac_wrist_weight,
    )


def _collect_policies(args: Args) -> list[PolicySpec]:
    if _checkpoint_ready(args.checkpoint_dir):
        return [PolicySpec(name="stage2", config_name=args.config_name, checkpoint_dir=args.checkpoint_dir)]
    if args.fail_on_missing_checkpoint:
        raise FileNotFoundError(f"Missing stage2 checkpoint: {args.checkpoint_dir}")
    logging.warning("Missing stage2 checkpoint: %s", args.checkpoint_dir)
    return []


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    _finalize_gpu_config(args)
    policy_gpu = (args.policy_gpu or os.environ.get("OPENPI_POLICY_GPU", "")).strip()
    render_gpu = (args.render_gpu or os.environ.get("OPENPI_RENDER_GPU", "")).strip()
    wm_gpu = (args.wm_gpu or policy_gpu).strip()
    vlac_gpu = (args.vlac_gpu or policy_gpu).strip()
    device_map = getattr(args, "device_map", {})
    if args.replay_on_sawyer:
        mode = "replay_on_sawyer"
    elif args.use_action_selection:
        mode = f"ctrlworld_vlac,n={args.n_candidates}"
    else:
        mode = "direct_policy"
    logging.info("Eval mode=%s", mode)
    logging.info(
        "GPUs: policy=%s wm=%s vlac=%s render=%s | visible=%s | device_map=%s | "
        "MUJOCO_EGL_DEVICE_ID=%s render_gpu_device_id=%s",
        policy_gpu,
        wm_gpu,
        vlac_gpu,
        render_gpu,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        device_map,
        os.environ.get("MUJOCO_EGL_DEVICE_ID", "(unset)"),
        args.render_gpu_device_id,
    )

    np.random.seed(args.seed)
    suite_names = tuple(_normalize_suite_name(s) for s in args.suites)
    if any(s.endswith("_cross") for s in suite_names):
        robot = args.libero_10_cross_robot.strip().lower()
        os.environ["LIBERO_CROSS_ROBOT"] = robot
        os.environ["LIBERO_10_CROSS_ROBOT"] = robot

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.replay_on_sawyer:
        run_name = args.run_name or f"{args.replay_source_suite}_replay_on_{args.replay_sawyer_suite}"
        tracks_video_root = out_root / run_name
        tracks_video_root.mkdir(parents=True, exist_ok=True)
        all_rows = _run_source_replay_on_sawyer(args, tracks_video_root)
        suite_names = (_normalize_suite_name(args.replay_sawyer_suite),)
    else:
        policies = _collect_policies(args)
        if not policies:
            raise ValueError("Nothing to evaluate: stage2 checkpoint is not available.")

        run_name = _resolve_run_name(args, policies)
        if args.use_action_selection and not args.run_name:
            run_name = f"{run_name}_ctrlworld_vlac_n{args.n_candidates}"
        tracks_video_root = out_root / run_name
        tracks_video_root.mkdir(parents=True, exist_ok=True)

        all_rows = []
        for spec in policies:
            for suite_name in suite_names:
                rows = _run_policy_on_suite(spec, args, suite_name, tracks_video_root)
                all_rows.extend(rows)

    per_task = _per_task_summary(all_rows)
    report_path = _write_run_text_report(
        out_root=out_root,
        run_name=run_name,
        suite_names=suite_names,
        rows=all_rows,
        per_task=per_task,
        seed=args.seed,
    )
    logging.info("Summary txt: %s", report_path)
    logging.info("Videos with track overlay under: %s/<suite>/", tracks_video_root)
    if args.save_original_rollout_video:
        logging.info(
            "Original rollout videos under: %s/<suite>/original_rollout/", tracks_video_root
        )
    if args.save_rollout_predictions:
        logging.info("Rollout HDF5 (actions + tracks) under: %s/<suite>/hdf5/", tracks_video_root)
    if args.replay_on_sawyer:
        logging.info(
            "Source replay videos: %s/%s/%s_replay/",
            tracks_video_root,
            args.replay_sawyer_suite,
            args.replay_source_suite,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
