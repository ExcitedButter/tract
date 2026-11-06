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
    """Prefer GLX/GLFW (works under xvfb) over EGL when render nodes are unavailable.

    robosuite forces ``MUJOCO_GL=egl`` when GPU rendering is on, unless the env is
    already ``osmesa`` or ``glx``. Set ``MUJOCO_GL=glx`` *before* importing robosuite
    and run under ``xvfb-run`` if there is no DISPLAY / no ``render`` group access.
    Override with ``OPENPI_MUJOCO_GL=egl`` when real EGL devices work.
    """
    requested = (os.environ.get("OPENPI_MUJOCO_GL") or os.environ.get("MUJOCO_GL") or "glx").strip().lower()
    if requested in ("", "auto"):
        requested = "glx"
    os.environ["MUJOCO_GL"] = requested
    if requested in ("glx", "glfw", "osmesa"):
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)


def _install_dual_gpu_env(policy_gpu: str, render_gpu: str) -> int:
    """Put policy on cuda:0; use a second GPU only when MuJoCo backend is EGL."""
    policy = policy_gpu.strip()
    render = render_gpu.strip()
    if not policy.isdigit() or not render.isdigit():
        raise ValueError(
            f"--policy-gpu and --render-gpu must be numeric GPU indices, got {policy_gpu!r} and {render_gpu!r}"
        )
    gl = (os.environ.get("MUJOCO_GL") or "").strip().lower()
    use_egl = gl == "egl"
    if use_egl and policy != render:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{policy},{render}"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = render
        os.environ["OPENPI_RENDER_GPU_DEVICE_ID"] = render
        return int(render)
    # GLX/GLFW/OSMesa (or single-GPU EGL): policy GPU only.
    os.environ["CUDA_VISIBLE_DEVICES"] = policy
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    os.environ["OPENPI_RENDER_GPU_DEVICE_ID"] = policy
    return int(policy)


def _apply_gpu_split_from_argv() -> None:
    cli_policy = _parse_cli_kv_flag("--policy-gpu")
    cli_render = _parse_cli_kv_flag("--render-gpu")
    policy = (cli_policy or os.environ.get("OPENPI_POLICY_GPU", "")).strip()
    render = (cli_render or os.environ.get("OPENPI_RENDER_GPU", "")).strip()
    if policy and render:
        _install_dual_gpu_env(policy, render)


def _warmup_glx_context(resolution: int = 128) -> None:
    """Create/destroy a tiny offscreen env before JAX/CUDA touches the process.

    Under ``MUJOCO_GL=glx`` + xvfb, importing torch/JAX *then* creating GLFW
    often SIGSEGVs. Warm up GL first (this must run before ``import torch``).
    """
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
# Critical: initialize GLFW before torch/JAX (xvfb + MUJOCO_GL=glx).
try:
    _warmup_glx_context(resolution=128)
except Exception as _warmup_exc:  # noqa: BLE001
    # Defer hard failure; env creation later will surface a clearer error.
    print(f"[evaluate] GL warmup failed before torch import: {_warmup_exc}", file=sys.stderr)

import torch
import tyro
from tract.policies import policy_config as _policy_config
from tract.training import config as training_config

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
from libero.libero.envs.libero_camera_view import apply_agentview_viewpoint
from libero.libero.envs.mesh_vertex_tracker import MeshVertexTracker
from libero.libero.envs.mesh_vertex_wrapper import MeshVertexWrapper
from robosuite.utils.mjcf_utils import find_elements, string_to_array

from libero_gripper_vertex_ids import (
    apply_dynamic_gripper_vertex_ids,
    find_mesh_vertex_tracker,
    refresh_tracked_vertices_obs,
)

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
    """Baseline LIBERO eval: single policy.infer per replan (no WM / VLAC selection).

    For Track-WM + VLAC candidate selection, use ``evaluate_full_track.py``.
    """

    # Fixed mode marker for logs/reports. This script never runs action selection.
    eval_mode: str = "direct_policy"
    config_name: str = "pi05_libero_stage2p5_joint"
    checkpoint_dir: str = str(OPENPI_ROOT / "checkpoints/pi05_libero_stage2p5_joint/pi05_final/29999")
    suites: tuple[str, ...] = ("libero_object_sawyer",)
    all_tasks: bool = True
    # For libero_integral use --all-tasks (20 tasks). task_indices are 0-based.
    task_indices: tuple[int, ...] = (0,)
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


@dataclasses.dataclass(frozen=True)
class _AgentviewCameraOverride:
    horizon_view: int
    vertical_view: int
    scale_factor: float
    end_point_rot: int
    end_point_vertical: int
    fovy_deg: float | None = None


# Default agentview pose before LIBERO viewpoint transforms (per arena type).
_AGENTVIEW_BASE_BY_ARENA: dict[str, tuple[list[float], list[float]]] = {
    "living_room": (
        [0.6065773716836134, 0.0, 0.96],
        [0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979],
    ),
    "kitchen": (
        [0.6586131746834771, 0.0, 1.6103500240372423],
        [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
    ),
    "table": (
        [0.6586131746834771, 0.0, 1.6103500240372423],
        [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
    ),
    "study": (
        [0.4586131746834771, 0.0, 1.6103500240372423],
        [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
    ),
    "coffee_table": (
        [1.5, 0.0, 0.9],
        [0.56, 0.43, 0.43, 0.56],
    ),
    "floor": (
        [0.8965773716836134, 5.216182733499864e-07, 0.65],
        [0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979],
    ),
}

# Per-(suite, task_id) agentview tweaks for eval video framing (does not change policy inputs).
# libero_integral task 6: LIVING_ROOM_SCENE5 dual mug (BDDL view_0_0_100_10_2 — gripper clipped).
_EVAL_AGENTVIEW_OVERRIDES: dict[tuple[str, int], _AgentviewCameraOverride] = {
    ("libero_integral", 6): _AgentviewCameraOverride(
        horizon_view=-18,
        vertical_view=0,
        scale_factor=1.32,
        end_point_rot=10,
        end_point_vertical=2,
        fovy_deg=51.0,
    ),
}


def _apply_eval_agentview_override_if_needed(
    env: Any,
    *,
    suite_name: str,
    task_id: int,
) -> None:
    override = _EVAL_AGENTVIEW_OVERRIDES.get((suite_name, int(task_id)))
    if override is None:
        return
    inner = _unwrap_inner_env(env)
    arena_type = str(getattr(inner, "_arena_type", "table"))
    base = _AGENTVIEW_BASE_BY_ARENA.get(arena_type)
    if base is None:
        logging.warning(
            "agentview override skipped: unknown arena %r for %s task_id=%s",
            arena_type,
            suite_name,
            task_id,
        )
        return
    pos_av, quat_av = base
    arena = inner.model.mujoco_arena
    apply_agentview_viewpoint(
        arena,
        pos_av=pos_av,
        quat_av=quat_av,
        horizon_view=override.horizon_view,
        vertical_view=override.vertical_view,
        scale_factor=override.scale_factor,
        end_point_rot=override.end_point_rot,
        end_point_vertical=override.end_point_vertical,
    )
    cam = find_elements(
        root=arena.worldbody,
        tags="camera",
        attribs={"name": "agentview"},
        return_first=True,
    )
    if cam is None:
        logging.warning("agentview override: camera element not found")
        return
    pos = string_to_array(cam.get("pos"))
    quat = string_to_array(cam.get("quat"))
    cam_id = inner.sim.model.camera_name2id("agentview")
    inner.sim.model.cam_pos[cam_id] = pos
    inner.sim.model.cam_quat[cam_id] = quat
    if override.fovy_deg is not None:
        inner.sim.model.cam_fovy[cam_id] = float(override.fovy_deg)
    inner.sim.forward()
    logging.info(
        "Applied agentview override for %s task_id=%s (arena=%s horizon=%d scale=%.2f fovy=%s)",
        suite_name,
        task_id,
        arena_type,
        override.horizon_view,
        override.scale_factor,
        override.fovy_deg,
    )


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
    left = base_img.copy()
    right = wrist_img.copy()
    if pred_tracks is not None and pred_tracks.size > 0:
        pred = np.asarray(pred_tracks, dtype=np.float32).reshape(-1, 39, 2)
        _draw_future_tracks_colormap(left, pred[:, :7])
        _draw_future_tracks_colormap(right, pred[:, 7:32])
        _draw_future_tracks_colormap(right, pred[:, 32:39])
    return np.concatenate([left, right], axis=1)


def _render_original_frame(base_img: np.ndarray, wrist_img: np.ndarray) -> np.ndarray:
    return np.concatenate([base_img, wrist_img], axis=1)


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
    agent_img = np.asarray(obs["agentview_image"])
    wrist_img = np.asarray(obs["robot0_eye_in_hand_image"])
    if flip_image_x and flip_image_y:
        agent_img = np.ascontiguousarray(agent_img[::-1, ::-1, :])
        wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1, :])
    else:
        if flip_image_y:
            agent_img = agent_img[::-1, :, :]
            wrist_img = wrist_img[::-1, :, :]
        if flip_image_x:
            agent_img = agent_img[:, ::-1, :]
            wrist_img = wrist_img[:, ::-1, :]
        agent_img = np.ascontiguousarray(agent_img)
        wrist_img = np.ascontiguousarray(wrist_img)
    agent_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(agent_img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

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


def _task_ids_for_suite(
    suite,
    *,
    all_tasks: bool,
    task_index: int,
    task_indices: tuple[int, ...],
) -> list[int]:
    if all_tasks:
        return list(range(suite.n_tasks))
    if task_indices:
        seen: set[int] = set()
        out: list[int] = []
        for tid in task_indices:
            clipped = int(np.clip(int(tid), 0, suite.n_tasks - 1))
            if clipped not in seen:
                seen.add(clipped)
                out.append(clipped)
        return out
    return [int(np.clip(task_index, 0, suite.n_tasks - 1))]


def _save_video(video_frames: list[np.ndarray], video_path: Path, fps: int) -> None:
    if not video_frames:
        return
    # Lazy import: moviepy/av before GLFW init can interact badly with GLX.
    from moviepy import ImageSequenceClip

    ImageSequenceClip(video_frames, fps=fps).write_videofile(
        str(video_path),
        codec="libx264",
        audio=False,
        logger=None,
    )


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
    lines.append("action_selection: none (direct_policy)")
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
    # GL was warmed at import time (before torch/JAX). Load policy next.
    policy = _load_policy(spec)

    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[suite_name]()
    task_ids = _task_ids_for_suite(
        suite,
        all_tasks=args.all_tasks,
        task_index=args.task_index,
        task_indices=args.task_indices,
    )
    logging.info("%s: evaluating task_ids=%s", suite_name, task_ids)

    rows: list[dict[str, Any]] = []
    point_xflip = args.flip_image_x if args.align_point_flips_with_image else args.xflip_points
    point_yflip = args.flip_image_y if args.align_point_flips_with_image else args.yflip_points
    max_steps = _max_steps_for_suite(suite_name)
    save_hdf5 = bool(args.save_rollout_predictions)
    save_original_video = bool(args.save_original_rollout_video)
    infer_return_tracks = True

    if save_hdf5:
        logging.info(
            "HDF5 per rollout: %s/<suite>/hdf5/*.hdf5 (actions, pred_actions, display-aligned tracks)",
            tracks_video_root,
        )

    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        n_inits = max(1, len(initial_states))
        n_roll = int(args.rollouts_per_task)
        attempt = 0
        attempt_seed = int(args.seed + task_id * 13)
        init_offset = 0
        save_media = True
        np.random.seed(attempt_seed)

        env, task_description = _build_mesh_env(
            task,
            args.env_resolution,
            attempt_seed,
            suite_name=suite_name,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        batch_rows: list[dict[str, Any]] = []
        try:
            for rollout_idx in range(n_roll):
                env.reset()
                init_i = (init_offset + rollout_idx) % n_inits
                _raw_init = initial_states[init_i]
                _assert_flat_init_matches_sim(
                    env,
                    _raw_init,
                    label=(
                        f"{suite_name} task_id={task_id} attempt={attempt} "
                        f"rollout={rollout_idx} init={init_i}"
                    ),
                )
                obs = env.set_init_state(_raw_init)
                _apply_eval_agentview_override_if_needed(
                    env, suite_name=suite_name, task_id=task_id
                )
                obs = _maybe_refresh_gripper_query_points(
                    env, obs, resolution=args.env_resolution
                )

                action_plan = collections.deque()
                video_frames: list[np.ndarray] = []
                original_video_frames: list[np.ndarray] = []
                executed_actions: list[np.ndarray] = []
                pred_action_chunks: list[np.ndarray] = []
                pred_track_chunks: list[np.ndarray | None] = []
                pred_env_steps: list[int] = []
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
                        infer_out = policy.infer(raw_item, return_tracks=infer_return_tracks)
                        action_chunk = np.asarray(infer_out["actions"], dtype=np.float32)
                        pred_action_chunks.append(action_chunk.copy())
                        pred_env_steps.append(control_step)
                        if "track_predictions" in infer_out:
                            tracks_raw = np.asarray(
                                infer_out["track_predictions"], dtype=np.float32
                            )
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
                    if save_media:
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
                video_name = (
                    f"{spec.name}_{suite_name}_{task_id:02d}_{rollout_idx:02d}_"
                    f"a{attempt:02d}_{status}_{task_slug}.mp4"
                )
                tracks_video_dir = tracks_video_root / suite_name
                video_path: str | None = None
                hdf5_path: str | None = None
                original_video_path: str | None = None
                if save_media:
                    tracks_video_dir.mkdir(parents=True, exist_ok=True)
                    vpath = tracks_video_dir / video_name
                    _save_video(video_frames, vpath, args.fps)
                    video_path = str(vpath)
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
                        )
                        hdf5_path = str(hdf5_file)
                    if save_original_video:
                        original_video_dir = (
                            tracks_video_root / suite_name / "original_rollout"
                        )
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
                    "init_index": int(init_i),
                    "attempt": int(attempt),
                    "attempt_seed": int(attempt_seed),
                    "init_offset": int(init_offset),
                    "success": bool(done),
                    "reward_sum": float(total_reward),
                    "steps": int(len(executed_actions)),
                    "control_steps": control_step,
                    "video_path": video_path,
                    "predictions_hdf5_path": hdf5_path,
                    "original_video_path": original_video_path,
                    "has_predicted_tracks_overlay": bool(last_pred_tracks is not None),
                }
                batch_rows.append(row)
                logging.info(
                    "%s %s task=%d attempt=%d rollout=%d init=%d success=%s "
                    "reward_sum=%.3f steps=%d",
                    spec.name,
                    suite_name,
                    task_id,
                    attempt,
                    rollout_idx,
                    init_i,
                    done,
                    total_reward,
                    control_step,
                )
        finally:
            env.close()

        rows.extend(batch_rows)

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
    """Apply dual-GPU env vars before LIBERO envs are constructed (idempotent)."""
    policy = (args.policy_gpu or os.environ.get("OPENPI_POLICY_GPU", "")).strip()
    render = (args.render_gpu or os.environ.get("OPENPI_RENDER_GPU", "")).strip()
    if not policy or not render:
        raise ValueError(
            "Set both --policy-gpu and --render-gpu (or OPENPI_POLICY_GPU and OPENPI_RENDER_GPU)."
        )
    physical = _install_dual_gpu_env(policy, render)
    gl = (os.environ.get("MUJOCO_GL") or "").strip().lower()
    # GLX/OSMesa: let robosuite pick the GLFW context; EGL uses the mapped device id.
    args.render_gpu_device_id = physical if gl == "egl" else -1


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
    logging.info(
        "Eval mode=%s (no action selection / no WM / no VLAC). "
        "Each replan uses a single policy.infer.",
        args.eval_mode,
    )
    logging.info(
        "GPU/render: policy GPU %s (cuda:0), MUJOCO_GL=%s, CUDA_VISIBLE_DEVICES=%s, "
        "MUJOCO_EGL_DEVICE_ID=%s, render_gpu_device_id=%s (cli render_gpu=%s)",
        policy_gpu,
        os.environ.get("MUJOCO_GL", ""),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        os.environ.get("MUJOCO_EGL_DEVICE_ID", "(unset)"),
        args.render_gpu_device_id,
        render_gpu,
    )

    np.random.seed(args.seed)
    suite_names = tuple(_normalize_suite_name(s) for s in args.suites)
    if any(s.endswith("_cross") or s == "libero_integral" for s in suite_names):
        robot = args.libero_10_cross_robot.strip().lower()
        os.environ["LIBERO_CROSS_ROBOT"] = robot
        os.environ["LIBERO_10_CROSS_ROBOT"] = robot

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    policies = _collect_policies(args)
    if not policies:
        raise ValueError("Nothing to evaluate: stage2 checkpoint is not available.")

    run_name = _resolve_run_name(args, policies)
    tracks_video_root = out_root / run_name
    tracks_video_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
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


if __name__ == "__main__":
    main(tyro.cli(Args))
