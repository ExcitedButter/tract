#!/usr/bin/env python3
"""
Direct RLDS-to-LeRobot preprocessing with CoTracker and simulator mesh seeds.

Pipeline (single stage):
1) Load LIBERO RLDS-as-LeRobot dataset (images/states/actions/tasks).
2) Derive scene-specific 7 mesh vertex seeds (agentview + eye-in-hand) from simulator.
3) Build query sets:
   - agentview: 7 mesh seeds
   - wrist: 25 uniform grid + 7 mesh seeds
4) Track queries with CoTracker over the episode frames (agent/wrist separately).
5) Emit LeRobot dataset with:
   - images, wrist_images, state, actions, task
   - per-frame tracks/vis (agentview_tracks/vis, wrist_tracks/vis)
   - mesh vertices (agentview_mesh_vertices_2d, wrist_mesh_vertices_2d)
   - grid query points (agentview_grid_mesh_32, wrist_grid_mesh_32)
   - track_targets_raw: horizon window (H, 7+32, 2) combining agent mesh 7 + wrist grid25+mesh7

This bypasses the intermediate ATM conversion and writes directly to a LeRobot repo.

Notes:
- Requires a working LIBERO simulator + OffScreenRenderEnv to get mesh vertex seeds.
- CoTracker2 (pretrained ``cotracker2`` torch.hub weights) from the vendored third_party/co-tracker.
- Designed for 256×256 frame resolution to match existing LeRobot LIBERO data.
"""

from __future__ import annotations

import tract.shared.local_cache_bootstrap  # noqa: F401

import argparse
import json
import os
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from einops import rearrange

from importlib.machinery import SourceFileLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Scene-specific vertex IDs (7 per camera)
DEFAULT_VERTEX_IDS = {
    "agentview": [3001, 3004, 3014, 3067, 3040, 3081, 2898],
    "eye_in_hand": [2955, 3055, 3040, 2981, 2966, 3052, 3022],
}
KITCHEN_VERTEX_IDS = {
    "agentview": [3249, 3252, 3200, 3315, 3272, 3329, 3134],
    "eye_in_hand": [2955, 3055, 3040, 2981, 2966, 3052, 3022],
}
LIVING_ROOM_VERTEX_IDS = {
    "agentview": [3135, 3140, 3141, 3202, 3205, 3215, 3034],
    "eye_in_hand": [2955, 3055, 3040, 2981, 2966, 3052, 3022],
}
STUDY_VERTEX_IDS = {
    "agentview": [3352, 3355, 3365, 3418, 3391, 3432, 3249],
    "eye_in_hand": [2955, 3055, 3040, 2981, 2966, 3052, 3022],
}

SCENE_VERTEX_IDS = {
    "default": DEFAULT_VERTEX_IDS,
    "kitchen": KITCHEN_VERTEX_IDS,
    "living_room": LIVING_ROOM_VERTEX_IDS,
    "living": LIVING_ROOM_VERTEX_IDS,
    "study": STUDY_VERTEX_IDS,
}


def _scene_from_task(task: str) -> str:
    t = task.lower()
    if "kitchen" in t:
        return "kitchen"
    if "living" in t or "livingroom" in t:
        return "living_room"
    if "study" in t:
        return "study"
    return "default"


def _make_grid_25() -> np.ndarray:
    u = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    v = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1)  # (25, 2)


EXTRA_LIBERO_PATH: str | None = None


def _load_cotracker(checkpoint: str | None, device: torch.device):
    """CoTracker2 offline predictor (stride 4, window 8) from the vendored ``third_party/co-tracker``.

    The released track dataset was generated with the pretrained ``cotracker2`` hub weights
    (facebook/cotracker ``cotracker2.pth``, downloaded by torch.hub). Note the hub entry ignores a
    ``checkpoint=`` kwarg, so an explicit ``checkpoint`` (a CoTracker2 state dict) is loaded here.
    """
    repo = Path(os.environ.get("COTRACKER_REPO", Path(__file__).resolve().parents[1] / "third_party" / "co-tracker"))
    if not (repo / "hubconf.py").is_file():
        hub_dir = Path(os.environ.get("TORCH_HUB_DIR", Path.home() / ".cache" / "torch" / "hub"))
        repo = hub_dir / "facebookresearch_co-tracker_main"
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"CoTracker repo (hubconf.py) not found at {repo}; set COTRACKER_REPO.")
    model = torch.hub.load(str(repo), "cotracker2", pretrained=checkpoint is None, source="local")
    if checkpoint is not None:
        model.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval().to(device)


def _track_queries(video: np.ndarray, queries_xy01: np.ndarray, model, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    video: [T, H, W, 3] uint8
    queries_xy01: [N, 2] normalized [0,1]
    """
    T, H, W, _ = video.shape
    vid = rearrange(torch.from_numpy(video).float().to(device) / 255.0, "t h w c -> 1 t c h w")
    q = torch.from_numpy(queries_xy01.copy()).float().to(device)
    # CoTracker expects pixel coords with leading time dim; start at frame 0.
    q_pix = q * torch.tensor([W, H], device=device).float()
    q_with_t = torch.cat([torch.zeros_like(q_pix[:, :1]), q_pix], dim=1)  # [N,3] (t,x,y)
    pred_tracks, pred_vis = model(vid, queries=q_with_t[None], backward_tracking=True)
    tracks = pred_tracks[0].detach().cpu().numpy()  # [T, N, 2] in px
    vis = pred_vis[0].detach().cpu().numpy()        # [T, N]
    tracks[:, :, 0] /= float(W)
    tracks[:, :, 1] /= float(H)
    return tracks, vis


def _track_queries_from_start(
    video: np.ndarray,
    queries_xy01: np.ndarray,
    start_t: int,
    model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track queries starting from an arbitrary frame `start_t`.
    Returns tracks and vis over video[start_t:].
    """
    if start_t < 0 or start_t >= video.shape[0]:
        raise ValueError(f"start_t out of range: {start_t} for T={video.shape[0]}")
    sub_video = video[start_t:]
    T, H, W, _ = sub_video.shape
    vid = rearrange(torch.from_numpy(sub_video).float().to(device) / 255.0, "t h w c -> 1 t c h w")
    q = torch.from_numpy(queries_xy01.copy()).float().to(device)
    q_pix = q * torch.tensor([W, H], device=device).float()
    q_with_t = torch.cat([torch.zeros_like(q_pix[:, :1]), q_pix], dim=1)  # t=0 in sub-video
    pred_tracks, pred_vis = model(vid, queries=q_with_t[None], backward_tracking=True)
    tracks = pred_tracks[0].detach().cpu().numpy()
    vis = pred_vis[0].detach().cpu().numpy()
    tracks[:, :, 0] /= float(W)
    tracks[:, :, 1] /= float(H)
    return tracks, vis


def _track_time_indexed_queries(
    video: np.ndarray,
    base_queries_xy01: np.ndarray,
    model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track fixed query set launched at every frame in one CoTracker pass.
    Returns:
      tracks: [T, T*N, 2]  (time, concatenated query-groups, xy01)
      vis:    [T, T*N]
    Query-group k corresponds to queries initialized at frame k.
    """
    T, H, W, _ = video.shape
    N = base_queries_xy01.shape[0]
    vid = rearrange(torch.from_numpy(video).float().to(device) / 255.0, "t h w c -> 1 t c h w")
    q = torch.from_numpy(base_queries_xy01.copy()).float().to(device)  # [N,2]
    q_pix = q * torch.tensor([W, H], device=device).float()  # [N,2]
    q_rep = q_pix[None].repeat(T, 1, 1).reshape(T * N, 2)  # [T*N,2]
    t_idx = (
        torch.arange(T, device=device, dtype=torch.float32)[:, None]
        .repeat(1, N)
        .reshape(T * N, 1)
    )  # [T*N,1]
    q_with_t = torch.cat([t_idx, q_rep], dim=1)  # [T*N,3]
    pred_tracks, pred_vis = model(vid, queries=q_with_t[None], backward_tracking=True)
    tracks = pred_tracks[0].detach().cpu().numpy()  # [T, T*N, 2] px
    vis = pred_vis[0].detach().cpu().numpy()        # [T, T*N]
    tracks[:, :, 0] /= float(W)
    tracks[:, :, 1] /= float(H)
    return tracks, vis


def _build_features() -> dict[str, dict]:
    return {
        "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "agentview_tracks": {"dtype": "float32", "shape": (7, 2), "names": ["points", "xy"]},
        "agentview_vis": {"dtype": "float32", "shape": (7,), "names": ["points"]},
        "wrist_tracks": {"dtype": "float32", "shape": (32, 2), "names": ["points", "xy"]},
        "wrist_vis": {"dtype": "float32", "shape": (32,), "names": ["points"]},
        "track_targets_raw": {"dtype": "float32", "shape": (16, 39, 2), "names": ["h", "points", "xy"]},
        "agentview_grid_mesh_32": {"dtype": "float32", "shape": (32, 2), "names": ["points", "xy"]},
        "wrist_grid_mesh_32": {"dtype": "float32", "shape": (32, 2), "names": ["points", "xy"]},
        "agentview_mesh_vertices_2d": {"dtype": "float32", "shape": (7, 2), "names": ["points", "xy"]},
        "wrist_mesh_vertices_2d": {"dtype": "float32", "shape": (7, 2), "names": ["points", "xy"]},
        "has_track_mesh": {"dtype": "float32", "shape": (1,), "names": ["flag"]},
    }


def _track_window(tracks: np.ndarray, start: int, horizon: int) -> np.ndarray:
    end = min(start + horizon, tracks.shape[0])
    window = tracks[start:end]
    if window.shape[0] == 0:
        return np.zeros((horizon, tracks.shape[1], tracks.shape[2]), dtype=tracks.dtype)
    if window.shape[0] < horizon:
        pad = np.repeat(window[-1:], horizon - window.shape[0], axis=0)
        window = np.concatenate([window, pad], axis=0)
    return window


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


@dataclass
class EpisodeBounds:
    start: int
    end: int
    task: str


def _episode_bounds(ds: LeRobotDataset, ep_idx: int) -> EpisodeBounds:
    ep_from = int(ds.episode_data_index["from"][ep_idx])
    ep_to = int(ds.episode_data_index["to"][ep_idx])
    task = ds.meta.episodes[ep_idx]["tasks"][0]
    return EpisodeBounds(ep_from, ep_to, task)


def _episode_file_from_global_index(ds_root: Path, global_idx: int) -> Path:
    chunk = global_idx // 1000
    chunk_dir = ds_root / "data" / f"chunk-{chunk:03d}"
    # Prefer global naming (episode_001000.parquet), but also support chunk-local naming
    # (episode_000000.parquet in chunk-001) for compatibility with older exports.
    global_name = chunk_dir / f"episode_{global_idx:06d}.parquet"
    local_name = chunk_dir / f"episode_{(global_idx % 1000):06d}.parquet"
    if global_name.exists():
        return global_name
    if local_name.exists():
        return local_name
    raise FileNotFoundError(
        "Episode parquet not found for global_idx="
        f"{global_idx}: tried {global_name} and {local_name}"
    )


def _coerce_mesh7(arr: np.ndarray, *, key: str, frame_idx: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    # Accept exact (7,2) or larger point sets where first 7 are canonical mesh points.
    if a.ndim != 2 or a.shape[-1] != 2:
        raise ValueError(f"{key} at frame={frame_idx} must be [P,2], got {a.shape}")
    if a.shape[0] < 7:
        raise ValueError(f"{key} at frame={frame_idx} has <7 points: {a.shape}")
    return a[:7]


def _mesh_seeds_for_scene(scene: str) -> tuple[list[int], list[int]]:
    ids = SCENE_VERTEX_IDS.get(scene, DEFAULT_VERTEX_IDS)
    return ids["agentview"], ids["eye_in_hand"]


def _normalize_task_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _resolve_bddl_for_task(
    task_name: str | None,
    benchmark_mod,
    bddl_roots: list[Path],
) -> str | None:
    if not task_name:
        return None
    target = _normalize_task_text(task_name)
    if not target:
        return None
    best_score = -1
    best_path: Path | None = None
    benchmark_dict = benchmark_mod.get_benchmark_dict()
    for _, suite_ctor in benchmark_dict.items():
        try:
            suite = suite_ctor()
        except Exception:
            continue
        for i in range(getattr(suite, "n_tasks", 0)):
            try:
                t = suite.get_task(i)
            except Exception:
                continue
            language = _normalize_task_text(getattr(t, "language", ""))
            if not language:
                continue
            score = 0
            if language == target:
                score = 3
            elif target in language or language in target:
                score = 2
            elif set(target.split()) & set(language.split()):
                score = 1
            if score < best_score:
                continue
            rel = Path(t.problem_folder) / t.bddl_file
            for root in bddl_roots:
                cand = root / rel
                if cand.exists():
                    best_score = score
                    best_path = cand
                    break
    return str(best_path) if best_path is not None else None


def _select_dynamic_ids_for_camera(
    env,
    camera_name: str,
    fallback_agent_ids: list[int],
    img_hw: tuple[int, int],
) -> list[int]:
    if not hasattr(env, "vertex_mapping"):
        return fallback_agent_ids
    base_env = env
    while hasattr(base_env, "env") and not hasattr(base_env, "sim"):
        base_env = base_env.env
    if not hasattr(base_env, "sim"):
        return fallback_agent_ids
    model = base_env.sim.model

    def _group_for_geom(name: str) -> str:
        n = (name or "").lower()
        if "finger1" in n:
            return "finger1"
        if "finger2" in n:
            return "finger2"
        if "hand" in n or "gripper" in n:
            return "base"
        return "other"

    candidates: dict[int, str] = {}
    for vid, mapping in env.vertex_mapping.items():
        geom_id = int(mapping.get("geom_id", -1))
        if geom_id < 0:
            continue
        geom_name = model.geom_id2name(geom_id)
        group = _group_for_geom(geom_name)
        if group != "other":
            candidates[int(vid)] = group
    if not candidates:
        return fallback_agent_ids

    vids = list(candidates.keys())
    if hasattr(env, "_update_vertex_positions"):
        env._update_vertex_positions()
    tracked = env._get_tracked_vertices(camera_name, vids)
    id_to_xy = {int(i): p for i, p in zip(tracked["vertex_ids"].tolist(), tracked["vertices_2d"])}
    rows = []
    w, h = float(img_hw[1]), float(img_hw[0])
    for vid in vids:
        if vid not in id_to_xy:
            continue
        x = float(id_to_xy[vid][0]) / w
        y = float(id_to_xy[vid][1]) / h
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue
        rows.append({"vid": vid, "group": candidates[vid], "x": x, "y": y})

    def _pick_three(group_rows: list[dict]) -> list[dict]:
        if len(group_rows) <= 3:
            return group_rows
        ys = np.array([r["y"] for r in group_rows], dtype=np.float32)
        lo = int(np.argmin(ys))
        hi = int(np.argmax(ys))
        mid_target = float((ys[lo] + ys[hi]) * 0.5)
        mid = min(range(len(group_rows)), key=lambda i: abs(float(group_rows[i]["y"]) - mid_target))
        idx = []
        for i in (lo, mid, hi):
            if i not in idx:
                idx.append(i)
        for i in range(len(group_rows)):
            if len(idx) >= 3:
                break
            if i not in idx:
                idx.append(i)
        return [group_rows[i] for i in idx[:3]]

    f1 = _pick_three([r for r in rows if r["group"] == "finger1"])
    f2 = _pick_three([r for r in rows if r["group"] == "finger2"])
    b = [r for r in rows if r["group"] == "base"]
    if len(f1) < 2 or len(f2) < 2 or not b:
        return fallback_agent_ids
    mx = (float(np.mean([r["x"] for r in f1])) + float(np.mean([r["x"] for r in f2]))) / 2.0
    my = (float(np.mean([r["y"] for r in f1])) + float(np.mean([r["y"] for r in f2]))) / 2.0
    base = sorted(b, key=lambda r: abs(r["x"] - mx) + 0.5 * abs(r["y"] - my))[0]
    picked = (f1 + f2 + [base])[:7]
    picked_ids = [int(r["vid"]) for r in picked]
    return picked_ids if len(picked_ids) == 7 else fallback_agent_ids


def _get_mesh_seeds_from_sim(scene: str, state_vec: np.ndarray, img_hw=(256, 256)) -> tuple[np.ndarray, np.ndarray]:
    last_exc = None
    OffScreenRenderEnv = None
    MeshVertexWrapper = None
    MeshVertexTracker = None
    benchmark_mod = None
    get_libero_path = None
    for mod_name in ("libero.envs.env_wrapper", "libero.libero.envs.env_wrapper"):
        try:
            env_mod = importlib.import_module(mod_name)
            OffScreenRenderEnv = env_mod.OffScreenRenderEnv  # type: ignore
            base_pkg = env_mod.__package__.rsplit(".env_wrapper", 1)[0]
            for benchmark_mod_name in ("libero.benchmark", "libero.libero.benchmark"):
                try:
                    benchmark_mod = importlib.import_module(benchmark_mod_name)
                    break
                except Exception:
                    benchmark_mod = None  # type: ignore
                    continue
            if benchmark_mod is None:
                raise ModuleNotFoundError("libero.benchmark not found")
            try:
                get_libero_path = importlib.import_module("libero").get_libero_path  # type: ignore
            except AttributeError:
                get_libero_path = importlib.import_module("libero.libero").get_libero_path  # type: ignore
            mesh_vertex_wrapper = None
            mesh_vertex_tracker = None
            for wrapper_mod in (
                f"{base_pkg}.mesh_vertex_wrapper",
                "libero.envs.mesh_vertex_wrapper",
                "libero.libero.envs.mesh_vertex_wrapper",
            ):
                try:
                    mesh_vertex_wrapper = importlib.import_module(wrapper_mod)
                    break
                except Exception:
                    mesh_vertex_wrapper = None  # type: ignore
                    continue
            for tracker_mod in (
                f"{base_pkg}.mesh_vertex_tracker",
                "libero.envs.mesh_vertex_tracker",
                "libero.libero.envs.mesh_vertex_tracker",
            ):
                try:
                    mesh_vertex_tracker = importlib.import_module(tracker_mod)
                    break
                except Exception:
                    mesh_vertex_tracker = None  # type: ignore
                    continue
            if mesh_vertex_wrapper is None or mesh_vertex_tracker is None:
                fallback_root = EXTRA_LIBERO_PATH or ""
                wrapper_path = Path(fallback_root) / "libero" / "envs" / "mesh_vertex_wrapper.py"
                tracker_path = Path(fallback_root) / "libero" / "envs" / "mesh_vertex_tracker.py"
                if mesh_vertex_wrapper is None and wrapper_path.exists():
                    mesh_vertex_wrapper = SourceFileLoader("libero.envs.mesh_vertex_wrapper_fallback", str(wrapper_path)).load_module()  # type: ignore
                if mesh_vertex_tracker is None and tracker_path.exists():
                    mesh_vertex_tracker = SourceFileLoader("libero.envs.mesh_vertex_tracker_fallback", str(tracker_path)).load_module()  # type: ignore
            if mesh_vertex_wrapper is None or mesh_vertex_tracker is None:
                raise ModuleNotFoundError("mesh vertex wrapper/tracker not found in LIBERO installation")
            MeshVertexWrapper = mesh_vertex_wrapper.MeshVertexWrapper  # type: ignore
            MeshVertexTracker = mesh_vertex_tracker.MeshVertexTracker  # type: ignore
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            OffScreenRenderEnv = None  # type: ignore
            continue
    if OffScreenRenderEnv is None:
        raise ImportError(
            "LIBERO simulator not available. Ensure libero is on PYTHONPATH and provides libero/envs/env_wrapper.py. "
            "Set --extra-libero-path to the LIBERO repo root (e.g., third_party/libero)."
        ) from last_exc

    benchmark_dict = benchmark_mod.get_benchmark_dict()
    suite = benchmark_dict["libero_goal"]()  # suite choice not critical for camera layout
    # Heuristic: pick first task to init cameras; real scene choice drives vertex ids.
    task = suite.get_task(0)
    rel_bddl = Path(task.problem_folder) / task.bddl_file
    bddl_roots: list[Path] = []
    try:
        bddl_roots.append(Path(get_libero_path("bddl_files")))
    except Exception:
        pass
    for module_name in ("libero.libero", "libero"):
        try:
            mod = importlib.import_module(module_name)
            mod_file = getattr(mod, "__file__", None)
            if mod_file:
                bddl_roots.append(Path(mod_file).resolve().parent / "bddl_files")
        except Exception:
            continue
    if env_mod and getattr(env_mod, "__file__", None):
        bddl_roots.append(Path(env_mod.__file__).resolve().parents[1] / "bddl_files")
    if EXTRA_LIBERO_PATH:
        bddl_roots.append(Path(EXTRA_LIBERO_PATH) / "libero" / "bddl_files")
        bddl_roots.append(Path(EXTRA_LIBERO_PATH) / "bddl_files")

    seen = set()
    uniq_roots = []
    for root in bddl_roots:
        k = str(root)
        if k not in seen:
            seen.add(k)
            uniq_roots.append(root)

    task_bddl_file = None
    for root in uniq_roots:
        cand = root / rel_bddl
        if cand.exists():
            task_bddl_file = str(cand)
            break
    if task_bddl_file is None:
        raise FileNotFoundError(
            f"Could not resolve BDDL file {rel_bddl} from candidates: {[str(p) for p in uniq_roots]}"
        )
    task_mapping = getattr(env_mod, "TASK_MAPPING", {})
    known_problems = set(task_mapping.keys()) if isinstance(task_mapping, dict) else set()

    def _problem_name_from_bddl(path: str) -> str | None:
        try:
            txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        m = re.search(r"\(problem\s+([^\s\)]+)\)", txt)
        return m.group(1) if m else None

    candidate_bddls = [task_bddl_file]
    for root in uniq_roots:
        if not root.exists():
            continue
        for cand in root.rglob("*.bddl"):
            prob = _problem_name_from_bddl(str(cand))
            if prob is None or (known_problems and prob in known_problems):
                candidate_bddls.append(str(cand))

    seen = set()
    candidate_bddls = [p for p in candidate_bddls if not (p in seen or seen.add(p))]

    env = None
    env_exc = None
    for cand in candidate_bddls:
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=cand,
                camera_heights=img_hw[0],
                camera_widths=img_hw[1],
                render_gpu_device_id=0,
            )
            break
        except Exception as exc:  # noqa: BLE001
            env_exc = exc
            continue
    if env is None:
        raise RuntimeError("Failed to create OffScreenRenderEnv from candidate BDDL files") from env_exc
    agent_ids, wrist_ids = _mesh_seeds_for_scene(scene)
    env = MeshVertexWrapper(env, max_points=10000, include_non_mesh_geoms=True)
    env = MeshVertexTracker(env, agentview_vertex_ids=agent_ids, eye_in_hand_vertex_ids=wrist_ids, img_height=img_hw[0], img_width=img_hw[1])
    env.reset()
    base_env = env
    while hasattr(base_env, "env") and not hasattr(base_env, "sim"):
        base_env = base_env.env
    expected_qpos = int(getattr(getattr(base_env, "sim", None).model, "nq", 0)) if hasattr(base_env, "sim") else 0
    state_arr = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    if expected_qpos > 0 and state_arr.shape[0] >= expected_qpos:
        try:
            env.set_init_state(state_arr)
        except Exception as exc:
            print(f"[warn] set_init_state failed ({exc}); using reset state for mesh seeds")
    else:
        print(
            f"[warn] simulator state unavailable or mismatched (got {state_arr.shape[0]}, expected >= {expected_qpos}); "
            "using reset state for mesh seeds"
        )
    if hasattr(env, "_update_vertex_positions"):
        env._update_vertex_positions()
    av = env._get_tracked_vertices("agentview", env.agentview_vertex_ids)["vertices_2d"]
    wr = env._get_tracked_vertices("robot0_eye_in_hand", env.eye_in_hand_vertex_ids)["vertices_2d"]
    env.close()
    av = np.asarray(av, dtype=np.float32) / float(img_hw[1])
    wr = np.asarray(wr, dtype=np.float32) / float(img_hw[1])
    return av, wr


def _get_mesh_sequence_from_sim(
    scene: str,
    state_vec: np.ndarray,
    actions: np.ndarray,
    task_name: str | None = None,
    img_hw=(256, 256),
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame simulator mesh vertices by stepping the simulator with episode actions."""
    last_exc = None
    OffScreenRenderEnv = None
    MeshVertexWrapper = None
    MeshVertexTracker = None
    benchmark_mod = None
    get_libero_path = None
    env_mod = None
    for mod_name in ("libero.envs.env_wrapper", "libero.libero.envs.env_wrapper"):
        try:
            env_mod = importlib.import_module(mod_name)
            OffScreenRenderEnv = env_mod.OffScreenRenderEnv  # type: ignore
            base_pkg = env_mod.__package__.rsplit(".env_wrapper", 1)[0]
            for benchmark_mod_name in ("libero.benchmark", "libero.libero.benchmark"):
                try:
                    benchmark_mod = importlib.import_module(benchmark_mod_name)
                    break
                except Exception:
                    benchmark_mod = None  # type: ignore
                    continue
            if benchmark_mod is None:
                raise ModuleNotFoundError("libero.benchmark not found")
            try:
                get_libero_path = importlib.import_module("libero").get_libero_path  # type: ignore
            except AttributeError:
                get_libero_path = importlib.import_module("libero.libero").get_libero_path  # type: ignore
            mesh_vertex_wrapper = None
            mesh_vertex_tracker = None
            for wrapper_mod in (
                f"{base_pkg}.mesh_vertex_wrapper",
                "libero.envs.mesh_vertex_wrapper",
                "libero.libero.envs.mesh_vertex_wrapper",
            ):
                try:
                    mesh_vertex_wrapper = importlib.import_module(wrapper_mod)
                    break
                except Exception:
                    mesh_vertex_wrapper = None  # type: ignore
                    continue
            for tracker_mod in (
                f"{base_pkg}.mesh_vertex_tracker",
                "libero.envs.mesh_vertex_tracker",
                "libero.libero.envs.mesh_vertex_tracker",
            ):
                try:
                    mesh_vertex_tracker = importlib.import_module(tracker_mod)
                    break
                except Exception:
                    mesh_vertex_tracker = None  # type: ignore
                    continue
            if mesh_vertex_wrapper is None or mesh_vertex_tracker is None:
                fallback_root = EXTRA_LIBERO_PATH or ""
                wrapper_path = Path(fallback_root) / "libero" / "envs" / "mesh_vertex_wrapper.py"
                tracker_path = Path(fallback_root) / "libero" / "envs" / "mesh_vertex_tracker.py"
                if mesh_vertex_wrapper is None and wrapper_path.exists():
                    mesh_vertex_wrapper = SourceFileLoader("libero.envs.mesh_vertex_wrapper_fallback", str(wrapper_path)).load_module()  # type: ignore
                if mesh_vertex_tracker is None and tracker_path.exists():
                    mesh_vertex_tracker = SourceFileLoader("libero.envs.mesh_vertex_tracker_fallback", str(tracker_path)).load_module()  # type: ignore
            if mesh_vertex_wrapper is None or mesh_vertex_tracker is None:
                raise ModuleNotFoundError("mesh vertex wrapper/tracker not found in LIBERO installation")
            MeshVertexWrapper = mesh_vertex_wrapper.MeshVertexWrapper  # type: ignore
            MeshVertexTracker = mesh_vertex_tracker.MeshVertexTracker  # type: ignore
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            OffScreenRenderEnv = None  # type: ignore
            continue
    if OffScreenRenderEnv is None:
        raise ImportError(
            "LIBERO simulator not available. Ensure libero is on PYTHONPATH and provides libero/envs/env_wrapper.py. "
            "Set --extra-libero-path to the LIBERO repo root (e.g., third_party/libero)."
        ) from last_exc

    benchmark_dict = benchmark_mod.get_benchmark_dict()
    suite = benchmark_dict["libero_goal"]()
    task = suite.get_task(0)
    rel_bddl = Path(task.problem_folder) / task.bddl_file
    bddl_roots: list[Path] = []
    try:
        bddl_roots.append(Path(get_libero_path("bddl_files")))
    except Exception:
        pass
    for module_name in ("libero.libero", "libero"):
        try:
            mod = importlib.import_module(module_name)
            mod_file = getattr(mod, "__file__", None)
            if mod_file:
                bddl_roots.append(Path(mod_file).resolve().parent / "bddl_files")
        except Exception:
            continue
    if env_mod and getattr(env_mod, "__file__", None):
        bddl_roots.append(Path(env_mod.__file__).resolve().parents[1] / "bddl_files")
    if EXTRA_LIBERO_PATH:
        bddl_roots.append(Path(EXTRA_LIBERO_PATH) / "libero" / "bddl_files")
        bddl_roots.append(Path(EXTRA_LIBERO_PATH) / "bddl_files")

    seen = set()
    uniq_roots = []
    for root in bddl_roots:
        k = str(root)
        if k not in seen:
            seen.add(k)
            uniq_roots.append(root)

    task_bddl_file = None
    for root in uniq_roots:
        cand = root / rel_bddl
        if cand.exists():
            task_bddl_file = str(cand)
            break
    if task_bddl_file is None:
        raise FileNotFoundError(
            f"Could not resolve BDDL file {rel_bddl} from candidates: {[str(p) for p in uniq_roots]}"
        )
    task_mapping = getattr(env_mod, "TASK_MAPPING", {})
    known_problems = set(task_mapping.keys()) if isinstance(task_mapping, dict) else set()

    def _problem_name_from_bddl(path: str) -> str | None:
        try:
            txt = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        m = re.search(r"\(problem\s+([^\s\)]+)\)", txt)
        return m.group(1) if m else None

    resolved_bddl = _resolve_bddl_for_task(task_name, benchmark_mod, uniq_roots)
    candidate_bddls = [resolved_bddl] if resolved_bddl else [task_bddl_file]
    for root in uniq_roots:
        if not root.exists():
            continue
        for cand in root.rglob("*.bddl"):
            prob = _problem_name_from_bddl(str(cand))
            if prob is None or (known_problems and prob in known_problems):
                candidate_bddls.append(str(cand))
    seen = set()
    candidate_bddls = [p for p in candidate_bddls if not (p in seen or seen.add(p))]

    env = None
    env_exc = None
    for cand in candidate_bddls:
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=cand,
                camera_heights=img_hw[0],
                camera_widths=img_hw[1],
                render_gpu_device_id=0,
            )
            break
        except Exception as exc:  # noqa: BLE001
            env_exc = exc
            continue
    if env is None:
        raise RuntimeError("Failed to create OffScreenRenderEnv from candidate BDDL files") from env_exc

    agent_ids, wrist_ids = _mesh_seeds_for_scene(scene)
    env = MeshVertexWrapper(env, max_points=10000, include_non_mesh_geoms=True)
    env = MeshVertexTracker(
        env,
        agentview_vertex_ids=agent_ids,
        eye_in_hand_vertex_ids=wrist_ids,
        img_height=img_hw[0],
        img_width=img_hw[1],
    )
    env.reset()
    base_env = env
    while hasattr(base_env, "env") and not hasattr(base_env, "sim"):
        base_env = base_env.env
    expected_qpos = int(getattr(getattr(base_env, "sim", None).model, "nq", 0)) if hasattr(base_env, "sim") else 0
    state_arr = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    if expected_qpos > 0 and state_arr.shape[0] >= expected_qpos:
        try:
            env.set_init_state(state_arr)
        except Exception as exc:
            print(f"[warn] set_init_state failed ({exc}); using reset state for mesh seeds")
    else:
        print(
            f"[warn] simulator state unavailable or mismatched (got {state_arr.shape[0]}, expected >= {expected_qpos}); "
            "using reset state for mesh seeds"
        )
    agent_ids = _select_dynamic_ids_for_camera(
        env,
        camera_name="agentview",
        fallback_agent_ids=agent_ids,
        img_hw=img_hw,
    )
    wrist_ids = _select_dynamic_ids_for_camera(
        env,
        camera_name="robot0_eye_in_hand",
        fallback_agent_ids=wrist_ids,
        img_hw=img_hw,
    )
    env.agentview_vertex_ids = agent_ids
    env.eye_in_hand_vertex_ids = wrist_ids

    actions_arr = np.asarray(actions, dtype=np.float32)
    if actions_arr.ndim != 2:
        raise ValueError(f"Expected actions shape [T, A], got {actions_arr.shape}")

    av_seq = []
    wr_seq = []
    for t in range(actions_arr.shape[0]):
        if hasattr(env, "_update_vertex_positions"):
            env._update_vertex_positions()
        av = env._get_tracked_vertices("agentview", env.agentview_vertex_ids)["vertices_2d"]
        wr = env._get_tracked_vertices("robot0_eye_in_hand", env.eye_in_hand_vertex_ids)["vertices_2d"]
        av_seq.append(np.asarray(av, dtype=np.float32) / float(img_hw[1]))
        wr_seq.append(np.asarray(wr, dtype=np.float32) / float(img_hw[1]))
        if t < actions_arr.shape[0] - 1:
            env.step(actions_arr[t])
    env.close()
    return np.stack(av_seq, axis=0), np.stack(wr_seq, axis=0)


def process_episode(
    ds: LeRobotDataset,
    ep_idx: int,
    horizon: int,
    cotracker_model,
    device: torch.device,
    grid25: np.ndarray,
    mesh_source: str = "sim",
    mesh_ds: LeRobotDataset | None = None,
) -> Iterable[dict]:
    bnds = _episode_bounds(ds, ep_idx)
    frames = []
    offline_agent_mesh = []
    offline_wrist_mesh = []
    # Pull source frames for observations/state/actions.
    for i in range(bnds.start, bnds.end):
        row = ds[i]
        agent_mesh_row = row.get("agentview_mesh_vertices_2d", None)
        wrist_mesh_row = row.get("wrist_mesh_vertices_2d", None)
        frames.append(
            (
                np.asarray(row["image"]),
                np.asarray(row["wrist_image"]),
                np.asarray(row["state"], dtype=np.float32),
                np.asarray(row["actions"], dtype=np.float32),
                row.get("task", bnds.task),
                None if agent_mesh_row is None else np.asarray(agent_mesh_row, dtype=np.float32),
                None if wrist_mesh_row is None else np.asarray(wrist_mesh_row, dtype=np.float32),
            )
        )
        offline_agent_mesh.append(agent_mesh_row is not None)
        offline_wrist_mesh.append(wrist_mesh_row is not None)
    images = np.stack([f[0] for f in frames], axis=0)
    wrist_images = np.stack([f[1] for f in frames], axis=0)
    states = np.stack([f[2] for f in frames], axis=0)
    actions = np.stack([f[3] for f in frames], axis=0)
    task = frames[0][4]

    # Normalize to HWC uint8
    images = np.stack([_to_hwc_uint8(f[0]) for f in frames], axis=0)
    wrist_images = np.stack([_to_hwc_uint8(f[1]) for f in frames], axis=0)
    states = np.stack([f[2] for f in frames], axis=0)
    actions = np.stack([f[3] for f in frames], axis=0)
    task = frames[0][4]

    if images.shape[1] != 256 or images.shape[2] != 256:
        raise ValueError(f"Expected 256x256 images, got {images.shape[1:3]}")

    if mesh_source == "dataset":
        if not all(offline_agent_mesh) or not all(offline_wrist_mesh):
            raise ValueError(
                "mesh_source=dataset requires source dataset columns "
                "`agentview_mesh_vertices_2d` and `wrist_mesh_vertices_2d` for all frames."
            )
        agent_mesh_seq = np.stack([f[5] for f in frames], axis=0)
        wrist_mesh_seq = np.stack([f[6] for f in frames], axis=0)
    elif mesh_source == "mesh_repo":
        if mesh_ds is None:
            raise ValueError("mesh_source=mesh_repo requires mesh_ds to be provided.")
        mesh_root = Path(str(mesh_ds.repo_id))
        mesh_ep_file = _episode_file_from_global_index(mesh_root, ep_idx)
        mesh_table = pq.read_table(mesh_ep_file, columns=["agentview_mesh_vertices_2d", "wrist_mesh_vertices_2d"])
        src_len = bnds.end - bnds.start
        if mesh_table.num_rows != src_len:
            raise ValueError(
                f"mesh_repo episode length mismatch for ep={ep_idx}: "
                f"source_len={src_len} mesh_len={mesh_table.num_rows}"
            )
        mesh_agent_raw = mesh_table["agentview_mesh_vertices_2d"].to_pylist()
        mesh_wrist_raw = mesh_table["wrist_mesh_vertices_2d"].to_pylist()
        mesh_agent = [_coerce_mesh7(v, key="agentview_mesh_vertices_2d", frame_idx=i) for i, v in enumerate(mesh_agent_raw)]
        mesh_wrist = [_coerce_mesh7(v, key="wrist_mesh_vertices_2d", frame_idx=i) for i, v in enumerate(mesh_wrist_raw)]
        agent_mesh_seq = np.stack(mesh_agent, axis=0).astype(np.float32)
        wrist_mesh_seq = np.stack(mesh_wrist, axis=0).astype(np.float32)
    else:
        scene = _scene_from_task(bnds.task)
        # Mesh vertices from simulator for each frame (step simulator with episode actions).
        agent_mesh_seq, wrist_mesh_seq = _get_mesh_sequence_from_sim(
            scene, states[0], actions, task_name=task, img_hw=(images.shape[1], images.shape[2])
        )

    # Dense wrist grid tracking: launch fixed grid25 at every frame.
    dense_grid_tracks, dense_grid_vis = _track_time_indexed_queries(
        wrist_images, grid25, cotracker_model, device
    )  # [T, T*25, 2], [T, T*25]

    # Build frames
    for local_t, global_t in enumerate(range(bnds.start, bnds.end)):
        # Dense per-frame logic:
        # - agent window uses simulator mesh7 sequence from current frame onward
        # - wrist window uses fixed grid25 queries launched at current frame, plus simulator mesh7 sequence
        agent_window = _track_window(agent_mesh_seq, local_t, horizon)  # (H,7,2)
        q0 = local_t * grid25.shape[0]
        q1 = q0 + grid25.shape[0]
        wrist_grid_future_src = dense_grid_tracks[:, q0:q1, :]  # (T,25,2) for queries started at local_t
        wrist_grid_window = _track_window(wrist_grid_future_src, local_t, horizon)  # (H,25,2)
        wrist_mesh_window = _track_window(wrist_mesh_seq, local_t, horizon)  # (H,7,2)
        wrist_window = np.concatenate([wrist_grid_window, wrist_mesh_window], axis=1)  # (H,32,2)
        track_targets_raw = np.concatenate([agent_window, wrist_window], axis=1)  # (H,39,2)
        wrist_grid_curr = dense_grid_tracks[local_t, q0:q1, :]  # (25,2)
        wrist_grid_vis_curr = np.asarray(dense_grid_vis[local_t, q0:q1], dtype=np.float32)  # (25,)
        wrist_tracks_curr = np.concatenate([wrist_grid_curr, wrist_mesh_seq[local_t]], axis=0)  # (32,2)
        wrist_vis_curr = np.concatenate(
            [wrist_grid_vis_curr, np.ones((7,), dtype=np.float32)],
            axis=0,
        )

        yield {
            "image": images[local_t],
            "wrist_image": wrist_images[local_t],
            "state": states[local_t],
            "actions": actions[local_t],
            "task": task,
            "agentview_tracks": agent_mesh_seq[local_t],
            "agentview_vis": np.ones((7,), dtype=np.float32),
            "wrist_tracks": wrist_tracks_curr,
            "wrist_vis": wrist_vis_curr,
            "track_targets_raw": track_targets_raw,
            "agentview_mesh_vertices_2d": agent_mesh_seq[local_t],
            "wrist_mesh_vertices_2d": wrist_mesh_seq[local_t],
            "agentview_grid_mesh_32": np.concatenate([grid25, agent_mesh_seq[local_t]], axis=0),
            "wrist_grid_mesh_32": np.concatenate([grid25, wrist_mesh_seq[local_t]], axis=0),
            "has_track_mesh": np.asarray([1.0], dtype=np.float32),
        }


def main():
    p = argparse.ArgumentParser(description="Reprocess LIBERO RLDS to LeRobot with CoTracker and simulator mesh seeds.")
    p.add_argument("--source-repo-id", default="physical-intelligence/libero")
    p.add_argument("--target-repo-id", default="physical-intelligence/libero_with_tracks_mesh_cotracker")
    p.add_argument(
        "--cotracker-checkpoint",
        default=None,
        help="Optional CoTracker2 state dict; default downloads the pretrained cotracker2 hub weights.",
    )
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument(
        "--mesh-source",
        choices=["sim", "dataset", "mesh_repo"],
        default="sim",
        help=(
            "Mesh timeline source: sim=simulator rollout; dataset=source dataset mesh columns; "
            "mesh_repo=load mesh columns from --mesh-repo-id aligned by episode/frame."
        ),
    )
    p.add_argument(
        "--mesh-repo-id",
        type=str,
        default=None,
        help="LeRobot dataset path used when --mesh-source=mesh_repo.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--start-episode", type=int, default=0, help="Start episode index (inclusive).")
    p.add_argument("--end-episode", type=int, default=None, help="End episode index (exclusive).")
    p.add_argument("--libero-root", type=str, default=None, help="Path to LIBERO codebase (adds to sys.path)")
    p.add_argument("--libero-data", type=str, default=None, help="Path to LIBERO assets (sets LIBERO_PATH)")
    p.add_argument(
        "--extra-libero-path",
        type=str,
        # LIBERO imports itself as ``libero.libero.*``, so the vendored repo root goes on sys.path.
        default=str(Path(__file__).resolve().parents[1] / "third_party" / "libero"),
        help="LIBERO code path to add to sys.path (vendored: third_party/libero).",
    )
    args = p.parse_args()

    global EXTRA_LIBERO_PATH
    EXTRA_LIBERO_PATH = args.extra_libero_path
    if args.libero_root and (Path(args.libero_root) / "libero" / "envs" / "mesh_vertex_wrapper.py").exists():
        EXTRA_LIBERO_PATH = args.libero_root

    sys_paths = []
    if args.libero_root:
        sys_paths.append(args.libero_root)
    if args.extra_libero_path:
        sys_paths.append(args.extra_libero_path)
    for pth in reversed(sys_paths):
        if pth and pth not in sys.path and Path(pth).exists():
            sys.path.insert(0, pth)

    if args.libero_data:
        os.environ.setdefault("LIBERO_PATH", args.libero_data)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid25 = _make_grid_25()

    print(f"Loading source LeRobot dataset: {args.source_repo_id}")
    src = LeRobotDataset(args.source_repo_id)
    mesh_ds = None
    if args.mesh_source == "mesh_repo":
        if not args.mesh_repo_id:
            raise ValueError("--mesh-repo-id is required when --mesh-source=mesh_repo.")
        print(f"Loading mesh LeRobot dataset: {args.mesh_repo_id}")
        mesh_ds = LeRobotDataset(args.mesh_repo_id)
        if len(mesh_ds.meta.episodes) < len(src.meta.episodes):
            raise ValueError(
                f"Mesh dataset has fewer episodes ({len(mesh_ds.meta.episodes)}) than source ({len(src.meta.episodes)})."
            )
    total_eps = len(src.meta.episodes)
    start_ep = max(0, int(args.start_episode))
    end_ep = total_eps if args.end_episode is None else min(int(args.end_episode), total_eps)
    if args.max_episodes is not None:
        end_ep = min(end_ep, start_ep + int(args.max_episodes))
    if end_ep <= start_ep:
        raise ValueError(f"Invalid episode range [{start_ep}, {end_ep}) for total {total_eps}")
    n_eps = end_ep - start_ep

    target_path = Path(args.target_repo_id)
    if not target_path.is_absolute():
        # Same root the training data loader reads (LeRobot's HF_LEROBOT_HOME).
        lerobot_home = os.environ.get(
            "HF_LEROBOT_HOME", str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "lerobot")
        )
        target_path = Path(lerobot_home) / args.target_repo_id
    if target_path.exists() and args.overwrite:
        import shutil
        shutil.rmtree(target_path)

    dst = LeRobotDataset.create(
        repo_id=str(target_path),
        robot_type="panda",
        fps=src.fps,
        features=_build_features(),
        image_writer_threads=10,
        image_writer_processes=5,
    )

    print(f"Loading CoTracker checkpoint from {args.cotracker_checkpoint} on {device}")
    cot_model = _load_cotracker(args.cotracker_checkpoint, device)

    for ep_idx in range(start_ep, end_ep):
        print(f"[ep {ep_idx-start_ep}/{n_eps}] abs={ep_idx}")
        for frame in process_episode(
            src,
            ep_idx,
            args.horizon,
            cot_model,
            device,
            grid25,
            mesh_source=args.mesh_source,
            mesh_ds=mesh_ds,
        ):
            dst.add_frame(frame)
        dst.save_episode()

    print(f"Done. Wrote {n_eps} episodes ({start_ep}:{end_ep}) to {args.target_repo_id}")


if __name__ == "__main__":
    main()
