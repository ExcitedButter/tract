"""
Full inference script: Delta Actions → Converter → Ctrl-World → Predicted Video

Usage:
    cd third_party/Ctrl-World
    source ../../.venv/bin/activate

    # RLDS episode (original):
    CUDA_VISIBLE_DEVICES=0 python scripts/inference_world_model.py \
        --source rlds --suite libero_goal_no_noops --episode 5

    # libero_viewpoint task 6 (put wine bottle, view_37 camera):
    CUDA_VISIBLE_DEVICES=0 python scripts/inference_world_model.py \
        --source libero_viewpoint --libero_viewpoint_task 6 \
        --openpi_root .

    # LeRobot parquet (default): episode_000000 agent+wrist, horizon 16 → results/ (320x192):
    CUDA_VISIBLE_DEVICES=0 python scripts/inference_world_model.py
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import cv2
import einops
import mediapy
import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_libero import wm_args
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.libero_action_converter import LiberoActionConverter

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
# Ctrl-World training / inference RGB (W x H), same as preprocess_libero.py
TARGET_W, TARGET_H = 320, 192
SOURCE_SIZE = 256
CROP_H = round(SOURCE_SIZE * TARGET_H / TARGET_W)  # 154
CROP_TOP = (SOURCE_SIZE - CROP_H) // 2  # 51
CTRL_WORLD_ROOT = Path(__file__).resolve().parents[1]
# tract repo root (third_party/Ctrl-World/scripts -> repo); shared checkpoints/ and lerobot_cache/ live here.
REPO_ROOT = CTRL_WORLD_ROOT.parents[1]
_DEFAULT_OUTPUT_DIR = CTRL_WORLD_ROOT / "results"
_DEFAULT_PARQUET = str(REPO_ROOT) + "/" + (
    "lerobot_cache/physical-intelligence/"
    "libero_cotracker_densegrid_merged/data/chunk-000/episode_000000.parquet"
)
_DEFAULT_LEROBOT_ROOT = str(REPO_ROOT) + "/" + (
    "lerobot_cache/physical-intelligence/"
    "libero_cotracker_densegrid_merged"
)
DEFAULT_HORIZON = 16
VIEWPOINT_MAX_STEPS = 520

_HF_SVD_REPO = "stabilityai/stable-video-diffusion-img2vid"


def resolve_ctrl_world_path(path: str | Path) -> Path:
    """Resolve data paths relative to Ctrl-World root when not run from that cwd."""
    p = Path(path).expanduser()
    if p.is_file():
        return p.resolve()
    if not p.is_absolute():
        alt = CTRL_WORLD_ROOT / p
        if alt.is_file():
            return alt.resolve()
    return p.resolve()


def resolve_ctrl_world_output(path: str | Path) -> Path:
    """Resolve output paths under Ctrl-World root (avoids scripts/scripts/ when cwd is scripts/)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (CTRL_WORLD_ROOT / p).resolve()

_HF_CLIP_REPO = "openai/clip-vit-base-patch32"
# Ctrl-World libero ckpt uses UNet conv_in with in_channels=8 (official SVD only).
_REQUIRED_SVD_IN_CHANNELS = 8
_EXTRA_SVD_CANDIDATES = [
    REPO_ROOT / "checkpoints" / "svd",
    CTRL_WORLD_ROOT / "checkpoints" / "svd",
    Path("/cephfs/shared/llm/stable-video-diffusion-img2vid"),
]
# NOT compatible: animate-anything SVD has in_channels=9
_SVD_SKIP_IN_CHANNELS = {9}
_EXTRA_CLIP_CANDIDATES = [
    REPO_ROOT / "checkpoints" / "clip-vit-base-patch32",
    CTRL_WORLD_ROOT / "checkpoints" / "clip-vit-base-patch32",
]


def _is_local_diffusers_repo(path: Path) -> bool:
    return path.is_dir() and (path / "model_index.json").is_file()


def _svd_in_channels(path: Path) -> int | None:
    cfg = path / "unet" / "config.json"
    if not cfg.is_file():
        return None
    with open(cfg, encoding="utf-8") as f:
        return int(json.load(f).get("in_channels", 8))


def _check_svd_compatible(path: Path) -> bool:
    if not _is_local_diffusers_repo(path):
        return False
    inc = _svd_in_channels(path)
    if inc in _SVD_SKIP_IN_CHANNELS or inc != _REQUIRED_SVD_IN_CHANNELS:
        print(
            f"Skip SVD at {path}: unet in_channels={inc} "
            f"(Ctrl-World libero ckpt requires {_REQUIRED_SVD_IN_CHANNELS})"
        )
        return False
    return True


def download_official_svd(dest: Path | None = None) -> str:
    """Download stabilityai/stable-video-diffusion-img2vid (8-channel UNet)."""
    dest = dest or (REPO_ROOT / "checkpoints" / "svd")
    dest = dest.expanduser().resolve()
    if _check_svd_compatible(dest):
        print(f"SVD already present: {dest}")
        return str(dest)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub to auto-download SVD") from e
    print(f"Downloading {_HF_SVD_REPO} -> {dest} (large, ~8GB)...")
    snapshot_download(
        repo_id=_HF_SVD_REPO,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    if not _check_svd_compatible(dest):
        raise RuntimeError(f"Downloaded SVD at {dest} still incompatible")
    return str(dest)


def resolve_svd_path(user_path: str | None, *, auto_download: bool = True) -> str:
    if user_path:
        p = Path(user_path).expanduser()
        if _is_local_diffusers_repo(p):
            if not _check_svd_compatible(p):
                raise ValueError(
                    f"SVD at {p} has wrong in_channels. Use official "
                    f"{_HF_SVD_REPO} (not animate-anything 9-ch)."
                )
            return str(p)
        if "/" not in user_path and "\\" not in user_path:
            return user_path
        raise FileNotFoundError(f"SVD path invalid (no model_index.json): {p}")
    for cand in _EXTRA_SVD_CANDIDATES:
        if _check_svd_compatible(cand):
            print(f"Using local SVD: {cand}")
            return str(cand)
    if auto_download:
        return download_official_svd()
    print(f"No local SVD found; pass --svd_path or use HF repo: {_HF_SVD_REPO}")
    return _HF_SVD_REPO


def resolve_clip_path(user_path: str | None) -> str:
    if user_path:
        p = Path(user_path).expanduser()
        if p.is_dir() and (p / "config.json").is_file():
            return str(p)
        if "/" not in user_path and "\\" not in user_path:
            return user_path
        raise FileNotFoundError(f"CLIP path invalid (no config.json): {p}")
    for cand in _EXTRA_CLIP_CANDIDATES:
        if cand.is_dir() and (cand / "config.json").is_file():
            print(f"Using local CLIP: {cand}")
            return str(cand)
    print(f"No local CLIP found; will download from HuggingFace: {_HF_CLIP_REPO}")
    return _HF_CLIP_REPO


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float32, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _state7_from_obs(obs: dict) -> np.ndarray:
    """Ctrl-World / converter use 7D: pos(3) + axis-angle(3) + gripper width(1)."""
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper.size >= 2:
        gripper_w = float(gripper[0] - gripper[1])
    elif gripper.size == 1:
        gripper_w = float(gripper[0])
    else:
        gripper_w = 0.0
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.array([gripper_w], dtype=np.float32),
        ]
    ).astype(np.float32)


def resize_frame_to_ctrlworld(img: np.ndarray) -> np.ndarray:
    """256x256 → vertical 5:3 center strip → resize to Ctrl-World 320x192 (H x W)."""
    img = np.asarray(img, dtype=np.uint8)
    if img.shape[0] == SOURCE_SIZE and img.shape[1] == SOURCE_SIZE:
        cropped = img[CROP_TOP : CROP_TOP + CROP_H, :, :]
        return cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)
    if img.shape[0] == TARGET_H and img.shape[1] == TARGET_W:
        return img
    return cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)


def stack_latent_views(agent_latent: torch.Tensor, wrist_latent: torch.Tensor) -> torch.Tensor:
    zero_latent = torch.zeros_like(agent_latent)
    stacked = torch.zeros(agent_latent.shape[0], 4, 72, 40)
    stacked[:, :, 0:24] = agent_latent
    stacked[:, :, 24:48] = zero_latent
    stacked[:, :, 48:72] = wrist_latent
    return stacked


def vae_encode_frames(frames: np.ndarray, vae, device, batch_size: int = 32) -> torch.Tensor:
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0 * 2 - 1
    latents = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size].to(device)
            z = vae.encode(batch).latent_dist.sample() * vae.config.scaling_factor
            latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def _setup_openpi_libero(openpi_root: Path):
    openpi_root = openpi_root.resolve()
    for p in (openpi_root / "third_party", openpi_root / "third_party" / "libero"):
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)
    if str(openpi_root) not in sys.path:
        sys.path.insert(0, str(openpi_root))
    cfg_root = Path(os.environ.get("LIBERO_CONFIG_PATH", str(openpi_root / ".libero_local")))
    cfg_root.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_root)


def _get_viewpoint_task(task_id: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    entry = libero_task_map["libero_viewpoint"][task_id]
    init_folder, stem = entry
    base_task = stem.split("_view_")[0]
    bench = benchmark.get_benchmark("libero_viewpoint")()
    task = bench.get_task(task_id)
    return task, stem, init_folder, base_task, get_libero_path


def _build_viewpoint_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.mesh_vertex_tracker import MeshVertexTracker
    from libero.libero.envs.mesh_vertex_wrapper import MeshVertexWrapper

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    env = MeshVertexWrapper(env, max_points=10000, include_non_mesh_geoms=True)
    env = MeshVertexTracker(
        env,
        agentview_vertex_ids=[3001, 3004, 3014, 3067, 3040, 3081, 2898],
        eye_in_hand_vertex_ids=[2955, 3055, 3040, 2981, 2966, 3052, 3022],
        img_height=resolution,
        img_width=resolution,
    )
    return env


def _load_openpi_policy(openpi_root: Path, config_name: str, checkpoint_dir: str):
    import dataclasses

    from tract import transforms as data_transforms
    from tract.policies import policy_config as _policy_config
    from tract.training import config as training_config

    cfg = training_config.get_config(config_name)
    eval_cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, include_track_targets=False))
    eval_data_cfg = eval_cfg.data.create(eval_cfg.assets_dirs, eval_cfg.model)
    policy = _policy_config.create_trained_policy(eval_cfg, checkpoint_dir)
    adapter = data_transforms.compose(
        [*eval_data_cfg.repack_transforms.inputs, *eval_data_cfg.data_transforms.inputs]
    )
    return policy, adapter


def _prepare_policy_obs(
    obs,
    task_description: str,
    *,
    resize_size: int = 224,
    flip_image_x: bool = True,
    flip_image_y: bool = True,
):
    from openpi_client import image_tools

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

    agent_rs = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agent_img, resize_size, resize_size)
    )
    wrist_rs = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)
    h, w = int(obs["agentview_image"].shape[0]), int(obs["agentview_image"].shape[1])
    agent_mesh = np.clip(
        np.asarray(obs["agentview_tracked_vertices_2d"], dtype=np.float32) / np.array([w, h], dtype=np.float32),
        0.0,
        1.0,
    )
    wrist_mesh = np.clip(
        np.asarray(obs["eye_in_hand_tracked_vertices_2d"], dtype=np.float32) / np.array([w, h], dtype=np.float32),
        0.0,
        1.0,
    )
    if flip_image_x:
        agent_mesh[:, 0] = 1.0 - agent_mesh[:, 0]
        wrist_mesh[:, 0] = 1.0 - wrist_mesh[:, 0]
    if flip_image_y:
        agent_mesh[:, 1] = 1.0 - agent_mesh[:, 1]
        wrist_mesh[:, 1] = 1.0 - wrist_mesh[:, 1]

    u = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    v = np.linspace(0.1, 0.9, 5, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    wrist_grid = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1).astype(np.float32)
    wrist_tracks_32 = np.concatenate([wrist_grid, wrist_mesh], axis=0).astype(np.float32)

    raw_item = {
        "image": agent_rs,
        "wrist_image": wrist_rs,
        "state": state,
        "actions": np.zeros((16, 7), dtype=np.float32),
        "prompt": str(task_description),
        "agentview_mesh_vertices_2d": agent_mesh.astype(np.float32),
        "wrist_mesh_vertices_2d": wrist_mesh.astype(np.float32),
        "wrist_tracks": wrist_tracks_32,
        "wrist_grid_mesh_32": wrist_tracks_32,
    }
    return raw_item, agent_img, wrist_img


def collect_libero_viewpoint_rollout(
    *,
    openpi_root: Path,
    task_id: int,
    rollout_idx: int,
    seed: int,
    config_name: str,
    checkpoint_dir: str,
    env_resolution: int = 256,
    max_steps: int = VIEWPOINT_MAX_STEPS,
    num_steps_wait: int = 10,
    replan_steps: int = 5,
    use_policy: bool = True,
) -> dict:
    import collections

    _setup_openpi_libero(openpi_root)
    from libero.libero import benchmark

    task, stem, init_folder, base_task, _ = _get_viewpoint_task(task_id)
    bench = benchmark.get_benchmark("libero_viewpoint")()
    initial_states = bench.get_task_init_states(task_id)
    env = _build_viewpoint_env(task, env_resolution, seed)

    policy, adapter = (None, None)
    if use_policy:
        policy, adapter = _load_openpi_policy(openpi_root, config_name, checkpoint_dir)

    env.reset()
    obs = env.set_init_state(initial_states[rollout_idx % len(initial_states)])
    action_plan = collections.deque()
    agent_frames, wrist_frames, states, actions = [], [], [], []

    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            if done:
                break
            continue

        states.append(_state7_from_obs(obs))
        agent_frames.append(resize_frame_to_ctrlworld(np.asarray(obs["agentview_image"])))
        wrist_frames.append(resize_frame_to_ctrlworld(np.asarray(obs["robot0_eye_in_hand_image"])))

        if use_policy and policy is not None:
            raw_item, _, _ = _prepare_policy_obs(obs, task.language)
            if not action_plan:
                infer_out = policy.infer(raw_item)
                action_chunk = infer_out["actions"]
                action_plan.extend(action_chunk[:replan_steps])
            action = np.asarray(action_plan.popleft(), dtype=np.float32)
        else:
            action = np.zeros(7, dtype=np.float32)

        actions.append(action.copy())
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break

    env.close()
    return {
        "task_id": task_id,
        "stem": stem,
        "init_folder": init_folder,
        "base_task": base_task,
        "language": task.language,
        "agent_frames": np.stack(agent_frames, axis=0),
        "wrist_frames": np.stack(wrist_frames, axis=0),
        "states_7d": np.stack(states, axis=0),
        "actions": np.stack(actions, axis=0),
    }


# OSC_POSE axis index → agentview screen direction (LIBERO tabletop, default camera).
_NUDGE_PRESETS = {
    "left": (1, -0.36),
    "right": (1, 0.36),
    "forward": (0, 0.36),
    "back": (0, -0.36),
    "up": (2, 0.36),
    "down": (2, -0.36),
}


def make_nudge_delta_actions(
    num_steps: int = 16,
    *,
    axis: int = 1,
    per_step_delta: float = -0.36,
) -> np.ndarray:
    """OSC_POSE 7D deltas: constant translation on one axis, gripper open (-1)."""
    actions = np.zeros((num_steps, 7), dtype=np.float32)
    actions[:, axis] = per_step_delta
    actions[:, 6] = -1.0
    return actions


def make_nudge_right_delta_actions(num_steps: int = 16, axis_delta: float = 0.12) -> np.ndarray:
    """Backward-compatible alias: nudge right on agentview (+y OSC)."""
    return make_nudge_delta_actions(num_steps, axis=1, per_step_delta=abs(axis_delta))


def load_rgb_frame(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_initial_state_viewpoint(
    openpi_root: Path,
    task_id: int,
    rollout_idx: int,
    seed: int,
    env_resolution: int = 256,
) -> tuple[np.ndarray, str]:
    """Read EE state at task init (for converter + world-model conditioning)."""
    _setup_openpi_libero(openpi_root)
    from libero.libero import benchmark

    task, _, _, _, _ = _get_viewpoint_task(task_id)
    bench = benchmark.get_benchmark("libero_viewpoint")()
    inits = bench.get_task_init_states(task_id)
    env = _build_viewpoint_env(task, env_resolution, seed)
    env.reset()
    obs = env.set_init_state(inits[rollout_idx % len(inits)])
    for _ in range(10):
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        if done:
            break
    state7 = _state7_from_obs(obs)
    lang = task.language
    env.close()
    return state7, lang


def build_latents_from_first_frames(
    agent_rgb: np.ndarray,
    wrist_rgb: np.ndarray | None,
    vae,
    device,
    num_frames: int = 17,
) -> torch.Tensor:
    """Encode first observation; pad time axis for world-model window (history + future)."""
    agent = resize_frame_to_ctrlworld(agent_rgb)
    wrist = resize_frame_to_ctrlworld(wrist_rgb) if wrist_rgb is not None else agent.copy()
    agent_lat = vae_encode_frames(agent[None], vae, device)
    wrist_lat = vae_encode_frames(wrist[None], vae, device)
    stacked0 = stack_latent_views(agent_lat, wrist_lat)[0]
    if num_frames <= 1:
        return stacked0.unsqueeze(0)
    pad = stacked0.unsqueeze(0).expand(num_frames - 1, -1, -1, -1).clone()
    return torch.cat([stacked0.unsqueeze(0), pad], dim=0)


def run_single_frame_world_model(
    *,
    init_frame: Path,
    wrist_frame: Path | None,
    model,
    pipeline,
    states_norm: np.ndarray,
    task_text: str,
    args,
    device,
    svd_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot: first-frame latent + normalized states → 16 predicted agent + wrist frames."""
    from diffusers.models import AutoencoderKLTemporalDecoder

    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)
    vae.eval()

    agent_rgb = load_rgb_frame(init_frame)
    wrist_rgb = load_rgb_frame(wrist_frame) if wrist_frame and wrist_frame.is_file() else None
    num_history = args.num_history
    num_frames = args.num_frames
    latents = build_latents_from_first_frames(
        agent_rgb, wrist_rgb, vae, device, num_frames=num_history + num_frames + 1
    )

    pred_agent, _, pred_wrist, _ = run_world_model_inference(
        model,
        pipeline,
        latents,
        states_norm,
        task_text,
        args,
        device,
        start_frame=0,
    )
    return pred_agent, pred_wrist


def load_frames_from_dir(frames_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    agent_dir = frames_dir / "agentview"
    wrist_dir = frames_dir / "wrist"
    if not agent_dir.is_dir() or not wrist_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {agent_dir} and {wrist_dir} (run openpi/2.py to split rollout video)"
        )

    def _load_sorted(folder: Path) -> np.ndarray:
        paths = sorted(folder.glob("frame_*.png"))
        if not paths:
            raise FileNotFoundError(f"No frame_*.png in {folder}")
        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
        return np.stack([resize_frame_to_ctrlworld(im) for im in imgs], axis=0)

    return _load_sorted(agent_dir), _load_sorted(wrist_dir)


def _decode_parquet_raw(cell) -> np.ndarray:
    """Decode LeRobot v2 parquet image cell to RGB uint8 without resize."""
    if isinstance(cell, dict) and "bytes" in cell:
        raw = cell["bytes"]
    elif isinstance(cell, (bytes, bytearray, memoryview)):
        raw = bytes(cell)
    else:
        arr = np.asarray(cell)
        if arr.dtype == np.uint8 and arr.ndim == 3:
            return arr
        raise TypeError(f"Unsupported parquet image cell type: {type(cell)}")
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed on parquet image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_parquet_image(cell) -> np.ndarray:
    return resize_frame_to_ctrlworld(_decode_parquet_raw(cell))


def _load_lerobot_task_text(meta_root: Path, task_index: int, episode_index: int | None = None) -> str:
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


def load_lerobot_parquet_episode(
    parquet_path: Path,
    *,
    meta_root: Path | None = None,
    horizon: int = 16,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]:
    """
    Load LeRobot parquet episode for Ctrl-World inference.

    Returns:
        states_7d: (T, 7) absolute EE states from 8D observation
        actions: (T, 7) delta actions
        task_text: language instruction
        agent_frames: (horizon+1, TARGET_H, TARGET_W, 3) agentview RGB
        wrist_frames: (horizon+1, TARGET_H, TARGET_W, 3) wrist RGB
    """
    import pandas as pd

    parquet_path = resolve_ctrl_world_path(parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if len(df) < horizon + 1:
        raise ValueError(
            f"Episode too short for horizon={horizon}: {len(df)} frames in {parquet_path}"
        )

    if meta_root is None:
        meta_root = parquet_path.parents[2]
    meta_root = Path(meta_root).expanduser().resolve()

    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
    episode_index = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else None
    task_text = _load_lerobot_task_text(meta_root, task_index, episode_index)
    if not task_text:
        task_text = f"lerobot episode {parquet_path.stem}"

    states_8 = np.stack([np.asarray(row, dtype=np.float32) for row in df["state"].values], axis=0)
    states_7d = np.column_stack([states_8[:, :6], states_8[:, 6] - states_8[:, 7]]).astype(np.float32)
    actions = np.stack([np.asarray(row, dtype=np.float32) for row in df["actions"].values], axis=0)

    n_frames = horizon + 1
    agent_frames = np.stack(
        [_decode_parquet_image(df["image"].iloc[i]) for i in range(n_frames)],
        axis=0,
    )
    wrist_frames = np.stack(
        [_decode_parquet_image(df["wrist_image"].iloc[i]) for i in range(n_frames)],
        axis=0,
    )
    return states_7d, actions, task_text, agent_frames, wrist_frames


def load_rlds_episode(rlds_dir, suite, episode_idx):
    """Load a single episode from RLDS TFRecords."""
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    tfrecords = sorted(glob.glob(os.path.join(rlds_dir, suite, "1.0.0", "*.tfrecord*")))
    idx = 0
    for tfr in tfrecords:
        for rec in tf.data.TFRecordDataset(tfr):
            if idx == episode_idx:
                ex = tf.train.SequenceExample()
                ex.ParseFromString(rec.numpy())
                s8 = np.array(ex.context.feature["steps/observation/state"].float_list.value, dtype=np.float32).reshape(-1, 8)
                a7 = np.array(ex.context.feature["steps/action"].float_list.value, dtype=np.float32).reshape(-1, 7)
                lang = ex.context.feature["steps/language_instruction"].bytes_list.value[0].decode()
                T = min(s8.shape[0], a7.shape[0])
                s7 = np.column_stack([s8[:T, :6], s8[:T, 6] - s8[:T, 7]])
                return s7, a7[:T], lang
            idx += 1
    raise ValueError(f"Episode {episode_idx} not found")


def load_episode_latents(dataset_dir, episode_id):
    """Load preprocessed latent videos for an episode."""
    for split in ["train", "val"]:
        latent_dir = os.path.join(dataset_dir, "latent_videos", split, episode_id)
        if os.path.exists(latent_dir):
            view0 = torch.load(os.path.join(latent_dir, "0.pt"), map_location="cpu")
            view1 = torch.load(os.path.join(latent_dir, "1.pt"), map_location="cpu")
            view2 = torch.load(os.path.join(latent_dir, "2.pt"), map_location="cpu")
            T = view0.shape[0]
            stacked = torch.zeros(T, 4, 72, 40)
            stacked[:, :, 0:24] = view0
            stacked[:, :, 24:48] = view1
            stacked[:, :, 48:72] = view2
            return stacked
    raise FileNotFoundError(f"Latents not found for {episode_id}")


def normalize_states(states_7d, stat_path):
    """Normalize 7D states to [-1, 1] using dataset statistics."""
    stat_file = resolve_ctrl_world_path(stat_path)
    with open(stat_file) as f:
        stat = json.load(f)
    p01 = np.array(stat["state_01"])
    p99 = np.array(stat["state_99"])
    normalized = 2 * (states_7d - p01) / (p99 - p01 + 1e-8) - 1
    return np.clip(normalized, -1, 1)


def decode_latents(latents, pipeline, decode_chunk_size=7):
    """Decode (B, F, 4, H, W) latents → (B, F, H*8, W*8, 3) uint8."""
    bsz, frame_num = latents.shape[:2]
    flat = latents.flatten(0, 1)
    decoded = []
    for i in range(0, flat.shape[0], decode_chunk_size):
        chunk = flat[i : i + decode_chunk_size] / pipeline.vae.config.scaling_factor
        decoded.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
    video = torch.cat(decoded, dim=0)
    video = video.reshape(bsz, frame_num, *video.shape[1:])
    video = ((video / 2.0 + 0.5).clamp(0, 1) * 255)
    return video.to(torch.float32).detach().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)


def run_world_model_inference(
    model, pipeline, latents, states_norm, text, args, device,
    start_frame=0,
):
    """Run Ctrl-World inference for 16 frames from start_frame.

    Args:
        model: CrtlWorld model
        pipeline: SVD pipeline (for decoding)
        latents: (T, 4, 72, 40) full episode stacked latents
        states_norm: (T, 7) normalized absolute EE states
        text: task instruction string
        args: wm_args config
        device: torch device
        start_frame: which frame to start from

    Returns:
        pred_agent: (num_frames, H, W, 3) predicted agentview
        gt_agent: (num_history + num_frames, H, W, 3) GT agentview latents decoded
        pred_wrist: (num_frames, H, W, 3) predicted wrist
        gt_wrist: (num_history + num_frames, H, W, 3) GT wrist latents decoded
    """
    num_history = args.num_history  # 1
    num_frames = args.num_frames   # 16

    # Extract window: [history, current, future...]
    his_idx = max(0, start_frame - 1)
    window_end = min(start_frame + num_frames + 1, latents.shape[0])
    window_latents = latents[his_idx:window_end].unsqueeze(0).to(device)  # (1, <=17, 4, 72, 40)

    # Pad if needed
    actual_len = window_latents.shape[1]
    if actual_len < num_history + num_frames:
        pad = torch.zeros(1, num_history + num_frames - actual_len, 4, 72, 40, device=device)
        window_latents = torch.cat([window_latents, pad], dim=1)

    his_latent = window_latents[:, :num_history]
    future_latent = window_latents[:, num_history:]
    current_latent = future_latent[:, 0]

    # Build action conditioning
    state_start = max(0, start_frame - num_history)
    state_end = min(start_frame + num_frames, states_norm.shape[0])
    action_window = states_norm[state_start:state_end]

    # Pad to num_history + num_frames
    if len(action_window) < num_history + num_frames:
        pad = np.tile(action_window[-1:], (num_history + num_frames - len(action_window), 1))
        action_window = np.concatenate([action_window, pad])
    action_window = action_window[:num_history + num_frames]

    actions = torch.from_numpy(action_window).float().unsqueeze(0).to(device)  # (1, 17, 7)

    # Encode actions + text
    with torch.no_grad():
        action_latent = model.action_encoder(
            actions, [text], model.tokenizer, model.text_encoder,
            frame_level_cond=args.frame_level_cond,
        )

        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(3 * args.height),
            num_frames=args.num_frames,
            history=his_latent,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )

    # Split views: m=0 agentview, m=1 unused, m=2 wrist (see stack_latent_views)
    pred_split = einops.rearrange(pred_latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)
    gt_full = torch.cat([his_latent, future_latent], dim=1)
    gt_split = einops.rearrange(gt_full, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)

    pred_agent = decode_latents(pred_split[0:1], pipeline)[0]
    gt_agent = decode_latents(gt_split[0:1], pipeline)[0]
    pred_wrist = decode_latents(pred_split[2:3], pipeline)[0]
    gt_wrist = decode_latents(gt_split[2:3], pipeline)[0]

    return pred_agent, gt_agent, pred_wrist, gt_wrist


def save_lerobot_dual_view_videos(
    out_dir: Path,
    run_tag: str,
    mode: str,
    horizon: int,
    *,
    gt_agent: np.ndarray,
    pred_agent: np.ndarray,
    gt_wrist: np.ndarray,
    pred_wrist: np.ndarray,
    fps: int,
) -> dict[str, Path]:
    """Write agent / wrist / stacked comparison MP4s (GT left, pred right per row)."""
    n = min(
        pred_agent.shape[0],
        pred_wrist.shape[0],
        gt_agent.shape[0],
        gt_wrist.shape[0],
        horizon,
    )
    gt_a, pr_a = gt_agent[:n], pred_agent[:n]
    gt_w, pr_w = gt_wrist[:n], pred_wrist[:n]

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
    mediapy.write_video(str(paths["agent"]), agent_cmp, fps=fps)
    mediapy.write_video(str(paths["wrist"]), wrist_cmp, fps=fps)
    mediapy.write_video(str(paths["both"]), both_cmp, fps=fps)
    mediapy.write_video(str(paths["pred_agent"]), pr_a, fps=fps)
    mediapy.write_video(str(paths["pred_wrist"]), pr_w, fps=fps)
    mediapy.write_video(str(paths["gt_agent"]), gt_a, fps=fps)
    mediapy.write_video(str(paths["gt_wrist"]), gt_w, fps=fps)
    return paths


def _load_viewpoint_episode(args_cli, device, vae):
    """Build latents + states for libero_viewpoint from frames, cache, or live rollout."""
    tag = f"libero_viewpoint_task{args_cli.libero_viewpoint_task:02d}"
    rollout_npz = Path(args_cli.rollout_npz) if args_cli.rollout_npz else None

    if args_cli.frames_dir:
        print(f"\nLoading frames from {args_cli.frames_dir}")
        agent_frames, wrist_frames = load_frames_from_dir(Path(args_cli.frames_dir))
        T = agent_frames.shape[0]
        if rollout_npz and rollout_npz.is_file():
            cache = np.load(rollout_npz)
            states_7d = cache["states_7d"][:T]
            actions = cache["actions"][: T - 1] if "actions" in cache else cache["actions"]
            task_text = str(cache["language"])
            print(f"  Loaded states/actions from {rollout_npz}")
        else:
            print("  No rollout_npz: collecting aligned states via LIBERO + openpi policy...")
            rollout = collect_libero_viewpoint_rollout(
                openpi_root=Path(args_cli.openpi_root),
                task_id=args_cli.libero_viewpoint_task,
                rollout_idx=args_cli.rollout_idx,
                seed=args_cli.seed,
                config_name=args_cli.openpi_config,
                checkpoint_dir=args_cli.openpi_checkpoint,
                env_resolution=args_cli.env_resolution,
                max_steps=min(VIEWPOINT_MAX_STEPS, T + args_cli.num_steps_wait),
                num_steps_wait=args_cli.num_steps_wait,
                replan_steps=args_cli.replan_steps,
                use_policy=True,
            )
            states_7d = rollout["states_7d"][:T]
            actions = rollout["actions"]
            task_text = rollout["language"]
            if rollout_npz:
                rollout_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    rollout_npz,
                    states_7d=rollout["states_7d"],
                    actions=rollout["actions"],
                    language=rollout["language"],
                    stem=rollout["stem"],
                )
                print(f"  Cached rollout -> {rollout_npz}")
    else:
        print(
            f"\nCollecting libero_viewpoint task {args_cli.libero_viewpoint_task} "
            f"rollout {args_cli.rollout_idx} in sim..."
        )
        rollout = collect_libero_viewpoint_rollout(
            openpi_root=Path(args_cli.openpi_root),
            task_id=args_cli.libero_viewpoint_task,
            rollout_idx=args_cli.rollout_idx,
            seed=args_cli.seed,
            config_name=args_cli.openpi_config,
            checkpoint_dir=args_cli.openpi_checkpoint,
            env_resolution=args_cli.env_resolution,
            max_steps=VIEWPOINT_MAX_STEPS,
            num_steps_wait=args_cli.num_steps_wait,
            replan_steps=args_cli.replan_steps,
            use_policy=not args_cli.no_policy,
        )
        agent_frames = rollout["agent_frames"]
        wrist_frames = rollout["wrist_frames"]
        states_7d = rollout["states_7d"]
        actions = rollout["actions"]
        task_text = rollout["language"]
        if rollout_npz:
            np.savez(
                rollout_npz,
                states_7d=states_7d,
                actions=actions,
                language=task_text,
                agent_frames=agent_frames,
                wrist_frames=wrist_frames,
                stem=rollout["stem"],
            )
            print(f"  Saved rollout cache -> {rollout_npz}")

    print(f"  Task: {task_text}")
    print(f"  Trajectory length: {len(states_7d)} steps, {agent_frames.shape[0]} frames")
    print("  Encoding frames with SVD VAE...")
    agent_lat = vae_encode_frames(agent_frames, vae, device)
    wrist_lat = vae_encode_frames(wrist_frames, vae, device)
    latents = stack_latent_views(agent_lat, wrist_lat)
    print(f"  Latent shape: {latents.shape}")
    return tag, task_text, states_7d, actions, latents


def main():
    parser = argparse.ArgumentParser(description="Ctrl-World + Converter full inference")
    parser.add_argument(
        "--source",
        choices=("single_frame", "rlds", "libero_viewpoint", "lerobot_parquet"),
        default="lerobot_parquet",
        help="lerobot_parquet (default): parquet agent+wrist, horizon actions | single_frame | libero_viewpoint | rlds",
    )
    parser.add_argument(
        "--init_frame",
        default=str(REPO_ROOT / "frames/agentview/frame_000000.png"),
        help="Agentview RGB for frame 0 (single_frame mode)",
    )
    parser.add_argument(
        "--wrist_frame",
        default=str(REPO_ROOT / "frames/wrist/frame_000000.png"),
        help="Wrist RGB for frame 0; falls back to agentview if missing",
    )
    parser.add_argument(
        "--output_video",
        default=str(_DEFAULT_OUTPUT_DIR / "single_frame_agent_wrist.mp4"),
        help="Output mp4 path (single_frame mode only)",
    )
    parser.add_argument(
        "--nudge_direction",
        choices=sorted(_NUDGE_PRESETS.keys()),
        default="left",
        help="Preset OSC translation (agentview: left/right = y axis)",
    )
    parser.add_argument(
        "--nudge_axis",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Override preset: OSC position axis 0=x, 1=y, 2=z",
    )
    parser.add_argument(
        "--nudge_delta",
        type=float,
        default=None,
        help="Per-step OSC delta on nudge_axis (sign = direction); default from --nudge_direction",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Output video FPS",
    )
    parser.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints/ctrl_world/libero_ctrlworld/checkpoint-20000.pt"))
    parser.add_argument(
        "--svd_path",
        default=None,
        help="Official 8-ch SVD folder or HF repo id (stabilityai/stable-video-diffusion-img2vid)",
    )
    parser.add_argument(
        "--download_svd",
        action="store_true",
        help="Download official SVD to checkpoints/svd if missing",
    )
    parser.add_argument(
        "--clip_path",
        default=None,
        help="Local CLIP folder or HF repo id (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--adapter",
        default=str(REPO_ROOT / "checkpoints" / "ctrl_world" / "libero_action_adapter.pt"),
        help="LIBERO delta→state MLP (not models/action_adapter/model2_15_9.pth which is Droid)",
    )
    parser.add_argument(
        "--stat_path",
        default=str(CTRL_WORLD_ROOT / "dataset_meta_info/libero/stat.json"),
    )
    parser.add_argument("--dataset_dir", default="dataset_example/libero")
    parser.add_argument("--rlds_dir", default="raw_data/modified_libero_rlds")
    parser.add_argument("--suite", default="libero_spatial_no_noops")
    parser.add_argument("--episode", type=int, default=3)
    parser.add_argument("--libero_viewpoint_task", type=int, default=6)
    parser.add_argument("--rollout_idx", type=int, default=0)
    parser.add_argument("--openpi_root", default=str(REPO_ROOT))
    parser.add_argument("--openpi_config", default="pi05_libero_stage2p5_joint")
    parser.add_argument(
        "--openpi_checkpoint",
        default=str(REPO_ROOT / "checkpoints/pi05_libero_stage2p5_joint/pi05_final/29999"),
    )
    parser.add_argument("--frames_dir", default=None, help="agentview/ + wrist/ PNG folders from 2.py")
    parser.add_argument("--rollout_npz", default=None, help="cache states/actions for frames_dir mode")
    parser.add_argument("--env_resolution", type=int, default=256)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no_policy", action="store_true", help="viewpoint: dummy actions (debug only)")
    parser.add_argument("--use_converter", action="store_true",
                        help="Use converter to derive states from delta actions (default for lerobot_parquet)")
    parser.add_argument(
        "--use_gt_states",
        action="store_true",
        help="Use GT absolute states from data instead of converter (lerobot_parquet / rlds)",
    )
    parser.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--parquet_path",
        default=_DEFAULT_PARQUET,
        help="LeRobot episode parquet: image + wrist_image + actions (lerobot_parquet mode)",
    )
    parser.add_argument(
        "--lerobot_meta_root",
        default=_DEFAULT_LEROBOT_ROOT,
        help="LeRobot dataset root containing meta/tasks.jsonl",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Number of delta actions / predicted frames from parquet (default 16)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=10,
        help="Output comparison video FPS (lerobot_parquet mode)",
    )
    args_cli = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    svd_resolved = resolve_svd_path(args_cli.svd_path, auto_download=args_cli.download_svd)
    clip_resolved = resolve_clip_path(args_cli.clip_path)

    print("Loading Ctrl-World model...")
    args = wm_args()
    args.svd_model_path = svd_resolved
    args.clip_model_path = clip_resolved
    model = CrtlWorld(args)
    print(f"  Loading checkpoint: {args_cli.ckpt}")
    state_dict = torch.load(args_cli.ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    pipeline = model.pipeline

    converter = LiberoActionConverter(device=str(device))
    converter.load_adapter(args_cli.adapter, device=str(device))
    print(f"  Converter loaded: {converter.has_adapter}")

    if args_cli.source == "lerobot_parquet":
        from diffusers.models import AutoencoderKLTemporalDecoder

        horizon = int(args_cli.horizon)
        args.num_frames = horizon
        parquet_path = Path(args_cli.parquet_path)
        out_dir = resolve_ctrl_world_output(args_cli.output_dir)
        print(f"\n[lerobot_parquet] parquet={parquet_path}")
        print(f"  views: agentview=image, wrist=wrist_image | horizon={horizon} | output={out_dir}")
        states_7d, actions, task_text, agent_frames, wrist_frames = load_lerobot_parquet_episode(
            parquet_path,
            meta_root=Path(args_cli.lerobot_meta_root),
            horizon=horizon,
        )
        run_tag = parquet_path.stem

        print(f"  Task: {task_text}")
        print(f"  Using {horizon} delta actions from parquet rows 0..{horizon - 1}")

        if args_cli.use_gt_states:
            print(f"  States: GT from parquet rows 0..{horizon}")
            ee_states = states_7d[: horizon + 1].astype(np.float32)
        else:
            print("  States: converter(initial_state, actions[:horizon])")
            ee_states = converter.trajectory(states_7d[0], actions[:horizon]).astype(np.float32)

        states_norm = normalize_states(ee_states, args_cli.stat_path).astype(np.float32)

        vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_resolved, subfolder="vae").to(device)
        vae.eval()
        print("  Encoding GT agent/wrist frames with SVD VAE...")
        agent_lat = vae_encode_frames(agent_frames, vae, device)
        wrist_lat = vae_encode_frames(wrist_frames, vae, device)
        latents = stack_latent_views(agent_lat, wrist_lat)

        print(f"  Running world model ({horizon} predicted frames, agent + wrist)...")
        pred_agent, _, pred_wrist, _ = run_world_model_inference(
            model,
            pipeline,
            latents,
            states_norm,
            task_text,
            args,
            device,
            start_frame=0,
        )
        gt_agent = agent_frames[1 : horizon + 1]
        gt_wrist = wrist_frames[1 : horizon + 1]

        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "gt_states" if args_cli.use_gt_states else "converter"
        paths = save_lerobot_dual_view_videos(
            out_dir,
            run_tag,
            mode,
            horizon,
            gt_agent=gt_agent,
            pred_agent=pred_agent,
            gt_wrist=gt_wrist,
            pred_wrist=pred_wrist,
            fps=int(args_cli.video_fps),
        )
        print(f"\nSaved pred agent: {paths['pred_agent']}")
        print(f"Saved pred wrist: {paths['pred_wrist']}")
        print(f"Saved GT agent:   {paths['gt_agent']}")
        print(f"Saved GT wrist:   {paths['gt_wrist']}")
        print(f"Saved agent  (GT|Pred): {paths['agent']}")
        print(f"Saved wrist  (GT|Pred): {paths['wrist']}")
        print(f"Saved stacked agent+wrist: {paths['both']}")
        return

    if args_cli.source == "single_frame":
        init_path = Path(args_cli.init_frame)
        wrist_path = Path(args_cli.wrist_frame)
        print(f"\n[single_frame] init={init_path}")
        init_state, task_text = get_initial_state_viewpoint(
            Path(args_cli.openpi_root),
            args_cli.libero_viewpoint_task,
            args_cli.rollout_idx,
            args_cli.seed,
            args_cli.env_resolution,
        )
        if args_cli.nudge_axis is not None and args_cli.nudge_delta is not None:
            nudge_axis, nudge_step = args_cli.nudge_axis, args_cli.nudge_delta
        elif args_cli.nudge_delta is not None:
            nudge_axis, nudge_step = _NUDGE_PRESETS[args_cli.nudge_direction]
            nudge_step = float(np.sign(nudge_step) or 1.0) * abs(args_cli.nudge_delta)
        else:
            nudge_axis, nudge_step = _NUDGE_PRESETS[args_cli.nudge_direction]
        delta_actions = make_nudge_delta_actions(16, axis=nudge_axis, per_step_delta=nudge_step)
        states_7d = converter.trajectory(init_state, delta_actions)
        states_norm = normalize_states(states_7d, args_cli.stat_path).astype(np.float32)
        print(f"  Task: {task_text}")
        print(f"  Nudge: direction={args_cli.nudge_direction} axis={nudge_axis} per_step={nudge_step}")
        print(f"  Initial EE pos: {init_state[:3]}")
        print(f"  Final EE pos (converted): {states_7d[-1, :3]}")
        print(f"  EE displacement (m): {states_7d[-1, :3] - states_7d[0, :3]}")
        print("  Running 16-frame world model prediction (agent + wrist)...")
        pred_agent, pred_wrist = run_single_frame_world_model(
            init_frame=init_path,
            wrist_frame=wrist_path,
            model=model,
            pipeline=pipeline,
            states_norm=states_norm,
            task_text=task_text,
            args=args,
            device=device,
            svd_path=svd_resolved,
        )
        init_agent = resize_frame_to_ctrlworld(load_rgb_frame(init_path))
        init_wrist = (
            resize_frame_to_ctrlworld(load_rgb_frame(wrist_path))
            if wrist_path.is_file()
            else init_agent.copy()
        )
        agent_video = np.concatenate([init_agent[None], pred_agent], axis=0)
        wrist_video = np.concatenate([init_wrist[None], pred_wrist], axis=0)
        both_video = np.concatenate([agent_video, wrist_video], axis=1)

        out_path = resolve_ctrl_world_output(args_cli.output_video)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_path.stem
        mediapy.write_video(str(out_path), both_video, fps=args_cli.fps)
        mediapy.write_video(str(out_dir / f"{stem}_agent.mp4"), agent_video, fps=args_cli.fps)
        mediapy.write_video(str(out_dir / f"{stem}_wrist.mp4"), wrist_video, fps=args_cli.fps)
        print(f"Saved agent+wrist stacked video ({both_video.shape[0]} frames): {out_path}")
        print(f"Saved agent-only: {out_dir / f'{stem}_agent.mp4'}")
        print(f"Saved wrist-only: {out_dir / f'{stem}_wrist.mp4'}")
        return

    if args_cli.source == "libero_viewpoint":
        from diffusers.models import AutoencoderKLTemporalDecoder

        vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_resolved, subfolder="vae").to(device)
        vae.eval()
        run_tag, task_text, states_7d, actions, latents = _load_viewpoint_episode(args_cli, device, vae)
        suite_short = run_tag
    else:
        suite_short = args_cli.suite.replace("_no_noops", "")
        print(f"\nLoading episode {args_cli.episode} from {args_cli.suite}")
        states_7d, actions, task_text = load_rlds_episode(args_cli.rlds_dir, args_cli.suite, args_cli.episode)
        episode_id = f"{suite_short}_{args_cli.episode:04d}"
        print(f"  Loading latents for {episode_id}")
        latents = load_episode_latents(args_cli.dataset_dir, episode_id)
        run_tag = f"{suite_short}_ep{args_cli.episode:04d}"

    T = len(states_7d)
    print(f"  Episode length: {T} steps")

    if args_cli.use_converter:
        print("\n  Using CONVERTER to derive states from delta actions")
        initial_state = states_7d[0]
        converted = converter.trajectory(initial_state, actions[: max(0, T - 1)])
        ee_states = converted[:T]
    else:
        print("\n  Using GROUND TRUTH absolute states")
        ee_states = states_7d

    states_norm = normalize_states(ee_states, args_cli.stat_path).astype(np.float32)

    print("\nRunning world model inference...")
    all_pred = []
    all_gt = []
    starts = list(range(1, min(T - 17, 80), 16))

    for start in starts:
        print(f"  Window start={start}")
        pred_agent, gt_agent_dec, _pred_wrist, _gt_wrist_dec = run_world_model_inference(
            model, pipeline, latents, states_norm, task_text, args, device,
            start_frame=start,
        )
        all_pred.append(pred_agent)
        all_gt.append(gt_agent_dec[1:])

    out_dir = os.path.join(
        args_cli.output_dir,
        f"task{args_cli.libero_viewpoint_task:02d}" if args_cli.source == "libero_viewpoint" else "",
    )
    os.makedirs(out_dir, exist_ok=True)

    mode = "converter" if args_cli.use_converter else "gt_states"
    for i, (pred, gt) in enumerate(zip(all_pred, all_gt)):
        n_frames = min(pred.shape[0], gt.shape[0])
        comparison = np.concatenate([gt[:n_frames], pred[:n_frames]], axis=1)
        out_path = os.path.join(out_dir, f"wm_inference_{run_tag}_roll{args_cli.rollout_idx}_{mode}_win{i}.mp4")
        mediapy.write_video(out_path, comparison, fps=4)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_windows = len(all_pred)
    fig, axes = plt.subplots(n_windows, 5, figsize=(20, 4 * n_windows))
    if n_windows == 1:
        axes = axes[np.newaxis, :]

    for row, (pred, gt) in enumerate(zip(all_pred, all_gt)):
        for col, t in enumerate([0, 3, 7, 11, 15]):
            if t < pred.shape[0] and t < gt.shape[0] - 1:
                combined = np.concatenate([gt[t + 1], pred[t]], axis=0)
                axes[row, col].imshow(combined)
            axes[row, col].set_title(f"Win {row} t={t}\nGT(top) Pred(bot)", fontsize=9)
            axes[row, col].axis("off")

    fig.suptitle(
        f"Ctrl-World | {task_text[:60]}\n{run_tag} | Mode: {mode} | {os.path.basename(args_cli.ckpt)}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    summary_path = os.path.join(out_dir, f"wm_inference_{run_tag}_roll{args_cli.rollout_idx}_{mode}_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved {n_windows} comparison videos to {out_dir}/")


if __name__ == "__main__":
    main()
