"""Build the dual-view track dataset.

Two passes:

**Pass 1 — resize** (`--resize`, default):
    Source: `data/task_data_320_square_tracks/<task>/<split>/<demo>.hdf5`
    Dest:   `data/task_data_{S}_dualview_tracks/<task>/<split>/<demo>.hdf5`

    Videos get cv2.resize'd from 320 to S per view. Tracks/actions/state copy
    verbatim.

**Pass 2 — VAE encode** (`--vae-encode`):
    Operates on the dest hdf5 files. Per demo, writes two new datasets in place:
    - root/video_latents_stacked:      (T, 4, 2S/8, S/8) fp16, scaling_factor applied
    - root/controlnet_latents_perwindow: (num_windows, 16, 4, 2S/8, S/8) fp16, raw mode (no scaling)

    Shared convention with pipeline:
    - video latents are scaled (training's `tensor_to_vae_latent` multiplies by scaling_factor).
    - controlnet latents are NOT scaled (training + pipeline call `vae.encode(...).latent_dist.mode()` raw).

Usage:
    python -m world_models.track_conditioned.preprocess.build_dualview_dataset \
        --src data/task_data_320_square_tracks \
        --dst data/task_data_224_dualview_tracks \
        --size 224 --num-workers 8

    python -m world_models.track_conditioned.preprocess.build_dualview_dataset \
        --dst data/task_data_224_dualview_tracks \
        --size 224 --vae-encode --gpu 0

    # Overlay sanity check (one demo):
    python -m world_models.track_conditioned.preprocess.build_dualview_dataset \
        --src data/task_data_320_square_tracks \
        --size 224 --validate task_1/train/session_demo_1.hdf5
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


VIDEO_KEYS = ["root/agentview/video", "root/eye_in_hand/video"]
COPY_KEYS = [
    "root/actions",
    "root/extra_states/ee_states",
    "root/task_emb_bert",
    "root/eye_in_hand/grid_tracks_perwindow",
    "root/mesh_data/vertex_tracking/agentview_vertices_2d",
    "root/mesh_data/vertex_tracking/eyeinhand_vertices_2d",
]

VIDEO_LAT_KEY = "root/video_latents_stacked"
CTRL_LAT_KEY = "root/controlnet_latents_perwindow"


# ------------------------------ Pass 1: resize ------------------------------

def resize_video(video: np.ndarray, size: int) -> np.ndarray:
    """(1, T, 3, H, W) uint8 → (1, T, 3, S, S) uint8."""
    assert video.ndim == 5 and video.shape[0] == 1 and video.shape[2] == 3, \
        f"unexpected video shape {video.shape}"
    _, T, C, H, W = video.shape
    if H == size and W == size:
        return video
    out = np.empty((1, T, 3, size, size), dtype=np.uint8)
    for t in range(T):
        frame = video[0, t].transpose(1, 2, 0)  # HWC
        resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        out[0, t] = resized.transpose(2, 0, 1)
    return out


def copy_with_resize(src_path: Path, dst_path: Path, size: int) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in VIDEO_KEYS:
            if key not in src:
                raise RuntimeError(f"missing {key} in {src_path}")
            video = np.asarray(src[key])
            resized = resize_video(video, size)
            dst.create_dataset(key, data=resized, compression="gzip", compression_opts=4)

        for key in COPY_KEYS:
            if key not in src:
                continue
            src.copy(key, dst, name=key)


def walk_demos(root: Path):
    tasks = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("task_")])
    for task in tasks:
        for split in ("train", "val"):
            split_dir = task / split
            if not split_dir.is_dir():
                continue
            for demo in sorted(split_dir.glob("*.hdf5")):
                yield demo


def rel_to(root: Path, demo: Path) -> Path:
    return demo.relative_to(root)


def _resize_worker(args):
    src, dst, size = args
    try:
        copy_with_resize(Path(src), Path(dst), size)
        return (src, None)
    except Exception as e:
        return (src, repr(e))


def build_resize(src_root: Path, dst_root: Path, size: int, num_workers: int) -> None:
    jobs = []
    for demo in walk_demos(src_root):
        rel = rel_to(src_root, demo)
        dst = dst_root / rel
        jobs.append((str(demo), str(dst), size))

    print(f"[resize] {len(jobs)} demos → {dst_root} @ {size}×{size} per view")
    if not jobs:
        return

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for i, (src, err) in enumerate(pool.imap_unordered(_resize_worker, jobs), 1):
                status = "OK" if err is None else f"FAIL {err}"
                print(f"[{i}/{len(jobs)}] {src} → {status}")
    else:
        for i, job in enumerate(jobs, 1):
            src, err = _resize_worker(job)
            status = "OK" if err is None else f"FAIL {err}"
            print(f"[{i}/{len(jobs)}] {src} → {status}")


# ------------------------------ Pass 2: VAE encode ------------------------------

def _load_vae(svd_path: str, device: str, dtype):
    import torch
    from diffusers import AutoencoderKLTemporalDecoder

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        svd_path, subfolder="vae", variant="fp16", torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def _stack_views_uint8(agent: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Stack (T, 3, S, S) + (T, 3, S, S) → (T, 3, 2S, S) uint8 on H axis."""
    return np.concatenate([agent, wrist], axis=2)


def _encode_frames_vae(frames_float: np.ndarray, vae, device, dtype, chunk: int,
                       apply_scaling: bool) -> np.ndarray:
    """(T, 3, H, W) float32 → (T, 4, H/8, W/8) float16 latents.

    `frames_float` is fed to `vae.encode` verbatim — the CALLER is responsible
    for the right range convention:
      - video frames: pass [-1, 1] (matches training's `tensor_to_vae_latent`)
      - heatmap:      pass [ 0, 1] (matches training's raw `vae.encode(condition_img)`
                                   — see pipeline `prepare_condition_image` comment
                                   "[0, 1] range && Torch data type")
    Passing the wrong range produces latents that don't match what the pretrained
    network was trained on.
    """
    import torch

    T = frames_float.shape[0]
    x = torch.from_numpy(frames_float).to(device=device, dtype=dtype)

    out_chunks = []
    with torch.no_grad():
        for i in range(0, T, chunk):
            batch = x[i:i + chunk]
            lat = vae.encode(batch).latent_dist.mode()
            if apply_scaling:
                lat = lat * vae.config.scaling_factor
            out_chunks.append(lat.to(torch.float16).cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


_HYBRID_VIS_FN = None


def _get_hybrid_vis_fn():
    """Load generate_hybrid_trajectory_vis once, memoize at module scope.

    Previous version re-imported the module on every window call (~150× per
    demo) which dominated CPU time for the controlnet-latent pass.
    """
    global _HYBRID_VIS_FN
    if _HYBRID_VIS_FN is not None:
        return _HYBRID_VIS_FN
    import importlib.util

    hybrid_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_loader", "video_realworld_hybrid_dataset.py",
    )
    spec = importlib.util.spec_from_file_location(
        "video_realworld_hybrid_dataset", hybrid_path)
    hybrid_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hybrid_mod)
    _HYBRID_VIS_FN = hybrid_mod.generate_hybrid_trajectory_vis
    return _HYBRID_VIS_FN


def _render_window_heatmap(
    agent_mesh_window: np.ndarray,   # (seq_len, 7, 2)
    wrist_mesh_window: np.ndarray,   # (seq_len, 7, 2)
    wrist_grid_window: np.ndarray,   # (seq_len, 25, 2)
    config: dict,
) -> np.ndarray:
    """Render stacked heatmap (seq_len, 3, 2S, S) float32 in [0,1]."""
    generate_hybrid_trajectory_vis = _get_hybrid_vis_fn()

    wrist_points = np.concatenate([wrist_grid_window, wrist_mesh_window], axis=1)  # (T, 32, 2)

    agent_hm = generate_hybrid_trajectory_vis(agent_mesh_window, config, "agentview")       # (T, 3, S, S)
    wrist_hm = generate_hybrid_trajectory_vis(wrist_points, config, "eye_in_hand")          # (T, 3, S, S)
    return np.concatenate([agent_hm, wrist_hm], axis=2)                                     # (T, 3, 2S, S)


def _pad_window(arr: np.ndarray, target: int) -> np.ndarray:
    """Repeat last frame to reach target T."""
    if arr.shape[0] >= target:
        return arr[:target]
    pad = np.repeat(arr[-1:], target - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _demo_complete(path: Path, seq_len: int, size: int) -> bool:
    """Return True if the file already has valid latent datasets.

    Guards against half-written files from interrupted runs: checks that both
    datasets exist and that shapes are internally consistent (ctrl num_windows
    matches video T, spatial dims match size/seq_len).
    """
    try:
        with h5py.File(path, "r") as f:
            if VIDEO_LAT_KEY not in f or CTRL_LAT_KEY not in f:
                return False
            vid = f[VIDEO_LAT_KEY]
            ctrl = f[CTRL_LAT_KEY]
            H_lat = (2 * size) // 8
            W_lat = size // 8
            if vid.ndim != 4 or tuple(vid.shape[1:]) != (4, H_lat, W_lat):
                return False
            if ctrl.ndim != 5 or tuple(ctrl.shape[1:]) != (seq_len, 4, H_lat, W_lat):
                return False
            T = int(vid.shape[0])
            expected = max(1, T - seq_len + 1)
            if int(ctrl.shape[0]) != expected:
                return False
            return True
    except Exception:
        return False


def precompute_demo(
    hdf5_path: Path,
    vae,
    device,
    dtype,
    heatmap_cfg: dict,
    seq_len: int,
    chunk: int,
) -> str:
    """Open hdf5 in rewrite-mode and write video+controlnet latent datasets.

    Returns one-line summary message.
    """
    import torch

    with h5py.File(hdf5_path, "r+") as f:
        if VIDEO_LAT_KEY in f:
            del f[VIDEO_LAT_KEY]
        if CTRL_LAT_KEY in f:
            del f[CTRL_LAT_KEY]

        agent = np.asarray(f["root/agentview/video"])[0]    # (T, 3, S, S)
        wrist = np.asarray(f["root/eye_in_hand/video"])[0]
        T, C, S, S2 = agent.shape
        assert C == 3 and S == S2, f"expected square per-view, got {agent.shape}"

        # ---- video latents ----
        # Video → [-1, 1] then scaling_factor applied (matches tensor_to_vae_latent).
        stacked = _stack_views_uint8(agent, wrist)                                 # (T, 3, 2S, S) uint8
        stacked_float = stacked.astype(np.float32) / 127.5 - 1.0                   # [-1, 1]
        vid_lat = _encode_frames_vae(
            stacked_float, vae, device, dtype, chunk=chunk, apply_scaling=True)    # (T, 4, 2S/8, S/8) fp16
        f.create_dataset(VIDEO_LAT_KEY, data=vid_lat, compression="gzip", compression_opts=4)

        # ---- controlnet latents per window ----
        agent_mesh = np.asarray(f["root/mesh_data/vertex_tracking/agentview_vertices_2d"])
        wrist_mesh = np.asarray(f["root/mesh_data/vertex_tracking/eyeinhand_vertices_2d"])

        perwindow_key = "root/eye_in_hand/grid_tracks_perwindow"
        pw_data = np.asarray(f[perwindow_key]) if perwindow_key in f else None  # (num_windows, 16, 25, 2) or None

        num_windows = max(1, T - seq_len + 1)
        H_lat = (2 * S) // 8
        W_lat = S // 8

        ctrl_out = np.empty((num_windows, seq_len, 4, H_lat, W_lat), dtype=np.float16)

        for w in range(num_windows):
            start = w
            agent_win = _pad_window(agent_mesh[start:start + seq_len], seq_len)  # (seq_len, 7, 2)
            wrist_mesh_win = _pad_window(wrist_mesh[start:start + seq_len], seq_len)

            if pw_data is not None:
                grid_w = pw_data[start] if start < pw_data.shape[0] else pw_data[-1]
                grid_win = _pad_window(grid_w, seq_len)                          # (seq_len, 25, 2)
            else:
                grid_win = np.zeros((seq_len, 25, 2), dtype=np.float32)

            # Heatmap float32 in [0, 1] (seq_len, 3, 2S, S). Feed VAE in [0, 1]
            # *without* rescaling to [-1, 1] — this matches what the network was
            # trained on (legacy path: `condition_img` is [0, 1] at line 1497
            # of train_csvd.py and goes into `vae.encode(...)` verbatim at line
            # 1621; pipeline's `prepare_condition_image` also returns [0, 1]
            # and feeds it raw to `self.vae.encode`). No scaling_factor either.
            hm = _render_window_heatmap(agent_win, wrist_mesh_win, grid_win, heatmap_cfg)
            hm = hm.astype(np.float32)                                              # (seq_len, 3, 2S, S) in [0, 1]
            win_lat = _encode_frames_vae(
                hm, vae, device, dtype, chunk=chunk, apply_scaling=False)
            ctrl_out[w] = win_lat

        f.create_dataset(CTRL_LAT_KEY, data=ctrl_out, compression="gzip", compression_opts=4)

    return f"T={T} num_windows={num_windows} vid_lat={vid_lat.shape} ctrl_lat={ctrl_out.shape}"


def _vae_worker(
    gpu_id: int,
    demos: list,
    svd_path: str,
    seq_len: int,
    heatmap_cfg: dict,
    chunk: int,
) -> None:
    """Single-process worker: pin to one GPU, process assigned demos sequentially.

    Runs as its own `multiprocessing.Process`. Each worker loads its own VAE
    onto its own GPU — no cross-process state, no GIL. Prints carry a `[gpuN]`
    prefix so interleaved output stays traceable.
    """
    import traceback

    # Pin BEFORE importing torch so only this GPU is visible.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"[gpu{gpu_id}]"

    print(f"{tag} loading VAE from {svd_path} → {device} ({dtype})", flush=True)
    vae = _load_vae(svd_path, device, dtype)
    _get_hybrid_vis_fn()  # warm module once

    total = len(demos)
    print(f"{tag} processing {total} demos", flush=True)

    for i, demo in enumerate(demos, 1):
        try:
            msg = precompute_demo(Path(demo), vae, device, dtype, heatmap_cfg, seq_len, chunk)
            print(f"{tag} [{i}/{total}] {demo} → OK  {msg}", flush=True)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"{tag} [{i}/{total}] {demo} → FAIL {exc!r}\n{tb}", flush=True)


def build_vae_encode(
    dst_root: Path,
    svd_path: str,
    gpus: list,
    seq_len: int,
    heatmap_cfg: dict,
    chunk: int,
    size: int,
) -> None:
    """Precompute latents across one or more GPUs via multiprocessing.

    Shards the todo-list round-robin across `gpus`, spawns one process per
    GPU. Processes (not threads): no GIL contention on the CPU-bound heatmap
    renderer, and each GPU holds its own VAE cleanly without CUDA context
    sharing headaches.
    """
    import multiprocessing as mp

    # Walk + skip completed (in parent, so each worker sees a fixed todo list).
    demos = list(walk_demos(dst_root))
    todo = [str(d) for d in demos if not _demo_complete(d, seq_len, size)]
    n_skip = len(demos) - len(todo)
    print(f"[vae-encode] {len(demos)} demos total, {n_skip} already complete, "
          f"{len(todo)} to process on GPUs {gpus}", flush=True)

    if not todo:
        return

    if len(gpus) == 1:
        _vae_worker(gpus[0], todo, svd_path, seq_len, heatmap_cfg, chunk)
        return

    # Round-robin shard → balances heterogeneous demo lengths across GPUs.
    shards = {g: [] for g in gpus}
    for i, demo in enumerate(todo):
        shards[gpus[i % len(gpus)]].append(demo)
    for g, s in shards.items():
        print(f"[vae-encode] gpu{g} → {len(s)} demos", flush=True)

    # `spawn` start method: forking after CUDA import poisons child contexts.
    ctx = mp.get_context("spawn")
    procs = []
    for g in gpus:
        p = ctx.Process(
            target=_vae_worker,
            args=(g, shards[g], svd_path, seq_len, heatmap_cfg, chunk),
            daemon=False,
        )
        p.start()
        procs.append(p)

    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)
    print(f"[vae-encode] workers exit codes: {exit_codes}", flush=True)
    if any(c != 0 for c in exit_codes):
        raise SystemExit(f"At least one worker failed: {exit_codes}")


# ------------------------------ Overlay validate ------------------------------

def render_overlay(demo_path: Path, size: int, out_path: Path, num_frames: int = 4) -> None:
    """Sanity-check overlay on a single demo."""
    with h5py.File(demo_path, "r") as f:
        agent = np.asarray(f["root/agentview/video"])[0]
        wrist = np.asarray(f["root/eye_in_hand/video"])[0]
        agent_mesh = np.asarray(f["root/mesh_data/vertex_tracking/agentview_vertices_2d"])
        wrist_mesh = np.asarray(f["root/mesh_data/vertex_tracking/eyeinhand_vertices_2d"])
        try:
            wrist_grid = np.asarray(f["root/eye_in_hand/grid_tracks_perwindow"][0])
        except Exception:
            wrist_grid = np.zeros((num_frames, 25, 2), dtype=np.float32)

    T = min(agent.shape[0], num_frames)
    panels = []
    for t in range(T):
        a = agent[t].transpose(1, 2, 0)
        w = wrist[t].transpose(1, 2, 0)
        a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA).copy()
        w = cv2.resize(w, (size, size), interpolation=cv2.INTER_AREA).copy()

        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        for x, y in agent_mesh[t]:
            if x > 0 or y > 0:
                cv2.circle(a, (int(x * size), int(y * size)), 3, (255, 0, 0), -1)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

        w = cv2.cvtColor(w, cv2.COLOR_RGB2BGR)
        grid_t = wrist_grid[t] if t < wrist_grid.shape[0] else wrist_grid[-1]
        for x, y in grid_t:
            if x > 0 or y > 0:
                cv2.circle(w, (int(x * size), int(y * size)), 2, (150, 150, 255), -1)
        for x, y in wrist_mesh[t]:
            if x > 0 or y > 0:
                cv2.circle(w, (int(x * size), int(y * size)), 3, (0, 0, 255), -1)
        w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)

        stacked = np.concatenate([a, w], axis=0)
        panels.append(stacked)

    strip = np.concatenate(panels, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"[validate] wrote {out_path}  shape={strip.shape}")


# ------------------------------ CLI ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/task_data_320_square_tracks")
    parser.add_argument("--dst", default=None,
                        help="default data/task_data_{size}_dualview_tracks")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validate", default=None,
                        help="relative hdf5 path under --src; render overlay PNG, skip build")
    parser.add_argument("--validate-out", default=None,
                        help="output PNG path for --validate (default alongside src)")

    parser.add_argument("--vae-encode", action="store_true",
                        help="Precompute video + controlnet VAE latents into dst hdf5 files")
    parser.add_argument("--svd-path",
                        default="checkpoints/svd")
    parser.add_argument("--gpus", type=str, default="0",
                        help="comma-separated GPU ids (e.g. '0,1,2,3'). One "
                             "process spawned per GPU; demos sharded round-robin.")
    parser.add_argument("--video-seq-length", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=16,
                        help="frames per VAE encode chunk")

    # Heatmap rendering config (must match training config!)
    parser.add_argument("--dot-range-agentview", type=int, default=12)
    parser.add_argument("--dot-range-wrist", type=int, default=8)
    parser.add_argument("--no-dilate", action="store_true",
                        help="disable Gaussian blur kernel")
    parser.add_argument("--no-view-colors", action="store_true")

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst) if args.dst \
        else Path(f"data/task_data_{args.size}_dualview_tracks")

    if args.validate:
        if not src_root.is_dir():
            sys.exit(f"src not found: {src_root}")
        demo = src_root / args.validate
        out = Path(args.validate_out) if args.validate_out else demo.with_suffix(".overlay.png")
        render_overlay(demo, args.size, out)
        return

    if args.vae_encode:
        if not dst_root.is_dir():
            sys.exit(f"dst not found: {dst_root} (run resize pass first)")
        heatmap_cfg = {
            "height": args.size,
            "width": args.size,
            "video_seq_length": args.video_seq_length,
            "dot_range_agentview": args.dot_range_agentview,
            "dot_range_wrist": args.dot_range_wrist,
            "dilate": not args.no_dilate,
            "use_view_specific_colors": not args.no_view_colors,
        }
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
        if not gpus:
            sys.exit(f"--gpus parsed to empty list: {args.gpus!r}")
        build_vae_encode(dst_root, args.svd_path, gpus,
                         args.video_seq_length, heatmap_cfg, args.chunk,
                         size=args.size)
        return

    if not src_root.is_dir():
        sys.exit(f"src not found: {src_root}")
    build_resize(src_root, dst_root, args.size, args.num_workers)


if __name__ == "__main__":
    main()
