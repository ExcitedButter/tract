# TrAct

TrAct is a π0.5-based vision-language-action policy that predicts 2D point tracks of the gripper
(agent view + wrist view) jointly with its actions. At test time the policy can be used directly, or
it proposes N candidate action chunks that are imagined by a world model and ranked by the VLAC critic:

```
observation ──► TrAct (π0.5 + track head) ──► N candidates (actions + tracks)
                    │                                   │
                    │        track-conditioned dual-view WM  (or action-conditioned Ctrl-World)
                    │                                   │  16-frame agent + wrist videos
                    │                     VLAC-8b critic: task progress per candidate
                    └──────────── execute the best candidate's first 5 actions, replan
```

The code is built on [openpi](https://github.com/Physical-Intelligence/openpi) (Python package `tract`).
This repository contains code only — no datasets, checkpoints or results.

**Contents**
1. [Repository layout](#repository-layout)
2. [Installation](#1-installation)
3. [Checkpoints and data](#2-checkpoints-and-data)
4. [Training TrAct](#3-training-tract)
5. [Evaluation on LIBERO](#4-evaluation-on-libero)
6. [World models](#5-world-models)
7. [VLAC critic](#6-vlac-critic)
8. [Generating the track-labelled training set](#7-generating-the-track-labelled-training-set)
9. [Reproducibility and troubleshooting](#8-reproducibility-and-troubleshooting)
10. [Physical experiments](#9-physical-experiments)

## Repository layout

```
scripts/
  train.py                      TrAct (π0.5) training                      -> section 3
  compute_norm_stats.py         normalisation stats for a train config
  evaluate.py                   LIBERO eval, policy only                    -> section 4
  evaluate_full_track.py        LIBERO eval, TrAct + track WM + VLAC        -> section 4
  evaluate_full_action.py       LIBERO eval, TrAct + Ctrl-World + VLAC      -> section 4
  eval_world_models.py          video metrics: Ctrl-World vs track WM       -> section 5
  lerobot_preprocess_cotracker.py   track labels (simulator + CoTracker2)   -> section 7
  serve_policy.py               websocket policy server (remote inference)
  vlac_pipeline/                candidate sampling, WM wrappers, VLAC critic (used by the eval scripts)
src/tract/                      models, data pipeline, training configs (fork of openpi's `openpi`)
packages/openpi-client/         websocket client / image utils (from openpi)
third_party/
  libero/                       LIBERO + the extra evaluation suites (bddl + init files; assets downloaded)
  track_dualview/               track-conditioned dual-view SVD world model (inference + training)
  Ctrl-World/                   action-conditioned world model (inference + LIBERO / real-world training)
  VLAC/                         VLAC critic (evo_vlac) + LIBERO fine-tuning scripts
  co-tracker/                   CoTracker (used for track labels)
requirements-full.txt           exact package set the full pipeline was verified with
```

## 1. Installation

Requirements: Linux, NVIDIA GPU(s) with ≥80 GB for training (H100/A100-80G), Python 3.11,
[uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> tract && cd tract
GIT_LFS_SKIP_SMUDGE=1 uv sync                               # core: TrAct training + policy-only eval
GIT_LFS_SKIP_SMUDGE=1 uv pip sync requirements-full.txt     # full: + world models, VLAC (ms-swift), metrics
source .venv/bin/activate
```

- Run everything with `python ...` from the activated venv (or `uv run --no-sync ...`). A plain `uv run`
  re-syncs to `uv.lock` and removes the extra packages of `requirements-full.txt`.
- All commands below are run from the repository root unless stated otherwise.
- LIBERO lives in `third_party/libero` and is not pip-installable (namespace package). The eval scripts
  add it to `sys.path` and keep its config in `.libero_local/`; for your own scripts use
  `export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH`.
- Headless simulation needs an X server: prefix LIBERO commands with `xvfb-run -a` (default
  `MUJOCO_GL=glx`). If EGL works on your machine you can set `OPENPI_MUJOCO_GL=egl` instead.
- Caches default to folders in the repo root (all git-ignored) unless the variables are already set in
  your shell: `OPENPI_DATA_HOME` → `./openpi-assets` + `./big_vision`, `HF_LEROBOT_HOME` → `./lerobot_cache`,
  `HF_HOME` → `./hf_cache`.

## 2. Checkpoints and data

### Checkpoints

| Component | Needed for | Download | Location |
|---|---|---|---|
| π0.5-LIBERO initialisation + norm stats | training | automatic (`gs://openpi-assets`) | `openpi-assets/checkpoints/pi05_libero/` |
| TrAct policy | evaluation | train it (section 3) | `checkpoints/pi05_libero_stage2p5_joint/<exp>/<step>` |
| Stable Video Diffusion img2vid (diffusers folders, ~14 GB) | both world models | `huggingface-cli download stabilityai/stable-video-diffusion-img2vid --include "model_index.json" "feature_extractor/*" "image_encoder/*" "scheduler/*" "unet/*" "vae/*" --local-dir checkpoints/svd` | `checkpoints/svd` |
| Track dual-view WM (LIBERO) | `evaluate_full_track.py` | trained checkpoint (section 5) | `checkpoints/track_dualview/hybrid-checkpoint-128000` |
| Ctrl-World DROID base | Ctrl-World fine-tuning | `huggingface-cli download yjguo/Ctrl-World checkpoint-10000.pt --local-dir checkpoints/ctrl_world/droid` | `checkpoints/ctrl_world/droid/checkpoint-10000.pt` |
| Ctrl-World LIBERO | `evaluate_full_action.py` | trained checkpoint (section 5) | `checkpoints/ctrl_world/libero_ctrlworld/checkpoint-20000.pt` |
| Ctrl-World LIBERO action adapter | `evaluate_full_action.py` (optional) | not released; without it an approximate analytical delta→state conversion is used and a warning is logged | `checkpoints/ctrl_world/libero_action_adapter.pt` |
| VLAC-8b | both selection evals, VLAC | `huggingface-cli download InternRobotics/VLAC-8b --local-dir checkpoints/VLAC-8b` | `checkpoints/VLAC-8b` |
| SD-2.1 tokenizer + text encoder | track-WM **training** only | `huggingface-cli download sd2-community/stable-diffusion-2-1-base --include "tokenizer/*" "text_encoder/*" model_index.json --local-dir checkpoints/stable-diffusion-2-1-base` | `checkpoints/stable-diffusion-2-1-base` |
| CLIP ViT-B/32 | Ctrl-World | automatic (`openai/clip-vit-base-patch32`) | HF cache |
| CoTracker2 | track labels | automatic (torch.hub `cotracker2` weights) | torch hub cache |

### Data

| Data | Needed for | How to get it | Location |
|---|---|---|---|
| LIBERO scene assets (LIBERO-plus) | all LIBERO evaluation | see below | `third_party/libero/libero/libero/assets/` |
| `physical-intelligence/libero` (LeRobot) | track-label generation | downloaded automatically by LeRobot | `lerobot_cache/physical-intelligence/libero` |
| `libero_cotracker_densegrid_merged` | TrAct training | generated, section 7 | `lerobot_cache/physical-intelligence/libero_cotracker_densegrid_merged` |
| `openvla/modified_libero_rlds` | Ctrl-World LIBERO training | `huggingface-cli download openvla/modified_libero_rlds --repo-type dataset --local-dir third_party/Ctrl-World/raw_data/modified_libero_rlds` | as given |
| ATM LIBERO demo images | VLAC fine-tuning | [ATM](https://github.com/Large-Trajectory-Model/ATM) preprocessed LIBERO (`<suite>/<task>_demo/images/demo_*/agentview_*.png`) | `data/atm/atm_libero/` |

LIBERO scene assets (the LIBERO-plus archive contains everything the suites below need):

```bash
huggingface-cli download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir /tmp/libero_plus
unzip -q /tmp/libero_plus/assets.zip -d /tmp/libero_plus
mv "$(find /tmp/libero_plus -type d -path '*LIBERO-plus-0/assets' | head -1)" third_party/libero/libero/libero/assets
```

## 3. Training TrAct

TrAct is the train config `pi05_libero_stage2p5_joint` in `src/tract/training/config.py`:
π0.5 with a track head (`track_head_dim=78`: 7 agent-view gripper points + 25 wrist grid points + 7 wrist
gripper points, 16-step horizon), action and track losses weighted 1:1, initialised from π0.5-LIBERO,
trained on `libero_cotracker_densegrid_merged` with the π0.5-LIBERO normalisation stats.

```bash
# reported model: 8 x H100, global batch 16, 30k steps (~2 samples per GPU)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi05_libero_stage2p5_joint \
  --exp-name=my_tract --overwrite
```

Useful options (all fields of `TrainConfig` are command-line flags):

| Flag | Default | Meaning |
|---|---|---|
| `--exp-name` | — | run name; checkpoints go to `checkpoints/pi05_libero_stage2p5_joint/<exp-name>/<step>/` |
| `--overwrite` / `--resume` | off | start fresh in an existing run folder / continue from its last checkpoint |
| `--batch-size` | 16 | global batch, must be divisible by the number of GPUs |
| `--num-train-steps` | 30000 | |
| `--save-interval`, `--keep-period` | 1000, 5000 | checkpoint frequency / which checkpoints are kept |
| `--fsdp-devices` | 1 | >1 shards parameters across GPUs (data-parallel otherwise) |
| `--weight-loader.params-path` | `gs://openpi-assets/checkpoints/pi05_libero/params` | initialisation |
| `--no-wandb-enabled` | wandb on | disable Weights & Biases logging |

- The first run converts the dataset into a HuggingFace `datasets` cache (~35 GB in `$HF_DATASETS_CACHE`).
- `pi05_libero_stage1_align` (alignment only) and `pi05_libero_stage2_track` (track head only, frozen
  backbone) are optional staged variants; the reported model does not use them.
- `scripts/compute_norm_stats.py --config-name <config>` computes normalisation stats for other configs
  (not needed for TrAct, which reuses the π0.5-LIBERO stats).

## 4. Evaluation on LIBERO

Three entry points share the same environment loop, suites and outputs:

```bash
# (a) policy only
xvfb-run -a python scripts/evaluate.py --suites libero_goal --policy-gpu 0 --render-gpu 0

# (b) TrAct + track dual-view world model + VLAC candidate selection
xvfb-run -a python scripts/evaluate_full_track.py --suites libero_integral --all-tasks \
  --n-candidates 8 --policy-gpu 0 --render-gpu 0 --wm-gpu 1 --vlac-gpu 1

# (c) TrAct + Ctrl-World (action-conditioned) + VLAC candidate selection
xvfb-run -a python scripts/evaluate_full_action.py --suites libero_integral --all-tasks \
  --n-candidates 8 --policy-gpu 0 --render-gpu 0 --wm-gpu 1 --vlac-gpu 1
```

Common options:

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint-dir` | `checkpoints/pi05_libero_stage2p5_joint/pi05_final/29999` | TrAct checkpoint |
| `--suites` | `libero_object_sawyer` | one or more suites (list below) |
| `--all-tasks` / `--no-all-tasks` | (a) on, (b)(c) off | evaluate every task of the suite |
| `--task-index N` | 0 | single task when not `--all-tasks`; (a)(b) also take `--task-indices i j ...` |
| `--rollouts-per-task` | 10 | episodes per task (init states 0, 1, ...) |
| `--seed` | 7 | |
| `--policy-gpu`, `--render-gpu` | 0, 0 | physical GPU ids (rendering uses the X display under GLX) |
| `--wm-gpu`, `--vlac-gpu` | = policy GPU | (b)(c): where the world model / VLAC run |
| `--out-dir`, `--run-name` | `results/`, auto | output location |

Selection options of (b)/(c): `--n-candidates` (20), `--no-use-action-selection` (fall back to (a)),
`--wm-num-inference-steps` (25), `--wm-batch-size` (8, (b) only), candidate diversity
`--sampling-mode tsr --tsr-k 5 --noise-scale 3 --noise-type sobol`, temporal smoothing `--smooth-sigma 2`,
VLAC view weights `--vlac-agent-weight 0.6 --vlac-wrist-weight 0.4`.
One 80 GB GPU holds policy + world model + VLAC; with 8 candidates a replan takes ~15–30 s, so a
520-step `libero_integral` episode takes 25–60 min (separate GPUs for WM and VLAC are faster).

**Outputs** (`results/` by default):

```
results/<run>_<suite>.txt            per-task success-rate summary
results/<run>/<suite>/*.mp4          rollout videos with the predicted tracks drawn (warm = now, cool = future)
results/<run>/<suite>/original_rollout/*.mp4   raw camera videos
results/<run>/<suite>/hdf5/*.hdf5    actions, pred_actions, pred_env_steps, tracks
                                     (+ best_candidate_idx, candidate_scores for (b)/(c))
```

**Suites** (all registered in `third_party/libero`):

| Suites | What changes |
|---|---|
| `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` | original LIBERO |
| `<base>_lan`, `<base>_object`, `<base>_swap`, `<base>_task` | LIBERO-PRO perturbations: instruction paraphrase, object appearance, object positions, task (`<base>` ∈ spatial/object/goal/10) |
| `libero_background`, `libero_robot`, `libero_viewpoint` | LIBERO-plus perturbations: background texture, robot initial state, camera viewpoint (hardest variant per base task, 40 tasks each) |
| `<base>_cross`, `<base>_sawyer` | cross-embodiment: UR5e / Sawyer arm, init states mapped from the Franka demos by IK |
| `libero_10_arm`, `libero_object_arm` | Franka with a black-recoloured arm |
| `libero_integral` | 20-task mix of the above (table below) |

<details><summary><code>libero_integral</code> tasks</summary>

| id | source suite / task | instruction |
|---|---|---|
| 0 | `libero_object_swap` #0 | pick the alphabet soup and place it in the basket |
| 1 | `libero_10_swap` #1 | put both the cream cheese box and the butter in the basket |
| 2 | `libero_10_object` #8 | put both moka pots on the stove |
| 3 | `libero_10_object` #9 | put the yellow and white mug in the microwave and close it |
| 4 | `libero_spatial_task` #8 | pick the akita black bowl next to the ramekin and place it on the plate |
| 5 | `libero_goal_task` #1 | put the plate on the stove |
| 6 | `libero_viewpoint` #14 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 7 | `libero_viewpoint` #7 | put the wine bottle on the top of the drawer |
| 8 | `libero_robot` #11 | put both the cream cheese box and the butter in the basket |
| 9 | `libero_robot` #15 | pick up the book and place it in the back compartment of the caddy |
| 10 | `libero_10_cross` #7 | put both the alphabet soup and the cream cheese box in the basket |
| 11 | `libero_10_cross` #0 | put both the alphabet soup and the tomato sauce in the basket |
| 12 | `libero_goal_cross` #1 | put the bowl on the stove |
| 13 | `libero_object_cross` #9 | pick the orange juice and place it in the basket |
| 14 | `libero_spatial_cross` #0 | pick the akita black bowl between the plate and the ramekin and place it on the plate |
| 15 | `libero_spatial_cross` #8 | pick the akita black bowl next to the plate and place it on the plate |
| 16 | `libero_object_cross` #4 | pick the ketchup and place it in the basket |
| 17 | `libero_object_cross` #5 | pick the tomato sauce and place it in the basket |
| 18 | `libero_spatial_cross` #3 | pick the akita black bowl on the cookies box and place it on the plate |
| 19 | `libero_goal_cross` #8 | put the bowl on the plate |

</details>

Remote inference (e.g. on a real robot): `python scripts/serve_policy.py policy:checkpoint
--policy.config=pi05_libero_stage2p5_joint --policy.dir=<checkpoint dir>` serves the policy over
websocket (client: `packages/openpi-client`).

## 5. World models

### Track-conditioned dual-view world model (`third_party/track_dualview`)

SVD + ControlNet conditioned on rendered track heat-maps; agent and wrist views are stacked vertically
(192×192 each). Inference runs without text conditioning, as in the reported results.

```bash
# inference on one LeRobot episode: GT | prediction videos for agent / wrist view
python third_party/track_dualview/scripts/infer_single_frame.py \
  --parquet_path lerobot_cache/physical-intelligence/libero_cotracker_densegrid_merged/data/chunk-000/episode_000401.parquet \
  --output_dir outputs/track_wm_demo

# training (accelerate, UNet + ControlNet); relative paths in the config resolve against the repo root
CUDA_VISIBLE_DEVICES=0,1,2,3 bash third_party/track_dualview/scripts/train.sh \
  third_party/track_dualview/config/train_track_realworld_dualview_192.yaml
```

Training configs (`third_party/track_dualview/config/`):

| Config | Data loader | Resolution | Initialisation |
|---|---|---|---|
| `train_track_realworld_dualview_192.yaml` | `realworld_dualview` | 2 × 192² (stacked 384×192) | `hybrid-checkpoint-128000` |
| `train_track_realworld_dualview_224.yaml` | `realworld_dualview` | 2 × 224² | `hybrid-checkpoint-128000` |
| `train_track_realworld_hybrid_320.yaml` | `realworld_hybrid` | 320², one view per sample | `checkpoint-180000` |

Data layout: one HDF5 per demo under `<dataset_path>/task_*/{train,val}/` with
`root/{agentview,eye_in_hand}/video` (1, T, 3, H, W) uint8, `root/mesh_data/vertex_tracking/{agentview,eyeinhand}_vertices_2d`
(T, 7, 2), `root/eye_in_hand/grid_tracks_perwindow` (W, 16, 25, 2), coordinates in [0, 1].
`preprocess/build_dualview_dataset.py` turns 320² data into the dual-view variant (resize + VAE latents).
Notes: needs the SD-2.1 tokenizer/text encoder (section 2); `train_csvd.py` deletes `output_dir` at start-up
(keep it on a dedicated folder, the configs use `outputs/track_dualview/`); checkpoints are saved every
`checkpointing_steps` as `checkpoint-<step>/{unet,controlnet}`, loadable with `TrackDualViewWM(checkpoint_path=...)`.
The LIBERO data loader used to train `hybrid-checkpoint-128000` is not part of this repository.

### Ctrl-World (`third_party/Ctrl-World`)

Action-conditioned SVD world model; LIBERO views are 320×192 (centre 5:3 band of the 256² image).

```bash
cd third_party/Ctrl-World
# inference on one LeRobot episode (policy deltas -> states via the action adapter / analytical converter)
python scripts/inference_world_model.py

# LIBERO fine-tuning
#  1) RLDS -> SVD-VAE latents in dataset_example/libero (needs TensorFlow: separate env with
#     tensorflow-cpu==2.15.1 + torch + diffusers + opencv)
python scripts/preprocess_libero.py --svd_path ../../checkpoints/svd
#  2) normalisation stats + sample lists. NOTE: dataset_meta_info/libero/stat.json is also the
#     normalisation used at inference; use --output_dir to experiment on a subset
python scripts/create_libero_meta_info.py
#  3) train from the DROID base model (writes model_ckpt/libero_ctrlworld/checkpoint-<step>.pt)
WANDB_MODE=offline accelerate launch scripts/train_wm_libero.py --svd_model_path ../../checkpoints/svd \
  --clip_model_path openai/clip-vit-base-patch32 --ckpt_path ../../checkpoints/ctrl_world/droid/checkpoint-10000.pt
```

Hyper-parameters (batch size, steps, learning rate, validation interval) are in `config_libero.py`.
The real-world variant uses `scripts/preprocess_realworld.py` (your Franka data: `data/task_data/` videos +
`data/task_data_320_square_tracks/` actions), `scripts/create_realworld_meta_info.py` and
`scripts/train_wm_realworld.py` with `config_realworld.py`.

### Comparing the two world models

```bash
python scripts/eval_world_models.py --num-horizon-windows 5     # PSNR / SSIM / LPIPS / FID / FVD per view
```

By default it predicts 16-frame windows of `libero_goal_task` task 1 from the LeRobot parquet
(`--suite`, `--task-index`, `--parquet_path` to change) and writes videos + `video/results*.txt`.

## 6. VLAC critic

The selection pipeline scores each imagined clip with the released
[`InternRobotics/VLAC-8b`](https://huggingface.co/InternRobotics/VLAC-8b): agent and wrist views are scored
separately (task progress 0–100) and combined 0.6 / 0.4.

```bash
# standalone example: pair-wise critic on an image sequence (+ optional reference trajectory)
python third_party/VLAC/evo_vlac/examples/image_pair.py
```

```python
from scripts.vlac_pipeline.vlac_dualview_critic import VLACDualViewCritic
critic = VLACDualViewCritic(model_path="checkpoints/VLAC-8b"); critic.load(device="cuda:0")
score = critic.score_dualview(agent_frames, wrist_frames, "put the bowl on the plate")  # lists of PIL images
```

Fine-tuning on LIBERO demos (ms-swift 3.3):

```bash
python third_party/VLAC/prepare_dataset.py --data_root data/atm/atm_libero/libero_goal   # -> third_party/VLAC/data/
python third_party/VLAC/train.py --use_lora --yes      # full fine-tuning without --use_lora; add --attn_impl eager without flash-attn
```

`prepare_dataset.py` builds image pairs with VLAC's own pair-wise prompt. With the default
`--label_mode progress` each pair is labelled with the signed progress difference in percent,
`round(100·(j−i)/(T−1))`, and half of the pairs are shown reversed (negative labels);
`--label_mode heuristic` keeps the earlier position-based random scores. Outputs go to
`third_party/VLAC/models/1/v*/checkpoint-*` (LoRA adapters when `--use_lora`).

## 7. Generating the track-labelled training set

`scripts/lerobot_preprocess_cotracker.py` writes a LeRobot dataset with `track_targets_raw`
(16 × 39 × 2: 7 agent-view gripper points | 25 wrist grid points | 7 wrist gripper points). The wrist grid is
tracked with CoTracker2 (pretrained torch.hub weights, run from `third_party/co-tracker`); the gripper
points come from the simulator (`--mesh-source sim`) or from a mesh dataset (`--mesh-source mesh_repo`).

```bash
MUJOCO_GL=glx xvfb-run -a python scripts/lerobot_preprocess_cotracker.py \
  --source-repo-id physical-intelligence/libero \
  --target-repo-id physical-intelligence/libero_cotracker_densegrid \
  --mesh-source mesh_repo --mesh-repo-id <mesh dataset> \
  --start-episode 0 --end-episode 211          # shard the 1693 episodes over GPUs / processes
```

A relative `--target-repo-id` is written under `$HF_LEROBOT_HOME` (where training reads it).
For the training set behind the reported model (`libero_cotracker_densegrid_merged`):
- the CoTracker wrist-grid tracks are reproduced exactly by this script;
- its gripper points were taken (`--mesh-source mesh_repo`) from an intermediate dataset,
  `libero_mesh_dynamic_plus_wrist25`, built from LIBERO demos with full simulator states; that dataset and
  its builder are not in this repository. `--mesh-source sim` cannot replace it for
  `physical-intelligence/libero`, which stores only 8-D end-effector states: the replay starts from the
  default scene (a `[warn]` is printed) and the gripper tracks do not follow the demonstrations;
- the dataset was produced as 8 shards merged into one LeRobot dataset; the merge step is not included.

## 8. Reproducibility and troubleshooting

- **Bit-exact reruns.** GPU numerics differ slightly between runs (~1e-3 in actions). For identical
  outputs set `XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0" CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Selection fell back?** If a world-model or VLAC call fails, the selector logs `Track WM failed` /
  `VLAC failed` and uses candidate 0; check the log and `best_candidate_idx` / `candidate_scores` in the HDF5s.
  All-zero VLAC scores in the first replans of a task can be legitimate (no visible progress yet).
- **Rendering crashes / black frames:** run under `xvfb-run -a` with `MUJOCO_GL=glx`.
- **Disk:** first training run ~35 GB `datasets` cache; LIBERO assets ~6 GB; SVD ~14 GB; VLAC-8b ~16 GB;
  π0.5-LIBERO params ~12 GB; each TrAct checkpoint ~12 GB.
- **Ctrl-World downloads SVD again:** pass `--svd_path checkpoints/svd` (or put it there); the resolver
  checks `checkpoints/svd` and `third_party/Ctrl-World/checkpoints/svd` before downloading.

## 9. Physical experiments

The real-robot results in the paper used checkpoints fine-tuned from the pretrained models in this
repository, with adjusted sampling parameters.

## License

Code from openpi is Apache-2.0 (`LICENSE`; `LICENSE_GEMMA.txt` for the Gemma weights). Each project under
`third_party/` keeps its own license file.
