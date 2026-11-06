"""
Train Ctrl-World on real-world Franka data.

Usage:
    cd third_party/Ctrl-World

    # Single GPU test:
    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
    accelerate launch --main_process_port 29501 scripts/train_wm_realworld.py \
        --svd_model_path checkpoints/svd \
        --clip_model_path checkpoints/clip-vit-base-patch32 \
        --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt

    # Multi-GPU:
    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    accelerate launch --main_process_port 29501 scripts/train_wm_realworld.py \
        --svd_model_path checkpoints/svd \
        --clip_model_path checkpoints/clip-vit-base-patch32 \
        --ckpt_path checkpoints/ctrl-world/checkpoint-10000.pt
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from models.ctrl_world import CrtlWorld

import numpy as np
import torch
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
import wandb
import swanlab
import mediapy
from config_realworld import wm_args
import math


def main(args):
    logger = get_logger(__name__, log_level="INFO")
    swanlab.sync_wandb()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=args.output_dir,
    )

    # Model and optimizer
    model = CrtlWorld(args)
    if args.ckpt_path is not None:
        print(f"Loading checkpoint from {args.ckpt_path}!")
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model.to(accelerator.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Logs
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        tag = args.tag
        run_name = f"train_{now}_{tag}"
        accelerator.init_trackers(
            args.wandb_project_name, config={}, init_kwargs={"wandb": {"name": run_name}}
        )
        os.makedirs(args.output_dir, exist_ok=True)
        num_params = sum(p.numel() for p in model.unet.parameters())
        print(f"UNet parameters: {num_params / 1e6:.2f}M")
        num_params = sum(p.numel() for p in model.action_encoder.parameters())
        print(f"Action encoder parameters: {num_params / 1e6:.2f}M")

    # Datasets — use realworld dataset loader
    from dataset.dataset_realworld import Dataset_mix

    train_dataset = Dataset_mix(args, mode="train")
    val_dataset = Dataset_mix(args, mode="val")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle
    )

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    ############################ Training ##############################
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        args.max_train_steps * args.gradient_accumulation_steps * total_batch_size / len(train_dataloader)
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  num_frames = {args.num_frames}, num_history = {args.num_history}")
    logger.info(f"  validation_steps = {args.validation_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")

    global_step = 0
    forward_step = 0
    train_loss = 0.0
    progress_bar = tqdm(
        range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_gen, _ = model(batch)
                avg_loss = accelerator.gather(loss_gen.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss_gen)
                params_to_clip = model.parameters()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                forward_step += 1

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log loss every 50 steps
                if global_step % 50 == 0:
                    progress_bar.set_postfix({"loss": train_loss / 50})
                    accelerator.log({"train_loss": train_loss / 50}, step=global_step)
                    train_loss = 0.0

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

                # Validation
                if global_step % args.validation_steps == 5 and accelerator.is_main_process:
                    model.eval()
                    with accelerator.autocast():
                        for vid_id in range(min(args.video_num, 2)):
                            validate_video_generation(
                                model, val_dataset, args, global_step, args.output_dir, vid_id, accelerator
                            )
                    model.train()

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break


def validate_video_generation(model, val_dataset, args, train_steps, videos_dir, id, accelerator):
    device = accelerator.device
    pipeline = model.module.pipeline if accelerator.num_processes > 1 else model.pipeline
    videos_row = args.video_num if not args.debug else 1
    videos_col = 2

    # Sample from val dataset
    batch_id = list(range(0, len(val_dataset), max(1, int(len(val_dataset) / videos_row / videos_col))))
    batch_id = batch_id[int(id * videos_col) : int((id + 1) * videos_col)]
    if not batch_id:
        return
    batch_list = [val_dataset.__getitem__(bid) for bid in batch_id]
    video_gt = torch.cat([t["latent"].unsqueeze(0) for t in batch_list], dim=0).to(device)
    text = [t["text"] for t in batch_list]
    actions = torch.cat([t["action"].unsqueeze(0) for t in batch_list], dim=0).to(device)
    his_latent_gt = video_gt[:, : args.num_history]
    future_latent_gt = video_gt[:, args.num_history :]
    current_latent = future_latent_gt[:, 0]

    print(f"Val: image {current_latent.shape}, action {actions.shape}")

    # Generate
    with torch.no_grad():
        bsz = actions.shape[0]
        action_latent = (
            model.module.action_encoder(
                actions, text, model.module.tokenizer, model.module.text_encoder, args.frame_level_cond
            )
            if accelerator.num_processes > 1
            else model.action_encoder(
                actions, text, model.tokenizer, model.text_encoder, args.frame_level_cond
            )
        )

        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(3 * args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
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

    # Split views and decode only real views (0 and 1), skip view 2 (zeros)
    pred_latents_split = einops.rearrange(pred_latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)
    video_gt_full = torch.cat([his_latent_gt, future_latent_gt], dim=1)
    video_gt_split = einops.rearrange(video_gt_full, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)

    # Only decode views 0 (agentview) and 2 (wrist), skip view 1 (zeros)
    # pred_latents_split shape: (B*3, F, 4, 24, 40)
    bsz_orig = actions.shape[0]
    view_indices = []
    for b in range(bsz_orig):
        view_indices.extend([b * 3, b * 3 + 2])  # skip b*3+1 (zeros)
    pred_2views = pred_latents_split[view_indices]
    gt_2views = video_gt_split[view_indices]

    # Decode GT
    decoded_gt = decode_latents(gt_2views, pipeline, args.decode_chunk_size)
    # Decode predicted
    decoded_pred = decode_latents(pred_2views, pipeline, args.decode_chunk_size)

    # Concat: predicted gets history prepended
    decoded_pred_with_his = np.concatenate(
        [decoded_gt[:, : args.num_history], decoded_pred], axis=1
    )

    # Stack GT on top, prediction on bottom
    videos = np.concatenate([decoded_gt, decoded_pred_with_his], axis=-3)
    # Concat all samples side-by-side
    videos = np.concatenate([v for v in videos], axis=-2).astype(np.uint8)

    os.makedirs(f"{videos_dir}/samples", exist_ok=True)
    filename = f"{videos_dir}/samples/train_steps_{train_steps}_{id}.mp4"
    mediapy.write_video(filename, videos, fps=4)


def decode_latents(latents, pipeline, decode_chunk_size):
    """Decode (B, F, 4, H, W) latents to (B, F, H*8, W*8, 3) uint8 numpy."""
    bsz, frame_num = latents.shape[:2]
    flat = latents.flatten(0, 1)
    decoded = []
    for i in range(0, flat.shape[0], decode_chunk_size):
        chunk = flat[i : i + decode_chunk_size] / pipeline.vae.config.scaling_factor
        decode_kwargs = {"num_frames": chunk.shape[0]}
        decoded.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
    video = torch.cat(decoded, dim=0)
    video = video.reshape(bsz, frame_num, *video.shape[1:])
    video = ((video / 2.0 + 0.5).clamp(0, 1) * 255)
    video = video.to(torch.float32).detach().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
    return video


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    args_new = parser.parse_args()
    args = wm_args()

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args

    args = merge_args(args, args_new)
    main(args)
