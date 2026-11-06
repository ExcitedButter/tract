#!/usr/bin/env python
'''
    This file is to train Stable Video Diffusion with Conditioning design by my peronal implementation which is based on diffusers' training example code.
'''

import argparse
import logging
import math
import os, sys
import time
import random
import shutil
import warnings
from PIL import Image 
from einops import rearrange, repeat
from pathlib import Path
import yaml
import imageio
import cv2


import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig


import diffusers
from diffusers import (
    AutoencoderKLTemporalDecoder,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, load_image, export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
if is_wandb_available():
    import wandb


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from svd.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionControlNetPipeline
from svd.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from svd.temporal_controlnet import ControlNetModel
from utils.img_utils import resize_with_antialiasing
from utils.optical_flow_utils import flow_to_image, filter_uv, bivariate_Gaussian
from data_loader.video_dataset import tokenize_captions
from data_loader.video_this_that_dataset import Video_ThisThat_Dataset, get_thisthat_sam
from train_code.train_svd import import_pretrained_text_encoder

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.25.0.dev0")

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


###################################################################################################################################################
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/train_image2video_gesturenet.yaml",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    args = parser.parse_args()
    return args


    
def log_validation(vae, unet, controlnet, image_encoder, text_encoder, tokenizer, config, accelerator, weight_dtype, step, 
                        val_dataset=None, parent_store_folder=None, force_close_flip=False, use_ambiguous_prompt=False,
                        config_name=None, timestamp=None):
    # This function will also be used in other files
    print("Running validation... ")


    # Init
    validation_source_folder = config["validation_img_folder"]
    # Realworld hybrid/dualview loaders pull validation from the main dataset's
    # `val` split, not this folder — skip the existence gate for them.
    _uses_val_folder = config["data_loader_type"] not in ("realworld_hybrid", "realworld_dualview")
    if _uses_val_folder and not os.path.exists(validation_source_folder):
        # If you don't have the validation dataset, we skip.
        return
    

    # Init the pipeline
    # IMPORTANT: Unwrap UNet if it's wrapped in DDP for validation
    pipeline = StableVideoDiffusionControlNetPipeline.from_pretrained(
        config["pretrained_model_name_or_path"],        # Still based on regular SVD config
        vae = vae,
        image_encoder = image_encoder,
        unet = accelerator.unwrap_model(unet),  # Unwrap UNet from DDP if wrapped
        revision = None,    # Set None directly now
        torch_dtype = weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)


    # Process validation based on data loader type
    frames_collection = []
    
    if config["data_loader_type"] == "libero" or config["data_loader_type"] == "libero_32point":
        # For LIBERO validation, use the pre-loaded dataset if available
        if val_dataset is None:
            # Fallback: create dataset if not provided
            if config["data_loader_type"] == "libero_32point":
                from data_loader.video_libero_32point_dataset import Video_LIBERO_32Point_Dataset
                val_config = config.copy()
                val_config["dataset_path"] = config.get("val_dataset_path", config["dataset_path"])
                val_config["split"] = config.get("val_split", "val")
                val_dataset = Video_LIBERO_32Point_Dataset(val_config, accelerator.device, tokenizer=tokenizer)
            else:
                from data_loader.video_libero_dataset import Video_LIBERO_Dataset, generate_gripper_trajectory_vis
                val_config = config.copy()
                val_config["dataset_path"] = config.get("val_dataset_path", config["dataset_path"])
                val_config["split"] = config.get("val_split", "val")
                val_dataset = Video_LIBERO_Dataset(val_config, accelerator.device, tokenizer=tokenizer)
        
        # Distribute validation samples across GPUs
        num_validation_samples = min(4, len(val_dataset))  # Use 4 to get 2 of each view
        
        # Each of the first 4 GPUs handles exactly one validation sample
        # GPUs 4-7 will wait
        if accelerator.process_index < num_validation_samples:
            val_sample_idx = accelerator.process_index
            
            # Define validation_store_folder for all processes
            if parent_store_folder is None:
                # Use timestamped validation folder if available
                base_val_folder = config["validation_store_folder"]
                if config_name and timestamp:
                    val_folder_name = f"{base_val_folder}_{config_name}_{timestamp}_{config['scheduler']}"
                else:
                    val_folder_name = f"{base_val_folder}_{config['scheduler']}"
                validation_store_folder = os.path.join(val_folder_name, "step_" + str(step), f"sample_{val_sample_idx}")
            else:
                validation_store_folder = os.path.join(parent_store_folder, f"sample_{val_sample_idx}")
            
            # Each process creates its own validation directory
            if os.path.exists(validation_store_folder):
                shutil.rmtree(validation_store_folder)
            os.makedirs(validation_store_folder, exist_ok=True)
            
            # Use random sampling for more diverse validation
            # Ensure we sample from the correct view type (alternating)
            import random
            random.seed(step + val_sample_idx)  # Use step as seed for reproducibility but different each validation

            # Get the original dataset length (before doubling for both control modes)
            original_length = len(val_dataset.index_mapping)

            # Get indices for this view type (even for agentview, odd for eye-in-hand)
            view_type = val_sample_idx % 2  # 0 for agentview, 1 for eye-in-hand

            # Get indices for BOTH control modes:
            # First half (mesh control)
            mesh_indices = [i for i in range(original_length) if i % 2 == view_type]
            # Second half (uniform control)
            uniform_indices = [i + original_length for i in range(original_length) if i % 2 == view_type]

            # Combine both control modes
            all_indices = mesh_indices + uniform_indices

            # Randomly select from all available indices (both mesh and uniform)
            if all_indices:
                val_idx = random.choice(all_indices)
                # Determine which control mode was selected
                control_type = "uniform" if val_idx >= original_length else "mesh"
                print(f"Validation sample {val_sample_idx}: Using {control_type} control")
            else:
                val_idx = val_sample_idx  # Fallback to original if no indices available
                control_type = "unknown"

            val_data = val_dataset[val_idx]

            # Get the actual task description for this validation sample
            # Handle doubled dataset: index_mapping only has original_length entries
            # (original_length was already defined above)
            actual_idx = val_idx if val_idx < original_length else val_idx - original_length
            dataset_idx, demo_idx = val_dataset.index_mapping[actual_idx]
            _, val_task_desc, _ = val_dataset.datasets[dataset_idx]  # Now includes track_len
            
            # Note which view is being used (alternates based on val_idx)
            view_name = "eye-in-hand" if val_idx % 2 == 1 else "agentview"
            print(f"Validation sample {val_sample_idx}: {view_name} view, task: {val_task_desc}")
            
            # Extract the first frame as reference image
            ref_image = val_data["video_frames"][0]  # First frame of video, should be in [-1, 1] range
            if ref_image.dim() == 3:  # CHW format
                ref_image = ref_image.permute(1, 2, 0)  # Convert to HWC
            # Convert from [-1, 1] to [0, 255] for PIL Image
            ref_image_np = ((ref_image + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
            ref_image = Image.fromarray(ref_image_np, mode='RGB')
            
            # Get other necessary data
            condition_img = val_data["controlnet_condition"].numpy()
            # Get alternative control visualization if available
            condition_img_alt = val_data.get("controlnet_condition_alternative", None)
            if condition_img_alt is not None:
                condition_img_alt = condition_img_alt.numpy()
            control_mode = val_data.get("control_mode", "unknown")

            reflected_motion_bucket_id = val_data["reflected_motion_bucket_id"].item()
            controlnet_image_index = val_data["controlnet_image_index"].numpy()
            coordinate_values = val_data["coordinate_values"].numpy()
            tokenized_prompt = val_data["prompt"].unsqueeze(0).to(accelerator.device)
            
            # Store prompt for sanity check
            f = open(os.path.join(validation_store_folder, "lang_cond.txt"), "a")
            f.write(val_task_desc)  # Write the actual task description
            f.close()
            print(f"Validation sample {val_sample_idx}: Task = '{val_task_desc}'")
            
            # Call the pipeline for LIBERO - PRIMARY control mode
            with torch.autocast("cuda"):
                frames = pipeline(
                    image = ref_image,
                    condition_img = condition_img,       # numpy [0,1] range
                    controlnet = accelerator.unwrap_model(controlnet),
                    prompt = tokenized_prompt,
                    use_text = config["use_text"],
                    text_encoder = text_encoder,
                    height = config["height"],
                    width = config["width"],
                    num_frames = config["video_seq_length"],
                    decode_chunk_size = 8,
                    motion_bucket_id = reflected_motion_bucket_id,
                    # controlnet_image_index is not a parameter of the pipeline
                    # coordinate_values is not a parameter of the pipeline
                    num_inference_steps = config["num_inference_steps"],
                    max_guidance_scale = config["inference_max_guidance_scale"],
                    fps = 7,
                    use_instructpix2pix = config["use_instructpix2pix"],
                    noise_aug_strength = config["inference_noise_aug_strength"],
                    controlnet_conditioning_scale = config["outer_conditioning_scale"],
                    inner_conditioning_scale = config["inner_conditioning_scale"],
                    guess_mode = config["inference_guess_mode"],
                    image_guidance_scale = config["image_guidance_scale"],
                ).frames[0]

            # Save frames for primary control mode
            for idx, frame in enumerate(frames):
                frame.save(os.path.join(validation_store_folder, f"{control_mode}_{idx:02d}.png"))

            # Generate video with ALTERNATIVE control mode if available
            if condition_img_alt is not None:
                alt_mode = "mesh" if control_mode == "uniform" else "uniform"
                print(f"Generating alternative video with {alt_mode} control...")

                with torch.autocast("cuda"):
                    frames_alt = pipeline(
                        image = ref_image,
                        condition_img = condition_img_alt,       # Alternative control
                        controlnet = accelerator.unwrap_model(controlnet),
                        prompt = tokenized_prompt,
                        use_text = config["use_text"],
                        text_encoder = text_encoder,
                        height = config["height"],
                        width = config["width"],
                        num_frames = config["video_seq_length"],
                        decode_chunk_size = 8,
                        motion_bucket_id = reflected_motion_bucket_id,
                        num_inference_steps = config["num_inference_steps"],
                        max_guidance_scale = config["inference_max_guidance_scale"],
                        fps = 7,
                        use_instructpix2pix = config["use_instructpix2pix"],
                        noise_aug_strength = config["inference_noise_aug_strength"],
                        controlnet_conditioning_scale = config["outer_conditioning_scale"],
                        inner_conditioning_scale = config["inner_conditioning_scale"],
                        guess_mode = config["inference_guess_mode"],
                        image_guidance_scale = config["image_guidance_scale"],
                    ).frames[0]

                # Save frames for alternative control mode
                for idx, frame in enumerate(frames_alt):
                    frame.save(os.path.join(validation_store_folder, f"{alt_mode}_{idx:02d}.png"))

                # Save GIF for alternative control mode
                imageio.mimsave(os.path.join(validation_store_folder, f'combined_{alt_mode}.gif'), frames_alt, duration=0.05)

            # Save GIF for primary control mode
            imageio.mimsave(os.path.join(validation_store_folder, f'combined_{control_mode}.gif'), frames, duration=0.05)
            
            # Save BOTH trajectory visualizations
            # 1. Primary trajectory (the one used for training)
            trajectory_frames = []
            for t in range(condition_img.shape[0]):
                # Convert trajectory frame from CHW to HWC and to PIL
                traj_frame = condition_img[t].transpose(1, 2, 0)
                traj_frame = (traj_frame * 255).clip(0, 255).astype(np.uint8)
                traj_pil = Image.fromarray(traj_frame)
                trajectory_frames.append(traj_pil)
                traj_pil.save(os.path.join(validation_store_folder, f"trajectory_{control_mode}_{t:02d}.png"))

            # Save primary trajectory GIF with mode label
            imageio.mimsave(os.path.join(validation_store_folder, f'trajectory_{control_mode}.gif'), trajectory_frames, duration=0.05)
            # Also save as trajectory.gif for backward compatibility
            imageio.mimsave(os.path.join(validation_store_folder, 'trajectory.gif'), trajectory_frames, duration=0.05)

            # 2. Alternative trajectory (for comparison)
            if condition_img_alt is not None:
                alt_mode = "mesh" if control_mode == "uniform" else "uniform"
                trajectory_frames_alt = []
                for t in range(condition_img_alt.shape[0]):
                    # Convert trajectory frame from CHW to HWC and to PIL
                    traj_frame_alt = condition_img_alt[t].transpose(1, 2, 0)
                    traj_frame_alt = (traj_frame_alt * 255).clip(0, 255).astype(np.uint8)
                    traj_pil_alt = Image.fromarray(traj_frame_alt)
                    trajectory_frames_alt.append(traj_pil_alt)
                    traj_pil_alt.save(os.path.join(validation_store_folder, f"trajectory_{alt_mode}_{t:02d}.png"))

                # Save alternative trajectory GIF
                imageio.mimsave(os.path.join(validation_store_folder, f'trajectory_{alt_mode}.gif'), trajectory_frames_alt, duration=0.05)
            
            # Save ground truth video
            gt_frames = []
            gt_video = val_data["video_frames"]  # Shape: (16, 3, H, W)
            for t in range(gt_video.shape[0]):
                # Convert from tensor to PIL
                gt_frame = gt_video[t].permute(1, 2, 0).numpy()  # CHW -> HWC
                gt_frame = ((gt_frame + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                gt_pil = Image.fromarray(gt_frame)
                gt_frames.append(gt_pil)
                gt_pil.save(os.path.join(validation_store_folder, f"gt_{t:02d}.png"))
            imageio.mimsave(os.path.join(validation_store_folder, 'ground_truth.gif'), gt_frames, duration=0.05)
            
            # Save combined video+trajectory side-by-side
            combined_frames = []
            for idx, (vid_frame, traj_frame) in enumerate(zip(frames, trajectory_frames)):
                # Resize both to same height if needed
                if vid_frame.size != traj_frame.size:
                    traj_frame = traj_frame.resize(vid_frame.size)
                # Combine horizontally
                combined = Image.new('RGB', (vid_frame.width * 2, vid_frame.height))
                combined.paste(vid_frame, (0, 0))
                combined.paste(traj_frame, (vid_frame.width, 0))
                combined_frames.append(combined)
            imageio.mimsave(os.path.join(validation_store_folder, 'video_trajectory_combined.gif'), combined_frames, duration=0.05)
            
            # Save comparison: GT vs Generated vs Trajectory
            comparison_frames = []
            for idx in range(len(frames)):
                gt_frame = gt_frames[idx]
                gen_frame = frames[idx]
                traj_frame = trajectory_frames[idx]
                
                # Resize all to same size if needed
                if gt_frame.size != gen_frame.size:
                    gt_frame = gt_frame.resize(gen_frame.size)
                if traj_frame.size != gen_frame.size:
                    traj_frame = traj_frame.resize(gen_frame.size)
                
                # Combine horizontally: GT | Generated | Trajectory
                comparison = Image.new('RGB', (gen_frame.width * 3, gen_frame.height))
                comparison.paste(gt_frame, (0, 0))
                comparison.paste(gen_frame, (gen_frame.width, 0))
                comparison.paste(traj_frame, (gen_frame.width * 2, 0))
                comparison_frames.append(comparison)
            imageio.mimsave(os.path.join(validation_store_folder, 'comparison.gif'), comparison_frames, duration=0.05)
            
            frames_collection.append(frames)
            
            # WandB video logging
            if config["report_to"] == "wandb" and is_wandb_available() and accelerator.is_main_process:
                import wandb
                
                # Log generated video
                video_path = os.path.join(validation_store_folder, 'combined.gif')
                wandb.log({
                    f"validation/video_sample_{val_sample_idx}": wandb.Video(video_path, fps=7, format="gif"),
                    "validation/step": step,
                }, step=step)
                
                # Log trajectory visualization
                traj_vis = (condition_img[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                wandb.log({
                    f"validation/trajectory_sample_{val_sample_idx}": wandb.Image(traj_vis),
                }, step=step)
    elif config["data_loader_type"] in ("realworld_hybrid", "realworld_dualview"):
        # Real-world hybrid/dualview validation — each GPU handles one sample.
        # Loader differs only in shape (dualview stacks two views vertically);
        # downstream val pipeline is shape-agnostic (uses config height/width).
        is_dualview = config["data_loader_type"] == "realworld_dualview"
        if is_dualview:
            from data_loader.video_realworld_dualview_dataset import (
                Video_Realworld_Dualview_Dataset as _RealworldDataset,
            )
        else:
            from data_loader.video_realworld_hybrid_dataset import (
                Video_Realworld_Hybrid_Dataset as _RealworldDataset,
            )

        if val_dataset is None:
            val_config = config.copy()
            val_config["split"] = "val"
            # Validation needs raw pixels (GT gif) + rendered heatmap (trajectory
            # gif), so force the dualview loader's slow render path regardless of
            # whether precomputed latents exist in the hdf5. At N=4 samples per
            # validation every 100 steps, the cost is negligible.
            if is_dualview:
                val_dataset = _RealworldDataset(
                    val_config, accelerator.device, tokenizer=tokenizer,
                    force_legacy=True)
            else:
                val_dataset = _RealworldDataset(
                    val_config, accelerator.device, tokenizer=tokenizer)

        num_validation_samples = min(4, len(val_dataset))

        if accelerator.process_index < num_validation_samples:
            val_sample_idx = accelerator.process_index

            base_val_folder = config["validation_store_folder"]
            if config_name and timestamp:
                val_folder_name = f"{base_val_folder}_{config_name}_{timestamp}_{config['scheduler']}"
            else:
                val_folder_name = f"{base_val_folder}_{config['scheduler']}"
            validation_store_folder = os.path.join(val_folder_name, "step_" + str(step), f"sample_{val_sample_idx}")

            if os.path.exists(validation_store_folder):
                shutil.rmtree(validation_store_folder)
            os.makedirs(validation_store_folder, exist_ok=True)

            import random
            random.seed(step + val_sample_idx)
            val_idx = random.randint(0, len(val_dataset) - 1)
            val_data = val_dataset[val_idx]

            # Reference image
            ref_image = val_data["video_frames"][0]
            if ref_image.dim() == 3:
                ref_image = ref_image.permute(1, 2, 0)
            ref_image_np = ((ref_image + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
            ref_image = Image.fromarray(ref_image_np, mode='RGB')

            condition_img = val_data["controlnet_condition"].numpy()
            reflected_motion_bucket_id = val_data["reflected_motion_bucket_id"].item()
            tokenized_prompt = val_data["prompt"].unsqueeze(0).to(accelerator.device)

            # Generate
            with torch.autocast("cuda"):
                frames = pipeline(
                    image=ref_image,
                    condition_img=condition_img,
                    controlnet=accelerator.unwrap_model(controlnet),
                    prompt=tokenized_prompt,
                    use_text=config["use_text"],
                    text_encoder=text_encoder,
                    height=config["height"],
                    width=config["width"],
                    num_frames=config["video_seq_length"],
                    decode_chunk_size=8,
                    motion_bucket_id=reflected_motion_bucket_id,
                    num_inference_steps=config["num_inference_steps"],
                    max_guidance_scale=config["inference_max_guidance_scale"],
                    fps=7,
                    use_instructpix2pix=config["use_instructpix2pix"],
                    noise_aug_strength=config["inference_noise_aug_strength"],
                    controlnet_conditioning_scale=config["outer_conditioning_scale"],
                    inner_conditioning_scale=config["inner_conditioning_scale"],
                    guess_mode=config["inference_guess_mode"],
                    image_guidance_scale=config["image_guidance_scale"],
                ).frames[0]

            # Save generated frames
            for idx, frame in enumerate(frames):
                frame.save(os.path.join(validation_store_folder, f"gen_{idx:02d}.png"))
            imageio.mimsave(os.path.join(validation_store_folder, 'combined.gif'), frames, duration=0.05)

            # Save trajectory visualization
            trajectory_frames = []
            for t in range(condition_img.shape[0]):
                traj_frame = condition_img[t].transpose(1, 2, 0)
                traj_frame = (traj_frame * 255).clip(0, 255).astype(np.uint8)
                traj_pil = Image.fromarray(traj_frame)
                trajectory_frames.append(traj_pil)
                traj_pil.save(os.path.join(validation_store_folder, f"trajectory_{t:02d}.png"))
            imageio.mimsave(os.path.join(validation_store_folder, 'trajectory.gif'), trajectory_frames, duration=0.05)

            # Save ground truth
            gt_frames = []
            gt_video = val_data["video_frames"]
            for t in range(gt_video.shape[0]):
                gt_frame = gt_video[t].permute(1, 2, 0).numpy()
                gt_frame = np.clip((gt_frame + 1.0) * 127.5, 0, 255).astype(np.uint8)
                gt_pil = Image.fromarray(gt_frame)
                gt_frames.append(gt_pil)
                gt_pil.save(os.path.join(validation_store_folder, f"gt_{t:02d}.png"))
            imageio.mimsave(os.path.join(validation_store_folder, 'ground_truth.gif'), gt_frames, duration=0.05)

            # Save comparison: GT | Generated | Trajectory
            comparison_frames = []
            for idx in range(len(frames)):
                gt_f = gt_frames[idx]
                gen_f = frames[idx]
                traj_f = trajectory_frames[idx]
                if gt_f.size != gen_f.size:
                    gt_f = gt_f.resize(gen_f.size)
                if traj_f.size != gen_f.size:
                    traj_f = traj_f.resize(gen_f.size)
                comp = Image.new('RGB', (gen_f.width * 3, gen_f.height))
                comp.paste(gt_f, (0, 0))
                comp.paste(gen_f, (gen_f.width, 0))
                comp.paste(traj_f, (gen_f.width * 2, 0))
                comparison_frames.append(comp)
            imageio.mimsave(os.path.join(validation_store_folder, 'comparison.gif'), comparison_frames, duration=0.05)

            frames_collection.append(frames)

    else:
        # Original logic for thisthat dataset
        import random
        for image_name in sorted(os.listdir(validation_source_folder)):
            # Compute validation_store_folder on ALL ranks (needed later for saving)
            if parent_store_folder is None:
                base_val_folder = config["validation_store_folder"]
                if config_name and timestamp:
                    val_folder_name = f"{base_val_folder}_{config_name}_{timestamp}_{config['scheduler']}"
                else:
                    val_folder_name = f"{base_val_folder}_{config['scheduler']}"
                validation_store_folder = os.path.join(val_folder_name, "step_" + str(step), image_name)
            else:
                validation_store_folder = os.path.join(parent_store_folder, image_name)

            if accelerator.is_main_process:
                if os.path.exists(validation_store_folder):
                    shutil.rmtree(validation_store_folder)
                os.makedirs(validation_store_folder)

            image_path = os.path.join(validation_source_folder, image_name, 'im_0.jpg')
            ref_image = load_image(image_path)      # [0, 255] Range
            ref_image = ref_image.resize((config["width"], config["height"]))


            # Prepare text prompt
            if config["use_text"]:
                # Read the file
                file_path = os.path.join(validation_source_folder, image_name, "lang.txt")
                file = open(file_path, 'r')
                prompt = file.readlines()[0]  # Only read the first line
                if use_ambiguous_prompt:
                    prompt = prompt.split(" ")[0] + " this to there"
                    print("We are creating ambiguous prompt, which is: ", prompt)
            else:
                prompt = ""
        # Use the same tokenize process as the dataset preparation stage
        tokenized_prompt = tokenize_captions(prompt, tokenizer, config, is_train=False).unsqueeze(0).to(accelerator.device)    # Use unsqueeze to expand dim

        # Store the prompt for the sanity check
        if accelerator.is_main_process:
            f = open(os.path.join(validation_store_folder, "lang_cond.txt"), "a")
            f.write(prompt)
            f.close()

        # Flip the image by chance (it is needed to check whether there is any object position words [left|right] in the prompt text)
        flip = False
        if not force_close_flip:    # force_close_flip is True in testing time; else, we cannot match in the same standard
            if random.random() < config["flip_aug_prob"]:
                if config["use_text"]:
                    if prompt.find("left") == -1 and prompt.find("right") == -1:    # Cannot have position word, like left and right (up and down is ok)
                        flip = True
                else:
                    flip = True
            if flip:
                print("Use flip in validation!")
                ref_image = ref_image.transpose(Image.FLIP_LEFT_RIGHT)


        if config["data_loader_type"] == "thisthat":
            condition_img, reflected_motion_bucket_id, controlnet_image_index, coordinate_values = get_thisthat_sam(config, 
                                                                                                                    os.path.join(validation_source_folder, image_name),
                                                                                                                    flip = flip, 
                                                                                                                    store_dir = validation_store_folder,
                                                                                                                    verbose = True)
        elif config["data_loader_type"] == "libero":
            # For LIBERO validation, use a random sample from validation set
            from data_loader.video_libero_dataset import Video_LIBERO_Dataset, generate_gripper_trajectory_vis
            
            # Create validation config with val split
            val_config = config.copy()
            val_config["dataset_path"] = config.get("val_dataset_path", config["dataset_path"])
            val_config["split"] = config.get("val_split", "val")
            
            # Create temporary dataset to load validation data
            val_dataset = Video_LIBERO_Dataset(val_config, accelerator.device, tokenizer=tokenizer)
            if len(val_dataset) > 0:
                # Get a random demo for validation (could iterate through multiple)
                import random
                val_idx = random.randint(0, min(len(val_dataset)-1, 10))  # Sample from first 10 demos
                val_data = val_dataset[val_idx]
                
                # Extract the first frame as reference image
                ref_image = val_data["video_frames"][0]  # First frame of video
                if ref_image.dim() == 3:  # CHW format
                    ref_image = ref_image.permute(1, 2, 0)  # Convert to HWC
                ref_image = (ref_image * 127.5 + 127.5).clamp(0, 255).numpy().astype(np.uint8)
                ref_image = Image.fromarray(ref_image)
                
                condition_img = val_data["controlnet_condition"].numpy()  # Already in correct format
                reflected_motion_bucket_id = val_data["reflected_motion_bucket_id"].item()
                controlnet_image_index = val_data["controlnet_image_index"].numpy()
                coordinate_values = val_data["coordinate_values"].numpy()
            else:
                # Fallback if no validation data
                print("Warning: No validation data found for LIBERO")
                condition_img = np.zeros((config["video_seq_length"], config["conditioning_channels"], 
                                         config["height"], config["width"]), dtype=np.float32)
                reflected_motion_bucket_id = 200
                controlnet_image_index = np.arange(config["video_seq_length"])
                coordinate_values = np.array([[64, 64]] * config["video_seq_length"])
        else:
            raise NotImplementedError("We don't support such data loader type")



        # Call the pipeline
        with torch.autocast("cuda"):
            frames = pipeline(
                                image = ref_image, 
                                condition_img = condition_img,       # numpy [0,1] range
                                controlnet = accelerator.unwrap_model(controlnet),
                                prompt = tokenized_prompt,
                                use_text = config["use_text"],
                                text_encoder = text_encoder,
                                height = config["height"],
                                width = config["width"],
                                num_frames = config["video_seq_length"], 
                                decode_chunk_size = 8, 
                                motion_bucket_id = reflected_motion_bucket_id,
                                num_inference_steps = config["num_inference_steps"],
                                max_guidance_scale = config["inference_max_guidance_scale"],
                                fps = 7,
                                use_instructpix2pix = config["use_instructpix2pix"],
                                noise_aug_strength = config["inference_noise_aug_strength"],
                                controlnet_conditioning_scale = config["outer_conditioning_scale"],
                                inner_conditioning_scale = config["inner_conditioning_scale"],
                                guess_mode = config["inference_guess_mode"],        # False in inference
                                image_guidance_scale = config["image_guidance_scale"],
                              ).frames[0]    

        if accelerator.is_main_process:
            for idx, frame in enumerate(frames):
                frame.save(os.path.join(validation_store_folder, str(idx)+".png"))
            imageio.mimsave(os.path.join(validation_store_folder, 'combined.gif'), frames, duration=0.05)

            # Save trajectory visualization
            trajectory_frames = []
            for t in range(condition_img.shape[0]):
                traj_frame = condition_img[t].transpose(1, 2, 0)
                traj_frame = (traj_frame * 255).clip(0, 255).astype(np.uint8)
                traj_pil = Image.fromarray(traj_frame)
                trajectory_frames.append(traj_pil)
                traj_pil.save(os.path.join(validation_store_folder, f"trajectory_{t:02d}.png"))
            imageio.mimsave(os.path.join(validation_store_folder, 'trajectory.gif'), trajectory_frames, duration=0.05)

            # Save ground truth video (only if val_data exists, i.e. libero/realworld_hybrid)
            if 'val_data' in dir():
                gt_frames = []
                gt_video = val_data["video_frames"]  # Shape: (16, 3, H, W)
                for t in range(gt_video.shape[0]):
                    gt_frame = gt_video[t].permute(1, 2, 0).numpy()  # CHW -> HWC
                    gt_frame = np.clip((gt_frame + 1.0) * 127.5, 0, 255).astype(np.uint8)
                    gt_pil = Image.fromarray(gt_frame)
                    gt_frames.append(gt_pil)
                    gt_pil.save(os.path.join(validation_store_folder, f"gt_{t:02d}.png"))
                imageio.mimsave(os.path.join(validation_store_folder, 'ground_truth.gif'), gt_frames, duration=0.05)

                # Save comparison: GT | Generated | Trajectory
                comparison_frames = []
                for idx in range(len(frames)):
                    gt_f = gt_frames[idx]
                    gen_f = frames[idx]
                    traj_f = trajectory_frames[idx]
                    if gt_f.size != gen_f.size:
                        gt_f = gt_f.resize(gen_f.size)
                    if traj_f.size != gen_f.size:
                        traj_f = traj_f.resize(gen_f.size)
                    comparison = Image.new('RGB', (gen_f.width * 3, gen_f.height))
                    comparison.paste(gt_f, (0, 0))
                    comparison.paste(gen_f, (gen_f.width, 0))
                    comparison.paste(traj_f, (gen_f.width * 2, 0))
                    comparison_frames.append(comparison)
                imageio.mimsave(os.path.join(validation_store_folder, 'comparison.gif'), comparison_frames, duration=0.05)

        frames_collection.append(frames)
        
        # WandB video logging
        if config["report_to"] == "wandb" and is_wandb_available() and accelerator.is_main_process:
            import wandb
            
            # Log generated video
            video_path = os.path.join(validation_store_folder, 'combined.gif')
            wandb.log({
                f"validation/video_{image_name}": wandb.Video(video_path, fps=7, format="gif"),
                "validation/step": step,
            }, step=step)
            
            # Log trajectory visualization
            if config["data_loader_type"] == "libero":
                # Create visualization of the trajectory condition
                traj_vis = (condition_img[0].transpose(1, 2, 0) * 255).astype(np.uint8)
                wandb.log({
                    f"validation/trajectory_{image_name}": wandb.Image(traj_vis),
                }, step=step)
            
            # Log first and last frames
            wandb.log({
                f"validation/first_frame_{image_name}": wandb.Image(frames[0]),
                f"validation/last_frame_{image_name}": wandb.Image(frames[-1]),
            }, step=step)


    # Cleaning process
    del pipeline
    torch.cuda.empty_cache()
    
    # IMPORTANT: All processes must wait for validation to complete
    # This prevents deadlock when some GPUs finish validation early
    accelerator.wait_for_everyone()

    return frames_collection   # Return resuly based on the need


def tensor_to_vae_latent(inputs, vae):
    video_length = inputs.shape[1]

    inputs = rearrange(inputs, "b f c h w -> (b f) c h w")
    latents = vae.encode(inputs).latent_dist.mode()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)    # Use f or b to rearrage should have the same effect
    latents = latents * vae.config.scaling_factor

    return latents



def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7 # In the range [0, 1]
    # TODO: "* (1 - 2e-7) + 1e-7" is not included in previous code, I add it back, don't why whether there is any influence now
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def get_add_time_ids(
        unet_config,
        expected_add_embed_dim,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance = False,
    ):

    # Construct Basic add_time_ids items
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]


    # Sanity Check
    passed_add_embed_dim = unet_config.addition_time_embed_dim * len(add_time_ids)
    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

    return add_time_ids
####################################################################################################################################################################



def main(config):
    # Read Config Setting
    # Add timestamp and config name to prevent overwriting
    from datetime import datetime
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    resume_from_checkpoint = config["resume_from_checkpoint"]
    # Create unique output directory with config name and timestamp
    base_output_dir = config["output_dir"]
    output_dir = f"{base_output_dir}_{config_name}_{timestamp}"
    logging_name = config.get("logging_name", "logs")
    mixed_precision = config["mixed_precision"]
    report_to = config["report_to"]
    pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
    pretrained_tokenizer_name_or_path = config["pretrained_tokenizer_name_or_path"]
    gradient_checkpointing = config["gradient_checkpointing"]
    learning_rate = config["learning_rate"]
    adam_beta1 = float(config["adam_beta1"])
    adam_beta2 = float(config["adam_beta2"])
    adam_weight_decay = float(config["adam_weight_decay"])
    adam_epsilon = float(config["adam_epsilon"])
    train_batch_size = config["train_batch_size"]
    dataloader_num_workers = config["dataloader_num_workers"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    num_train_iters = config["num_train_iters"]
    lr_warmup_steps = config["lr_warmup_steps"]
    checkpointing_steps = config["checkpointing_steps"]
    process_fps = config["process_fps"]
    train_noise_aug_strength = config["train_noise_aug_strength"]
    use_8bit_adam = config["use_8bit_adam"]
    scale_lr = config["scale_lr"]
    conditioning_dropout_prob = config["conditioning_dropout_prob"]
    checkpoints_total_limit = config["checkpoints_total_limit"]
    validation_step = config["validation_step"]
    partial_finetune = config['partial_finetune']
    load_unet_path = config['load_unet_path']

    if mixed_precision == 'None':   # For mixed precision use
        mixed_precision = 'no'


    # Default Setting
    revision = None
    variant = "fp16"        # TODO: 这里进行了调整，不知道会有多少区别，现在跟unet training保持一致
    lr_scheduler = "constant"
    max_grad_norm = 1.0
    tracker_project_name = "img2video"
    num_videos_per_prompt = 1
    seed = 42
    # No CFG in training now



    # Define the accelerator
    logging_dir = Path(output_dir, logging_name)
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps = gradient_accumulation_steps,
        mixed_precision = mixed_precision,
        log_with = report_to,
        project_config = accelerator_project_config,
    )
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    # Handle the repository creation
    if accelerator.is_main_process and resume_from_checkpoint != "latest":      # For the latest checkpoint version, we don't need to delete our folders
        # Validation file - with timestamp it's unique, so no need to remove
        base_val_folder = config["validation_store_folder"]
        validation_store_folder = f"{base_val_folder}_{config_name}_{timestamp}_{config['scheduler']}"
        print("Validation will be saved to:", validation_store_folder)
        # No need to remove since timestamp makes it unique
        if False:  # Disabled removal since we have unique timestamps
            archive_name = validation_store_folder + "_archive"
            if os.path.exists(archive_name):
                shutil.rmtree(archive_name)
            print("We will move to archive ", archive_name)
            os.rename(validation_store_folder, archive_name)
        os.makedirs(validation_store_folder)

        # Output Dir
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

        # Log - use unique runs directory per training session
        runs_dir = f"runs_{config_name}_{timestamp}" if config_name and timestamp else "runs"
        if os.path.exists(runs_dir):
            shutil.rmtree(runs_dir)
        
        # Copy the config to here
        os.system(" cp config/train_image2video_gesturenet.yaml " + validation_store_folder + "/")


    # Load All Module Needed
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path, subfolder="feature_extractor", revision=revision
    )   # This instance has now weight, they are just seeting file
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="image_encoder", revision=revision, variant=variant
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
    )
    if load_unet_path != None:
        print("We will use pretrained UNet path by our, at ", load_unet_path)
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            load_unet_path, 
            subfolder = "unet", 
            low_cpu_mem_usage = True,       
        )   # For the variant, we don't have fp16 version, so we will read from fp32
    else:
        print("We will still use provided UNet path")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder = "unet", 
            low_cpu_mem_usage = True,
            variant = variant,
        )

    # Prepare for the tokenizer if use text
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_tokenizer_name_or_path,
        subfolder = "tokenizer",
        revision = revision,
        use_fast = False,
    )

    if config["use_text"]:
        # Clip Text Encoder
        text_encoder_cls = import_pretrained_text_encoder(pretrained_tokenizer_name_or_path, revision)
        text_encoder = text_encoder_cls.from_pretrained(
            pretrained_tokenizer_name_or_path, subfolder = "text_encoder", revision = revision, variant = variant
        )
    else:
        text_encoder = None

    # Init for the Controlnet (check if has pretrained path to load)
    if config["load_controlnet_path"] != None:
        print("We will load pre-trained controlnet from ", config["load_controlnet_path"])
        controlnet = ControlNetModel.from_pretrained(config["load_controlnet_path"], subfolder="controlnet")
    else:
        controlnet = ControlNetModel.from_unet(unet, load_weights_from_unet=True, conditioning_channels=config["conditioning_channels"])


    # Store the config due to the disappearance after accelerator prepare
    unet_config = unet.config
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    # Get training configuration from config
    train_unet = config.get('train_unet', True)  # Default: train both
    train_controlnet = config.get('train_controlnet', True)  # Default: train both

    # Validate configuration
    if not train_unet and not train_controlnet:
        print("WARNING: Neither model set to train. Defaulting to train_controlnet=True")
        train_controlnet = True

    print(f"\n{'='*60}")
    print("MODEL TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Train UNet: {train_unet}")
    print(f"  Train ControlNet: {train_controlnet}")
    print(f"{'='*60}\n")

    # Freeze vae + feature_extractor + image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # Configure UNet training based on config
    if train_unet:
        unet.requires_grad_(True)
        print("UNet: TRAINING mode enabled")
    else:
        unet.requires_grad_(False)
        print("UNet: FROZEN (using pretrained)")

    # ControlNet will be configured later in the code
    controlnet.requires_grad_(False)    # Will turn back to requires grad later if train_controlnet=True

    if config["use_text"]:
        text_encoder.requires_grad_(False)


    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    # Move vae + unet + image_encoder to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    # Only cast UNet to lower precision if it's not being trained
    if not any(p.requires_grad for p in unet.parameters()):
        unet.to(accelerator.device, dtype=weight_dtype)     # Cast to FP16 only if not training
    else:
        unet.to(accelerator.device)  # Keep FP32 for training
        print("Keeping UNet in FP32 for training")
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    if config["use_text"]:
        text_encoder.to(accelerator.device, dtype=weight_dtype)



    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                print(f"[SAVE HOOK] Saving {len(models)} models to {output_dir}")

                # Debug: Print all models and their types
                for idx, model in enumerate(models):
                    print(f"[SAVE HOOK DEBUG] Model {idx}: {model.__class__.__name__}")

                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    # Better detection: check class name to distinguish UNet from ControlNet
                    model_class_name = model.__class__.__name__
                    print(f"[SAVE HOOK] Processing model {i}: {model_class_name}")

                    if "UNet" in model_class_name:
                        # This is the UNet
                        if train_unet:  # Only save if UNet was trained
                            sub_dir = "unet"
                            print(f"[SAVE HOOK] Saving trained UNet ({model_class_name}) to {sub_dir}/")
                            model.save_pretrained(os.path.join(output_dir, sub_dir))
                            print(f"[SAVE HOOK] UNet saved successfully to {output_dir}/{sub_dir}")
                        else:
                            print(f"[SAVE HOOK] Skipping UNet (not trained)")
                    elif "ControlNet" in model_class_name:
                        # This is the ControlNet
                        if train_controlnet:  # Only save if ControlNet was trained
                            sub_dir = "controlnet"
                            print(f"[SAVE HOOK] Saving trained ControlNet ({model_class_name}) to {sub_dir}/")
                            model.save_pretrained(os.path.join(output_dir, sub_dir))
                            print(f"[SAVE HOOK] ControlNet saved successfully to {output_dir}/{sub_dir}")
                        else:
                            print(f"[SAVE HOOK] Skipping ControlNet (not trained)")
                    else:
                        print(f"[SAVE HOOK] Unknown model type: {model_class_name}")
                        print(f"[SAVE HOOK] Model attributes: has conv_in={hasattr(model, 'conv_in')}, has time_proj={hasattr(model, 'time_proj')}")

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # Check if this is UNet or ControlNet and load only if it was trained
                if hasattr(model, 'conv_in') and hasattr(model, 'time_proj'):
                    # This is the UNet
                    if train_unet and os.path.exists(os.path.join(input_dir, "unet")):
                        from diffusers import UNet2DConditionModel
                        print(f"[LOAD HOOK] Loading trained UNet from {input_dir}/unet")
                        load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                        model.register_to_config(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                    else:
                        print(f"[LOAD HOOK] Skipping UNet load (not trained or not found)")
                else:
                    # This is the ControlNet
                    if train_controlnet and os.path.exists(os.path.join(input_dir, "controlnet")):
                        print(f"[LOAD HOOK] Loading trained ControlNet from {input_dir}/controlnet")
                        load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                        model.register_to_config(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                    else:
                        print(f"[LOAD HOOK] Skipping ControlNet load (not trained or not found)")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    if gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()


    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )
    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Also check UNet if it's being trained
    if any(p.requires_grad for p in unet.parameters()):
        if accelerator.unwrap_model(unet).dtype != torch.float32:
            raise ValueError(
                f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
            )
    
    


    ################################ Make Training dataset  ######################################
    if config["data_loader_type"] == "thisthat":    # Only keep thisthat mode now
        train_dataset = Video_ThisThat_Dataset(config, accelerator.device, tokenizer=tokenizer)
        val_dataset = None  # ThistThat uses folder-based validation
    elif config["data_loader_type"] == "libero":
        from data_loader.video_libero_dataset import Video_LIBERO_Dataset
        train_dataset = Video_LIBERO_Dataset(config, accelerator.device, tokenizer=tokenizer)
        
        # Also load validation dataset at the beginning
        val_config = config.copy()
        val_config["dataset_path"] = config.get("val_dataset_path", config["dataset_path"])
        val_config["split"] = config.get("val_split", "val")
        val_dataset = Video_LIBERO_Dataset(val_config, accelerator.device, tokenizer=tokenizer)
        print(f"Validation dataset loaded with {len(val_dataset)} samples")
    elif config["data_loader_type"] == "libero_32point":
        from data_loader.video_libero_32point_dataset import Video_LIBERO_32Point_Dataset
        train_dataset = Video_LIBERO_32Point_Dataset(config, accelerator.device, tokenizer=tokenizer)

        # Also load validation dataset at the beginning
        val_config = config.copy()
        val_config["dataset_path"] = config.get("val_dataset_path", config["dataset_path"])
        val_config["split"] = config.get("val_split", "val")
        val_dataset = Video_LIBERO_32Point_Dataset(val_config, accelerator.device, tokenizer=tokenizer)
        print(f"Validation dataset loaded with {len(val_dataset)} samples (32-point tracking)")
    elif config["data_loader_type"] in ("realworld_hybrid", "realworld_dualview"):
        if config["data_loader_type"] == "realworld_dualview":
            from data_loader.video_realworld_dualview_dataset import (
                Video_Realworld_Dualview_Dataset as _RealworldDataset,
            )
            _tag = "realworld dualview"
        else:
            from data_loader.video_realworld_hybrid_dataset import (
                Video_Realworld_Hybrid_Dataset as _RealworldDataset,
            )
            _tag = "realworld hybrid"

        train_dataset = _RealworldDataset(config, accelerator.device, tokenizer=tokenizer)

        val_config = config.copy()
        val_config["split"] = "val"
        # Dualview validation uses the slow render path so GT + trajectory gifs work.
        if config["data_loader_type"] == "realworld_dualview":
            val_dataset = _RealworldDataset(
                val_config, accelerator.device, tokenizer=tokenizer, force_legacy=True)
        else:
            val_dataset = _RealworldDataset(
                val_config, accelerator.device, tokenizer=tokenizer)
        print(f"Validation dataset loaded with {len(val_dataset)} samples ({_tag})")
    else:
        raise NotImplementedError("We don't support such data loader type")

    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler = sampler,
        batch_size = train_batch_size,
        num_workers = dataloader_num_workers * accelerator.num_processes,
    )
    
    # Do not wrap dataloader - let epoch mechanism handle cycling
    # The training loop already handles epochs properly
    logger.info("Using standard dataloader with epoch-based iteration")
    ##############################################################################################
    


    ####################################### Optimizer Setting ##############################################################
    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # 8bit adam to save more memory
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    
    # Make ControlNet require Grad
    controlnet.requires_grad_(True)


    ###################### For partial fine-tune setting #######################
    parameters_list = []

    # Only process ControlNet parameters if we're training it
    if train_controlnet:
        for name, para in controlnet.named_parameters():
            if partial_finetune:    # The partial finetune we use is to only train attn layers, which will be ~190M params (TODO:needs to check later for exact value)
                if not name.find("attn") != -1:     # Only block the spatial Transformer
                    para.requires_grad = False
                else:
                    parameters_list.append(para)
                    para.requires_grad = True
            else:
                parameters_list.append(para)
                para.requires_grad = True

    # Add UNet parameters if it's trainable
    unet_has_trainable_params = any(p.requires_grad for p in unet.parameters())
    if unet_has_trainable_params:
        print("Adding UNet parameters to optimizer")
        unet_params = 0
        for name, para in unet.named_parameters():
            if para.requires_grad:
                parameters_list.append(para)
                unet_params += para.numel()
        print(f"Added {unet_params:,} UNet parameters to optimizer")

    # Double check the weight that will be trained
    total_params_for_training = 0
    for name, param in controlnet.named_parameters():
        if param.requires_grad:
            total_params_for_training += param.numel()
            print(name + " requires grad update")
    print("Total parameter that will be trained in controlnet has ", total_params_for_training)

    # Also count UNet params if training
    if unet_has_trainable_params:
        unet_params_training = 0
        for name, param in unet.named_parameters():
            if param.requires_grad:
                unet_params_training += param.numel()
        print(f"Total parameters that will be trained in UNet: {unet_params_training:,}")
        print(f"Total parameters being trained: {total_params_for_training + unet_params_training:,}")
    #############################################################################

    # Optimizer creation
    optimizer = optimizer_cls(
        parameters_list,
        lr = learning_rate,
        betas = (adam_beta1, adam_beta2),
        weight_decay = adam_weight_decay,
        eps = adam_epsilon,
    )


    # Scheduler and Training steps
    dataset_length = len(train_dataset)
    print("Dataset length read from the train side is ", dataset_length)
    num_update_steps_per_epoch = math.ceil(dataset_length / (train_batch_size * gradient_accumulation_steps))
    # Fixed: max_train_steps should be num_train_iters, NOT multiplied by batch_size
    max_train_steps = num_train_iters

    # Learning Rate Scheduler - Using cosine decay to prevent NaN in later training
    lr_scheduler = get_scheduler(
        "cosine",  # Changed from "constant" to "cosine" for gradual decay
        optimizer = optimizer,
        num_warmup_steps = lr_warmup_steps * accelerator.num_processes,
        num_training_steps = max_train_steps * accelerator.num_processes,
        num_cycles = 1,
        power = 1.0,
    )
    #######################################################################################################################
    


    # Prepare everything with our `accelerator`.
    # Include UNet if it has trainable parameters
    unet_is_trainable = any(p.requires_grad for p in unet.parameters())
    if unet_is_trainable:
        print("UNet is trainable, including it in accelerator.prepare()")
        unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, controlnet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        print("UNet is NOT trainable, only preparing ControlNet")
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )   



    # We need to RECALCULATE our total training steps as the size of the training dataloader may have changed.
    print("accelerator.num_processes is ", accelerator.num_processes)
    print("num_train_iters is ", num_train_iters)
    # Fixed: Include batch_size in epoch calculation
    # num_train_epochs = total_iterations * batch_size * num_gpus / dataset_size
    num_train_epochs = math.ceil(num_train_iters * train_batch_size * accelerator.num_processes * gradient_accumulation_steps / dataset_length) 
    print(f"num_train_epochs is {num_train_epochs} (with batch_size={train_batch_size})")

    # We need to initialize the trackers we use, and also store our configuration.
    if accelerator.is_main_process: # Only on the main process!
        tracker_config = dict(vars(args))
        
        # Enhanced WandB initialization
        if report_to == "wandb" and is_wandb_available():
            import wandb
            # Initialize wandb with more config
            # Convert OmegaConf to plain dict to avoid serialization issues
            if hasattr(config, 'to_container'):
                wandb_config = config.to_container(resolve=True)
            else:
                wandb_config = dict(config)
            
            # Ensure all values are JSON serializable
            for k, v in list(wandb_config.items()):
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    wandb_config[k] = str(v)
            
            wandb.init(
                project=config.get("wandb_project", "this-that-vdm"),
                name=config.get("wandb_run_name", None),
                config=wandb_config,
                tags=config.get("wandb_tags", []),
                notes=f"Training on {config.get('dataset_name', 'dataset')} at {config.get('height', 128)}x{config.get('width', 128)} resolution",
                save_code=True,
            )
            
            # Log model architecture summary
            wandb.watch(controlnet, log="all", log_freq=config.get("wandb_log_interval", 100))
        
        accelerator.init_trackers(tracker_project_name, tracker_config)



    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Dataset Length = {dataset_length}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")



    # Load the Closest / Best weight       TODO: need to check how to use checkpoints from pre-trained weights!!!
    global_step = 0     # Catch the current iteration
    first_epoch = 0
    if resume_from_checkpoint:          # Resume Checkpoints!!!!!
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            print("We will resume the latest weight ", path)

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            full_ckpt_path = os.path.join(output_dir, path)
            if os.path.isdir(full_ckpt_path):
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(full_ckpt_path)
            else:
                # External checkpoint — model weights already loaded via load_unet_path/load_controlnet_path
                # Fresh optimizer + scheduler (same approach as action model)
                accelerator.print(f"Checkpoint {path} not in output_dir, skipping optimizer state load (model weights already loaded, fresh scheduler)")
            # Take the last "-"-separated token so names like
            # "checkpoint-5000" and "hybrid-checkpoint-128000" both parse.
            global_step = int(path.rsplit("-", 1)[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    if accelerator.is_main_process:
        print("Initial Learning rate is ", optimizer.param_groups[0]['lr'])
        print("global_step will start from ", global_step)

    progress_bar = tqdm(
                            range(initial_global_step, max_train_steps),
                            initial=initial_global_step,
                            desc="Steps",
                            # Only show the progress bar once on each machine.
                            disable=not accelerator.is_local_main_process,
                        )

    

    # Prepare tensorboard log - use unique directory if multiple configs running
    runs_dir = f"runs_{config_name}_{timestamp}" if config_name and timestamp else "runs"
    writer = SummaryWriter(log_dir=runs_dir) 


    ################################### Auxiliary Function ################################################################################################
    def encode_clip(pixel_values, prompt):
        ''' Encoder hidden states input source
            pixel_values:   first frame pixel information
            prompt:         language prompt with takenized
        '''

        ########################################## Prepare the Text Embedding #####################################################
        # pixel_values is in the range [-1, 1]
        pixel_values = resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0   # [-1, 1] -> [0, 1]

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        # The following is the same as _encode_image in SVD pipeline
        pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        encoder_hidden_states = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        #############################################################################################################################



        ########################################## Prepare the Text embedding if needed #############################################
        if config["use_text"]:
            text_embeddings = text_encoder(prompt)[0]
            
            # Concat two embeddings together on dim 1
            encoder_hidden_states = torch.cat((text_embeddings, encoder_hidden_states), dim=1)      # 目前先用text_embeddings 再用encoder_hidden_states感觉好一点

            # Layer norm on the last dim
            layer_norm = nn.LayerNorm((78, 1024)).to(device=accelerator.device, dtype=weight_dtype)
            encoder_hidden_states_norm = layer_norm(encoder_hidden_states)

            # Return
            return encoder_hidden_states_norm

        else:   # Just return back default on
            return encoder_hidden_states
        #############################################################################################################################

    #########################################################################################################################################################


    ############################################################################################################################
    # For the training, we mimic the code from test2image in diffusers  TODO: check the data loader conflict
    for epoch in range(first_epoch, num_train_epochs):
        controlnet.train()
        unet.train()  # ADDED: Train UNet along with ControlNet
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # batch is a dictionary with video_frames and controlnet_condition
                # Debug: Check input dimensions and order from dataloader (commented out for cleaner output)
                # print(f"[Rank {accelerator.process_index}] Input shapes from dataloader:")
                # print(f"  video_frames shape: {batch['video_frames'].shape} (expect [B, T=16, C=3, H=480, W=480])")
                # print(f"  controlnet_condition shape: {batch['controlnet_condition'].shape}")
                
                use_precomputed = "video_latents" in batch

                if use_precomputed:
                    # Fast path: VAE latents for video and controlnet heatmap were
                    # precomputed by build_dualview_dataset --vae-encode. First frame
                    # is still raw pixels (needed for CLIP + conditional-frame noise aug).
                    latents = batch["video_latents"].to(weight_dtype).to(accelerator.device, non_blocking=True)                 # (B, T, 4, H/8, W/8) already scaled
                    controlnet_latents_in = batch["controlnet_latents"].to(weight_dtype).to(accelerator.device, non_blocking=True)  # (B, T, 4, H/8, W/8) unscaled
                    first_frame = batch["first_frame"].to(weight_dtype).to(accelerator.device, non_blocking=True)               # (B, 3, H, W) in [-1, 1]
                    video_frames = None
                    condition_img = None
                else:
                    if batch["video_frames"].shape[1] != 16:
                        print(f"[Rank {accelerator.process_index}] ERROR at step {global_step}: batch['video_frames'] dim 1 is {batch['video_frames'].shape[1]} instead of 16!")
                    if batch["controlnet_condition"].shape[1] != 16:
                        print(f"[Rank {accelerator.process_index}] ERROR at step {global_step}: batch['controlnet_condition'] dim 1 is {batch['controlnet_condition'].shape[1]} instead of 16!")

                    video_frames = batch["video_frames"].to(weight_dtype).to(accelerator.device, non_blocking=True)        # [-1, 1] range
                    condition_img = batch["controlnet_condition"].to(weight_dtype).to(accelerator.device, non_blocking=True)        # [0, 1] range
                    first_frame = video_frames[:, 0, :, :, :]

                    # Images to VAE latent space
                    latents = tensor_to_vae_latent(video_frames, vae)       # For all frames
                    controlnet_latents_in = None

                reflected_motion_bucket_id = batch["reflected_motion_bucket_id"]
                controlnet_image_index = batch["controlnet_image_index"]
                prompt = batch["prompt"].to(accelerator.device)


                ##################################### Add Noise ########################################
                bsz, num_frames = latents.shape[:2]


                # Encode the first frame
                conditional_pixel_values = first_frame      # First frame (raw pixels, [-1, 1])
                # Following AnimateSomething, we use constant to repace cond_sigmas
                conditional_pixel_values = conditional_pixel_values + torch.randn_like(conditional_pixel_values) * train_noise_aug_strength # cond_sigmas
                conditional_latents = vae.encode(conditional_pixel_values).latent_dist.mode()
                conditional_latents = repeat(conditional_latents, 'b c h w->b f c h w', f=num_frames)       # conditional_latents没有noise的成分的


                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = rand_log_normal(shape=[bsz,], loc=config["noise_mean"], scale=config["noise_std"]).to(weight_dtype).to(latents.device)
                sigmas = torch.clamp(sigmas, min=1e-4, max=10.0)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + torch.randn_like(latents) * sigmas
                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)        # multiplied by c_in in paper


                # For the encoder hidden states based on the first frame and prompt
                encoder_hidden_states = encode_clip(first_frame.float(), prompt)     # First Frame + Text Prompt


                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if conditioning_dropout_prob != 0:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)

                    # Sample masks for the encoder_hidden_states (to replace prompts in InstructPix2Pix). 
                    prompt_mask = random_p < 2 * conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final encoder_hidden_states conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states) # encoder_hidden_states has already been used with .unsqueeze(1)
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states) 

                    # Sample masks for the original image latents.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - ((random_p >= conditioning_dropout_prob).to(image_mask_dtype) * (random_p < 3 * conditioning_dropout_prob).to(image_mask_dtype))
                    # Reshape to match conditional_latents shape (batch, frames, channels, height, width)
                    image_mask = image_mask.reshape(bsz, 1, 1, 1, 1)

                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents   

                # The Concatenation is move downward with the masking feature


                # GT noise
                target = latents
                ##########################################################################################


                ################################ Other Embedding and Conditioning ###################################
                reflected_motion_bucket_id = torch.sum(reflected_motion_bucket_id)/len(reflected_motion_bucket_id)
                reflected_motion_bucket_id = int(reflected_motion_bucket_id.cpu().detach().numpy())
                # print("Training reflected_motion_bucket_id is ", reflected_motion_bucket_id)

                added_time_ids = get_add_time_ids(
                                                    unet_config,
                                                    expected_add_embed_dim,
                                                    process_fps,
                                                    reflected_motion_bucket_id,
                                                    train_noise_aug_strength,       # Note: noise strength
                                                    weight_dtype,
                                                    bsz,                            # runtime batch size (handles partial last batch)
                                                    num_videos_per_prompt,
                                                )       # The same as SVD pipeline's _get_add_time_ids
                added_time_ids = added_time_ids.to(accelerator.device)

                timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)
                ##########################################################################################



                ################################### Get ControlNet Output ###################################

                # Transform controlnet_image_index to the data format we want
                controlnet_image_index = list(controlnet_image_index.cpu().detach().numpy()[0])
                assert latents.shape[1] >= len(controlnet_image_index)

                # Designing the 0/1 mask for Sparse Conditioning
                # NOTE: currently dead — not consumed by controlnet() below. Kept for
                # parity with original code. Built from latents.shape so both paths
                # (precomputed latent / legacy pixel) produce the same-shape tensor.
                _b, _t = latents.shape[0], latents.shape[1]
                controlnet_conditioning_mask_shape = [_b, _t, 1, latents.shape[-2], latents.shape[-1]]
                controlnet_conditioning_mask = torch.zeros(controlnet_conditioning_mask_shape).to(dtype=weight_dtype).to(accelerator.device)
                controlnet_conditioning_mask[:, controlnet_image_index] = 1


                # Add vae latent mask to controlnet noise
                if config["mask_controlnet_vae"]:
                    b, f, c, h, w = conditional_latents.shape

                    # Create a mask: Value less than the threshold is set to be True
                    mask = torch.rand((b, f, 1, h, w), device=accelerator.device) < (1-config["mask_proportion"])      # channel sync
                    # mask[:,0,:,:,:] = 1     # For the first frame, we still keep it

                    # Multiply to the conditional latents, we will just make the mean and variance zero to present those with zero masking
                    masked_conditional_latents = conditional_latents * mask
                    controlnet_inp_noisy_latents = torch.cat([inp_noisy_latents, masked_conditional_latents], dim=2)
                else:
                    controlnet_inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)


                # VAE encode (skip if precomputed latents are in the batch)
                if use_precomputed:
                    controlnet_cond = controlnet_latents_in.flatten(0, 1)                 # (B*T, 4, H/8, W/8)
                else:
                    controlnet_cond = condition_img.flatten(0, 1)
                    controlnet_cond = vae.encode(controlnet_cond).latent_dist.mode()


                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample = controlnet_inp_noisy_latents,          
                    timestep = timesteps,
                    encoder_hidden_states = encoder_hidden_states,     
                    added_time_ids = added_time_ids,
                    controlnet_cond = controlnet_cond,
                    return_dict = False,
                    conditioning_scale = config["outer_conditioning_scale"],
                    inner_conditioning_scale = config["inner_conditioning_scale"],
                    guess_mode = False,         # No Guess Mode
                )   

                #############################################################################################



                ###################################### Predict Noise ########################################
                # Add vae latent mask to controlnet noise
                if config["mask_unet_vae"]:
                    b, f, c, h, w = conditional_latents.shape

                    # Create a mask
                    mask = torch.rand((b, f, 1, h, w), device=accelerator.device) < (1-config["mask_proportion"])      # channel sync
                    # mask[:,0,:,:,:] = 1     # For the first frame, we still keep it

                    # Multiply to the conditional latents, we will just make the mean and variance zero to present those with zero masking
                    if not config["mask_controlnet_vae"]:  
                        masked_conditional_latents = conditional_latents * mask
                    unet_inp_noisy_latents = torch.cat([inp_noisy_latents, masked_conditional_latents], dim=2)
                else:
                    unet_inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

                # Add with controlnet middle output layers
                model_pred = unet(
                                    unet_inp_noisy_latents,
                                    timesteps, 
                                    encoder_hidden_states, 
                                    added_time_ids = added_time_ids,
                                    down_block_additional_residuals = [
                                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                                    ],
                                    mid_block_additional_residual = mid_block_res_sample.to(dtype=weight_dtype),
                                ).sample    


                # Denoise the latents
                eps = 1e-6
                sigmas_float = sigmas.float()
                c_out = -sigmas_float / ((sigmas_float**2 + 1).sqrt() + eps)
                c_out = c_out.to(weight_dtype)
                c_skip = 1 / (sigmas_float**2 + 1 + eps)
                c_skip = c_skip.to(weight_dtype)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents

                # Loss weighing (matching action model)
                sigma_inv_sq = torch.clamp(1.0 / (sigmas_float**2 + eps), min=0.0, max=1e6)
                weighing = (1 + sigmas**2) * sigma_inv_sq
                weighing = torch.clamp(weighing, min=0.0, max=1000.0)
                ##########################################################################################


                ############################### Calculate Loss and Update Optimizer #######################
                loss_components = (weighing.float() * (denoised_latents.float() - target.float())**2).reshape(target.shape[0], -1)
                loss_components = torch.nan_to_num(loss_components, nan=0.0, posinf=1000.0, neginf=0.0)
                loss = torch.mean(loss_components, dim=1).mean()
                loss = torch.clamp(loss, min=0.0, max=1000.0)
                if torch.isnan(loss) or torch.isinf(loss):
                    if accelerator.is_main_process:
                        print(f"⚠️  NaN/Inf loss at step {global_step}, replacing with 1.0")
                    loss = torch.tensor(1.0, device=loss.device, dtype=loss.dtype)
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Update Tensorboard
                writer.add_scalar('Loss/train-Loss-Step', avg_loss.item()/ gradient_accumulation_steps, global_step)        # 我觉得loss的report就用这个avg_loss就行了
                
                # Enhanced WandB logging
                if report_to == "wandb" and is_wandb_available() and accelerator.is_main_process:
                    log_dict = {
                        "train/loss_step": avg_loss.item() / gradient_accumulation_steps,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }
                    
                    # Log gradient norms periodically
                    if global_step % config.get("wandb_log_interval", 100) == 0:
                        grad_norm = 0.0
                        for p in controlnet.parameters():
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        # Check if UNet has trainable params
                        if any(p.requires_grad for p in unet.parameters()):
                            for p in unet.parameters():
                                if p.grad is not None:
                                    grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        log_dict["train/gradient_norm"] = grad_norm
                    
                    wandb.log(log_dict, step=global_step)

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:  # For ControlNet and UNet
                    # Check if UNet has trainable params
                    if any(p.requires_grad for p in unet.parameters()):
                        # Clip gradients for both models
                        params_to_clip = list(controlnet.parameters()) + list(unet.parameters())
                    else:
                        params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()     # Apply cosine decay to learning rate
                optimizer.zero_grad(set_to_none=True)
                ##########################################################################################


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            # Update Log
            logs = {"step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            ########################################## Checkpoints #########################################
            # Outside sync_gradients so global_step alignment is correct
            if global_step != 0 and global_step % checkpointing_steps == 0:
                if accelerator.is_main_process:
                    start = time.time()
                    if checkpoints_total_limit is not None:
                        os.makedirs(output_dir, exist_ok=True)
                        checkpoints = os.listdir(output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                            for removing_checkpoint in removing_checkpoints:
                                shutil.rmtree(os.path.join(output_dir, removing_checkpoint))

                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    unwrapped_unet = accelerator.unwrap_model(unet)
                    unwrapped_unet.save_pretrained(os.path.join(save_path, "unet"))

                    unwrapped_controlnet = accelerator.unwrap_model(controlnet)
                    unwrapped_controlnet.save_pretrained(os.path.join(save_path, "controlnet"))

                    import yaml
                    with open(os.path.join(save_path, "training_config.yaml"), "w") as cf:
                        yaml.dump(config, cf)

                    logger.info(f"Saved state to {save_path}")
                    print("Save time use " + str(time.time() - start) + " s")
                    torch.cuda.empty_cache()
            ########################################################################################################


           ##################################### Validation per XXX iterations #######################################
            # Run validation on ALL processes to avoid deadlock.
            # global_step is a micro-iteration counter; validation_step is given
            # in optimizer steps, so scale by gradient_accumulation_steps.
            _val_period = validation_step * gradient_accumulation_steps
            if global_step > -1 and global_step % _val_period == 0:
                # Each GPU handles a subset of validation samples
                log_validation(
                                vae,
                                unet,
                                controlnet,
                                image_encoder,
                                text_encoder,
                                tokenizer,
                                config,
                                accelerator,
                                weight_dtype,
                                global_step,
                                val_dataset=val_dataset,  # Pass the pre-loaded validation dataset
                                use_ambiguous_prompt = config["mix_ambiguous"],
                                config_name=config_name,
                                timestamp=timestamp,
                            )
                                
            ###############################################################################################################

            # Update Steps and Break if needed      global step should be updated together
            global_step += 1

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    ############################################################################################################################


if __name__ == "__main__":
    args = parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    main(config)
