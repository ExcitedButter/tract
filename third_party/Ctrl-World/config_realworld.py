import torch
import os
import json
from dataclasses import dataclass


@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths
    svd_model_path = "checkpoints/svd"
    clip_model_path = "checkpoints/clip-vit-base-patch32"
    ckpt_path = "model_ckpt/realworld_ctrlworld/checkpoint-10000.pt"

    # dataset parameters
    dataset_root_path = "dataset_example"
    dataset_names = "realworld"
    dataset_meta_info_path = "dataset_meta_info"
    dataset_cfgs = dataset_names
    prob = [1.0]
    annotation_name = "annotation"
    num_workers = 4
    down_sample = 1  # data is already at 10Hz, no further downsampling
    skip_step = 1

    # logs
    debug = False
    tag = "realworld_ctrlworld_v3"
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "ctrl_world_realworld"

    # training parameters
    learning_rate = 5e-6      # lower than default 1e-5 for fine-tuning
    gradient_accumulation_steps = 4
    mixed_precision = "fp16"
    train_batch_size = 4
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 50000
    checkpointing_steps = 5000
    validation_steps = 100
    max_grad_norm = 1.0
    video_num = 4             # number of validation videos to generate

    ############################ model args ##############################
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 1.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    num_frames = 16           # predict 16 future frames (was 5)
    num_history = 1           # 1 history frame (was 6) for fair comparison with track WM
    action_dim = 7            # 7D: [pos(3), axis_angle(3), gripper(1)]
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.bfloat16

    ########################### rollout args ############################
    task_type: str = "replay"
    gripper_max_dict = {"replay": 1.0}
    z_min_dict = {}
    policy_type = "pi05"
    action_adapter = None
    pred_step = 16
    policy_skip_step = 1
    interact_num = 5

    # wm
    data_stat_path = "dataset_meta_info/realworld/stat.json"
    val_model_path = ckpt_path
    history_idx = [0]  # only 1 history frame

    # save
    save_dir = "synthetic_traj"

    def __post_init__(self):
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        self.z_min = self.z_min_dict.get(self.task_type, 0.18)
        self.task_name = f"Rollouts_realworld_{self.task_type}"

        if self.task_type == "replay":
            self.val_dataset_dir = "dataset_example/realworld"
            self.val_id = []
            self.start_idx = []
            self.instruction = []
