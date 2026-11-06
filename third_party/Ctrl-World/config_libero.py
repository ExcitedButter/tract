import torch
from dataclasses import dataclass


@dataclass
class wm_args:
    ########################### training args ##############################
    svd_model_path = "checkpoints/svd"
    clip_model_path = "checkpoints/clip-vit-base-patch32"
    ckpt_path = "checkpoints/ctrl-world/checkpoint-10000.pt"

    dataset_root_path = "dataset_example"
    dataset_names = "libero"
    dataset_meta_info_path = "dataset_meta_info"
    dataset_cfgs = dataset_names
    prob = [1.0]
    annotation_name = "annotation"
    num_workers = 4
    down_sample = 1
    skip_step = 1

    debug = False
    tag = "libero_ctrlworld"
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "ctrl_world_libero"

    learning_rate = 5e-6
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"
    train_batch_size = 4
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 100000
    checkpointing_steps = 10000
    validation_steps = 500
    max_grad_norm = 1.0
    video_num = 4

    ############################ model args ##############################
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 1.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    num_frames = 16
    num_history = 1
    action_dim = 7
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

    data_stat_path = "dataset_meta_info/libero/stat.json"
    val_model_path = ckpt_path
    history_idx = [0]

    save_dir = "synthetic_traj"

    def __post_init__(self):
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.08)
        self.z_min = self.z_min_dict.get(self.task_type, 0.18)
        self.task_name = f"Rollouts_libero_{self.task_type}"

        if self.task_type == "replay":
            self.val_dataset_dir = "dataset_example/libero"
            self.val_id = []
            self.start_idx = []
            self.instruction = []
