"""VLAC-8b critic wrapper.

Loads InternVL2-based VLAC model via MS-Swift.
Scores video frames for task progress (0-100 cumulative score).
Accepts PIL frames at any resolution — resizes internally to 448×448.
"""

import logging
import os
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Local install: openpi/VLAC (evo_vlac package + model weights).
VLAC_DIR = os.path.join(PROJECT_ROOT, "third_party", "VLAC")

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "VLAC-8b")


class VLACCritic:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._critic = None

    def load(self, device: str = "cuda"):
        if VLAC_DIR not in sys.path:
            sys.path.insert(0, VLAC_DIR)

        from evo_vlac.utils.model_utils import GAC_model

        critic = GAC_model(tag="critic")
        critic.init_model(
            model_path=self.model_path,
            model_type="internvl2",
            device_map=device,
        )
        critic.temperature = 0.5
        critic.top_k = 1
        critic.set_config()
        critic.set_system_prompt()
        self._critic = critic
        logger.info("VLAC critic loaded from %s", self.model_path)

    @property
    def ready(self) -> bool:
        return self._critic is not None

    def score_frames(self, frames: list, task_description: str,
                     ref_image_list: list = None, ref_num: int = 0) -> float:
        """Score a list of PIL frames for task progress.

        Args:
            frames: List of PIL.Image objects (any resolution, resized internally to 448×448).
            task_description: task instruction string.
            ref_image_list: Optional reference images (e.g. [current_frame, goal_image]).
            ref_num: Number of reference images (0 = no reference).

        Returns:
            float: cumulative progress score (0-100). Higher = more progress toward task.
                   Returns -inf on failure.
        """
        if not self.ready:
            return float("-inf")
        try:
            critic_list, value_list = self._critic.get_trajectory_critic(
                task=task_description,
                image_list=frames,
                ref_image_list=ref_image_list,
                batch_num=5,
                ref_num=ref_num,
                think=False,
                skip=5,
                rich=False,
                reverse_eval=False,
                frame_skip=True,
                addition_scale=1,
                bias=0,
                related_critic=False,
                positive_clip=0,
                negative_clip=0,
                value_simple=True,
            )
            if not value_list:
                return float("-inf")
            return float(value_list[-1])
        except Exception as exc:
            logger.warning("VLAC scoring failed: %s", exc)
            return float("-inf")

    def score_video(self, video_path: str, task_description: str) -> float:
        """Score a video file (MP4) for task progress.

        Convenience method that extracts frames internally.

        Args:
            video_path: path to MP4 file.
            task_description: task instruction string.

        Returns:
            float: cumulative progress score (0-100).
        """
        if not self.ready:
            return float("-inf")
        try:
            _, value_list, _, _ = self._critic.web_trajectory_critic(
                task_description=task_description,
                main_video_path=str(video_path),
                reference_video_path=None,
                batch_num=5,
                ref_num=0,
                think=False,
                skip=5,
                frame_skip=True,
                done_flag=False,
                video_output=False,
            )
            if not value_list:
                return float("-inf")
            return float(value_list[-1])
        except Exception as exc:
            logger.warning("VLAC video scoring failed: %s", exc)
            return float("-inf")

    def score_batch(self, video_list: list, task_description: str,
                    ref_image_list: list = None, ref_num: int = 0) -> list:
        """Score B videos. Each video is a list of PIL frames.

        Args:
            video_list: list of B frame-lists (each is list of PIL Images).
            task_description: task instruction string.
            ref_image_list: Optional reference images passed to each scoring call.
            ref_num: Number of reference images.

        Returns:
            list of B float scores (0-100). -inf on failure.
        """
        return [self.score_frames(frames, task_description, ref_image_list, ref_num)
                for frames in video_list]
