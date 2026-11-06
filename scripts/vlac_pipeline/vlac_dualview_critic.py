"""Dual-view VLAC critic.

Scores agentview (third-person) and wrist (eye-in-hand) frames independently
using a single VLACCritic model, then combines them via a weighted sum.
Default weighting favors the agentview (0.6) over the wrist (0.4), since the
third-person camera captures global task progress while the wrist camera
provides fine-grained manipulation detail.

If one view's score is -inf (scoring failure), the result falls back to the
other view's score alone.
"""

import logging
import math

from .vlac_critic import VLACCritic

logger = logging.getLogger(__name__)


class VLACDualViewCritic:
    """Dual-view VLAC critic. Scores agentview and wrist independently, combines."""

    def __init__(self, model_path=None, agent_weight=0.6, wrist_weight=0.4):
        self.critic = VLACCritic(model_path=model_path) if model_path else VLACCritic()
        self.agent_weight = agent_weight
        self.wrist_weight = wrist_weight

    def load(self, device="cuda"):
        self.critic.load(device=device)

    @property
    def ready(self):
        return self.critic.ready

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _combine(self, agent_score: float, wrist_score: float) -> float:
        """Weighted combination with fallback on -inf."""
        agent_ok = math.isfinite(agent_score)
        wrist_ok = math.isfinite(wrist_score)

        if agent_ok and wrist_ok:
            return self.agent_weight * agent_score + self.wrist_weight * wrist_score
        if agent_ok:
            logger.warning("Wrist score is -inf, using agentview only")
            return agent_score
        if wrist_ok:
            logger.warning("Agentview score is -inf, using wrist only")
            return wrist_score

        return float("-inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_dualview(self, agentview_frames, wrist_frames, task_description,
                       ref_image_list=None, ref_num=0):
        """Score both views, return weighted combination.

        If one view fails (-inf), falls back to the other view's score.

        Args:
            agentview_frames: list of PIL.Image for the third-person view.
            wrist_frames: list of PIL.Image for the eye-in-hand view.
            task_description: task instruction string.
            ref_image_list: optional reference images forwarded to the critic.
            ref_num: number of reference images.

        Returns:
            float: combined score (0-100 range). -inf if both views fail.
        """
        agent_score = self.critic.score_frames(
            agentview_frames, task_description,
            ref_image_list=ref_image_list, ref_num=ref_num,
        )
        wrist_score = self.critic.score_frames(
            wrist_frames, task_description,
            ref_image_list=ref_image_list, ref_num=ref_num,
        )
        combined = self._combine(agent_score, wrist_score)
        logger.debug(
            "Dual-view scores — agent=%.2f  wrist=%.2f  combined=%.2f",
            agent_score, wrist_score, combined,
        )
        return combined

    def score_dualview_batch(self, agentview_list, wrist_list, task_description,
                             ref_image_list=None, ref_num=0):
        """Score B videos with both views.

        Args:
            agentview_list: list of B frame-lists (each a list of PIL Images).
            wrist_list: list of B frame-lists.
            task_description: task instruction string.
            ref_image_list: optional reference images forwarded to the critic.
            ref_num: number of reference images.

        Returns:
            list of B combined scores.
        """
        agent_scores = self.critic.score_batch(
            agentview_list, task_description,
            ref_image_list=ref_image_list, ref_num=ref_num,
        )
        wrist_scores = self.critic.score_batch(
            wrist_list, task_description,
            ref_image_list=ref_image_list, ref_num=ref_num,
        )
        return [
            self._combine(a, w)
            for a, w in zip(agent_scores, wrist_scores)
        ]
