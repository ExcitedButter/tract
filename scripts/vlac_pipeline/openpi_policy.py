"""OpenPI policy wrapper for VLAC reranking pipeline.

Default: ``pi05_libero_stage2p5_joint`` (LIBERO mesh + track policy).
Also supports real-world configs when passed via ``config_name``.
"""

import logging
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Repo layout: <openpi>/src/tract/...  (not <openpi>/openpi/src)
OPENPI_SRC = os.path.join(PROJECT_ROOT, "src")

logger = logging.getLogger(__name__)


class OpenPIPolicy:
    def __init__(
        self,
        config_name: str = "pi05_libero_stage2p5_joint",
        checkpoint_dir: str = None,
        mem_fraction: float = 0.4,
    ):
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            PROJECT_ROOT,
            "checkpoints",
            "pi05_libero_stage2p5_joint",
            "pi05_final",
            "29999",
        )
        # JAX/XLA GPU memory fraction. Defaults to 40% to leave room for PyTorch
        # models (WM, VLAC) that share the same GPU.
        self.mem_fraction = mem_fraction
        self._policy = None

    def load(self, device: str = "cuda"):
        if OPENPI_SRC not in sys.path:
            sys.path.insert(0, OPENPI_SRC)

        # Cap JAX/XLA GPU memory to leave room for PyTorch models.
        # Must be set BEFORE any jax import.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(self.mem_fraction))
        os.environ.setdefault("JAX_PLATFORMS", device if device == "cpu" else "")
        os.environ.setdefault("HF_LEROBOT_HOME", os.path.join(PROJECT_ROOT, "lerobot_cache"))

        import dataclasses
        from tract.training import config as _config
        from tract.policies import policy_config as _policy_config

        cfg = _config.get_config(self.config_name)

        # For eval inference, disable include_track_targets — it requires ground-truth
        # future track_targets_raw which we don't have at inference time.
        # The model still predicts tracks via its track head; we just skip the loss target.
        if hasattr(cfg.data, "include_track_targets"):
            eval_cfg = dataclasses.replace(
                cfg, data=dataclasses.replace(cfg.data, include_track_targets=False)
            )
        else:
            eval_cfg = cfg

        eval_data_cfg = eval_cfg.data.create(eval_cfg.assets_dirs, eval_cfg.model)
        self._policy = _policy_config.create_trained_policy(
            eval_cfg,
            self.checkpoint_dir,
            repack_transforms=eval_data_cfg.repack_transforms,
        )

        # Detect whether the loaded model has a track head.
        # pi05_realworld_ee_squarecrop has track_head_dim=None (actions only),
        # while pi05_realworld_track_joint has track_head_dim=78.
        model_cfg = getattr(cfg, "model", None)
        self.has_track_head = getattr(model_cfg, "track_head_dim", None) is not None
        logger.info(
            "OpenPI policy loaded: %s from %s (track_head=%s)",
            self.config_name, self.checkpoint_dir, self.has_track_head,
        )

    @property
    def ready(self) -> bool:
        return self._policy is not None

    def infer(self, observation: dict, return_tracks: bool | None = None) -> dict:
        """Run single inference. Returns dict with 'actions' and optionally 'track_predictions'.

        If return_tracks is None (default), auto-detects based on whether the
        loaded model has a track head configured.
        """
        if return_tracks is None:
            return_tracks = self.has_track_head
        return self._policy.infer(observation, return_tracks=return_tracks)

    def generate_candidates(
        self,
        observation: dict,
        n_candidates: int = 5,
        noise_scale: float = 1.0,
        noise_type: str = "random",
        num_steps: int = None,
        sampling_mode: str = "ode",
        sde_noise_level: float = 0.0,
        tsr_k: float = 1.2,
        tsr_sigma: float = 0.25,
        state_noise_std: float = 0.2,
        time_gamma: float = 1.0,
        sigma_step_max: float = 0.0,
        image_noise_std: float = 0.0,
        tsr_anneal: str = "none",
    ) -> list:
        """Generate n candidates in a single batched forward pass.

        Tiles the observation N times along the batch dimension so the prefix
        KV cache is computed once (batched) and N different noise samples are
        denoised in parallel, instead of N sequential infer() calls.

        Each candidate is a dict with:
            - "actions": (16, 10) numpy array of 10D absolute EE actions
            - "track_predictions": (16, 78) numpy array of track predictions
        """
        import jax
        import jax.numpy as jnp
        from tract.models import model as _model

        policy = self._policy
        return_tracks = self.has_track_head

        # 1. Apply input transforms once
        inputs = jax.tree.map(lambda x: x, observation)
        inputs = policy._input_transform(inputs)

        # 2. Tile observation N times along batch dim
        inputs_batched = jax.tree.map(
            lambda x: jnp.repeat(jnp.asarray(x)[None, ...], n_candidates, axis=0),
            inputs,
        )

        # 2b. Add independent Gaussian noise to each candidate's images
        if image_noise_std > 0.0:
            policy._rng, img_rng = jax.random.split(policy._rng)
            for key in list(inputs_batched.keys()):
                if "image" not in key:
                    continue
                img_batch = inputs_batched[key]
                if not isinstance(img_batch, jnp.ndarray):
                    continue
                if img_batch.ndim < 3:
                    continue
                img_rng, noise_rng = jax.random.split(img_rng)
                noise_img = image_noise_std * jax.random.normal(noise_rng, img_batch.shape)
                inputs_batched[key] = jnp.clip(img_batch + noise_img, 0.0, 255.0)

        # 3. Generate N action noise samples
        policy._rng, noise_rng, sample_rng = jax.random.split(policy._rng, 3)
        model = policy._model
        if noise_type == "sobol":
            from scipy.stats import qmc, norm as sp_norm
            total_dim = model.action_horizon * model.action_dim
            sampler = qmc.Sobol(d=total_dim, scramble=True, seed=int(jax.random.bits(noise_rng)))
            sobol_uniform = sampler.random(n_candidates)
            sobol_normal = sp_norm.ppf(np.clip(sobol_uniform, 1e-6, 1 - 1e-6))
            noise = jnp.array(
                sobol_normal.reshape(n_candidates, model.action_horizon, model.action_dim).astype(np.float32)
            ) * noise_scale
        else:
            noise = jax.random.normal(
                noise_rng, (n_candidates, model.action_horizon, model.action_dim),
            ) * noise_scale

        # 4. Single batched forward pass
        sample_kwargs = dict(policy._sample_kwargs)
        sample_kwargs["noise"] = noise
        sample_kwargs["return_tracks"] = return_tracks
        if num_steps is not None:
            sample_kwargs["num_steps"] = num_steps
        sample_kwargs["sampling_mode"] = sampling_mode
        sample_kwargs["sde_noise_level"] = sde_noise_level
        sample_kwargs["tsr_k"] = tsr_k
        sample_kwargs["tsr_sigma"] = tsr_sigma
        sample_kwargs["time_gamma"] = time_gamma
        sample_kwargs["sigma_step_max"] = sigma_step_max
        sample_kwargs["tsr_anneal"] = tsr_anneal
        obs_batch = _model.Observation.from_dict(inputs_batched)
        raw_output = policy._sample_actions(sample_rng, obs_batch, **sample_kwargs)

        # 5. Split into N candidates, apply output transforms per candidate
        is_dict = isinstance(raw_output, dict)
        candidates = []
        for i in range(n_candidates):
            out_i = {}
            if "state" in inputs_batched:
                out_i["state"] = np.asarray(inputs_batched["state"][i])
            if is_dict:
                for k, v in raw_output.items():
                    out_i[k] = np.asarray(v[i])
            else:
                out_i["actions"] = np.asarray(raw_output[i])
            out_i = policy._output_transform(out_i)
            candidates.append({
                "actions": np.asarray(out_i["actions"]),
                "track_predictions": np.asarray(
                    out_i.get("track_predictions", np.zeros((16, 78)))
                ),
            })
        return candidates
