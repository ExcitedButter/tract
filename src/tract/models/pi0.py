import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from tract.models import model as _model
from tract.models import pi0_config
import tract.models.gemma as _gemma
import tract.models.siglip as _siglip
from tract.shared import array_typing as at

logger = logging.getLogger("tract")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.track_head_dim = config.track_head_dim
        self.action_loss_weight = config.action_loss_weight
        self.track_loss_weight = config.track_loss_weight
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        if self.track_head_dim is not None:
            self.track_in_proj = nnx.Linear(self.track_head_dim, action_expert_config.width, rngs=rngs)
            if not config.pi05:
                self.track_time_mlp_in = nnx.Linear(
                    2 * action_expert_config.width,
                    action_expert_config.width,
                    rngs=rngs,
                )
                self.track_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.track_out_proj = nnx.Linear(action_expert_config.width, self.track_head_dim, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _suffix_output_slices(
        self, suffix_out: at.Float[at.Array, "b s d"], *, has_tracks: bool
    ) -> tuple[at.Float[at.Array, "b ah d"], at.Float[at.Array, "b ah d"] | None]:
        """Split action-expert outputs. When tracks are present they come before actions."""
        suffix_start = 0 if self.pi05 else 1
        h = self.action_horizon
        if has_tracks:
            track_out = suffix_out[:, suffix_start : suffix_start + h]
            action_out = suffix_out[:, suffix_start + h : suffix_start + 2 * h]
            return action_out, track_out
        action_out = suffix_out[:, suffix_start : suffix_start + h]
        return action_out, None

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        noisy_tracks: at.Float[at.Array, "b ah td"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        track_expert_tokens = None
        if noisy_tracks is not None:
            if self.track_head_dim is None:
                raise ValueError("Track targets were provided but track_head_dim is not configured.")
            track_tokens = self.track_in_proj(noisy_tracks)
            if self.pi05:
                track_expert_tokens = track_tokens
            else:
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=noisy_tracks.shape[1])
                track_time_tokens = jnp.concatenate([track_tokens, time_tokens], axis=-1)
                track_time_tokens = self.track_time_mlp_in(track_time_tokens)
                track_time_tokens = nnx.swish(track_time_tokens)
                track_time_tokens = self.track_time_mlp_out(track_time_tokens)
                track_expert_tokens = track_time_tokens

        # Track tokens are placed before action tokens so cumsum(ar_mask) lets actions attend
        # to tracks while tracks cannot attend to actions (see make_attn_mask).
        if track_expert_tokens is not None:
            tokens.append(track_expert_tokens)
            input_mask.append(jnp.ones(track_expert_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [True] + ([False] * (track_expert_tokens.shape[1] - 1))

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to suffix tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.ActionTargets,
        *,
        train: bool = False,
        return_components: bool = False,
    ) -> at.Float[at.Array, "*b ah"] | dict[str, at.Array]:
        if isinstance(actions, dict):
            action_targets = actions["actions"]
            track_targets = actions.get("track_targets")
        else:
            action_targets = actions
            track_targets = None

        preprocess_rng, noise_rng, track_noise_rng, time_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = action_targets.shape[:-2]
        noise = jax.random.normal(noise_rng, action_targets.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * action_targets
        u_t = noise - action_targets

        x_track_t = None
        u_track_t = None
        if track_targets is not None:
            if self.track_head_dim is None:
                raise ValueError("Received track targets but model was created without a track head.")
            if track_targets.shape[-2] != self.action_horizon:
                raise ValueError(
                    f"Track horizon {track_targets.shape[-2]} does not match action horizon {self.action_horizon}."
                )
            track_noise = jax.random.normal(track_noise_rng, track_targets.shape)
            x_track_t = time_expanded * track_noise + (1 - time_expanded) * track_targets
            u_track_t = track_noise - track_targets

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            x_t,
            time,
            noisy_tracks=x_track_t,
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        del prefix_out
        has_tracks = track_targets is not None
        action_slice, track_slice = self._suffix_output_slices(suffix_out, has_tracks=has_tracks)
        v_t = self.action_out_proj(action_slice)
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        total_loss = self.action_loss_weight * action_loss
        track_loss = jnp.zeros_like(action_loss)
        if track_targets is not None:
            if self.track_loss_weight <= 0:
                if return_components:
                    return {
                        "total_loss": total_loss,
                        "action_loss": action_loss,
                        "track_loss": track_loss,
                    }
                return total_loss
            if self.track_head_dim is None:
                raise ValueError("Track loss requested but track_head_dim is not configured.")
            assert track_slice is not None
            v_track = self.track_out_proj(track_slice)
            track_loss = jnp.mean(jnp.square(v_track - u_track_t), axis=-1)
            total_loss = total_loss + self.track_loss_weight * track_loss

        if return_components:
            return {
                "total_loss": total_loss,
                "action_loss": action_loss,
                "track_loss": track_loss,
            }

        return total_loss

    # @override
    # def sample_actions(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     *,
    #     num_steps: int | at.Int[at.Array, ""] = 10,
    #     noise: at.Float[at.Array, "b ah ad"] | None = None,
    #     return_tracks: bool = False,
    # ) -> _model.Actions:
    #     observation = _model.preprocess_observation(None, observation, train=False)
    #     # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    #     # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    #     dt = -1.0 / num_steps
    #     batch_size = observation.state.shape[0]
    #     if noise is None:
    #         rng, action_noise_rng = jax.random.split(rng)
    #         noise = jax.random.normal(action_noise_rng, (batch_size, self.action_horizon, self.action_dim))

    #     track_noise = None
    #     if return_tracks:
    #         if self.track_head_dim is None:
    #             raise ValueError("return_tracks=True requested but track_head_dim is not configured.")
    #         rng, track_noise_rng = jax.random.split(rng)
    #         track_noise = jax.random.normal(track_noise_rng, (batch_size, self.action_horizon, self.track_head_dim))

    #     # first fill KV cache with a forward pass of the prefix
    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions = jnp.cumsum(prefix_mask, axis=1) - 1
    #     _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    #     def step_actions_only(carry):
    #         x_t, time = carry
    #         suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
    #             observation, x_t, jnp.broadcast_to(time, batch_size)
    #         )
    #         # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
    #         # other
    #         suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #         # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
    #         # prefix tokens
    #         prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    #         # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
    #         # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
    #         full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    #         assert full_attn_mask.shape == (
    #             batch_size,
    #             suffix_tokens.shape[1],
    #             prefix_tokens.shape[1] + suffix_tokens.shape[1],
    #         )
    #         # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
    #         positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    #         (prefix_out, suffix_out), _ = self.PaliGemma.llm(
    #             [None, suffix_tokens],
    #             mask=full_attn_mask,
    #             positions=positions,
    #             kv_cache=kv_cache,
    #             adarms_cond=[None, adarms_cond],
    #         )
    #         assert prefix_out is None
    #         v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    #         return x_t + dt * v_t, time + dt

    #     def step_actions_and_tracks(carry):
    #         x_t, x_track_t, time = carry
    #         suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
    #             observation,
    #             x_t,
    #             jnp.broadcast_to(time, batch_size),
    #             noisy_tracks=x_track_t,
    #         )
    #         suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #         prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    #         full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    #         assert full_attn_mask.shape == (
    #             batch_size,
    #             suffix_tokens.shape[1],
    #             prefix_tokens.shape[1] + suffix_tokens.shape[1],
    #         )
    #         positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    #         (prefix_out, suffix_out), _ = self.PaliGemma.llm(
    #             [None, suffix_tokens],
    #             mask=full_attn_mask,
    #             positions=positions,
    #             kv_cache=kv_cache,
    #             adarms_cond=[None, adarms_cond],
    #         )
    #         assert prefix_out is None
    #         suffix_start = 0 if self.pi05 else 1
    #         action_slice = suffix_out[:, suffix_start : suffix_start + self.action_horizon]
    #         track_slice = suffix_out[
    #             :,
    #             suffix_start + self.action_horizon : suffix_start + (2 * self.action_horizon),
    #         ]
    #         v_t = self.action_out_proj(action_slice)
    #         v_track_t = self.track_out_proj(track_slice)
    #         return x_t + dt * v_t, x_track_t + dt * v_track_t, time + dt

    #     def cond(carry):
    #         time = carry[-1]
    #         # robust to floating-point error
    #         return time >= -dt / 2

    #     if track_noise is not None:
    #         x_0, x_track_0, _ = jax.lax.while_loop(cond, step_actions_and_tracks, (noise, track_noise, 1.0))
    #         return {"actions": x_0, "track_predictions": x_track_0}

    #     x_0, _ = jax.lax.while_loop(cond, step_actions_only, (noise, 1.0))
    #     return x_0
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        return_tracks: bool = False,
        sampling_mode: str = "ode",
        sde_noise_level: float = 0.0,
        tsr_k: float = 1.0,
        tsr_sigma: float = 0.25,
        time_gamma: float = 1.0,
        sigma_step_max: float = 0.0,
        tsr_anneal: str = "none",
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            if return_tracks:
                rng, action_noise_rng = jax.random.split(rng)
            else:
                action_noise_rng = rng
            noise = jax.random.normal(action_noise_rng, (batch_size, self.action_horizon, self.action_dim))

        track_noise = None
        if return_tracks:
            if self.track_head_dim is None:
                raise ValueError("return_tracks=True requested but track_head_dim is not configured.")
            rng, track_noise_rng = jax.random.split(rng)
            track_noise = jax.random.normal(track_noise_rng, (batch_size, self.action_horizon, self.track_head_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def _get_velocity(x_t, time):
            """Shared velocity computation for all sampling modes."""
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step_actions_only(carry):
            x_t, time = carry
            v_t = _get_velocity(x_t, time)
            return x_t + dt * v_t, time + dt

        def step_actions_tsr(carry):
            x_t, time = carry
            v_t = _get_velocity(x_t, time)
            v_t = _apply_tsr(v_t, x_t, time)
            return x_t + dt * v_t, time + dt

        def step_actions_noisy_ode(carry):
            x_t, time, step_rng = carry
            v_t = _get_velocity(x_t, time)
            # Standard ODE step
            x_next = x_t + dt * v_t
            # Add decaying noise — strong at t=1, zero at t=0
            # Generate INDEPENDENT noise per batch element using split keys
            step_rng, noise_rng = jax.random.split(step_rng)
            per_batch_keys = jax.random.split(noise_rng, batch_size)
            eps = jax.vmap(lambda k: jax.random.normal(k, x_t.shape[1:]))(per_batch_keys)
            noise_magnitude = sde_noise_level * time * time * jnp.sqrt(jnp.abs(dt))
            x_next = x_next + noise_magnitude * eps
            return x_next, time + dt, step_rng

        def step_actions_sde(carry):
            x_t, time, step_rng = carry
            v_t = _get_velocity(x_t, time)

            # SDE: convert ODE to stochastic (from pi_RL paper)
            sigma_t = sde_noise_level * jnp.sqrt(time / jnp.maximum(1.0 - time, 1e-6))
            score = -(x_t + (1.0 - time) * v_t) / jnp.maximum(time, 1e-6)
            drift_correction = (sigma_t**2 / 2.0) * score

            step_rng, noise_rng = jax.random.split(step_rng)
            eps = jax.random.normal(noise_rng, x_t.shape)
            x_next = x_t + dt * (v_t + drift_correction) + sigma_t * jnp.sqrt(jnp.abs(dt)) * eps

            return x_next, time + dt, step_rng

        def step_actions_howard(carry, t_pair):
            """Howard's recommended fix: ODE step + annealed noise sigma_step(t) = sigma_max * sqrt(t)."""
            x_t, step_rng = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_t = _get_velocity(x_t, t_curr)
            x_next = x_t + dt_i * v_t
            step_rng, noise_rng = jax.random.split(step_rng)
            eps = jax.random.normal(noise_rng, x_t.shape)
            noise_mag = sigma_step_max * jnp.sqrt(jnp.maximum(t_curr, 1e-8)) * jnp.sqrt(jnp.abs(dt_i))
            x_next = x_next + noise_mag * eps
            return (x_next, step_rng), None

        def step_actions_scan_ode(carry, t_pair):
            """Deterministic ODE step for scan-based (gamma time grid) integration."""
            x_t = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_t = _get_velocity(x_t, t_curr)
            return x_t + dt_i * v_t, None

        def _annealed_k(t_curr):
            """Compute effective TSR k at time t, with optional annealing."""
            if tsr_anneal == "none":
                return tsr_k
            elif tsr_anneal == "linear":
                # k(t) = 1 + (k_max - 1) * t  — full TSR at t=1, standard ODE at t=0
                return 1.0 + (tsr_k - 1.0) * t_curr
            elif tsr_anneal == "cosine":
                # Smooth cosine decay from k_max to 1.0
                return 1.0 + (tsr_k - 1.0) * 0.5 * (1.0 + jnp.cos(jnp.pi * (1.0 - t_curr)))
            elif tsr_anneal == "sqrt":
                # sqrt decay — holds diversity longer then drops
                return 1.0 + (tsr_k - 1.0) * jnp.sqrt(t_curr)
            elif tsr_anneal == "early_half":
                # Full TSR for t > 0.5, standard ODE for t <= 0.5
                return jnp.where(t_curr > 0.5, tsr_k, 1.0)
            elif tsr_anneal == "early_third":
                # Full TSR for t > 0.67, standard ODE for t <= 0.67
                return jnp.where(t_curr > 0.67, tsr_k, 1.0)
            elif tsr_anneal == "quadratic":
                # k(t) = 1 + (k_max - 1) * t^2  — gentle start, rapid drop near t=0
                return 1.0 + (tsr_k - 1.0) * t_curr * t_curr
            else:
                return tsr_k

        def _apply_tsr(v_t, x_t, t_curr):
            """Apply TSR velocity rescaling with annealed k."""
            k_eff = _annealed_k(t_curr)
            eta_t = ((1.0 - t_curr) / jnp.maximum(t_curr, 1e-6)) ** 2
            r_t = (eta_t * tsr_sigma**2 + 1.0) / (eta_t * tsr_sigma**2 / k_eff + 1.0)
            score_part = (1.0 - t_curr) * v_t + x_t
            return (1.0 / jnp.maximum(1.0 - t_curr, 1e-6)) * (r_t * score_part - x_t)

        def step_actions_scan_tsr(carry, t_pair):
            """TSR step for scan-based (gamma time grid) integration."""
            x_t = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_t = _get_velocity(x_t, t_curr)
            v_t = _apply_tsr(v_t, x_t, t_curr)
            return x_t + dt_i * v_t, None

        def step_actions_scan_tsr_noisy(carry, t_pair):
            """TSR + annealed noise injection combined."""
            x_t, step_rng = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_t = _get_velocity(x_t, t_curr)
            v_t = _apply_tsr(v_t, x_t, t_curr)
            x_next = x_t + dt_i * v_t
            step_rng, noise_rng = jax.random.split(step_rng)
            eps = jax.random.normal(noise_rng, x_t.shape)
            noise_mag = sigma_step_max * jnp.sqrt(jnp.maximum(t_curr, 1e-8)) * jnp.sqrt(jnp.abs(dt_i))
            x_next = x_next + noise_mag * eps
            return (x_next, step_rng), None

        def step_actions_scan_sde(carry, t_pair):
            """SDE step for scan-based (gamma time grid) integration."""
            x_t, step_rng = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_t = _get_velocity(x_t, t_curr)
            sigma_t = sde_noise_level * jnp.sqrt(t_curr / jnp.maximum(1.0 - t_curr, 1e-6))
            score = -(x_t + (1.0 - t_curr) * v_t) / jnp.maximum(t_curr, 1e-6)
            drift_correction = (sigma_t**2 / 2.0) * score
            step_rng, noise_rng = jax.random.split(step_rng)
            eps = jax.random.normal(noise_rng, x_t.shape)
            x_next = x_t + dt_i * (v_t + drift_correction) + sigma_t * jnp.sqrt(jnp.abs(dt_i)) * eps
            return (x_next, step_rng), None

        def _get_velocity_with_tracks(x_t, x_track_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size),
                noisy_tracks=x_track_t,
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
            )
            action_slice, track_slice = self._suffix_output_slices(suffix_out, has_tracks=True)
            assert track_slice is not None
            v_action = self.action_out_proj(action_slice)
            v_track = self.track_out_proj(track_slice)
            return v_action, v_track

        def step_actions_and_tracks(carry):
            x_t, x_track_t, time = carry
            v_action, v_track = _get_velocity_with_tracks(x_t, x_track_t, time)
            return x_t + dt * v_action, x_track_t + dt * v_track, time + dt

        def step_tracks_tsr(carry):
            x_t, x_track_t, time = carry
            v_action, v_track = _get_velocity_with_tracks(x_t, x_track_t, time)
            v_action = _apply_tsr(v_action, x_t, time)
            v_track = _apply_tsr(v_track, x_track_t, time)
            return x_t + dt * v_action, x_track_t + dt * v_track, time + dt

        def step_tracks_noisy_ode(carry):
            x_t, x_track_t, time, step_rng = carry
            v_action, v_track = _get_velocity_with_tracks(x_t, x_track_t, time)
            x_action_next = x_t + dt * v_action
            x_track_next = x_track_t + dt * v_track
            noise_mag = sde_noise_level * time * time * jnp.sqrt(jnp.abs(dt))
            step_rng, noise_rng = jax.random.split(step_rng)
            x_action_next = x_action_next + noise_mag * jax.random.normal(noise_rng, x_t.shape)
            x_track_next = x_track_next + noise_mag * jax.random.normal(noise_rng, x_track_t.shape)
            return x_action_next, x_track_next, time + dt, step_rng

        def step_tracks_sde(carry):
            x_t, x_track_t, time, step_rng = carry
            v_action, v_track = _get_velocity_with_tracks(x_t, x_track_t, time)
            sigma_t = sde_noise_level * jnp.sqrt(time / jnp.maximum(1.0 - time, 1e-6))
            score_a = -(x_t + (1.0 - time) * v_action) / jnp.maximum(time, 1e-6)
            drift_a = (sigma_t**2 / 2.0) * score_a
            score_tr = -(x_track_t + (1.0 - time) * v_track) / jnp.maximum(time, 1e-6)
            drift_tr = (sigma_t**2 / 2.0) * score_tr
            step_rng, noise_rng = jax.random.split(step_rng)
            sqrt_dt = jnp.sqrt(jnp.abs(dt))
            x_action_next = x_t + dt * (v_action + drift_a) + sigma_t * sqrt_dt * jax.random.normal(noise_rng, x_t.shape)
            x_track_next = x_track_t + dt * (v_track + drift_tr) + sigma_t * sqrt_dt * jax.random.normal(noise_rng, x_track_t.shape)
            return x_action_next, x_track_next, time + dt, step_rng

        def step_tracks_scan_ode(carry, t_pair):
            x_t, x_track_t = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_action, v_track = _get_velocity_with_tracks(x_t, x_track_t, t_curr)
            return (x_t + dt_i * v_action, x_track_t + dt_i * v_track), None

        def step_tracks_scan_tsr(carry, t_pair):
            x_t, x_track_t = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_action, v_track = _get_velocity_with_tracks(x_t, x_track_t, t_curr)
            v_action = _apply_tsr(v_action, x_t, t_curr)
            v_track = _apply_tsr(v_track, x_track_t, t_curr)
            return (x_t + dt_i * v_action, x_track_t + dt_i * v_track), None

        def step_tracks_scan_tsr_noisy(carry, t_pair):
            x_t, x_track_t, step_rng = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_action, v_track = _get_velocity_with_tracks(x_t, x_track_t, t_curr)
            v_action = _apply_tsr(v_action, x_t, t_curr)
            v_track = _apply_tsr(v_track, x_track_t, t_curr)
            x_action_next = x_t + dt_i * v_action
            x_track_next = x_track_t + dt_i * v_track
            step_rng, noise_rng = jax.random.split(step_rng)
            noise_mag = sigma_step_max * jnp.sqrt(jnp.maximum(t_curr, 1e-8)) * jnp.sqrt(jnp.abs(dt_i))
            x_action_next = x_action_next + noise_mag * jax.random.normal(noise_rng, x_t.shape)
            x_track_next = x_track_next + noise_mag * jax.random.normal(noise_rng, x_track_t.shape)
            return (x_action_next, x_track_next, step_rng), None

        def step_tracks_scan_sde(carry, t_pair):
            x_t, x_track_t, step_rng = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_action, v_track = _get_velocity_with_tracks(x_t, x_track_t, t_curr)
            sigma_t = sde_noise_level * jnp.sqrt(t_curr / jnp.maximum(1.0 - t_curr, 1e-6))
            score_a = -(x_t + (1.0 - t_curr) * v_action) / jnp.maximum(t_curr, 1e-6)
            drift_a = (sigma_t**2 / 2.0) * score_a
            score_tr = -(x_track_t + (1.0 - t_curr) * v_track) / jnp.maximum(t_curr, 1e-6)
            drift_tr = (sigma_t**2 / 2.0) * score_tr
            step_rng, noise_rng = jax.random.split(step_rng)
            sqrt_dt_i = jnp.sqrt(jnp.abs(dt_i))
            x_action_next = x_t + dt_i * (v_action + drift_a) + sigma_t * sqrt_dt_i * jax.random.normal(noise_rng, x_t.shape)
            x_track_next = x_track_t + dt_i * (v_track + drift_tr) + sigma_t * sqrt_dt_i * jax.random.normal(noise_rng, x_track_t.shape)
            return (x_action_next, x_track_next, step_rng), None

        def step_tracks_howard(carry, t_pair):
            x_t, x_track_t, step_rng = carry
            t_curr, t_next = t_pair
            dt_i = t_next - t_curr
            v_action, v_track = _get_velocity_with_tracks(x_t, x_track_t, t_curr)
            x_action_next = x_t + dt_i * v_action
            x_track_next = x_track_t + dt_i * v_track
            step_rng, noise_rng = jax.random.split(step_rng)
            noise_mag = sigma_step_max * jnp.sqrt(jnp.maximum(t_curr, 1e-8)) * jnp.sqrt(jnp.abs(dt_i))
            x_action_next = x_action_next + noise_mag * jax.random.normal(noise_rng, x_t.shape)
            x_track_next = x_track_next + noise_mag * jax.random.normal(noise_rng, x_track_t.shape)
            return (x_action_next, x_track_next, step_rng), None

        def cond(carry):
            time = carry[-1]
            # robust to floating-point error
            return time >= -dt / 2

        if track_noise is not None:
            use_scan_tracks = (time_gamma != 1.0) or (sigma_step_max > 0.0) or (tsr_anneal != "none")

            if use_scan_tracks:
                t_linear = jnp.linspace(1.0, 0.0, num_steps + 1)
                t_grid = jnp.where(
                    time_gamma == 1.0, t_linear, jnp.sign(t_linear) * jnp.abs(t_linear) ** time_gamma
                )
                t_pairs = (t_grid[:-1], t_grid[1:])

                if sampling_mode == "tsr" and sigma_step_max > 0.0:
                    rng, combo_rng = jax.random.split(rng)
                    (x_0, x_track_0, _), _ = jax.lax.scan(step_tracks_scan_tsr_noisy, (noise, track_noise, combo_rng), t_pairs)
                elif sigma_step_max > 0.0:
                    rng, howard_rng = jax.random.split(rng)
                    (x_0, x_track_0, _), _ = jax.lax.scan(step_tracks_howard, (noise, track_noise, howard_rng), t_pairs)
                elif sampling_mode == "sde":
                    rng, sde_rng = jax.random.split(rng)
                    (x_0, x_track_0, _), _ = jax.lax.scan(step_tracks_scan_sde, (noise, track_noise, sde_rng), t_pairs)
                elif sampling_mode == "tsr":
                    (x_0, x_track_0), _ = jax.lax.scan(step_tracks_scan_tsr, (noise, track_noise), t_pairs)
                else:
                    (x_0, x_track_0), _ = jax.lax.scan(step_tracks_scan_ode, (noise, track_noise), t_pairs)
            else:
                if sampling_mode == "sde":
                    rng, sde_rng = jax.random.split(rng)
                    def cond_tracks_sde(carry):
                        return carry[-2] >= -dt / 2
                    x_0, x_track_0, _, _ = jax.lax.while_loop(cond_tracks_sde, step_tracks_sde, (noise, track_noise, 1.0, sde_rng))
                elif sampling_mode == "noisy_ode":
                    rng, noisy_rng = jax.random.split(rng)
                    def cond_tracks_noisy(carry):
                        return carry[-2] >= -dt / 2
                    x_0, x_track_0, _, _ = jax.lax.while_loop(cond_tracks_noisy, step_tracks_noisy_ode, (noise, track_noise, 1.0, noisy_rng))
                elif sampling_mode == "tsr":
                    x_0, x_track_0, _ = jax.lax.while_loop(cond, step_tracks_tsr, (noise, track_noise, 1.0))
                else:
                    x_0, x_track_0, _ = jax.lax.while_loop(cond, step_actions_and_tracks, (noise, track_noise, 1.0))

            return {"actions": x_0, "track_predictions": x_track_0}

        use_scan = (time_gamma != 1.0) or (sigma_step_max > 0.0) or (tsr_anneal != "none")

        if use_scan:
            t_linear = jnp.linspace(1.0, 0.0, num_steps + 1)
            t_grid = jnp.where(
                time_gamma == 1.0, t_linear, jnp.sign(t_linear) * jnp.abs(t_linear) ** time_gamma
            )
            t_pairs = (t_grid[:-1], t_grid[1:])

            if sampling_mode == "tsr" and sigma_step_max > 0.0:
                rng, combo_rng = jax.random.split(rng)
                (x_0, _), _ = jax.lax.scan(step_actions_scan_tsr_noisy, (noise, combo_rng), t_pairs)
            elif sigma_step_max > 0.0:
                rng, howard_rng = jax.random.split(rng)
                (x_0, _), _ = jax.lax.scan(step_actions_howard, (noise, howard_rng), t_pairs)
            elif sampling_mode == "sde":
                rng, sde_rng = jax.random.split(rng)
                (x_0, _), _ = jax.lax.scan(step_actions_scan_sde, (noise, sde_rng), t_pairs)
            elif sampling_mode == "tsr":
                x_0, _ = jax.lax.scan(step_actions_scan_tsr, noise, t_pairs)
            else:
                x_0, _ = jax.lax.scan(step_actions_scan_ode, noise, t_pairs)
        else:
            if sampling_mode == "sde":
                rng, sde_rng = jax.random.split(rng)
                def cond_sde(carry):
                    return carry[-2] >= -dt / 2
                x_0, _, _ = jax.lax.while_loop(cond_sde, step_actions_sde, (noise, 1.0, sde_rng))
            elif sampling_mode == "noisy_ode":
                rng, noisy_rng = jax.random.split(rng)
                def cond_noisy(carry):
                    return carry[-2] >= -dt / 2
                x_0, _, _ = jax.lax.while_loop(cond_noisy, step_actions_noisy_ode, (noise, 1.0, noisy_rng))
            elif sampling_mode == "tsr":
                x_0, _ = jax.lax.while_loop(cond, step_actions_tsr, (noise, 1.0))
            else:
                x_0, _ = jax.lax.while_loop(cond, step_actions_only, (noise, 1.0))
        return x_0