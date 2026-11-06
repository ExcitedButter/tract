from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from tract.models import tokenizer as _tokenizer
from tract.shared import array_typing as at
from tract.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        # Prefer the per-sample task string when present. This avoids dependence on
        # potentially stale/corrupted task_index values in merged datasets.
        if "task" in data:
            task_value = data["task"]
            if isinstance(task_value, str) and task_value.strip():
                return {**data, "prompt": task_value}
            raise ValueError(f'Invalid "task" value for prompt extraction: {task_value!r}')

        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without either "task" or "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class BuildLiberoQueryState(DataTransformFn):
    """Builds query-state as agent mesh(7) + wrist grid(25) + wrist mesh(7)."""

    output_key: str = "observation/query_state"
    grid_points: int = 25

    def __call__(self, data: DataDict) -> DataDict:
        key_pairs = {
            "agent": ("observation/agentview_mesh_vertices_2d", "agentview_mesh_vertices_2d"),
            "wrist": ("observation/wrist_mesh_vertices_2d", "wrist_mesh_vertices_2d"),
            # Prefer canonical wrist_tracks; keep wrist_grid_mesh_32 as legacy compatibility alias.
            "wrist_grid": ("observation/wrist_tracks", "wrist_tracks", "observation/wrist_grid_mesh_32", "wrist_grid_mesh_32"),
        }

        def _pick(keys: tuple[str, ...]) -> str | None:
            for k in keys:
                if k in data:
                    return k
            return None

        selected = {name: _pick(keys) for name, keys in key_pairs.items()}
        missing = [name for name, key in selected.items() if key is None]
        if missing:
            raise KeyError(f"Missing required query-point inputs: {missing}")

        def _current_frame(arr: np.ndarray) -> np.ndarray:
            a = np.asarray(arr, dtype=np.float32)
            # If a temporal sequence is provided (T, P, 2), use the current frame (t=0)
            # for query-state construction.
            if a.ndim >= 3:
                return a[0]
            return a

        def _flatten(arr: np.ndarray) -> np.ndarray:
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 1:
                return a
            return a.reshape(*a.shape[:-2], -1) if a.ndim >= 2 else a

        agent = _flatten(_current_frame(data[selected["agent"]]))
        wrist = _flatten(_current_frame(data[selected["wrist"]]))
        grid_key = selected["wrist_grid"]
        assert grid_key is not None
        grid_raw = _current_frame(np.asarray(data[grid_key], dtype=np.float32))
        if grid_raw.ndim < 2 or grid_raw.shape[-2] < self.grid_points:
            raise ValueError(
                f"Grid mesh must have at least {self.grid_points} points, got shape {grid_raw.shape}"
            )
        grid = _flatten(grid_raw[..., : self.grid_points, :])

        data[self.output_key] = np.concatenate([agent, grid, wrist], axis=-1).astype(np.float32, copy=False)
        return data


@dataclasses.dataclass(frozen=True)
class BuildTrackTargets(DataTransformFn):
    """Builds per-step track targets.

    Modes:
    - raw_combined: use `track_targets_raw` directly (legacy)
    - mesh_timeline_both_views:
      agent future from mesh timeline, wrist grid future from `track_targets_raw`
      (preferred, points 7:32), and wrist mesh future from mesh timeline.
      Falls back to canonical wrist tracks if `track_targets_raw` is unavailable.
    - mesh_timeline_agent_plus_wrist_grid_donor:
      compatibility alias for `mesh_timeline_both_views`.
    """

    raw_key: str = "track_targets_raw"
    agent_key: str = "agentview_tracks"
    wrist_key: str = "wrist_tracks"
    agent_mesh_key: str = "agentview_mesh_vertices_2d"
    wrist_mesh_key: str = "wrist_mesh_vertices_2d"
    wrist_tracks_key: str = "wrist_tracks"
    wrist_tracks_fallback_key: str = "wrist_grid_mesh_32"
    target_key: str = "track_targets"
    drop_source: bool = False
    agent_points: int = 7
    wrist_points: int = 32  # 25 grid + 7 mesh
    wrist_grid_points: int = 25
    track_horizon: int | None = None
    mode: Literal["raw_combined", "mesh_timeline_both_views", "mesh_timeline_agent_plus_wrist_grid_donor"] = "raw_combined"

    @staticmethod
    def _ensure_seq(x: np.ndarray, *, key: str) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            return arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"{key} must be (T, P, 2) or (P, 2); got {arr.shape}")
        return arr

    @staticmethod
    def _flatten_or_reshape_targets(raw: np.ndarray) -> np.ndarray:
        if raw.ndim == 2:  # (T, D)
            return raw
        if raw.ndim == 3:  # (T, P, 2)
            return raw.reshape(raw.shape[0], -1)
        raise ValueError(f"track_targets_raw must be (T, D) or (T, P, 2); got {raw.shape}")

    @staticmethod
    def _pad_to_horizon(seq: np.ndarray, horizon: int) -> np.ndarray:
        if seq.shape[0] >= horizon:
            return seq[:horizon]
        pad_needed = horizon - seq.shape[0]
        pad = np.repeat(seq[-1:], pad_needed, axis=0)
        return np.concatenate([seq, pad], axis=0)

    def __call__(self, data: DataDict) -> DataDict:
        if self.mode == "raw_combined":
            # Legacy path: use prebuilt raw targets with explicit horizon.
            if self.raw_key not in data:
                raise KeyError(f"Missing raw track source key: {self.raw_key}")
            raw = np.asarray(data[self.raw_key], dtype=np.float32)
            combined = self._flatten_or_reshape_targets(raw)

            if self.track_horizon is not None:
                combined = self._pad_to_horizon(combined, self.track_horizon)

            data[self.target_key] = combined.astype(np.float32, copy=False)
            if self.drop_source:
                data.pop(self.raw_key, None)
            return data

        if self.mode in ("mesh_timeline_both_views", "mesh_timeline_agent_plus_wrist_grid_donor"):
            missing = [k for k in (self.agent_mesh_key, self.wrist_mesh_key) if k not in data]
            if missing:
                raise KeyError(f"Missing track source key(s): {missing}")

            if self.raw_key not in data:
                raise KeyError(
                    f"Missing required raw wrist-grid future source key: {self.raw_key!r}. "
                    "No fallback is allowed for mesh_timeline_both_views."
                )
            raw_tracks = self._ensure_seq(np.asarray(data[self.raw_key], dtype=np.float32), key=self.raw_key)

            agent_mesh = self._ensure_seq(np.asarray(data[self.agent_mesh_key], dtype=np.float32), key=self.agent_mesh_key)
            wrist_mesh = self._ensure_seq(np.asarray(data[self.wrist_mesh_key], dtype=np.float32), key=self.wrist_mesh_key)

            seq_for_horizon = [agent_mesh.shape[0], wrist_mesh.shape[0], raw_tracks.shape[0]]
            horizon = min(seq_for_horizon)
            if self.track_horizon is not None:
                horizon = min(horizon, self.track_horizon)
                if horizon <= 0:
                    raise ValueError("Track horizon must be > 0")

            agent_mesh = agent_mesh[:horizon, : self.agent_points]
            wrist_mesh = wrist_mesh[:horizon, : self.agent_points]
            if raw_tracks.shape[1] < (self.agent_points + self.wrist_grid_points):
                raise ValueError(
                    f"{self.raw_key} must have at least {self.agent_points + self.wrist_grid_points} points; "
                    f"got {raw_tracks.shape}"
                )
            wrist_grid = raw_tracks[:horizon, self.agent_points : self.agent_points + self.wrist_grid_points]
            combined_points = np.concatenate([agent_mesh, wrist_grid, wrist_mesh], axis=1)  # (T,39,2)
            combined = combined_points.reshape(combined_points.shape[0], -1)

            if self.track_horizon is not None:
                combined = self._pad_to_horizon(combined, self.track_horizon)

            data[self.target_key] = combined.astype(np.float32, copy=False)
            if self.drop_source:
                data.pop(self.raw_key, None)
                data.pop(self.agent_mesh_key, None)
                data.pop(self.wrist_mesh_key, None)
            return data

        raise ValueError(f"Unknown BuildTrackTargets mode: {self.mode}")

def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
