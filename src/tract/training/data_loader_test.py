import dataclasses

import jax
import numpy as np

from tract.models import pi0_config
from tract.training import config as _config
from tract.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_split_preprocessed_batch_actions_only():
    batch = {
        "state": np.zeros((2, 8), dtype=np.float32),
        "image": {
            "base_0_rgb": np.zeros((2, 224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((2, 224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((2, 224, 224, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.ones((2,), dtype=bool),
            "left_wrist_0_rgb": np.ones((2,), dtype=bool),
            "right_wrist_0_rgb": np.ones((2,), dtype=bool),
        },
        "tokenized_prompt": np.zeros((2, 8), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((2, 8), dtype=bool),
        "token_ar_mask": np.zeros((2, 8), dtype=np.int32),
        "token_loss_mask": np.zeros((2, 8), dtype=bool),
        "actions": np.zeros((2, 16, 32), dtype=np.float32),
    }
    obs, targets = _data_loader.split_preprocessed_batch(batch)
    assert np.asarray(obs.state).shape == (2, 8)
    assert np.asarray(targets).shape == (2, 16, 32)


def test_split_preprocessed_batch_with_tracks():
    batch = {
        "state": np.zeros((1, 78), dtype=np.float32),
        "image": {
            "base_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.ones((1,), dtype=bool),
            "left_wrist_0_rgb": np.ones((1,), dtype=bool),
            "right_wrist_0_rgb": np.ones((1,), dtype=bool),
        },
        "tokenized_prompt": np.zeros((1, 8), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((1, 8), dtype=bool),
        "token_ar_mask": np.zeros((1, 8), dtype=np.int32),
        "token_loss_mask": np.zeros((1, 8), dtype=bool),
        "actions": np.zeros((1, 16, 32), dtype=np.float32),
        "track_targets": np.zeros((1, 16, 78), dtype=np.float32),
    }
    obs, targets = _data_loader.split_preprocessed_batch(batch)
    assert np.asarray(obs.state).shape == (1, 78)
    assert sorted(targets.keys()) == ["actions", "track_targets"]
    assert np.asarray(targets["track_targets"]).shape == (1, 16, 78)
