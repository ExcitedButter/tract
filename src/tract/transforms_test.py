import numpy as np
import pytest

import tract.models.tokenizer as _tokenizer
import tract.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task": "Task from sample", "task_index": 1})
    assert data["prompt"] == "Task from sample"

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})

    with pytest.raises(ValueError, match='Invalid "task" value'):
        transform({"task": "", "task_index": 1})


def test_build_track_targets_mesh_timeline_both_views():
    transform = _transforms.BuildTrackTargets(
        mode="mesh_timeline_both_views",
        track_horizon=3,
    )

    # T=2 input, horizon=3 verifies padding behavior.
    agent_mesh = np.array(
        [
            [[1.0, 10.0]] * 7,
            [[2.0, 20.0]] * 7,
        ],
        dtype=np.float32,
    )
    wrist_mesh = np.array(
        [
            [[3.0, 30.0]] * 7,
            [[4.0, 40.0]] * 7,
        ],
        dtype=np.float32,
    )
    track_targets_raw = np.zeros((2, 39, 2), dtype=np.float32)
    track_targets_raw[0, 7:32, :] = np.array([5.0, 50.0], dtype=np.float32)
    track_targets_raw[1, 7:32, :] = np.array([6.0, 60.0], dtype=np.float32)

    out = transform(
        {
            "track_targets_raw": track_targets_raw,
            "agentview_mesh_vertices_2d": agent_mesh,
            "wrist_mesh_vertices_2d": wrist_mesh,
        }
    )
    targets = out["track_targets"]
    assert targets.shape == (3, 39 * 2)

    # Step 0
    step0 = targets[0].reshape(39, 2)
    assert np.allclose(step0[:7], np.array([[1.0, 10.0]] * 7))
    assert np.allclose(step0[7:32], np.array([[5.0, 50.0]] * 25))
    assert np.allclose(step0[32:39], np.array([[3.0, 30.0]] * 7))

    # Step 1
    step1 = targets[1].reshape(39, 2)
    assert np.allclose(step1[:7], np.array([[2.0, 20.0]] * 7))
    assert np.allclose(step1[7:32], np.array([[6.0, 60.0]] * 25))
    assert np.allclose(step1[32:39], np.array([[4.0, 40.0]] * 7))

    # Step 2 padded from last
    step2 = targets[2].reshape(39, 2)
    assert np.allclose(step2, step1)


def test_build_libero_query_state_uses_wrist_tracks():
    transform = _transforms.BuildLiberoQueryState()

    agent_mesh = np.array([[[1.0, 10.0]] * 7], dtype=np.float32)
    wrist_mesh = np.array([[[3.0, 30.0]] * 7], dtype=np.float32)
    wrist_tracks = np.zeros((1, 32, 2), dtype=np.float32)
    wrist_tracks[0, :25, :] = np.array([5.0, 50.0], dtype=np.float32)

    out = transform(
        {
            "agentview_mesh_vertices_2d": agent_mesh,
            "wrist_mesh_vertices_2d": wrist_mesh,
            "wrist_tracks": wrist_tracks,
        }
    )
    query_state = out["observation/query_state"]
    assert query_state.shape == (39 * 2,)

    points = query_state.reshape(39, 2)
    assert np.allclose(points[:7], np.array([[1.0, 10.0]] * 7))
    assert np.allclose(points[7:32], np.array([[5.0, 50.0]] * 25))
    assert np.allclose(points[32:39], np.array([[3.0, 30.0]] * 7))


def test_build_track_targets_requires_raw_track_targets_in_mesh_timeline_mode():
    transform = _transforms.BuildTrackTargets(
        mode="mesh_timeline_both_views",
        track_horizon=2,
    )

    agent_mesh = np.array(
        [
            [[1.0, 10.0]] * 7,
            [[2.0, 20.0]] * 7,
        ],
        dtype=np.float32,
    )
    wrist_mesh = np.array(
        [
            [[3.0, 30.0]] * 7,
            [[4.0, 40.0]] * 7,
        ],
        dtype=np.float32,
    )

    with pytest.raises(KeyError, match="Missing required raw wrist-grid future source key"):
        transform(
            {
                "agentview_mesh_vertices_2d": agent_mesh,
                "wrist_mesh_vertices_2d": wrist_mesh,
            }
        )
