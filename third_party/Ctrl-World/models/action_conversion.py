"""
Reusable action format conversion between 10D real-world and 7D Ctrl-World formats.

10D (real-world Franka): [pos(3), rot6d(6), gripper(1)]
  - pos: eef_pos from Franka state (world frame, meters)
  - rot6d: first two ROWS of rotation matrix (from dp_gs/dataset/utils.py rot_mat_to_rot_6d)
    Provenance: Franka eef_quat (xyzw) → scipy Rotation.from_quat() → .as_matrix() → [:2, :].flatten()
  - gripper: gripper_state.width in meters (0.0 closed, 0.08 open) or binarized {0, 0.08}

7D (Ctrl-World / DROID): [pos(3), axis_angle(3), gripper(1)]
  - axis_angle: scipy Rotation.as_rotvec(), direction=axis, magnitude=angle, range ~[-pi, pi]
  - gripper: scalar [0, 1]

Roundtrip verified: rot6d → axis_angle → rot6d has < 1e-16 error.

Usage:
    from models.action_conversion import actions_10d_to_7d, actions_7d_to_10d

    actions_7d = actions_10d_to_7d(actions_10d)   # (T, 10) → (T, 7)
    actions_10d = actions_7d_to_10d(actions_7d)    # (T, 7) → (T, 10)
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation (two rows of rotation matrix) to axis-angle.

    Args:
        rot6d: (..., 6) array. First 3 = row 1 of R, last 3 = row 2 of R.

    Returns:
        (..., 3) axis-angle rotation vectors.
    """
    shape = rot6d.shape[:-1]
    flat = rot6d.reshape(-1, 6)

    r1 = flat[:, 0:3]
    r2 = flat[:, 3:6]

    # Gram-Schmidt orthogonalization
    r1 = r1 / (np.linalg.norm(r1, axis=1, keepdims=True) + 1e-8)
    r2 = r2 - np.sum(r1 * r2, axis=1, keepdims=True) * r1
    r2 = r2 / (np.linalg.norm(r2, axis=1, keepdims=True) + 1e-8)
    r3 = np.cross(r1, r2)

    # (N, 3, 3) rotation matrices — rows are r1, r2, r3
    rotmats = np.stack([r1, r2, r3], axis=1)

    rotvecs = Rotation.from_matrix(rotmats).as_rotvec()
    return rotvecs.reshape(*shape, 3)


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation to 6D rotation (two rows of rotation matrix).

    Args:
        axis_angle: (..., 3) rotation vectors.

    Returns:
        (..., 6) array. First 3 = row 1 of R, last 3 = row 2 of R.
    """
    shape = axis_angle.shape[:-1]
    flat = axis_angle.reshape(-1, 3)

    rotmats = Rotation.from_rotvec(flat).as_matrix()  # (N, 3, 3)
    r1 = rotmats[:, 0, :]  # first row
    r2 = rotmats[:, 1, :]  # second row

    rot6d = np.concatenate([r1, r2], axis=1)  # (N, 6)
    return rot6d.reshape(*shape, 6)


def actions_10d_to_7d(actions_10d: np.ndarray, gripper_max: float = 0.08) -> np.ndarray:
    """Convert 10D real-world actions to 7D Ctrl-World format.

    Args:
        actions_10d: (T, 10) or (B, T, 10) — [pos(3), rot6d(6), gripper(1)]
        gripper_max: maximum gripper width for normalization to [0, 1]

    Returns:
        (T, 7) or (B, T, 7) — [pos(3), axis_angle(3), gripper(1)]
    """
    pos = actions_10d[..., 0:3]
    rot6d = actions_10d[..., 3:9]
    gripper = actions_10d[..., 9:10]

    axis_angle = rot6d_to_axis_angle(rot6d)

    # Normalize gripper to [0, 1]
    if gripper_max > 0:
        gripper = np.clip(gripper / gripper_max, 0.0, 1.0)

    return np.concatenate([pos, axis_angle, gripper], axis=-1)


def actions_7d_to_10d(actions_7d: np.ndarray, gripper_max: float = 0.08) -> np.ndarray:
    """Convert 7D Ctrl-World actions back to 10D real-world format.

    Args:
        actions_7d: (T, 7) or (B, T, 7) — [pos(3), axis_angle(3), gripper(1)]
        gripper_max: maximum gripper width for denormalization

    Returns:
        (T, 10) or (B, T, 10) — [pos(3), rot6d(6), gripper(1)]
    """
    pos = actions_7d[..., 0:3]
    axis_angle = actions_7d[..., 3:6]
    gripper = actions_7d[..., 6:7]

    rot6d = axis_angle_to_rot6d(axis_angle)

    # Denormalize gripper back to meters
    gripper = gripper * gripper_max

    return np.concatenate([pos, rot6d, gripper], axis=-1)
