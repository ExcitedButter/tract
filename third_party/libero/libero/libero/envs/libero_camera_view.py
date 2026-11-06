"""Camera viewpoint helpers (LIBERO-plus Camera Viewpoints compatibility)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def scale_distance_from_pivot(
    original_quat=None, original_pos=None, scale_factor: float = 1.3
):
    result = {}
    pivot_point = np.array([0, 0, 0.8])
    if original_quat is not None:
        result["new_quat"] = original_quat
    if original_pos is not None:
        original_pos = np.array(original_pos)
        vec_to_point = original_pos - pivot_point
        scaled_vec = vec_to_point * scale_factor
        new_pos = pivot_point + scaled_vec
        result["new_pos"] = (
            new_pos.tolist() if isinstance(new_pos, np.ndarray) else new_pos
        )
    return result


def rotate_around_y(original_quat=None, original_pos=None, degrees=0):
    result = {}
    axis = np.array([0, 1, 0])
    axis_point = np.array([0, 0, 0.8])
    custom_rotation = Rotation.from_rotvec(np.radians(-degrees) * axis)
    if original_quat is not None:
        original_rot = Rotation.from_quat(
            [original_quat[1], original_quat[2], original_quat[3], original_quat[0]]
        )
        combined_rot = custom_rotation * original_rot
        new_quat = combined_rot.as_quat()
        result["new_quat"] = [
            float(new_quat[3]),
            float(new_quat[0]),
            float(new_quat[1]),
            float(new_quat[2]),
        ]
    if original_pos is not None:
        translated_pos = np.array(original_pos) - axis_point
        rotated_pos = custom_rotation.apply(translated_pos)
        final_pos = rotated_pos + axis_point
        result["new_pos"] = (
            final_pos.tolist() if isinstance(final_pos, np.ndarray) else final_pos
        )
    return result


def rotate_around_z(original_quat=None, original_pos=None, degrees=0):
    result = {}
    z_rotation = Rotation.from_euler("z", degrees, degrees=True)
    if original_quat is not None:
        original_rot = Rotation.from_quat(
            [original_quat[1], original_quat[2], original_quat[3], original_quat[0]]
        )
        combined_rot = z_rotation * original_rot
        new_quat = combined_rot.as_quat()
        result["new_quat"] = [
            float(new_quat[3]),
            float(new_quat[0]),
            float(new_quat[1]),
            float(new_quat[2]),
        ]
    if original_pos is not None:
        rotated_pos = z_rotation.apply(original_pos)
        result["new_pos"] = (
            rotated_pos.tolist() if isinstance(rotated_pos, np.ndarray) else rotated_pos
        )
    return result


def apply_agentview_viewpoint(
    mujoco_arena,
    *,
    pos_av: list[float],
    quat_av: list[float],
    horizon_view,
    vertical_view,
    scale_factor: float,
    end_point_rot,
    end_point_vertical,
) -> None:
    up_view = vertical_view
    view = horizon_view

    if int(up_view) != 0:
        result_up = rotate_around_y(
            original_quat=quat_av, original_pos=pos_av, degrees=int(up_view)
        )
        pos_up = [round(x, 4) for x in result_up["new_pos"]]
        quat_up = [round(x, 4) for x in result_up["new_quat"]]
        result_view = rotate_around_z(
            original_quat=quat_up, original_pos=pos_up, degrees=int(view)
        )
        pos_view = [round(x, 4) for x in result_view["new_pos"]]
        quat_view = [round(x, 4) for x in result_view["new_quat"]]
    else:
        result_view = rotate_around_z(
            original_quat=quat_av, original_pos=pos_av, degrees=int(view)
        )
        pos_view = [round(x, 4) for x in result_view["new_pos"]]
        quat_view = [round(x, 4) for x in result_view["new_quat"]]

    if float(scale_factor) != 1.0:
        result = scale_distance_from_pivot(
            original_quat=quat_view,
            original_pos=pos_view,
            scale_factor=float(scale_factor),
        )
        pos_view = [round(x, 4) for x in result["new_pos"]]
        quat_view = [round(x, 4) for x in result["new_quat"]]

    if int(end_point_rot) != 0:
        result_view = rotate_around_z(original_quat=quat_view, degrees=int(end_point_rot))
        quat_view = [round(x, 4) for x in result_view["new_quat"]]
    if int(end_point_vertical) != 0:
        result_view = rotate_around_y(
            original_quat=quat_view, degrees=int(end_point_vertical)
        )
        quat_view = [round(x, 4) for x in result_view["new_quat"]]

    mujoco_arena.set_camera(
        camera_name="agentview",
        pos=pos_view,
        quat=quat_view,
    )
