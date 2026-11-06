"""Recolor Franka arm + gripper geoms for black-arm LIBERO suites (libero_10_arm, libero_object_arm)."""

from __future__ import annotations

import re

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

# Dark matte black (RGBA).
BLACK_ARM_RGBA = np.array([0.06, 0.06, 0.06, 1.0], dtype=np.float32)

_ROBOT_VIS_PATTERN = re.compile(r"^robot0_g\d+_vis$", re.IGNORECASE)

# PandaGripper is a separate MuJoCo subtree (not under robot0_*).
_GRIPPER_PREFIXES = ("gripper0_",)
_GRIPPER_EXCLUDE_SUBSTR = ("eef", "pad_collision")  # keep contact pads unchanged optional


def _geom_name(model, geom_id: int) -> str:
    if hasattr(model, "geom_id2name"):
        try:
            return model.geom_id2name(geom_id) or ""
        except (KeyError, ValueError):
            pass
    raw = getattr(model, "_model", None)
    if raw is not None and mujoco is not None:
        return mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    return ""


def should_recolor_arm_geom(geom_name: str, robot_prefix: str = "robot0") -> bool:
    """Panda arm (robot0_*) + gripper fingers/hand (gripper0_*)."""
    name = geom_name or ""
    name_l = name.lower()

    if _ROBOT_VIS_PATTERN.match(name):
        return True

    if name_l.startswith(robot_prefix.lower()):
        if "mount" in name_l or "camera" in name_l:
            return False
        return "collision" in name_l and "link" in name_l

    for gp in _GRIPPER_PREFIXES:
        if name_l.startswith(gp):
            if any(x in name_l for x in _GRIPPER_EXCLUDE_SUBSTR):
                return False
            return True

    return False


def _write_geom_black(model, geom_id: int) -> None:
    model.geom_rgba[geom_id] = BLACK_ARM_RGBA
    model.geom_matid[geom_id] = -1
    raw = getattr(model, "_model", None)
    if raw is not None:
        raw.geom_rgba[geom_id] = BLACK_ARM_RGBA
        raw.geom_matid[geom_id] = -1


def apply_black_robot_arm(sim, robot_prefix: str = "robot0") -> int:
    """Black out Panda arm meshes and PandaGripper (gripper0_*)."""
    model = sim.model
    count = 0
    for geom_id in range(model.ngeom):
        name = _geom_name(model, geom_id)
        if not should_recolor_arm_geom(name, robot_prefix=robot_prefix):
            continue
        _write_geom_black(model, geom_id)
        count += 1

    if count and hasattr(sim, "forward"):
        sim.forward()
    return count
