"""Sawyer pose aligned with UR5e (``libero_10_cross``) and Franka base offsets in LIBERO.

``base_xpos_offset`` matches ``MountedUR5e`` / ``OnTheGroundUR5e``. Joint homes are IK-calibrated
to the UR5e EEF on ``libero_10_cross`` (see ``scripts/calibrate_libero_sawyer_init.py --base-suite libero_10_cross``).
"""

from __future__ import annotations

import numpy as np

# Franka home used by LIBERO (MountedPanda / OnTheGroundPanda).
PANDA_LIBERO_INIT_QPOS = np.array(
    [0.0, -0.16103739, 0.0, -2.44459747, 0.0, 2.2267522, np.pi / 4],
    dtype=np.float64,
)

# Sawyer on-the-ground: libero_object / floor scenes (arena_type=floor, base empty).
# IK to match OnTheGroundPanda EEF pos + gripper axis on libero_object task 0
# (old copy of Panda joints with wrist=-1.57 pointed gripper upward).
SAWYER_INIT_QPOS_FLOOR = np.array(
    [
        -0.09584186,
        -1.34240374,
        2.36772609,
        -1.82897968,
        3.14,
        1.1146124,
        -2.20294191,
    ],
    dtype=np.float64,
)

# Sawyer on-the-ground: living-room tabletop (arena_type=living_room, z~0.42).
SAWYER_INIT_QPOS_LIVING_ROOM = np.array(
    [
        -0.20176912,
        -1.40403526,
        -0.15525753,
        2.07884272,
        0.0039564,
        0.92203027,
        -1.74108171,
    ],
    dtype=np.float64,
)

# Backward-compatible alias (living-room calibration).
SAWYER_INIT_QPOS_ON_GROUND = SAWYER_INIT_QPOS_LIVING_ROOM

# Kitchen-style mounted tables (libero_10_sawyer kitchen tasks).
SAWYER_INIT_QPOS_MOUNTED = np.array(
    [
        -0.07654566,
        -1.56804598,
        2.10952505,
        -2.08657761,
        3.05206647,
        1.08574635,
        -2.44337762,
    ],
    dtype=np.float64,
)

# Study mounted table (same calibration target; slightly worse FK match than kitchen).
SAWYER_INIT_QPOS_MOUNTED_STUDY = np.array(
    [
        -0.06137142,
        -1.62650068,
        2.01549683,
        -2.16443112,
        3.00912749,
        1.03177198,
        -2.5079558,
    ],
    dtype=np.float64,
)


def sawyer_init_qpos_floor() -> np.ndarray:
    return SAWYER_INIT_QPOS_FLOOR.copy()


def sawyer_init_qpos_living_room() -> np.ndarray:
    return SAWYER_INIT_QPOS_LIVING_ROOM.copy()


def sawyer_init_qpos_on_ground() -> np.ndarray:
    return sawyer_init_qpos_living_room()


def sawyer_init_qpos_mounted(*, study_table: bool = False) -> np.ndarray:
    if study_table:
        return SAWYER_INIT_QPOS_MOUNTED_STUDY.copy()
    return SAWYER_INIT_QPOS_MOUNTED.copy()


def sawyer_table_x_offset(table_length: float, *, front_margin: float = 0.16) -> tuple[float, float, float]:
    """Match MountedPanda ``kitchen_table`` / ``table`` base placement."""
    return (-front_margin - table_length / 2, 0.0, 0.0)


def sawyer_study_table_x_offset(table_length: float) -> tuple[float, float, float]:
    """Match MountedPanda ``study_table`` base placement."""
    return (-0.25 - table_length / 2, 0.0, 0.0)


def sawyer_living_room_x_offset(table_length: float) -> tuple[float, float, float]:
    return (-0.16 - table_length / 2, 0.0, 0.42)
