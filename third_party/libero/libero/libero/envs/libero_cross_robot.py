"""Robot selection for ``*_cross`` LIBERO benchmarks (non-Franka arms, default UR5e)."""

from __future__ import annotations

import os

_SUPPORTED = {
    "panda": "Panda",
    "franka": "Panda",
    "ur5": "UR5e",
    "ur5e": "UR5e",
}

_UNSUPPORTED = {
    "sawyer": "Use the matching *_sawyer suite for Sawyer.",
    "widowx": "WidowX is not bundled in robosuite.",
    "window-x": "WidowX is not bundled in robosuite.",
    "windowx": "WidowX is not bundled in robosuite.",
}


def normalize_cross_robot_choice(choice: str | None = None) -> str:
    key = (
        choice
        or os.environ.get("LIBERO_CROSS_ROBOT")
        or os.environ.get("LIBERO_10_CROSS_ROBOT", "ur5e")
    ).strip().lower()
    if key in _UNSUPPORTED:
        raise ValueError(_UNSUPPORTED[key])
    if key not in _SUPPORTED:
        raise ValueError(
            f"Unknown cross-suite robot {choice!r}. "
            f"Supported: {sorted(_SUPPORTED)}. Set LIBERO_CROSS_ROBOT."
        )
    return _SUPPORTED[key]


def resolve_cross_robot_names(choice: str | None = None) -> list[str]:
    """Return robosuite robot list for OffScreenRenderEnv, e.g. ``['UR5e']``."""
    return [normalize_cross_robot_choice(choice)]
