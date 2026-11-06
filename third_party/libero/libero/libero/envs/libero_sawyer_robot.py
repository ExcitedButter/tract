"""Fixed Sawyer arm for ``*_sawyer`` LIBERO benchmarks.

Runtime maps ``Sawyer`` -> ``MountedSawyer`` / ``OnTheGroundSawyer`` per scene.
Base pose and home ``init_qpos`` are tuned in ``libero_sawyer_pose.py``.

Eval inits should match Franka (same object layout + EEF)::

    python scripts/regenerate_libero_sawyer_inits_from_panda.py --suite libero_spatial
"""

from __future__ import annotations


def resolve_sawyer_robot_names(_bddl_path: str | None = None) -> list[str]:
    # Keep canonical robot key; LIBERO / robosuite scene logic maps this to
    # MountedSawyer or OnTheGroundSawyer as needed.
    return ["Sawyer"]
