"""Patched Sawyer MJCF: MuJoCo 3.x needs head-mesh drop + shell inertia; 2.x uses stock XML."""

from __future__ import annotations

import os
import re
from pathlib import Path

import mujoco
from robosuite.utils.mjcf_utils import xml_path_completion

_PATCHED: Path | None = None


def _mujoco_major() -> int:
    return int(str(mujoco.__version__).split(".", maxsplit=1)[0])


def patched_sawyer_robot_xml() -> str:
    """Return Sawyer robot.xml path compatible with the installed MuJoCo."""
    global _PATCHED
    src = Path(xml_path_completion("robots/sawyer/robot.xml"))
    if _mujoco_major() < 3:
        return str(src)

    if _PATCHED is not None and _PATCHED.is_file():
        return str(_PATCHED)

    mesh_dir = src.parent.resolve()
    text = src.read_text()

    # Head display meshes trigger "mesh volume too small" in MuJoCo 3.6; not needed for control.
    text = re.sub(
        r'^\s*<mesh name="head_[^"]+"[^>]*/>\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^\s*<geom mesh="head_[^"]+"[^>]*/>\s*\n',
        "",
        text,
        flags=re.MULTILINE,
    )

    def _abs_mesh_path(match: re.Match[str]) -> str:
        rel = match.group(1)
        return f'file="{(mesh_dir / rel).as_posix()}"'

    text = re.sub(r'file="(obj_meshes/[^"]+)"', _abs_mesh_path, text)

    def _mesh_inertia_shell(match: re.Match[str]) -> str:
        tag = match.group(0)
        if 'inertia="' in tag:
            return tag
        return tag[:-2] + ' inertia="shell"/>'

    text = re.sub(r'<mesh name="[^"]+" file="[^"]+"[^>]*/>', _mesh_inertia_shell, text)

    cache_root = Path(
        os.environ.get(
            "LIBERO_CONFIG_PATH",
            str(Path(__file__).resolve().parents[4] / ".libero_local"),
        )
    )
    out = cache_root / "sawyer_robot_patched_mujoco3.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    _PATCHED = out
    return str(out)
