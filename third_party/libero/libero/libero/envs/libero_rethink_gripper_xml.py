"""Patched RethinkGripper MJCF for MuJoCo 3.6+ (mesh inertia=shell); stock XML on MuJoCo 2.x."""

from __future__ import annotations

import os
import re
from pathlib import Path

import mujoco
from robosuite.utils.mjcf_utils import xml_path_completion

_PATCHED: Path | None = None


def _mujoco_major() -> int:
    return int(str(mujoco.__version__).split(".", maxsplit=1)[0])


def patched_rethink_gripper_xml() -> str:
    global _PATCHED
    src = Path(xml_path_completion("grippers/rethink_gripper.xml"))
    if _mujoco_major() < 3:
        return str(src)

    if _PATCHED is not None and _PATCHED.is_file():
        return str(_PATCHED)

    asset_dir = src.parent.resolve()
    text = src.read_text()

    def _abs_mesh_path(match: re.Match[str]) -> str:
        rel = match.group(1)
        return f'file="{(asset_dir / rel).as_posix()}"'

    text = re.sub(
        r'file="((?:meshes|obj_meshes)/[^"]+)"',
        _abs_mesh_path,
        text,
    )

    def _mesh_inertia_shell(match: re.Match[str]) -> str:
        tag = match.group(0)
        if 'inertia="' in tag:
            return tag
        return tag[:-2] + ' inertia="shell"/>'

    text = re.sub(r'<mesh [^>]*/>', _mesh_inertia_shell, text)

    cache_root = Path(
        os.environ.get(
            "LIBERO_CONFIG_PATH",
            str(Path(__file__).resolve().parents[4] / ".libero_local"),
        )
    )
    out = cache_root / "rethink_gripper_patched.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    _PATCHED = out
    return str(out)
