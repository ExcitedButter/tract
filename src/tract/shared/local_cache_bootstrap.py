"""Default HuggingFace / LeRobot / OpenPI cache locations (matches ``scripts/train.py``).

Import this **before** ``lerobot``, ``huggingface_hub``, or ``tract.training.data_loader``
so ``HF_LEROBOT_HOME`` and friends are set before those packages read them.

Override the base directory with env ``OPENPI_CACHE_BASE`` (defaults to the scratch
layout used in this repo's trainer).
"""

from __future__ import annotations

import os
import pathlib

# Repo root (src/tract/shared/ -> repo); keeps caches independent of the cwd.
_OPENPI_CACHE_BASE = os.environ.get("OPENPI_CACHE_BASE", str(pathlib.Path(__file__).resolve().parents[3]))

os.environ.setdefault("OPENPI_DATA_HOME", _OPENPI_CACHE_BASE)
os.environ.setdefault("HF_HOME", f"{_OPENPI_CACHE_BASE}/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_OPENPI_CACHE_BASE}/hf_cache")
os.environ.setdefault("HF_LEROBOT_HOME", f"{_OPENPI_CACHE_BASE}/lerobot_cache")
os.environ.setdefault("TFDS_DATA_DIR", f"{_OPENPI_CACHE_BASE}/tfds_data")
