# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Preset pack accessor — thin shim over ``nodes/preset_pack.py``.

The standalone upstream maintained its own preset_pack module under
``services/``; here we re-export the same public surface but resolve the
active pack via the project-level ``nodes/preset_pack.py`` (which reads
``config.ini`` at the repo root). This keeps every entrypoint —
the editor API, existing custom nodes, and the legacy
``/sam3d/autosave`` route — agreeing on a single resolver.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import paths

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresetPackPaths:
    pack_dir: Path
    npz_path: Path
    chara_settings_dir: Path


def active_pack_paths() -> PresetPackPaths:
    pack_dir = paths.active_preset_dir()
    return PresetPackPaths(
        pack_dir=pack_dir,
        npz_path=pack_dir / "face_blendshapes.npz",
        chara_settings_dir=pack_dir / "chara_settings_presets",
    )


def _valid_name(name: str) -> bool:
    if not name:
        return False
    return "/" not in name and "\\" not in name and ".." not in name


def list_presets() -> list[str]:
    d = active_pack_paths().chara_settings_dir
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if p.is_file())


def load_preset(name: str) -> dict[str, Any]:
    if not _valid_name(name):
        raise ValueError(f"invalid preset name: {name!r}")
    p = active_pack_paths().chara_settings_dir / f"{name}.json"
    if not p.is_file():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
