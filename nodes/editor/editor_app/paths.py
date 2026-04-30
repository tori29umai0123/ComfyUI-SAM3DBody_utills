# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Path resolution for the editor backend.

The standalone upstream resolves everything off the repo root; here we
defer to ComfyUI's ``folder_paths`` for model storage and reuse
``nodes/preset_pack.py`` for the preset pack root.

Layout:

  <ComfyUI>/models/sam3dbody/                 — SAM 3D Body weights
  <ComfyUI>/models/birefnet/BiRefNet_lite/    — BiRefNet weights (shared with nodes/processing/birefnet_mask.py)
  <repo>/presets/<active_pack>/...            — preset pack (resolved per call)
  <repo>/nodes/editor_tmp/{mesh.obj,mask.png} — single-file tmp artifacts
"""

from __future__ import annotations

from pathlib import Path

import folder_paths  # type: ignore[import-not-found]

from ...preset_pack import (  # type: ignore[import-not-found]
    active_pack_dir,
    body_preset_settings_dir,
    repo_root,
)

REPO_ROOT: Path = repo_root()
TMP_DIR: Path = REPO_ROOT / "nodes" / "editor_tmp"

MODELS_DIR: Path = Path(folder_paths.models_dir)
SAM3D_DIR: Path = MODELS_DIR / "sam3dbody"
# Match nodes/processing/birefnet_mask.py so the editor and the existing
# BiRefNet auto-mask node share a single model snapshot rather than each
# downloading their own ~170 MB copy. If you change one, change both.
BIREFNET_DIR: Path = MODELS_DIR / "birefnet" / "BiRefNet_lite"
# SAM 2 (hiera-tiny) snapshot — used by the Pose Editor + to produce
# per-person instance masks from user-drawn bounding boxes.
SAM2_DIR: Path = MODELS_DIR / "sam2" / "tiny"

for _d in (TMP_DIR, SAM3D_DIR, BIREFNET_DIR, SAM2_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def tmp_dir() -> Path:
    return TMP_DIR


def sam3d_dir() -> Path:
    return SAM3D_DIR


def birefnet_dir() -> Path:
    return BIREFNET_DIR


def sam2_dir() -> Path:
    return SAM2_DIR


def active_preset_dir() -> Path:
    return active_pack_dir()


def active_body_preset_settings_dir() -> Path:
    return body_preset_settings_dir()


# Public URL prefix the editor frontend uses to fetch tmp artifacts;
# matches the route registered in ``editor_server.py``.
TMP_URL_PREFIX = "/sam3d/editor/tmp"


def tmp_url(name: str, *, version: str | None = None) -> str:
    """Cache-bust-friendly URL for a file under ``tmp_dir()``."""
    base = f"{TMP_URL_PREFIX}/{name}"
    return f"{base}?v={version}" if version else base
