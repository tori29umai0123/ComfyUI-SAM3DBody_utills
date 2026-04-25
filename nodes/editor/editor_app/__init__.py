# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Editor backend services.

Ported from ``E:/SAM3DBody_utills/src/sam3dbody_app/services``. Path
resolution lives in ``paths.py`` — the rest of this package treats
``ComfyUI/models/`` as the model root and ``<repo>/presets/<active>/``
as the preset root, with a per-process tmp dir under
``<repo>/nodes/editor_tmp/``.
"""
