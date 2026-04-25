# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""ComfyUI SAM 3D Body - Robust Full-Body Human Mesh Recovery."""

from comfy_env import wrap_nodes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

wrap_nodes()

# Server routes for API endpoints
try:
    from .nodes import server  # /sam3d/autosave, /sam3d/preset/{name}
except Exception:
    pass

# Editor page + API routes (/sam3d/editor/*, /sam3d/api/*)
try:
    from .nodes.editor import editor_server  # noqa: F401
except Exception as _exc:
    print(f"[SAM3DBody] editor routes failed to load: {_exc}")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
