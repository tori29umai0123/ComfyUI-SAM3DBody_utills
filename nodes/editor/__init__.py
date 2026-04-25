# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""SAM 3D Body editor nodes (Pose / Character).

Each editor node ships an HTML page that mirrors the corresponding tab
of the upstream standalone app (``E:/SAM3DBody_utills``) and stores its
confirmed JSON in a hidden widget. The Python side is intentionally
thin: ``execute()`` just returns the cached JSON or raises if the user
hasn't confirmed yet.
"""

from .pose_editor_node import (
    NODE_CLASS_MAPPINGS as POSE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as POSE_DISPLAY_MAPPINGS,
)
from .character_editor_node import (
    NODE_CLASS_MAPPINGS as CHAR_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CHAR_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {**POSE_MAPPINGS, **CHAR_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**POSE_DISPLAY_MAPPINGS, **CHAR_DISPLAY_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
