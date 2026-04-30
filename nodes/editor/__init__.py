# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""SAM 3D Body editor nodes (Pose / Body Preset).

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
from .pose_plus_editor_node import (
    NODE_CLASS_MAPPINGS as POSE_PLUS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as POSE_PLUS_DISPLAY_MAPPINGS,
)
from .body_preset_editor_node import (
    NODE_CLASS_MAPPINGS as BODY_PRESET_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BODY_PRESET_DISPLAY_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {**POSE_MAPPINGS, **POSE_PLUS_MAPPINGS, **BODY_PRESET_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {
    **POSE_DISPLAY_MAPPINGS,
    **POSE_PLUS_DISPLAY_MAPPINGS,
    **BODY_PRESET_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
