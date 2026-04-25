# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Pose Editor terminal node.

The frontend widget (``web/js/pose_editor_widget.js``) opens
``/sam3d/editor/pose`` in a popup, where the user runs the image →
segmentation → pose pipeline and confirms a result. The popup posts
``{type: "sam3d-pose-confirmed", node_id, pose_json}`` back to ComfyUI
which writes the JSON into this node's hidden ``pose_json`` widget; on
the next workflow run ``execute()`` hands the string out and also
surfaces the captured image's width / height as INT outputs (handy for
downstream nodes — Render width/height, EmptyImage backgrounds, etc. —
that otherwise have to parse the JSON themselves).
"""

from __future__ import annotations

import json


class SAM3DBodyPoseEditor:
    """Terminal node that surfaces a confirmed pose JSON string + the
    source image's pixel dimensions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "node_id": "UNIQUE_ID",
                "pose_json": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("pose_json", "width", "height")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "SAM3DBody/Editor"

    @classmethod
    def IS_CHANGED(cls, node_id="", pose_json="", **_):
        # Re-run whenever the confirmed JSON content changes.
        return pose_json or "<unset>"

    def execute(self, node_id="", pose_json="", **_):
        if not pose_json:
            raise RuntimeError(
                "[SAM3DBody] Pose Editor: ポーズが未確定です。"
                "ノード上の『Open Pose Editor』ボタンを押し、"
                "エディタで『確定して閉じる』を押してください。\n"
                "Pose Editor: pose has not been confirmed yet. "
                "Click 'Open Pose Editor' on the node and press "
                "'Confirm & Close' inside the editor."
            )
        # Pull width/height out of the confirmed JSON. The editor
        # frontend stores them under ``image_size``; we also accept the
        # legacy top-level ``width``/``height`` keys as a fallback so
        # workflows authored against earlier (or hand-edited) JSON keep
        # working. ``0`` lets downstream nodes treat it as "unspecified".
        width = 0
        height = 0
        try:
            payload = json.loads(pose_json) if pose_json else {}
            if isinstance(payload, dict):
                size = payload.get("image_size") or {}
                if isinstance(size, dict):
                    width  = int(size.get("width")  or 0)
                    height = int(size.get("height") or 0)
                if width <= 0:
                    width = int(payload.get("width") or 0)
                if height <= 0:
                    height = int(payload.get("height") or 0)
        except (ValueError, TypeError) as exc:
            print(f"[SAM3DBody] Pose Editor: image_size parse failed: {exc}")
        return (pose_json, max(0, width), max(0, height))


NODE_CLASS_MAPPINGS = {
    "SAM3DBodyPoseEditor": SAM3DBodyPoseEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyPoseEditor": "SAM 3D Body: Pose Editor",
}
