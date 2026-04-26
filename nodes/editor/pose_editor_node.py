# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Pose Editor terminal node.

The frontend widget (``web/js/pose_editor_widget.js``) opens
``/sam3d/editor/pose`` in a popup, where the user runs the image →
segmentation → pose pipeline and confirms a result. The popup posts
``{type: "sam3d-pose-confirmed", node_id, pose_json, pose_image,
input_image, hand_l_image, hand_r_image}`` back to ComfyUI which writes
the JSON / base64 PNG strings into this node's hidden widgets; on the
next workflow run ``execute()`` hands the JSON out, surfaces the
captured image's width / height as INT outputs, and decodes the base64
PNGs into ComfyUI IMAGE tensors:

* ``pose_image``   — the 3D viewer capture, optionally cropped.
* ``input_image``  — the raw uploaded image.
* ``hand_l_image`` — the user-supplied left hand crop (after any
  in-editor mirroring), empty if none was provided.
* ``hand_r_image`` — same for the right hand.
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import torch
from PIL import Image


def _b64_to_image_tensor(b64_str: str) -> torch.Tensor:
    """Decode a (possibly data-URL prefixed) base64 PNG string into a
    ComfyUI IMAGE tensor of shape (1, H, W, 3), float32 in [0, 1].

    Returns a 1×1 black tensor when the string is empty / invalid so the
    downstream graph still sees a tensor instead of None — Render / Save
    nodes throw less helpful errors when handed a bare ``None``.
    """
    if not b64_str:
        return torch.zeros(1, 1, 1, 3, dtype=torch.float32)
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]
    except Exception as exc:  # noqa: BLE001
        print(f"[SAM3DBody] Pose Editor: image decode failed: {exc}")
        return torch.zeros(1, 1, 1, 3, dtype=torch.float32)


class SAM3DBodyPoseEditor:
    """Terminal node that surfaces a confirmed pose JSON string + the
    source image's pixel dimensions + captured pose / input images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "node_id": "UNIQUE_ID",
                "pose_json":    ("STRING", {"default": ""}),
                "pose_image":   ("STRING", {"default": ""}),
                "input_image":  ("STRING", {"default": ""}),
                "hand_l_image": ("STRING", {"default": ""}),
                "hand_r_image": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "pose_json", "width", "height",
        "pose_image", "input_image", "hand_l_image", "hand_r_image",
    )
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "SAM3DBody/Editor"

    @classmethod
    def IS_CHANGED(cls, node_id="", pose_json="", pose_image="", input_image="",
                   hand_l_image="", hand_r_image="", **_):
        # Re-run whenever any of the captured payloads change. Hashing
        # the (often-large) base64 images would be cheaper to send to the
        # cache layer, but ComfyUI hashes the IS_CHANGED return itself —
        # piping the lengths in is enough to invalidate.
        return (
            (pose_json or "<unset>")
            + f"|img:{len(pose_image or '')}"
            + f"|in:{len(input_image or '')}"
            + f"|hl:{len(hand_l_image or '')}"
            + f"|hr:{len(hand_r_image or '')}"
        )

    def execute(self, node_id="", pose_json="", pose_image="", input_image="",
                hand_l_image="", hand_r_image="", **_):
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
        pose_img_t   = _b64_to_image_tensor(pose_image)
        input_img_t  = _b64_to_image_tensor(input_image)
        hand_l_img_t = _b64_to_image_tensor(hand_l_image)
        hand_r_img_t = _b64_to_image_tensor(hand_r_image)
        return (
            pose_json, max(0, width), max(0, height),
            pose_img_t, input_img_t, hand_l_img_t, hand_r_img_t,
        )


NODE_CLASS_MAPPINGS = {
    "SAM3DBodyPoseEditor": SAM3DBodyPoseEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyPoseEditor": "SAM 3D Body: Pose Editor",
}
