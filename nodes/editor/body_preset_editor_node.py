# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Body Preset Editor terminal node — analogous to PoseEditor but emits
``body_preset_json`` (body_params + bone_lengths + blendshapes) plus a
``body_preset_image`` IMAGE output captured from the editor's 3D viewer
(white background, full-body framing)."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image


def _b64_to_image_tensor(b64_str: str) -> torch.Tensor:
    """Decode a (possibly data-URL prefixed) base64 PNG string into a
    ComfyUI IMAGE tensor of shape (1, H, W, 3), float32 in [0, 1].
    Returns a 1×1 black tensor on miss/parse-failure."""
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
        print(f"[SAM3DBody] Body Preset Editor: image decode failed: {exc}")
        return torch.zeros(1, 1, 1, 3, dtype=torch.float32)


class SAM3DBodyBodyPresetEditor:
    """Terminal node that surfaces a confirmed body preset JSON string +
    a captured 3D viewer image (white bg, full-body framing)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "node_id": "UNIQUE_ID",
                "body_preset_json":  ("STRING", {"default": ""}),
                "body_preset_image": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("body_preset_json", "body_preset_image")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "SAM3DBody/Editor"

    @classmethod
    def IS_CHANGED(cls, node_id="", body_preset_json="", body_preset_image="", **_):
        return (body_preset_json or "<unset>") + f"|img:{len(body_preset_image or '')}"

    def execute(self, node_id="", body_preset_json="", body_preset_image="", **_):
        if not body_preset_json:
            raise RuntimeError(
                "[SAM3DBody] Body Preset Editor: ボディプリセットが未確定です。"
                "ノード上の『Open Body Preset Editor』ボタンを押し、"
                "エディタで『確定して閉じる』を押してください。\n"
                "Body Preset Editor: body preset has not been confirmed yet. "
                "Click 'Open Body Preset Editor' on the node and press "
                "'Confirm & Close' inside the editor."
            )
        body_preset_img_t = _b64_to_image_tensor(body_preset_image)
        return (body_preset_json, body_preset_img_t)


NODE_CLASS_MAPPINGS = {
    "SAM3DBodyBodyPresetEditor": SAM3DBodyBodyPresetEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyBodyPresetEditor": "SAM 3D Body: Body Preset Editor",
}
