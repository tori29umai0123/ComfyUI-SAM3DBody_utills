# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""Pose Editor + terminal node.

Inputs: none. Everything happens in the popup editor (image upload, per-
person bbox + height + optional left/right hand crops, SAM 2 masking +
SAM 3D Body inference). On confirm the editor postMessages back to
ComfyUI a bundle of base64 PNG strings:

    pose_image    — single composite Three.js capture
    pose_images   — JSON array of per-person Three.js captures (transparent)
    input_image   — the raw uploaded image
    hand_l_images — JSON array of per-person left-hand crops
    hand_r_images — JSON array of per-person right-hand crops

JSON pose data is intentionally NOT exposed in this revision — multi-person
JSON layout is still being worked out. Downstream nodes consume the IMAGE /
IMAGE-list outputs directly.
"""

from __future__ import annotations

import base64
import io
import json
from typing import List

import numpy as np
import torch
from PIL import Image

from ..preset_pack import (
    clean_blender_exe_path,
    get_blender_exe_path,
    set_blender_exe_path,
)


def _placeholder_image_tensor() -> torch.Tensor:
    return torch.zeros(1, 1, 1, 3, dtype=torch.float32)


def _b64_to_image_tensor(b64_str: str) -> torch.Tensor:
    """Decode a (possibly data-URL prefixed) base64 PNG into a ComfyUI IMAGE
    tensor of shape (1, H, W, 3) float32 in [0, 1]. Empty / invalid input
    returns a 1×1 black tensor so downstream graphs see a real tensor.
    """
    if not b64_str:
        return _placeholder_image_tensor()
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]
    except Exception as exc:  # noqa: BLE001
        print(f"[SAM3DBody] Pose Editor +: image decode failed: {exc}")
        return _placeholder_image_tensor()


def _decode_b64_list(json_str: str) -> List[torch.Tensor]:
    """Decode a JSON-encoded array of base64 strings into a list of IMAGE
    tensors. Always returns at least one element (1×1 black) so ComfyUI's
    list-output runtime sees a non-empty list.
    """
    if not json_str:
        return [_placeholder_image_tensor()]
    try:
        decoded = json.loads(json_str)
    except (TypeError, ValueError) as exc:
        print(f"[SAM3DBody] Pose Editor +: JSON decode failed: {exc}")
        return [_placeholder_image_tensor()]
    if not isinstance(decoded, list) or not decoded:
        return [_placeholder_image_tensor()]
    return [_b64_to_image_tensor(s if isinstance(s, str) else "") for s in decoded]


class SAM3DBodyPoseEditorPlus:
    """Multi-person variant of the Pose Editor terminal node. No inputs —
    every parameter is set inside the editor popup. Outputs IMAGE tensors
    (single + lists) for the composite render, the input image, and the
    per-person hand crops the user uploaded inside the editor.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blender_exe": ("STRING", {
                    "default": get_blender_exe_path(),
                    "tooltip": "blender.exe のパス。エディタ内の人物別 FBX / BVH "
                               "出力で使用されます。\n"
                               "Windows の \"C:\\Program Files\\...\\blender.exe\" の"
                               "ように引用符を含む貼り付けでも自動で除去されます。\n"
                               "ノード実行時に config.ini [blender] exe_path に保存さ"
                               "れ、次回以降のデフォルトになります。",
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "pose_image":    ("STRING", {"default": ""}),  # composite base64 PNG
                "pose_images":   ("STRING", {"default": ""}),  # JSON array of per-person base64 PNGs
                "input_image":   ("STRING", {"default": ""}),  # source image base64 PNG
                "hand_l_images": ("STRING", {"default": ""}),  # JSON array of per-person base64 PNGs
                "hand_r_images": ("STRING", {"default": ""}),  # JSON array of per-person base64 PNGs
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "pose_image", "pose_images", "input_image",
        "hand_l_images", "hand_r_images",
    )
    OUTPUT_IS_LIST = (False, True, False, True, True)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "SAM3DBody/Editor"

    @classmethod
    def IS_CHANGED(cls, blender_exe="", node_id="", pose_image="", pose_images="",
                   input_image="", hand_l_images="", hand_r_images="", **_):
        # Length-fingerprint each payload so a fresh confirmation invalidates
        # the ComfyUI cache. Hashing the (often-large) base64 strings would
        # be more "correct" but costs more than it saves.
        return (
            f"img:{len(pose_image or '')}|"
            f"imgs:{len(pose_images or '')}|"
            f"in:{len(input_image or '')}|"
            f"hl:{len(hand_l_images or '')}|"
            f"hr:{len(hand_r_images or '')}|"
            f"be:{clean_blender_exe_path(blender_exe)}"
        )

    def execute(self, blender_exe="", node_id="", pose_image="", pose_images="",
                input_image="", hand_l_images="", hand_r_images="", **_):
        # Persist the cleaned Blender path so the in-editor FBX / BVH buttons
        # (and the next freshly-added export node) pick up the user's value.
        try:
            set_blender_exe_path(blender_exe)
        except Exception as exc:
            print(f"[SAM3DBody] config.ini blender path save failed: {exc}")

        if not pose_image:
            raise RuntimeError(
                "[SAM3DBody] Pose Editor +: ポーズが未確定です。"
                "ノード上の『Open Pose Editor +』ボタンを押し、"
                "エディタで『確定して閉じる』を押してください。\n"
                "Pose Editor +: pose has not been confirmed yet. "
                "Click 'Open Pose Editor +' on the node and press "
                "'Confirm & Close' inside the editor."
            )

        pose_img_t   = _b64_to_image_tensor(pose_image)
        input_img_t  = _b64_to_image_tensor(input_image)
        pose_imgs_l  = _decode_b64_list(pose_images)
        hand_l_l     = _decode_b64_list(hand_l_images)
        hand_r_l     = _decode_b64_list(hand_r_images)

        return (pose_img_t, pose_imgs_l, input_img_t, hand_l_l, hand_r_l)


NODE_CLASS_MAPPINGS = {
    "SAM3DBodyPoseEditorPlus": SAM3DBodyPoseEditorPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyPoseEditorPlus": "SAM 3D Body: Pose Editor +",
}
